import copy
import logging
import math
import argparse
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm



def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer.num_heads
        self.attention_head_size = int(config.dim / self.num_attention_heads)  # 768 / 6 = 128
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.dim, self.all_head_size)
        self.key = Linear(config.dim, self.all_head_size)
        self.value = Linear(config.dim, self.all_head_size)

        self.out = Linear(config.dim, config.dim)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.num_attention_heads, config.n_classes, config.n_classes))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores + self.position_embeddings  # RPE

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.dim, config.hidden_size)
        self.fc2 = Linear(config.hidden_size, config.dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()

        self.attention_norm = LayerNorm(config.dim, eps=1e-6)
        self.ffn_norm = LayerNorm(config.dim, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class self_attention(nn.Module):
    def __init__(self, config):
        super(self_attention, self).__init__()
        num_layers = 1
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.dim, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, class_center):
        for layer_block in self.layer:
            class_center = layer_block(class_center)

        encoded = self.encoder_norm(class_center)
        
        return encoded



# center: B * N * C
# feature_embeddings: B * C * H * W * L
# mask_embeddings: B * N * H * W * L
def center_update(value, soft_embeddings, center):
    center_residual = torch.matmul(soft_embeddings.flatten(2), value.flatten(2).transpose(-1, -2))
    center = center + center_residual
    return center



class CCM(nn.Module):
    def __init__(self, config, in_channel):
        super(CCM, self).__init__()
        self.SA = self_attention(config)
        self.key_projection = DecoderResBlock(in_channel, config.dim)
        self.value_projection = DecoderResBlock(in_channel, config.dim)
        self.resblock3 = DecoderResBlock(config.n_classes, config.n_classes)
        self.resblock4 = DecoderResBlock(config.n_classes, config.n_classes)

        self.classes = config.n_classes
        self.softmax = Softmax(dim=-2)                # softmax on the dimension of class numbers
        
    def forward(self, feature, class_center):
        b, c, h, w, l = feature.size()
        class_center = self.SA(class_center)
        key = self.key_projection(feature)
        value = self.value_projection(feature)

        mask_embeddings = torch.matmul(class_center, key.flatten(2))
        mask_embeddings = mask_embeddings / math.sqrt(self.classes)
        soft_embeddings = self.softmax(mask_embeddings)
        
        soft_embeddings = soft_embeddings.contiguous().view(b, self.classes, h, w, l)
        mask_embeddings = soft_embeddings + self.resblock4(self.resblock3(soft_embeddings))
        refined_center = center_update(value, soft_embeddings, class_center)
        return mask_embeddings, refined_center




class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv3 = Conv3dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):

        feature_in = self.conv3(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + feature_in
        x = self.relu(x)

        return x




class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, bn, relu)

class Conv3dbn(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)


