# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import argparse
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm
from torch.nn.modules.utils import _pair, _triple
from scipy import ndimage
from . import vit_seg_configs as configs
# import vit_seg_configs as configs
from . vit_seg_modeling_resnet_skip import ResNetV2
# from vit_seg_modeling_resnet_skip import ResNetV2

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_attention_heads, config.n_patches, config.n_patches))

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

        # print(query_layer.shape, key_layer.shape)            # torch.Size([2, 12, 700, 64]) torch.Size([2, 12, 700, 64])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print(attention_scores.shape)                                       # torch.Size([2, 12, 700, 700])
        
        attention_scores = attention_scores + self.position_embeddings
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
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

class edge_detect(nn.Module):
    def __init__(self, config, in_channels=3, use_batchnorm=True):
        super(edge_detect, self).__init__()
        out_channels = [16, 32, 32]
        outputs = 64
        self.outputs = outputs
        num_heads = 2
        ratio = 16
        self.ratio = ratio
        self.normalize = True
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels[0],
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels[0],
            out_channels[1],
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv3dReLU(
            config.skip_channels[2],         # 64
            out_channels[1],
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = Conv3dReLU(
            out_channels[1],
            out_channels[2],
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv5 = Conv3dReLU(
            out_channels[2] + out_channels[1],
            num_heads,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.deep_q = nn.Conv3d(out_channels[2] + out_channels[1], out_channels[0], kernel_size=1)
        self.deep_k = nn.Conv3d(out_channels[2] + out_channels[1], out_channels[0], kernel_size=1)
        self.deep_v = Conv3dReLU(out_channels[2] + out_channels[1], outputs, kernel_size=1, padding=0, use_batchnorm=use_batchnorm)
        self.softmax = Softmax(dim=-1)
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.down = nn.AvgPool3d(kernel_size=ratio, stride=ratio)
        
        
    def forward(self, x0, x1):
        # x0: features[3]   x1: features[2]
        b, c, l, h, w = x0.size()
        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        
        x1 = self.up(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        
   
        combine = torch.cat((x0, x1), 1)
        edge = self.conv5(combine)
        
        combine = self.down(combine)
        query = self.deep_q(combine).flatten(2).permute(0, 2, 1)  # b * n * c
        key = self.deep_k(combine).flatten(2)  # b * c * n
        value = self.deep_v(combine).flatten(2).permute(0, 2, 1)
        
        attention_map = torch.matmul(query, key)
        if self.normalize:
            attention_map = attention_map * (1. / query.size(2))
        # print(attention_map.shape)

        attention_map = self.softmax(attention_map)
        edge_features = torch.matmul(attention_map, value).permute(0, 2, 1).contiguous().view(b, self.outputs, np.int(l / self.ratio),
                                                                                              np.int(h / self.ratio),
                                                                                              np.int(w / self.ratio))

        return edge, edge_features




class edge_detect_module(nn.Module):
    def __init__(self, config, in_channels=3, use_batchnorm=True):
        super(edge_detect_module, self).__init__()
        out_channels = [16, 32, 32, 64]
        outputs = 64
        num_heads = 2
        self.conv1 = Conv3dReLU(
            in_channels,
            out_channels[0],
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels[0],
            out_channels[1],
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv3dReLU(
            config.skip_channels[2],         # 64
            out_channels[2],
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = Conv3dReLU(
            out_channels[2],
            out_channels[3],
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv5 = Conv3dReLU(
            out_channels[3] + out_channels[1],
            num_heads,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv6 = Conv3dReLU(
            out_channels[3] + out_channels[1],
            outputs,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.down = nn.AvgPool3d(kernel_size=16, stride=16)

    def forward(self, x0, x1):      
        # x0: features[3]   x1: features[2]
        x0 = self.conv1(x0)
        x0 = self.conv2(x0)
        
        x1 = self.up(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        
        # print(x0.shape, x1.shape)
        edge_features = self.conv6(torch.cat((x0, x1), 1))
        edge_features = self.down(edge_features)
        edge = self.conv5(torch.cat((x0, x1), 1))
        return edge, edge_features

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config


        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1], img_size[2] // 16 // grid_size[2])     # patch size: (1, 1, 1)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16, patch_size[2] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) * (img_size[2] // patch_size_real[2])
            self.hybrid = True
        else:
            patch_size = _triple(config.patches["size"])              # _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16              # 1024
        self.patch_embeddings = Conv3d(in_channels=in_channels,        # + config.skip_channels[0] + config.skip_channels[1] + config.skip_channels[2],
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.dropout = Dropout(config.transformer["dropout_rate"])

        

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None


        x = self.patch_embeddings(x)

        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x
        embeddings = self.dropout(embeddings)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        # with torch.no_grad():
        query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t().cuda()
        key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t().cuda()
        value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t().cuda()
        out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t().cuda()

        query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1).cuda()
        key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1).cuda()
        value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1).cuda()
        out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1).cuda()

        
        self.attn.query.weight = nn.Parameter(query_weight)
        self.attn.key.weight = nn.Parameter(key_weight)
        self.attn.value.weight = nn.Parameter(value_weight)
        self.attn.out.weight = nn.Parameter(out_weight)
        self.attn.query.bias = nn.Parameter(query_bias)
        self.attn.key.bias = nn.Parameter(key_bias)
        self.attn.value.bias = nn.Parameter(value_bias)
        self.attn.out.bias = nn.Parameter(out_bias)

        self.attention_norm.weight = nn.Parameter(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]).cuda())
        self.attention_norm.bias = nn.Parameter(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]).cuda())
        
        # with torch.no_grad():
        mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t().cuda()
        mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t().cuda()
        mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t().cuda()
        mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t().cuda()

        self.ffn.fc1.weight = nn.Parameter(mlp_weight_0)
        self.ffn.fc2.weight = nn.Parameter(mlp_weight_1)
        self.ffn.fc1.bias = nn.Parameter(mlp_bias_0)
        self.ffn.fc2.bias = nn.Parameter(mlp_bias_1)


        self.ffn_norm.weight = nn.Parameter(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]).cuda())
        self.ffn_norm.bias = nn.Parameter(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]).cuda())


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        # attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder1 = Encoder(config, vis)
        self.encoder2 = Encoder(config, vis)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.n_patches, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])
 

    def forward(self, input_ids, global_):
        embedding_output, features = self.embeddings(input_ids)               # return the learnable positional embedding
        
        
        
        encoded = self.encoder1(embedding_output) + self.encoder2(self.dropout(global_ + self.position_embeddings))  # (B, n_patch, hidden)

        return encoded, features


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
        # relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm3d(out_channels)

        super(Conv3dbn, self).__init__(conv, bn)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
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
            out_channels,
            out_channels,
            kernel_size=1,          # 1
            padding=0,   # 0
            use_batchnorm=use_batchnorm,
        )
        self.conv4 = Conv3dReLU(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        self.conv5 = Conv3dbn(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        feature_in = self.conv5(x)
        # print(feature_in.shape)
        if skip is not None:

            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + feature_in
        x = self.relu(x)

        return x

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        super().__init__(conv3d)

class DecoderCup(nn.Module):
    def __init__(self, config, img_size):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv3dReLU(
            config.hidden_size + 64,
            head_channels,
            kernel_size=1,
            padding=0,
            use_batchnorm=True,
        )
        self.decoder_channels = config.decoder_channels             # (256, 128, 64, 16)
        self.in_channels = [head_channels] + list(self.decoder_channels[:-1])             # change into a list  [512, 256, 128, 64]
        self.out_channels = self.decoder_channels

        if self.config.n_skip != 0:
            self.skip_channels = self.config.skip_channels           # [512, 256, 64, 16]
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                self.skip_channels[3-i]=0

        else:
            self.skip_channels=[0,0,0,0]

        blocks = [
            DecoderResBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(self.in_channels, self.out_channels, self.skip_channels)
            # DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]

        self.blocks = nn.ModuleList(blocks)


    def forward(self, hidden_states, features=None, input=None, edge_features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, l, hidden)
        _, _, L, H, W = input.shape
        l, h, w = int(L / 16), int(H / 16), int(W / 16)

        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, l, h, w)       # b * hidden_size * (img_size/patch_size)^3
        x = torch.cat((x, edge_features), 1)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):

            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None

            x = decoder_block(x, skip=skip)

        return x

class verteformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config, img_size)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config
        
        self.edge_detect = edge_detect(config, in_channels=3, use_batchnorm=True)                     # edge_detect_module

    def forward(self, x, global_):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)                  # x = x.repeat(1,3,1,1)
            global_ = global_.repeat(1, 768, 1, 1, 1)

        global_ = global_.flatten(2)
        global_ = global_.transpose(-1, -2)
        
        
        input = x
        x, features = self.transformer(x, global_)  # (B, n_patch, hidden)

        edge, edge_features = self.edge_detect(features[3], features[2])
        
        x = self.decoder(x, features, input, edge_features)
        

        logits = self.segmentation_head(x)
        return logits, edge

    def load_from(self, weights):
        # with torch.no_grad():

        res_weight = weights
        # self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
        # self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

        self.transformer.encoder1.encoder_norm.weight = nn.Parameter(np2th(weights["Transformer/encoder_norm/scale"]).cuda())
        self.transformer.encoder1.encoder_norm.bias = nn.Parameter(np2th(weights["Transformer/encoder_norm/bias"]).cuda())
        self.transformer.encoder2.encoder_norm.weight = nn.Parameter(np2th(weights["Transformer/encoder_norm/scale"]).cuda())
        self.transformer.encoder2.encoder_norm.bias = nn.Parameter(np2th(weights["Transformer/encoder_norm/bias"]).cuda())


        # Encoder whole
        for bname, block in self.transformer.encoder1.named_children():
            for uname, unit in block.named_children():
                unit.load_from(weights, n_block=uname)
        for bname, block in self.transformer.encoder2.named_children():
            for uname, unit in block.named_children():
                unit.load_from(weights, n_block=uname)




CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}



        
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--train_root_path', type=str,
                        default='/lustre/home/medical_dataset/verse19', help='root dir for train data')
    parser.add_argument('--val_root_path', type=str,
                        default='/lustre/home/medical_dataset/verse19', help='root dir for val data')
    parser.add_argument('--dataset', type=str,
                        default='verse19', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='/lustre/home/verteformer/lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=26, help='output channel of network')        # 25 + 1
    parser.add_argument('--max_iterations', type=int,
                        default=10000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='batch_size per gpu')             # 24
    parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=0.01,        # 0.01
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=[128, 160, 96], help='input patch size of network input')                 # network input image size : 128
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()



    num_classes = 3
    config_vit = CONFIGS['R50-ViT-B_16']
    config_vit.n_classes = num_classes
    config_vit.n_skip = args.n_skip
    config_vit.batch_size = args.batch_size
    # number of patches
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size[0] / args.vit_patches_size), int(args.img_size[1] / args.vit_patches_size), int(args.img_size[2] / args.vit_patches_size))

    config_vit.n_patches = int(args.img_size[0] / args.vit_patches_size) * int(args.img_size[1] / args.vit_patches_size) * int(args.img_size[2] / args.vit_patches_size)
    net = verteformer(config_vit, img_size=[128, 160, 96], num_classes=num_classes).cuda()
    input = torch.rand((1, 1, 128, 160, 96)).cuda()
    output = net(input)
    print(output)
        
        

