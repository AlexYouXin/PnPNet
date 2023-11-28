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


