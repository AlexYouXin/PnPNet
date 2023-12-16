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




# intra loss for centers
class intra_loss(nn.Module):
    def __init__(self, in_channels, dim, num_classes, scale, use_batchnorm=True):
        super(intra_loss, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        self.scale = scale
        self.dim = dim
        self.LN1 = LayerNorm(dim, eps=1e-6)
        self.LN2 = LayerNorm(dim, eps=1e-6)
        self.conv = Conv3dReLU(
            in_channels,
            dim,
            kernel_size=1,
            padding=0,
            use_batchnorm=use_batchnorm,
        )
        
    def forward(self, center, feature, label):
        # skipped feature: BN + RELU
        label = one_hot(label, self.num_classes)
        label = F.interpolate(label.float(), scale_factor=self.scale, mode="trilinear")
        pred_center = self.LN1(torch.matmul(label.flatten(2), self.conv(feature).flatten(2).permute(0, 2, 1)))
        difference = (pred_center - self.LN2(center))

        difference_threshold = torch.clamp(difference, min=-1.0, max=1.0)
        loss = torch.mean(torch.square(difference_threshold))
        return loss




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

