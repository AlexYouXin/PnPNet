import copy
import logging
import math
import argparse
from os.path import join as pjoin
import torch
import torch.nn as nn
import torch.nn.functional as F
from . resnet_skip import ResNetV2
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, Conv3d, LayerNorm
from . modules.SDM_module import SDM
from . modules.CCM_module import CCM
from . modules.class_center_loss import intra_loss

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}




class network(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, training=True, config=None):
        super(network, self).__init__()
        self.dim = 512
        self.hybrid_model = ResNetV2(block_units=(2, 3, 5), width_factor=1)
        self.decoder_channels = (256, 128, 64, 16)
        self.skip_channels = [256, 128, 64, 16]
        channels = [16, 32, 64, 128, 256, 512]
        
        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels[0], kernel_size=1, padding=0),
            Conv3dReLU(channels[0], channels[0], kernel_size=1, padding=0)
        )
        
        self.decoder1 = nn.Sequential(
            Conv3dReLU(self.dim + self.skip_channels[0], self.decoder_channels[0], kernel_size=1, padding=0),
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[0], kernel_size=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[0] + self.skip_channels[1], self.decoder_channels[1], kernel_size=1, padding=0),
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[1], kernel_size=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[1] + self.skip_channels[2], self.decoder_channels[2], kernel_size=1, padding=0),
            Conv3dReLU(self.decoder_channels[2], self.decoder_channels[2], kernel_size=1, padding=0)
        )
        self.decoder4 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[2] + channels[0], channels[0], kernel_size=1, padding=0),
            Conv3dReLU(channels[0], channels[0], kernel_size=1, padding=0)
        )
        
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.cluster1 = CCM(config, self.skip_channels[0])
        self.cluster2 = CCM(config, self.skip_channels[1])
        self.cluster3 = CCM(config, self.skip_channels[2])
        
        self.cluster_center = nn.Parameter(torch.randn(1, config.n_classes, config.dim))

        self.segmentation_head = nn.Conv3d(channels[0], out_channel, kernel_size=3, padding=1)
        self.sdn1 = SDM(channels[4], channels[5])
        self.sdn2 = SDM(channels[3], channels[4])
        self.sdn3 = SDM(channels[2], channels[3])

        self.conv1 = DecoderResBlock(3 * config.n_classes, 3 * config.n_classes)
        self.conv2 = nn.Conv3d(3 * config.n_classes, config.n_classes, kernel_size=3, padding=1)

        self.intra_loss0 = intra_loss(self.skip_channels[2], config.dim, config.n_classes, 1/2)



    def forward(self, x, label):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        t1 = self.encoder1(x)
    
        x, features = self.hybrid_model(t1)      

        x = self.up(x)
        class_token0, refined_center = self.cluster1(features[0], self.cluster_center)

        class_token0 = F.interpolate(class_token0, scale_factor=8, mode="trilinear")
        features[0] = self.sdn1(features[0], x)
        x = torch.cat((x, features[0]), 1)
        x = self.decoder1(x)


        x = self.up(x)
        class_token1, refined_center = self.cluster2(features[1], refined_center)
        class_token1 = F.interpolate(class_token1, scale_factor=4, mode="trilinear")
        features[1] = self.sdn2(features[1], x)
        x = torch.cat((x, features[1]), 1)
        x = self.decoder2(x)

        
        x = self.up(x)
        class_token2, refined_center = self.cluster3(features[2], refined_center)
        loss0 = self.intra_loss0(refined_center, features[2], label)
        class_token2 = F.interpolate(class_token2, scale_factor=2, mode="trilinear")
        features[2] = self.sdn3(features[2], x)
        x = torch.cat((x, features[2]), 1)
        x = self.decoder3(x)

        
        x = self.up(x)
        x = torch.cat((x, t1), 1)
        x = self.decoder4(x)

        
        x = self.segmentation_head(x)

        # fuse class token features
        class_token = torch.cat((torch.cat((class_token0, class_token1), 1), class_token2), 1)
        class_token = self.conv2(self.conv1(class_token))

        x = torch.sigmoid(x) * class_token + class_token

        return x, loss0




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

        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, skip=None):

        feature_in = self.conv3(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + feature_in
        x = self.relu(x)


        return x





class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)



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



