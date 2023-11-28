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




class SDC(nn.Module):
    def __init__(self, in_channels, guidance_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(SDC, self).__init__() 
        self.conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv1 = Conv3dGN(guidance_channels, in_channels, kernel_size=3, padding=1)
        self.theta = theta
        self.guidance_channels = guidance_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # initialize
        x_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        x_initial = self.kernel_initialize(x_initial)

        self.x_kernel_diff = nn.Parameter(x_initial)
        self.x_kernel_diff[:, :, 0, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 0, 2].detach()
        self.x_kernel_diff[:, :, 0, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 0, 0].detach()
        self.x_kernel_diff[:, :, 0, 2, 2].detach()
        self.x_kernel_diff[:, :, 2, 0, 2].detach()
        self.x_kernel_diff[:, :, 2, 2, 0].detach()
        self.x_kernel_diff[:, :, 2, 2, 2].detach()
        
        guidance_initial = torch.randn(in_channels, 1, kernel_size, kernel_size, kernel_size)
        guidance_initial = self.kernel_initialize(guidance_initial)        

        self.guidance_kernel_diff = nn.Parameter(guidance_initial)
        self.guidance_kernel_diff[:, :, 0, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 0].detach()
        self.guidance_kernel_diff[:, :, 0, 2, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 0, 2].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 0].detach()
        self.guidance_kernel_diff[:, :, 2, 2, 2].detach()


    def kernel_initialize(self, kernel):
        kernel[:, :, 0, 0, 0] = -1
        
        kernel[:, :, 0, 0, 2] = 1
        kernel[:, :, 0, 2, 0] = 1
        kernel[:, :, 2, 0, 0] = 1
        
        kernel[:, :, 0, 2, 2] = -1        
        kernel[:, :, 2, 0, 2] = -1        
        kernel[:, :, 2, 2, 0] = -1
        
        kernel[:, :, 2, 2, 2] = 1
        
        return kernel
        

    def forward(self, x, guidance):
        guidance_channels = self.guidance_channels
        in_channels = self.in_channels
        kernel_size = self.kernel_size
        
        guidance = self.conv1(guidance)

        x_diff = F.conv3d(input=x, weight=self.x_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)

        guidance_diff = F.conv3d(input=guidance, weight=self.guidance_kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=1, groups=in_channels)
        out = self.conv(x_diff * guidance_diff * guidance_diff)
        return out


class SDM(nn.Module):
    def __init__(self, in_channel=3, guidance_channels=2):
        super(SDM, self).__init__()
        self.sdc1 = SDC(in_channel, guidance_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_channel)

    def forward(self, feature, guidance):
        boundary_enhanced = self.sdc1(feature, guidance)
        boundary = self.relu(self.bn(boundary_enhanced))
        boundary_enhanced = boundary + feature          

        return boundary_enhanced


