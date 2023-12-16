import torch
import torch.nn as nn
import torch.nn.functional as F
from . resnet_skip import ResNetV2


class resunet(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, training=True):
        super(network, self).__init__()
        self.dim = 512
        self.hybrid_model = ResNetV2(block_units=(2, 3, 5), width_factor=1)
        self.decoder_channels = (256, 128, 64, 16)
        self.skip_channels = [256, 128, 64, 16]
        channels = [16, 32, 64, 128, 256, 512]
        
        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels[0], kernel_size=3, padding=1),
            Conv3dReLU(channels[0], channels[0], kernel_size=1, padding=0)
        )
        
        self.decoder1 = nn.Sequential(
            Conv3dReLU(self.dim + self.skip_channels[0], self.decoder_channels[0], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[0], self.decoder_channels[0], kernel_size=1, padding=0)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[0] + self.skip_channels[1], self.decoder_channels[1], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[1], self.decoder_channels[1], kernel_size=1, padding=0)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[1] + self.skip_channels[2], self.decoder_channels[2], kernel_size=3, padding=1),
            Conv3dReLU(self.decoder_channels[2], self.decoder_channels[2], kernel_size=1, padding=0)
        )  # b, 1, 28, 28
        self.decoder4 = nn.Sequential(
            Conv3dReLU(self.decoder_channels[2] + channels[0], channels[0], kernel_size=3, padding=1),
            Conv3dReLU(channels[0], channels[0], kernel_size=1, padding=0)
        )
        
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.segmentation_head = nn.Conv3d(channels[0], out_channel, kernel_size=3, padding=1)


    def forward(self, x):
        # print(x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
        t1 = self.encoder1(x)
    
        x, features = self.hybrid_model(t1)
        
        x = self.up(x)
        # print(x.shape)
        x = torch.cat((x, features[0]), 1)
        x = self.decoder1(x)

        
        x = self.up(x)
        x = torch.cat((x, features[1]), 1)
        x = self.decoder2(x)

        
        x = self.up(x)
        x = torch.cat((x, features[2]), 1)
        x = self.decoder3(x)

        
        x = self.up(x)
        x = torch.cat((x, t1), 1)
        x = self.decoder4(x)
        # output4 = self.map4(x)
        
        x = self.segmentation_head(x)

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

        
if __name__ == '__main__':
    num_classes = 3
    net = resunet(in_channel=3, out_channel=num_classes, training=False).cuda()
    input = torch.rand((1, 1, 128, 160, 96)).cuda()
    output = net(input)
    print(output)
        
        
