import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        channels = [16, 32, 64, 128, 256, 512]
        self.encoder1 = nn.Sequential(
            Conv3dReLU(in_channel, channels[0], kernel_size=3, padding=1),
            Conv3dReLU(channels[0], channels[1], kernel_size=3, padding=1)
        )                                                            # b, 16, 10, 10
        self.encoder2 = nn.Sequential(
            Conv3dReLU(channels[1], channels[1], kernel_size=3, padding=1),
            Conv3dReLU(channels[1], channels[2], kernel_size=3, padding=1)
        )     # b, 8, 3, 3
        self.encoder3 = nn.Sequential(
            Conv3dReLU(channels[2], channels[2], kernel_size=3, padding=1),
            Conv3dReLU(channels[2], channels[3], kernel_size=3, padding=1)
        )
        self.encoder4 = nn.Sequential(
            Conv3dReLU(channels[3], channels[3], kernel_size=3, padding=1),
            Conv3dReLU(channels[3], channels[4], kernel_size=3, padding=1)
        )
        self.encoder5 = nn.Sequential(
            Conv3dReLU(channels[4], channels[4], kernel_size=3, padding=1),
            Conv3dReLU(channels[4], channels[5], kernel_size=3, padding=1)
        )
        
        self.decoder1 = nn.Sequential(
            Conv3dReLU(channels[5] + channels[4], channels[4], kernel_size=3, padding=1),
            Conv3dReLU(channels[4], channels[4], kernel_size=3, padding=1)
        )
        self.decoder2 = nn.Sequential(
            Conv3dReLU(channels[4] + channels[3], channels[3], kernel_size=3, padding=1),
            Conv3dReLU(channels[3], channels[3], kernel_size=3, padding=1)
        )
        self.decoder3 = nn.Sequential(
            Conv3dReLU(channels[3] + channels[2], channels[2], kernel_size=3, padding=1),
            Conv3dReLU(channels[2], channels[2], kernel_size=3, padding=1)
        )  # b, 1, 28, 28
        self.decoder4 = nn.Sequential(
            Conv3dReLU(channels[2] + channels[1], channels[1], kernel_size=3, padding=1),
            Conv3dReLU(channels[1], channels[1], kernel_size=3, padding=1)
        )
        
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.segmentation_head = nn.Conv3d(channels[1], out_channel, kernel_size=3, padding=1)
        '''
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )
        '''
    def forward(self, x):
        # print(x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1,1)
            
        out = self.encoder1(x)
        t1 = out
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder2(out)
        t2 = out
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder3(out)
        t3 = out
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder4(out)
        t4 = out
        # print('t4 shape: ', t4.shape)
        out = self.down(out)
        
        # print(out.shape)
        
        out = self.encoder5(out)
        

        out = self.up(out)
        # print(out.shape)
        out = torch.cat((out, t4), 1)
        out = self.decoder1(out)
        # output1 = self.map1(out)
        
        out = self.up(out)
        out = torch.cat((out, t3), 1)
        out = self.decoder2(out)
        # output2 = self.map2(out)
        
        out = self.up(out)
        out = torch.cat((out, t2), 1)
        out = self.decoder3(out)
        # output3 = self.map3(out)
        
        out = self.up(out)
        out = torch.cat((out, t1), 1)
        out = self.decoder4(out)
        # output4 = self.map4(out)
        
        out = self.segmentation_head(out)
        
        # print(out.shape)         # torch.Size([2, 26, 158, 158, 110])
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        '''
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return out
        '''
        return out