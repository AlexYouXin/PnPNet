from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He



        
if __name__ == '__main__':
    num_classes = 3
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
    # The choice for these hyper-parameters is related to the preprocessing stage of nnUNet
    # base_num_features num_pool num_conv_per_stage
    # pool_op_kernel_sizes
    # conv_kernel_sizes
    net = Generic_UNet(1, 32, num_classes,
                                    5,
                                    2, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, False, False, lambda x: x, InitWeights_He(1e-2),
                                    [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]], False, True, True).cuda()


    input = torch.rand((1, 1, 128, 160, 96)).cuda()
    output = net(input)
    print(output)
        
