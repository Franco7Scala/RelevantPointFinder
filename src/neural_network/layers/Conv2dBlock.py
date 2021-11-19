import torch

from torch import nn


class Conv2dBlock(nn.Module):
    def __init__(self, depth, layer_filters, filters_growth,
                 strides_start, strides_end, input_shape, first_layer=False):
        super(Conv2dBlock, self).__init__()
        layers = []
        c_in_channels = layer_filters
        for i in range(depth):
            if first_layer:
                layers.append(nn.Conv2d(in_channels=1,
                                        kernel_size=3,
                                        out_channels=layer_filters,
                                        padding=1,
                                        dilation=1,
                                        stride=strides_start))
                torch.nn.init.xavier_uniform_(layers[-1].weight)
                first_layer = False
                c_in_channels = layer_filters
            else:
                if i == depth - 1:
                    layer_filters += filters_growth
                    layers.append(nn.Conv2d(in_channels=c_in_channels,
                                            out_channels=layer_filters,
                                            kernel_size=3,
                                            padding=1,
                                            dilation=1,
                                            stride=strides_end))
                    torch.nn.init.xavier_uniform_(layers[-1].weight)
                    c_in_channels = layer_filters
                else:
                    layers.append(nn.Conv2d(in_channels=c_in_channels,
                                            out_channels=layer_filters,
                                            kernel_size=3,
                                            padding=1,
                                            dilation=1,
                                            stride=strides_start))
                    torch.nn.init.xavier_uniform_(layers[-1].weight)
                    c_in_channels = layer_filters
            layers.append(nn.BatchNorm2d(layer_filters))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)