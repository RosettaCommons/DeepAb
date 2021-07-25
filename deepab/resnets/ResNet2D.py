import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock2D(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=5, dilation=1, stride=1):
        super(ResBlock2D, self).__init__()

        padding = ((kernel_size - 1) * dilation) // 2

        self.activation = F.relu
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=kernel_size,
                               dilation=dilation,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.activation(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self,
                 in_channels,
                 block,
                 num_blocks,
                 planes=64,
                 kernel_size=5,
                 dilation_cycle=5):
        super(ResNet2D, self).__init__()
        # Check if the number of initial planes is a power of 2, done for faster computation on GPU
        if not (planes != 0 and ((planes & (planes - 1)) == 0)):
            raise ValueError(
                'The initial number of planes must be a power of 2')

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.planes = planes

        self.conv1 = nn.Conv2d(in_channels,
                               self.planes,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.planes)

        resnet = self._make_layer(block,
                                  self.planes,
                                  num_blocks,
                                  stride=1,
                                  kernel_size=kernel_size,
                                  dilation_cycle=dilation_cycle)

        setattr(self, 'layer0', resnet)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size,
                    dilation_cycle):
        layers = []
        for i in range(num_blocks):
            dilation = int(math.pow(
                2, i % dilation_cycle)) if dilation_cycle > 0 else 1
            layers.append(
                block(planes,
                      planes,
                      stride=stride,
                      kernel_size=kernel_size,
                      dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        # Only thing in self.layers is resnet
        out = self.layer0(out)
        return out
