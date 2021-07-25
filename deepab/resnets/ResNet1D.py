import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(ResBlock1D, self).__init__()

        padding = kernel_size // 2

        self.activation = F.relu
        self.conv1 = nn.Conv1d(in_planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes,
                               planes,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = self.activation(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self,
                 in_channels,
                 block,
                 num_blocks,
                 planes=64,
                 kernel_size=3):
        super(ResNet1D, self).__init__()
        # Check if the number of initial planes is a power of 2
        if not (planes != 0 and ((planes & (planes - 1)) == 0)):
            raise ValueError(
                'The initial number of planes must be a power of 2')

        self.activation = F.relu
        self.kernel_size = kernel_size
        self.planes = planes

        self.conv1 = nn.Conv1d(in_channels,
                               self.planes,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(self.planes)

        resnet = self._make_layer(block,
                                  self.planes,
                                  num_blocks,
                                  stride=1,
                                  kernel_size=kernel_size)

        setattr(self, 'layer0', resnet)

    def _make_layer(self, block, planes, num_blocks, stride, kernel_size):
        layers = []
        for i in range(num_blocks):
            layers.append(
                block(planes, planes, stride=stride, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        # Only thing in self.layers is resnet
        out = self.layer0(out)
        return out
