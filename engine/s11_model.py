from __future__ import print_function
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        residue = self.block(x)
        residue = self.block(residue)
        return torch.add(x, residue)


class Layer(nn.Module):
    def __init__(self, in_channels, channels, shortcut=False):
        super(Layer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),      #i:32x32x3    k,n:3,(3x3x3x32)      o:30x30x32      RF:5
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.shortcut = shortcut
        if shortcut:
            self.res_block = ResBlock(channels)

    def forward(self, x):
        x = self.conv(x)
        if self.shortcut:
            R = self.res_block(x)
            return torch.add(x, R)
        return x


class S11ResNet(nn.Module):
    def __init__(self):
        super(S11ResNet, self).__init__()

        kernels = 64
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=kernels, kernel_size=3, stride=1, padding=1, bias=False),      #i:32x32x3    k,n:3,(3x3x3x32)      o:30x30x32      RF:5
            nn.BatchNorm2d(kernels),
            nn.ReLU()
        )
        self.layer1 = Layer(64, 128, shortcut=True)
        self.layer2 = Layer(128, 256, shortcut=False)
        self.layer3 = Layer(256, 512, shortcut=True)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=1)
        self.fc_layer = nn.Conv2d(in_channels=512, out_channels=10, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.fc_layer(x)
        x = x.view(x.size(0), -1)
        return x

    # def _make_layer(self, channels, shortcut=False):