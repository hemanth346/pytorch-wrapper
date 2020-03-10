from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary

class summary(object):
    def __init__(self, model, input_size=(3, 32, 32), device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        summary(model.to_device(device), input_size=input_size)        


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



def conv_block(in_channels, out_channels, dropout=0, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout)
        )
# class DepthwiseSeparableConv(in_channels, out_channels, *args, **kwargs):
#     # https://discuss.pytorch.org/t/depthwise-and-separable-convolutions-in-pytorch/7315/2?u=hemanth346
#     depthwise = nn.Conv2d(in_channels, in_channels*kernel_size, kernel_size=(kernel_size, 1), groups=in_channels, *args, **kwargs)
#     pointwise = nn.Conv2d(in_channels*kernel_size, out_channels, kernel_size=(1, kernel_size), *args, **kwargs)
#     return depthwise, pointwise

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = conv_block(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2, bias=False)      #i:32x32x3    k,n:3,32(3x3x3x32)      o:32x32x32      RF:5      
        self.conv2 = conv_block(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2, dropout=0.1,  bias=False)     #i:32x32x32   k,n:3,64(3x3x32x64)     o:32x32x64      RF:9 
        self.pool = nn.MaxPool2d(2, 2)                                                                                  #i:32x32x64   k,n:2,                  o:16x16x64      RF:10

        self.conv3 = conv_block(in_channels=64, out_channels=64*3, kernel_size=3, padding=1, bias=False, groups=64)     #i:16x16x64   k,n:3,32((3x3x1)64x3)  o:16x16x(64*3)     RF:14
                                                                                                            # Using groups, each group returns 3 channels taking total output to 64x3
        self.conv4 = conv_block(in_channels=64*3, out_channels=128, kernel_size=1, bias=False)                          #i:16x16x(64*3)   k,n:1,128(1x1x32x128)  o:16x16x128     RF:14
        self.pool = nn.MaxPool2d(2, 2)                                                                                  #i:16x16x128   k,n:2,                o:8x8x128      RF:16

        self.conv5 = conv_block(in_channels=128, out_channels=32, kernel_size=1, dropout=0.1, bias=False)                 #i:8x8x128   k,n:1,32(3x3x1x32)   o:8x8x32     RF:16
        self.conv6 = conv_block(in_channels=32, out_channels=64, kernel_size=3, dropout=0.1, padding=1, bias=False)                  #i:8x8x32    k,n:3,64(3x3x32x64)  o:8x8x64     RF:24
        self.conv7 = conv_block(in_channels=64, out_channels=128, kernel_size=3, dropout=0.2, padding=1, bias=False)                 #i:8x8x64   k,n:3,128(3x3x64x128) o:8x8x128    RF:32
        self.pool = nn.MaxPool2d(2, 2)                                                                                  #i:8x8x128   k,n:2,               o:4x4x128    RF:36

        # No relu, BN before GAP/Last layer
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False)                  #i:4x4x128  k,n:3,64    o:4x4x64    RF:42
        self.gap = nn.AvgPool2d(kernel_size=4)                                                                          #i:4x4x64   k,n:4,      o:1x1x64    RF:46
        self.final = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, bias=False)                              #i:1x1x64   k,n:1,10    o:1x1x10    RF:60

    def forward(self, x):
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv4(self.conv3(x)))
        x = self.pool(self.conv7(self.conv6(self.conv5(x))))
        x = self.final(self.gap(self.conv8(x)))
        x = x.view(-1, 10)                           # Don't want 10x1x1..
        return F.log_softmax(x)

