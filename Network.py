import torch
import torch.nn as nn
from torch.functional import Tensor
from Network2 import MyNetwork2

"""This script defines the network.
"""

class MyNetwork(nn.Module):
    # Baseline structure from the follownig urls:

    # https://amaarora.github.io/2020/08/02/densenets.html
    # https://github.com/bamos/densenet.pytorch/blob/master/densenet.py
    # https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

    def __init__(self, configs):
        super(MyNetwork, self).__init__()
        # initial setting
        self.configs = configs
        self.input_channel = 64
        self.conv = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # define blocks
        self.block1 = self.stack(64,  128,  depth=16,  stride=1, size=3)
        self.block2 = self.stack(128, 256,  depth=32,  stride=2, size=4)
        self.block3 = self.stack(256, 512,  depth=24,  stride=2, size=10)
        self.block4 = self.stack(512, 1024, depth=128, stride=2, size=3)
        self.blocks = [self.block1, self.block2, self.block3, self.block4]
        self.pooling = nn.AvgPool2d(4)
        self.fc = nn.Linear(1536, 10)


    def stack(self, inputc, outputc, depth, stride, size):
        stacks   = list()
        strides  = [1] * size; strides[0] = stride
        is_first = True

        for i in range(size):
            if i != 0: is_first = False
            stacks.append(MyNetwork2(self.input_channel, inputc, outputc, depth, strides[i], is_first))
            self.input_channel = outputc + (i + 2) * depth

        return nn.Sequential(*stacks)

    # Using foward instead of  __call__ method
    # def __call__(self, inputs, training):
    def forward(self, inputs, training=None):
        '''
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
        '''
        x = self.conv(inputs)
        x = self.batch_norm(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
