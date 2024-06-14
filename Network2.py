import torch
import torch.nn as nn
from torch.functional import Tensor

"""This script defines the network.
"""

class MyNetwork2(nn.Module):
    def __init__(self, prevc, inputc, outputc, depth, stride, is_first):
        super(MyNetwork2, self).__init__()
        self.projection_shorcut = nn.Sequential()
        self.outputc = outputc
        combined = outputc + depth

        self.conv1 = nn.Conv2d(prevc, inputc, 1, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(inputc)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(inputc, inputc, 3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(inputc)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(inputc, combined, 1, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(combined)
        self.relu3 = nn.ReLU()

        if is_first:
            self.projection_shorcut = nn.Sequential(nn.Conv2d(prevc, combined, 1, stride=stride), nn.BatchNorm2d(combined))
        self.relu4 = nn.ReLU()

    # Using foward instead of __call__ method
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
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x_ = self.projection_shorcut(inputs)
        x  = torch.cat([x_[:,:self.outputc,:,:]+x[:,:self.outputc,:,:], x_[:,self.outputc:,:,:], x[:,self.outputc:,:,:]], 1)
        x  = self.relu4(x)

        return x
