"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 64, (3,3), 1, 1),
            PreAct(64),
            nn.Conv2d(64, 128, (1,1), 1, 0),
            nn.MaxPool2d((3,3), 2, 1),
            PreAct(128),
            PreAct(128),
            nn.Conv2d(128, 256, (1,1), 1, 0),
            nn.MaxPool2d((3,3), 2, 1),
            PreAct(256),
            PreAct(256),
            nn.Conv2d(256, 512, (1,1), 1, 0),
            nn.MaxPool2d((3,3), 2, 1),
            PreAct(512),
            PreAct(512),
            nn.MaxPool2d((3,3), 2, 1),
            PreAct(512),
            PreAct(512),
            nn.MaxPool2d((3,3), 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        print(self)
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        
        out = self.net(x)

        return out

class PreAct(nn.Module):
    def __init__(self, channels, k_size=(3,3), stride=1, padding=1):
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, k_size, stride, padding)
        )
    
    def forward(self, x):
      z = self.net(x)
      return x + z

