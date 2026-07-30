# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:41:00 2026

@author: salim
"""

import torch
from torch import nn
import torch.nn.functional as F

class CNN(nn.Module):
   def __init__(self, in_channels: int, num_classes: int):

       """
       Building blocks of convolutional neural network.

       Parameters:
           * in_channels: Number of channels in the input image (for grayscale images, 1)
           * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
       """
       super().__init__()

       # 1st convolutional layer
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       # Max pooling layer
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       # 2nd convolutional layer
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       # Fully connected layer
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
       #self.fc2 = nn.Linear(128, num_cla)sses)

   def forward(self, x: torch.Tensor) -> torch.Tensor:
       """
       Define the forward pass of the neural network.

       Parameters:
           x: Input tensor.

       Returns:
           torch.Tensor
               The output tensor after passing through the network.
       """
       x = F.relu(self.conv1(x))  # Apply first convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = F.relu(self.conv2(x))  # Apply second convolution and ReLU activation
       x = self.pool(x)           # Apply max pooling
       x = torch.flatten(x, start_dim=1)  # Flatten the tensor
       x = self.fc1(x)            # Apply fully connected layer
       #x = self.fc2(x)            # Apply fully connected layer
       return x
