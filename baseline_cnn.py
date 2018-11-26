################################################################################
# Description: 
# This module contains the class BasicCNN, trained and evaluated on the
# ChestX-ray14 dataset with simple validation and performance metrics. No
# solution to the class imbalance problem is applied during BasicCNN training.
#
################################################################################

# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os

class BasicCNN(nn.Module):
    ''' 
    A basic convolutional neural network model for baseline comparison.
    '''
    
    def __init__(self):
        super(BasicCNN, self).__init__()
        
        # conv1: 1 input channel, 12 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=8)
        
        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(12)
        
        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)

        # conv2: X input channels, 10 output channels, [8x8] kernel
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=8)
        self.conv2_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv2.weight)

        # conv3: X input channels, 8 output channels, [6x6] kernel
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=6)
        self.conv3_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv3.weight)

        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

        # Define 2 fully connected layers:
        self.fc1 = nn.Linear(in_features=8*164*164, out_features=128)
        self.fc1_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc1.weight)

        # Output layer
        self.fc2 = nn.Linear(in_features=128, out_features=14).cuda()
        torch_init.xavier_normal_(self.fc2.weight)


    def forward(self, batch):
        '''
        Pass the batch of images through each layer of the network, applying 
        non-linearities after each layer.
        
        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        '''
        
        # Apply first convolution, followed by ReLU non-linearity
        batch = func.relu(self.conv1_normed(self.conv1(batch)))
        
        # Apply conv2 and conv3 similarly
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        
        # Pass the output of conv3 to the pooling layer
        batch = self.pool(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))
        
        # Connect the reshaped features of the pooled conv3 to fc1
        batch = func.relu(self.fc1_normed(self.fc1(batch)))
        
        batch = self.fc2(batch)

        # Return the class predictions
        return func.sigmoid(batch)
    

    def num_flat_features(self, inputs):
        
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        
        for s in size:
            num_features *= s
        
        return num_features
