################################################################################
# Description: 
# This module contains the class ArchOneCNN, trained and evaluated on the
# ChestX-ray14 dataset with k-fold validation and undersampling of the
# training set. 
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
from xray_dataloader import \
ChestXrayDataset, create_split_loaders, create_k_loaders

import matplotlib.pyplot as plt
import numpy as np
import os

class ArchOneCNN(nn.Module):
    def __init__(self):
        super(ArchOneCNN, self).__init__()
        # Layer 1: two conv, one bn, ReLU activation, pool
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, 
            kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=24, 
            kernel_size=3, stride=1, padding=1)
        self.conv1_normed = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        torch_init.xavier_normal_(self.conv1_2.weight)
        torch_init.xavier_normal_(self.conv1_1.weight)

        # Layer 2: two conv, one bn, ReLU activation, pool
        self.conv2_1 = nn.Conv2d(in_channels=24, out_channels=32, 
            kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, 
            kernel_size=3, stride=1, padding=1)
        self.conv2_normed = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        torch_init.xavier_normal_(self.conv2_1.weight)
        torch_init.xavier_normal_(self.conv2_2.weight)

        # Layer 3: three conv, one bn, ReLU activation, pool
        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=16, 
            kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=16, out_channels=14, 
            kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=14, out_channels=12, 
            kernel_size=3, stride=1, padding=1)
        self.conv3_normed = nn.BatchNorm2d(12)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        torch_init.xavier_normal_(self.conv3_1.weight)
        torch_init.xavier_normal_(self.conv3_2.weight)
        torch_init.xavier_normal_(self.conv3_3.weight)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=5292, out_features=1024)
        self.fc1_normed = nn.BatchNorm1d(1024)
        torch_init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc2_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc2.weight)
        
        self.fc3 = nn.Linear(in_features=128, out_features=14)
        torch_init.xavier_normal_(self.fc3.weight)


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
        # Apply layer 1
        batch = ((self.conv1_1(batch)))
        batch = func.relu(self.conv1_normed(self.conv1_2(batch)))
        batch = (self.pool1(batch))

        # Apply layer 2
        batch = self.conv2_1(batch)
        batch = func.relu(self.conv2_normed(self.conv2_2(batch)))
        batch = self.pool2(batch)

        # Apply layer 3
        batch = self.conv3_1(batch)
        batch = self.conv3_2(batch)
        batch = func.relu(self.conv3_normed(self.conv3_3(batch)))
        batch = self.pool3(batch)

        # Flatten output for FC layers
        batch = batch.view(-1, self.num_flat_features(batch))
    
        # Fully connected layers
        batch = func.relu(self.fc1_normed(self.fc1(batch)))
        batch = func.relu(self.fc2_normed(self.fc2(batch)))
        batch = self.fc3(batch)

        return func.sigmoid(batch)


    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s
  
        return num_features
