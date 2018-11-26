################################################################################
# Description: 
# This module contains the class ArchTwoCNN, trained and evaluated on the
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

class ArchTwoCNN(nn.Module):
    def __init__(self):
        super(ArchTwoCNN, self).__init__()

        # Layer 1: one conv, one bn, ReLU activation, pool
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64,
            kernel_size=7)
        torch_init.xavier_normal_(self.conv1_1.weight)
        self.bn1 = nn.BatchNorm2d(64)
        self.activation1 = func.relu
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        # Layer 2: three conv, one bn, ReLU activation, pool
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128,
            kernel_size=5)
        torch_init.xavier_normal_(self.conv2_1.weight)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, 
            kernel_size=5)
        torch_init.xavier_normal_(self.conv2_2.weight)
        self.conv2_3 = nn.Conv2d(in_channels=128, out_channels=128, 
            kernel_size=5)
        torch_init.xavier_normal_(self.conv2_3.weight)
        self.bn2 = nn.BatchNorm2d(128)
        self.activation2 = func.relu
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3: three conv, one bn, ReLU activation, pool
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256,
            kernel_size=3)
        torch_init.xavier_normal_(self.conv3_1.weight)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, 
            kernel_size=3)
        torch_init.xavier_normal_(self.conv3_2.weight)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, 
            kernel_size=3)
        torch_init.xavier_normal_(self.conv3_3.weight)
        self.bn3 = nn.BatchNorm2d(256)
        self.activation3 = func.relu
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 4: four conv, one bn, ReLU activation, pool
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=256,
            kernel_size=3)
        torch_init.xavier_normal_(self.conv4_1.weight)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=128, 
            kernel_size=3)
        torch_init.xavier_normal_(self.conv4_2.weight)
        self.conv4_3 = nn.Conv2d(in_channels=128, out_channels=64, 
            kernel_size=3)
        torch_init.xavier_normal_(self.conv4_3.weight)
        self.conv4_4 = nn.Conv2d(in_channels=64, out_channels=32, 
            kernel_size=3)
        torch_init.xavier_normal_(self.conv4_4.weight)
        self.bn4 = nn.BatchNorm2d(32)
        self.activation4 = func.relu
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer
        self.fc1 = nn.Linear(in_features=6272, out_features=800)
        self.fc1_bn = nn.BatchNorm1d(800)
        self.fc1_activation = func.relu
        torch_init.xavier_normal_(self.fc1.weight)

        self.fc2 = nn.Linear(in_features=800, out_features=14)
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

        # Apply layer 1 
        batch = self.pool1(self.activation1(self.bn1(self.conv1_1(batch))))

        # Apply layer 2 
        batch = self.conv2_3(self.conv2_2(self.conv2_1(batch)))
        batch = self.pool2(self.activation2(self.bn2(batch)))

        # Apply layer 3
        batch = self.conv3_3(self.conv3_2(self.conv3_1(batch)))
        batch = self.pool3(self.activation3(self.bn3(batch)))

        # Apply layer 4
        batch = self.conv4_4(self.conv4_3(self.conv4_2(self.conv4_1(batch))))
        batch = self.pool4(self.activation4(self.bn4(batch)))
        batch = batch.view(-1, self.num_flat_features(batch))

        # Apply FC layers
        batch = self.fc1_activation(self.fc1_bn(self.fc1(batch)))
        batch = self.fc2(batch)

        return func.sigmoid(batch)


    def num_flat_features(self, inputs):

        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features
