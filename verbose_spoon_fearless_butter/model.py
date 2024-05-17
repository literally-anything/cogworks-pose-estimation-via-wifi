import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(2, 4, 5)
        self.conv12 = nn.Conv1d(4, 8, 5)
        self.conv13 = nn.Conv1d(8, 16, 5)
        self.conv14 = nn.Conv1d(16, 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 4)
        self.conv3 = nn.Conv1d(64, 128, 4)

        self.drop1 = nn.Dropout(p=0.25)
        self.drop2 = nn.Dropout(p=0.25)
        self.drop3 = nn.Dropout(p=0.25)

        self.avg_pool = nn.AvgPool1d(36)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        """ The model's forward pass functionality.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(N, T)
            The batch of size-N.

        Returns
        -------
        mygrad.Tensor, shape=(N, 2)
            The model's predictions for each of the N pieces of data in the batch.
        """
        # CHANGE DOCSTRING

        x = F.selu(self.conv1(x))
        print(1, x.shape)

        x = F.selu(self.conv12(x))
        print(12, x.shape)
        x = F.selu(self.conv13(x))
        print(13, x.shape)

        x = F.selu(self.conv2(x))
        print(2,x.shape)

        x = F.selu(self.conv3(x))
        print(3,x.shape)

        # x = self.avg_pool(x)
        print(4,x.shape)
        x = torch.squeeze(x)  # squeeze to remove dimension 1 at end, becomes 100x24 array
        print(5,x.shape)

        x = F.selu(self.fc1(x))
        print(6,x.shape)
        x = F.selu(self.fc2(x))
        print(7,x.shape)
        x = self.fc3(x)
        print(8,x.shape)
        return x


net = Net()
