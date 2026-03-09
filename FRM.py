import torch
import torch.nn as nn
import torch.nn.functional as F

class FRM(nn.Module):
    """Feature Reassembly Module for small target enhancement."""
    def __init__(self, in_channels):
        super(FRM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Split
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))

        # Reassemble
        out = x1 + x2 + x3
        return out
