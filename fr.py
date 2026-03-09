import torch
import torch.nn as nn

class FR(nn.Module):
    def __init__(self, c1, c2, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2 * scale * scale, 1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.ps(self.conv(x))
import torch
import torch.nn as nn

class FR(nn.Module):
    """
    Feature Reassembly (YOLOv5-safe)
    - Upsamples spatially
    - Preserves channel count
    """
    def __init__(self, c1, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1 * scale * scale, kernel_size=1, stride=1, padding=0)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.ps(self.conv(x))
