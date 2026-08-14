import torch.nn.functional as F
from torch import nn


class ScenePoolingCNN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()

        self.pad = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(in_channels, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, 64, 5, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(self.pad(x)))
        x = F.relu(self.conv2(self.pad(x)))
        x = F.relu(self.conv3(self.pad(x)))
        x = self.pool(x)
        return x.view(x.size(0), -1)
