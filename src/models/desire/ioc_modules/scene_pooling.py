import torch.nn.functional as F
from torch import nn

from models.desire.params import DESIREParams


class ScenePoolingCNN(nn.Module):
    """Simple CNN to process the map context.
    """

    def __init__(self, params: DESIREParams):
        super().__init__()
        _in = params.in_channels
        _h = params.intermediate_size
        _out = params.out_channels

        self.pad = nn.ReflectionPad2d(2)
        self.conv1 = nn.Conv2d(_in, _h, 5, 2)
        self.conv2 = nn.Conv2d(_h, _out, 5, 1)

    def forward(self, x):
        x = F.relu(self.conv1(self.pad(x)))
        x = F.relu(self.conv2(self.pad(x)))
        return x
