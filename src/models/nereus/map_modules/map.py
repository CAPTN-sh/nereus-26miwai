import torch
import torch.nn as nn
import torch.nn.functional as F

from data.map.rasterize import Rasterizer
from models.nereus.map_modules.utils import build_base_grid, sample_scene_features
from models.nereus.params import NEREUSParams

class ScenePoolingCNN(nn.Module):
    """
    Map module that extracts a local region from environmental map layers
    (e.g., water depth, distance to shore) based on the current absolute position
    and encodes it using a convolutional neural network.
    """
    def __init__(self, rasterizer:Rasterizer, config: NEREUSParams, in_channels = None, out_channels = None):
        super().__init__()
        self.rasterizer = rasterizer
        self.radius=config.map_radius
        in_channels=in_channels if in_channels else config.map_cnn_in
        out_channels=out_channels if out_channels else config.map_cnn_out

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(in_channels, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.conv3 = nn.Conv2d(32, out_channels, 3, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer("base_grid", build_base_grid(self.rasterizer, self.radius))

    def forward(self, scene, abs_pos, query_feat = None):
        x = sample_scene_features(scene, abs_pos, self.rasterizer, self.base_grid)

        x = F.relu(self.conv1(self.pad(x)))
        x = F.relu(self.conv2(self.pad(x)))
        x = F.relu(self.conv3(self.pad(x)))

        x = self.pool(x)
        return x.view(x.size(0), -1)

class MapAttention(nn.Module):
    """
    Map module that extracts a local region from environmental map layers
    (e.g., water depth, distance to shore) based on the current absolute position
    and applies attention with the prior hidden state as the query.
    """
    def __init__(self, rasterizer, config: NEREUSParams):
        super().__init__()
        self.rasterizer = rasterizer
        self.radius=config.map_radius
        in_channels=config.map_cnn_in
        out_channels=config.map_cnn_out
        query_dim=config.rnn_hidden_size

        self.map_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 3, padding=1),
        )

        self.query_proj = nn.Linear(query_dim, out_channels)
        self.key_proj   = nn.Linear(out_channels, out_channels)
        self.value_proj = nn.Linear(out_channels, out_channels)

        self.register_buffer("base_grid", build_base_grid(self.rasterizer, self.radius))

    def forward(self, scene, abs_pos, query_feat):
        x = sample_scene_features(scene, abs_pos, self.rasterizer, self.base_grid)
        
        feat = self.map_encoder(x)  # B,C,H,W
        B, C, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1,2)  # B, HW, C

        Q = self.query_proj(query_feat).unsqueeze(1)  # B,1,C
        K = self.key_proj(tokens)
        V = self.value_proj(tokens)

        attn = torch.softmax(Q @ K.transpose(-2,-1) / (C**0.5), dim=-1)
        context = (attn @ V).squeeze(1)

        return context