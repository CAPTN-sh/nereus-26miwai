import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.maps.rasterize import Rasterizer

class ScenePoolingCNN(nn.Module):
    def __init__(self, rasterizer:Rasterizer, radius: int, in_channels: int, out_channels: int):
        super().__init__()
        self.rasterizer = rasterizer
        self.radius = radius

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(in_channels, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 2)
        self.conv3 = nn.Conv2d(32, out_channels, 3, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer("base_grid", self.build_base_grid())

    def build_base_grid(self):
        R = int(self.radius / self.rasterizer.pos_res)
        ys = torch.linspace(-R + 0.5, R - 0.5, 2 * R)
        xs = torch.linspace(-R + 0.5, R - 0.5, 2 * R)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1)

    def forward(self, scene, abs_pos):

        x = self.sample_scene_features(scene, abs_pos)

        x = F.relu(self.conv1(self.pad(x)))
        x = F.relu(self.conv2(self.pad(x)))
        x = F.relu(self.conv3(self.pad(x)))

        x = self.pool(x)
        return x.view(x.size(0), -1)

class MapAttention(nn.Module):
    def __init__(self, in_channels, vessel_dim, hidden_dim):
        super().__init__()

        self.map_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.query_proj = nn.Linear(vessel_dim, hidden_dim)
        self.key_proj   = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, scene, abs_pos, vessel_feat):
        x = self.sample_scene_features(scene, abs_pos)
        
        feat = self.map_encoder(x)  # B,C,H,W
        B, C, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1,2)  # B, HW, C

        Q = self.query_proj(vessel_feat).unsqueeze(1)  # B,1,C
        K = self.key_proj(tokens)
        V = self.value_proj(tokens)

        attn = torch.softmax(Q @ K.transpose(-2,-1) / (C**0.5), dim=-1)
        context = (attn @ V).squeeze(1)

        return context
    
    def sample_scene_features(self, scene, abs_pos):
        N, C, H, W = scene.shape

        # shift grid
        x_idx, y_idx = self.rasterizer.pos_to_index(abs_pos)
        grid = self.base_grid[None] + torch.stack([x_idx, y_idx], dim=-1)[:, None, None, :]

        # F.grid_sample needs grid to be normalized to [-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0

        scene_sample = F.grid_sample(
            scene,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return scene_sample
