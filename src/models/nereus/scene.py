import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.maps.rasterize import Rasterizer

class ScenePoolingCNN(nn.Module):
    def __init__(self, rasterizer:Rasterizer, in_channels: int, out_channels: int):
        super().__init__()
        self.rasterizer = rasterizer

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(in_channels, 16, 5, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.conv3 = nn.Conv2d(32, out_channels, 5, 1)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer("base_grid", self.build_base_grid())

    def build_base_grid(self):
        R = int(1000 / self.rasterizer.pos_res)
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