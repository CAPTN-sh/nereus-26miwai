import torch
import torch.nn.functional as F

from data.map.rasterize import Rasterizer


def build_base_grid(rasterizer: Rasterizer, radius):
    R = int(radius / rasterizer.pos_res)
    ys = torch.linspace(-R + 0.5, R - 0.5, 2 * R)
    xs = torch.linspace(-R + 0.5, R - 0.5, 2 * R)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1)

def sample_scene_features(scene, abs_pos, rasterizer: Rasterizer, base_grid):
    """Extracts a local region from map layers based on the current absolute position.
    """
    N, C, H, W = scene.shape

    # shift grid
    x_idx, y_idx = rasterizer.pos_to_index(abs_pos)
    grid = base_grid[None] + torch.stack([x_idx, y_idx], dim=-1)[:, None, None, :]

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
