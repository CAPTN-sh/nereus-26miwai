import torch

from data.map.rasterize import Rasterizer

from utils.config import TRAIN_BBOX

RASTER = Rasterizer(TRAIN_BBOX)

def rasterize_occupancy(fut_pos):
    """
    Renders multiple future positions into a single occupancy grid.
    """
    x_bins, y_bins, *_ = RASTER.get_total_grid_sizes()

    x_idx = torch.floor((fut_pos[:, 0] - RASTER.x_min) / RASTER.pos_res).to(torch.int64)
    y_idx = torch.floor((fut_pos[:, 1] - RASTER.y_min) / RASTER.pos_res).to(torch.int64)
    
    x_idx = x_idx.clamp(0, x_bins - 1)
    y_idx = y_idx.clamp(0, y_bins - 1)

    grid = torch.zeros((x_bins, y_bins))
    
    grid[x_idx, y_idx] = 1.0
    grid = grid / (grid.sum() + 1e-8)

    return grid