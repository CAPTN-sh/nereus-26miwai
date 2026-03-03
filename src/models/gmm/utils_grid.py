from pathlib import Path
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from data.map.rasterize import Rasterizer
from utils.config import TRAIN_BBOX, DEFAULT_CRS, AREA_CRS

GRID_RES = 10

def normalize_density(grid, total_hours):
    """
    Normalizes the density by area and time.
    """
    area_km2 = (GRID_RES * GRID_RES) / 1e6
    grid = grid / (area_km2 * total_hours)

    grid = gaussian_filter(grid, sigma=3)

    eps = 1e-6
    grid = np.log(grid + eps) - np.log(grid.mean() + eps)
    return grid

def cluter_to_grid(gmm, train_loader, scene, device, n_clusters):
    """
    Generates a density map for each cluster from historic trajectories.
    """
    rasterizer = Rasterizer(TRAIN_BBOX, pos_res = GRID_RES)
    
    x, y, *_ = rasterizer.get_total_grid_sizes()
    grids = np.zeros((n_clusters, y, x), dtype="float32")

    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            probs = gmm.predict_proba(batch, scene)
            ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
            obs_pos = batch.x_pos[ego_idx, :, :]

            grid_x, grid_y = rasterizer.pos_to_index(obs_pos)
            grid_x = grid_x.detach().cpu().numpy().reshape(-1)
            grid_y = grid_y.detach().cpu().numpy().reshape(-1)

            probs_expanded = np.repeat(probs, 60, axis=0)

            for k in range(n_clusters):
                np.add.at(grids[k], (grid_y, grid_x), probs_expanded[:, k])

    for k in range(n_clusters):
        grids[k] = normalize_density(grids[k], 5064)
    minx, miny, maxx, maxy = transform_bounds(DEFAULT_CRS, AREA_CRS, *TRAIN_BBOX)
    transform = from_origin(minx, maxy, GRID_RES, GRID_RES)
    for k in range(n_clusters):
        grid = grids[k].astype("float32")
        with rasterio.open(
            Path(f"checkpoints/gmm/cluster_{n_clusters}/density_cluster_{k}.tif"),
            "w",
            driver="GTiff",
            height=y,
            width=x,
            count=1,
            dtype="float32",
            crs=AREA_CRS,
            transform=transform,
        ) as dst:
            dst.write(grid, 1)

    return grids