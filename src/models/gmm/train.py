import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from loaders.graph_loader.loader import graph_loader
from models.traisformer.model import TrAISformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from tqdm import tqdm
from utils.config import DATA_FOLDER_PATH

import matplotlib.pyplot as plt
from models.utils.maps.rasterize import Rasterizer
from models.utils.maps.scene_gernerator import SceneLoader
from scipy.ndimage import gaussian_filter
from models.gmm.model import AIS_GMM

DEFAULT_CRS = "EPSG:4326"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def normalize_density(grid, grid_res, total_hours):
    """
    - normalize by size and time (count/km²/h)
    - smooth with gaussian_filter
    - normalize by mean + log
    """
    area_km2 = (grid_res * grid_res) / 1e6
    grid = grid / (area_km2 * total_hours)

    grid = gaussian_filter(grid, sigma=3)

    eps = 1e-6
    grid = np.log(grid + eps) - np.log(grid.mean() + eps)
    return grid


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print("load dataset")

    flag = "train"
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
    train_loader, _ = graph_loader(
        data_folder=data_folder,
        flag=flag,
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        batch_size= 512,
        pin_memory=True,
        pred_len= 1,
        obs_len= 60,
        max_edge_dist = 0,
    )

    path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
    sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))
    scene_contiguous = np.ascontiguousarray(sl.load_scene(path))
    scene = torch.from_numpy(scene_contiguous).to(device).to(torch.float32)

    print("load traisfromer")
    best_ckpt_path = Path("checkpoints/traisformer/traisformer_dest_best.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    prior_module = TrAISformer(ckpt["config"]).to(device)
    prior_module.load_state_dict(ckpt["model_state_dict"])
    prior_module.eval()
    prior_module.requires_grad_(False)

    K = 32
    print(f"fit k_means {K}")
    gmm = AIS_GMM(prior_module, n_clusters = K)
    gmm.fit(train_loader, scene=scene, device=device, max_samples = 10000)

    state_dict_full = {
        "prior_config": gmm.prior_model.config,
        "prior_state_dict": gmm.prior_model.state_dict(),
        "gmm": gmm.gmm,
        "n_clusters": gmm.k
    }
    torch.save(state_dict_full, f"data/gmm/cluster_{K}/ais_gmm.pt")

    print("create grid")
    grid_res = 10
    local_crs = "EPSG:3035"
    bbox = [10.12, 54.31, 10.33, 54.46]
    rasterizer = Rasterizer(bbox, pos_res = grid_res)
    
    x, y, *_ = rasterizer.get_total_grid_sizes()
    grids = np.zeros((K, y, x), dtype="float32")

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

            for k in range(K):
                np.add.at(grids[k], (grid_y, grid_x), probs_expanded[:, k])

    for k in range(K):
        grids[k] = normalize_density(grids[k], grid_res, 5064)

    minx, miny, maxx, maxy = transform_bounds(DEFAULT_CRS, local_crs, *bbox)
    transform = from_origin(minx, maxy, grid_res, grid_res)
    for k in range(K):
        grid = grids[k].astype("float32")
        with rasterio.open(
            Path(f"data/gmm/cluster_{K}/density_cluster_{k}.tif"),
            "w",
            driver="GTiff",
            height=y,
            width=x,
            count=1,
            dtype="float32",
            crs=local_crs,
            transform=transform,
        ) as dst:
            dst.write(grid, 1)

    for k in range(K):
        grid = grids[k]

        plt.figure(figsize=(6, 8))
        plt.imshow(
            grid,
            origin="lower",
            cmap="turbo"
        )
        plt.colorbar(label="Normalized Density")
        plt.title(f"GMM Cluster {k}")
        plt.xlabel("Grid Cell X")
        plt.ylabel("Grid Cell Y")
        plt.tight_layout()
        plt.savefig(f"data/gmm/cluster_{K}/density_cluster_{k}.png")

"""
CUDA_VISIBLE_DEVICES=1 python src/models/gmm/train.py

"""