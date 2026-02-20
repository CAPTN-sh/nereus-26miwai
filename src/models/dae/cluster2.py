import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from loaders.graph_loader.loader import graph_loader
from models.dae.model import DAE
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from sklearn.mixture import GaussianMixture
from torch import nn
from tqdm import tqdm
from utils.config import DATA_FOLDER_PATH

DEFAULT_CRS = "EPSG:4326"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class DAE_GMM(nn.Module):
    def __init__(self, dae: DAE, k):
        super().__init__()
        self.dae = dae
        self.k = k
        self.gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            max_iter=200,
            n_init=5,
            reg_covar=1e-6,
        )

    def fit(self, train_loader, device, max_samples):
        self.dae.eval()

        all_z = []
        total_samples = 0

        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(device)

                rec, z = self.dae.inference(batch)
                z = torch.nn.functional.normalize(z, dim=1)
                all_z.append(z.cpu())
                total_samples += z.size(0)

                if total_samples >= max_samples:
                    break

        z_cpu = torch.cat(all_z, dim=0).numpy()
        self.gmm.fit(z_cpu)

    def predict_proba(self, data):
        """Soft cluster assignment p(k | z)."""
        self.dae.eval()

        with torch.no_grad():
            rec, z = self.dae.inference(data)
            z = torch.nn.functional.normalize(z, dim=1)
            z = z.cpu().numpy()

        probs = self.gmm.predict_proba(z)
        return probs

import matplotlib.pyplot as plt
import numpy as np
from models.utils.maps.rasterize import Rasterizer

def cluster_and_plot(dae, rasterizer: Rasterizer):
    with torch.no_grad():
        K = 16
        gmm = DAE_GMM(dae, k = K)
        gmm.fit(train_loader, device=device, max_samples = 2000)

        print("cluster_and_plot")
        
        x, y, *_ = rasterizer.get_total_grid_sizes()
        grids = np.zeros((K, y, x), dtype="float32")

        n = 0
        for batch in tqdm(train_loader):
            n += 1
            if n > 2000:
                break
            batch = batch.to(device)

            probs = gmm.predict_proba(batch)

            ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
            obs_pos = batch.x_pos[ego_idx, :, :]

            grid_x, grid_y = rasterizer.pos_to_index(obs_pos)
            grid_x = grid_x.detach().cpu().numpy().reshape(-1)
            grid_y = grid_y.detach().cpu().numpy().reshape(-1)

            probs_expanded = np.repeat(probs, 60, axis=0)

            for k in range(K):
                np.add.at(grids[k], (grid_y, grid_x), probs_expanded[:, k])

    for k in range(K):
        grid = grids[k]

        plt.figure(figsize=(6, 8))
        plt.imshow(
            grid.clip(0, 100),
            origin="lower",  # matches rasterio convention
            cmap="viridis"
        )
        plt.colorbar(label="Accumulated probability")
        plt.title(f"GMM Cluster {k}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(f"GMM_Cluster_{k}.png")

if __name__ == "__main__":
    print("start")
    flag = "val"

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

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

    print("load dae")
    best_ckpt_path = Path("checkpoints/dae") / f"dae_32_best.pt"
    ckpt = torch.load(best_ckpt_path, map_location=device)
    dae = DAE(ckpt["config"])
    dae.load_state_dict(ckpt["model_state_dict"])
    dae = dae.to(device)

    print("fit k_means")
    K = 10
    gmm = DAE_GMM(dae, k = K)
    gmm.fit(train_loader, device=device, max_samples = 2000)

    print("create grid")

    from models.utils.maps.rasterize import Rasterizer
    grid_res = 10
    local_crs = "EPSG:3035"
    bbox = [10.12, 54.31, 10.33, 54.46]
    rasterizer = Rasterizer(bbox, pos_res = grid_res)
    
    x, y, *_ = rasterizer.get_total_grid_sizes()
    grids = np.zeros((K, y, x), dtype="float32")

    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)

            probs = gmm.predict_proba(batch)

            ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
            obs_pos = batch.x_pos[ego_idx, :, :]

            grid_x, grid_y = rasterizer.pos_to_index(obs_pos)
            grid_x = grid_x.detach().cpu().numpy().reshape(-1)
            grid_y = grid_y.detach().cpu().numpy().reshape(-1)

            probs_expanded = np.repeat(probs, 60, axis=0)

            for k in range(K):
                np.add.at(grids[k], (grid_y, grid_x), probs_expanded[:, k])

    minx, miny, maxx, maxy = transform_bounds(DEFAULT_CRS, local_crs, *bbox)
    transform = from_origin(minx, maxy, grid_res, grid_res)
    for k in range(K):
        grid = grids[k].astype("float32")
        print("max:", grid.max())
        with rasterio.open(
            Path("data/gmm") / f"density_cluster_{k}.tif",
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
