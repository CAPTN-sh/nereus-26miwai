from sklearn.mixture import GaussianMixture
import pandas as pd
import torch
from torch import nn
from loaders.graph_loader.loader import graph_loader
from pathlib import Path
from utils.config import DATA_FOLDER_PATH
from models.dae.model import DAE
import os
from tqdm import tqdm

from math import ceil
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from scipy.ndimage import gaussian_filter
import rasterio
import numpy as np

from utils.config import AIS_SOURCE
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

                z = self.dae.inference(batch)
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
            z = self.dae.inference(data)
            z = torch.nn.functional.normalize(z, dim=1)
            z = z.cpu().numpy()

        probs = self.gmm.predict_proba(z)
        return probs

if __name__ == "__main__":

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
    train_loader, _ = graph_loader(
        data_folder=data_folder,
        flag="train",
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

    print("fit k_means")
    K = 10
    gmm = DAE_GMM(dae, k = K)
    gmm.fit(train_loader, device=device, max_samples = 1000)

    print("fit k_means")
    all_ids = []

    for batch in train_loader:
        all_ids.append(batch.target_id.cpu())

    all_ids = torch.cat(all_ids)

    unique_ids, inverse_indices = torch.unique(all_ids, return_inverse=True)
    num_traj = unique_ids.size(0)
    id_to_idx = {int(id_): i for i, id_ in enumerate(unique_ids)}

    cluster_sum = torch.zeros(num_traj, K, device=device)
    cluster_count = torch.zeros(num_traj, device=device)

    with torch.no_grad():
        for batch in tqdm(train_loader):
            batch = batch.to(device)

            z = dae.inference(batch)
            z = torch.nn.functional.normalize(z, dim=1)

            probs = torch.from_numpy(
                gmm.predict_proba(z.cpu().numpy())
            ).to(device)

            traj_ids = batch.target_id.long().cpu()

            # map using dict (fast enough at batch level)
            mapped_ids = torch.tensor(
                [id_to_idx[int(t)] for t in traj_ids],
                device=device
            )

            cluster_sum.index_add_(0, mapped_ids, probs)
            cluster_count.index_add_(0, mapped_ids, torch.ones_like(mapped_ids, dtype=torch.float))

    def generate_density_maps_gmm(
        df,
        unique_ids,
        cluster_probs_traj,  # (num_traj, K)
        K,
        bbox,
        folder_out,
        bbox_name,
        local_crs,
        grid_res,
    ):
        folder_out.mkdir(parents=True, exist_ok=True)
        minx, miny, maxx, maxy = transform_bounds(
            DEFAULT_CRS, local_crs, *bbox
        )

        width = int(ceil((maxx - minx) / grid_res))
        height = int(ceil((maxy - miny) / grid_res))

        # Prepare K grids
        grids = np.zeros((K, height, width), dtype="float32")

        # --- Map traj_id → traj_idx ---
        # unique_ids is sorted from torch.unique
        unique_ids_np = unique_ids.cpu().numpy()
        df_ids = df["target_id"].to_numpy()

        traj_indices = np.searchsorted(unique_ids_np, df_ids)

        # --- Project coordinates ---
        to_laea = Transformer.from_crs(DEFAULT_CRS, local_crs, always_xy=True)
        x, y = to_laea.transform(df["lon"].to_numpy(), df["lat"].to_numpy())

        grid_x = ((x - minx) // grid_res).astype(int)
        grid_y = ((miny - y) // grid_res).astype(int)

        # Clip valid cells
        valid = (
            (grid_x >= 0) & (grid_x < width) &
            (grid_y >= 0) & (grid_y < height)
        )

        grid_x = grid_x[valid]
        grid_y = grid_y[valid]
        traj_indices = traj_indices[valid]

        # --- Accumulate weighted densities ---
        cluster_probs_np = cluster_probs_traj.cpu().numpy()

        for k in range(K):
            weights = cluster_probs_np[traj_indices, k]
            np.add.at(grids[k], (grid_y, grid_x), weights)

        # --- Normalize and save ---
        area_km2 = (grid_res * grid_res) / 1e6
        window_hours = len(df["timestamp"].dt.date.unique()) * 24

        transform = from_origin(minx, maxy, grid_res, grid_res)

        for k in range(K):
            grid = grids[k] / (area_km2 * window_hours)
            grid = gaussian_filter(grid, sigma=3)

            eps = 1e-6
            grid = np.log(grid + eps) - np.log(grid.mean() + eps)

            with rasterio.open(
                folder_out / f"density_cluster_{k}.tif",
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs=local_crs,
                transform=transform,
            ) as dst:
                dst.write(grid.astype("float32"), 1)

    cluster_probs_traj = cluster_sum / cluster_count.clamp(min=1).unsqueeze(1)
    cluster_probs_traj = cluster_probs_traj.clamp(min=1e-12)
    file_name = f"{AIS_SOURCE}_{data_folder.name}_train"
    path = data_folder / f"{file_name}_ship_features.parquet"
    df = pd.read_parquet(path)


    local_crs = "EPSG:3035"
    grid_res = 10.0
    bbox = [10.12, 54.31, 10.33, 54.46]

    generate_density_maps_gmm(
        df=df,
        unique_ids=unique_ids,
        cluster_probs_traj=cluster_probs_traj,
        K=K,
        bbox=bbox,
        folder_out= Path("data/maps/2_standardized/fh_10/kiel/gmm"),
        bbox_name="kiel",
        local_crs=local_crs,
        grid_res=grid_res,
    )
