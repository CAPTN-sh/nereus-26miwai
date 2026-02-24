import logging
import os
from pathlib import Path
import torch

from utils.logger import logger
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from loaders.graph_loader.loader import graph_loader
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer

from models.nereus.model import NEREUS
from models.nereus.social import GAT, EgoSocialPooling
from models.nereus.intent import DensityIntent
from models.traisformer.model import TrAISformer
from models.gmm.model import AIS_GMM, MAP_GMM

from utils.config import DATA_FOLDER_PATH
from eval.cpa import compute_batch_collision_risk
import matplotlib.pyplot as plt
import os
import geopandas as gpd

import matplotlib.cm as cm
import matplotlib.colors as mcolors

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DE_NORMALIZE = 100

def plot_mdn_k_trajectories(
    file_name,
    obs_pos,        # [T_obs, 2]
    fut_pos,        # [T_fut, 2]
    pred_abs_k,     # [K, T_fut, 2]
    pi_k,
    region,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- background map ----
    map_path = f"/home/bbi/nereus/assets/maps/2_standardized/dma_10/{region}/land.geojson"
    background = gpd.read_file(map_path).to_crs("EPSG:25832")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    # ---- observed ----
    ax.scatter(
        obs_pos[:, 0],
        obs_pos[:, 1],
        color="blue",
        marker="o",
        s=7,
        alpha=0.7,
        label="Observed",
    )

    # ---- ground truth future ----
    ax.scatter(
        fut_pos[:, 0],
        fut_pos[:, 1],
        color="green",
        marker="o",
        s=7,
        alpha=0.7,
        label="Ground Truth",
    )

    # ---- K predicted trajectories ----
    cmap = cm.plasma
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    K = pred_abs_k.shape[0]
    for k in range(K):
        traj = pred_abs_k[k]
        color = cmap(norm(pi_k[k]))

        ax.scatter(
            traj[:, 0],
            traj[:, 1],
            color=color,
            s=7,
            alpha=0.7,
            zorder=2,
            label=f"Mode {k} (π={pi_k[k]:.2f})"
        )

    # ---- styling ----
    """"
    minx, miny, maxx, maxy = background.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    """

    # ---- compute bounds from trajectories ----
    all_x = np.concatenate([
        obs_pos[:, 0],
        fut_pos[:, 0],
        pred_abs_k[:, :, 0].reshape(-1)
    ])

    all_y = np.concatenate([
        obs_pos[:, 1],
        fut_pos[:, 1],
        pred_abs_k[:, :, 1].reshape(-1)
    ])

    minx, maxx = all_x.min(), all_x.max()
    miny, maxy = all_y.min(), all_y.max()

    # Add small padding (meters)
    pad = 200  # adjust (100–500 works well for AIS)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # Keep correct aspect ratio (IMPORTANT for map coordinates)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("MDN K-Trajectory Prediction in {region}")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Mode Probability (π)")

    ax.legend()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{file_name}.png", dpi=200)
    plt.close()


def ade_per_agent(pred_abs, data):
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    dist = dist * data.y_mask

    ade_per_agent = dist.sum(dim=1) / data.y_mask.sum(dim=1).clamp_min(1)

    return ade_per_agent

def fde_per_agent(pred_abs, data, t):
    full_traj_mask = data.y_mask.sum(dim=1) >= t
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    fde = dist[:, t-1][full_traj_mask]
    return fde

def k_ade_per_agent(pred_abs_k, data):
    gt = data.y_pos.unsqueeze(1)
    dist = torch.norm(pred_abs_k - gt, dim=-1)
    dist = dist * data.y_mask.unsqueeze(1)

    ade_k = dist.sum(dim=2) / data.y_mask.sum(dim=1, keepdim=True).clamp_min(1)
    k_ade_min = ade_k.min(dim=1).values
    return k_ade_min

def k_fde_per_agent(pred_abs_k, data, t):
    gt = data.y_pos.unsqueeze(1)
    dist = torch.norm(pred_abs_k - gt, dim=-1)
    dist = dist * data.y_mask.unsqueeze(1)

    full_traj_mask = data.y_mask.sum(dim=1) >= t
    fde_k = dist[:, :, t-1]
    k_fde_min = fde_k[full_traj_mask].min(dim=1).values

    return k_fde_min

def full_eval(data_folder, model, region, bbox):
    for ship_group in ["sailing", "cargo", "passenger", "other"]: # "all", 
        logging.info("#"*20)
        logging.info(f"[SHIP GROUP]: {ship_group}")

        B, T = 512, 30

        test_loader, _ = graph_loader(
            data_folder=data_folder,
            flag="test",
            min_date=pd.Timestamp("2022-01-01"),
            max_date=pd.Timestamp("2024-01-01"),
            batch_size=B,
            pin_memory=True,
            pred_len=T,
            obs_len=60,
            max_edge_dist=500,
            shuffle=True,
            ship_group = ship_group,
        )

        path = DATA_FOLDER_PATH / f"maps/2_standardized/dma_10/{region}/"
        sl = SceneLoader(Rasterizer(bbox))

        scene_contiguous = np.ascontiguousarray(sl.load_scene(path))
        scene = torch.from_numpy(scene_contiguous).to(device).to(torch.float32)

        model = model.to(device)
        model.eval()

        with torch.inference_mode():
            total_pred_risk = 0.0
            total_min_pred_dist = 0.0
            total_high_risk = 0.0
            total_close_dist = 0.0
            total_collision_count = 0.0
            total_graphs = 0

            ade_sum = 0.0
            n_ade = 0
            fde_1_sum = 0.0
            n_fde_1 = 0
            fde_3_sum = 0.0
            n_fde_3 = 0
            fde_5_sum = 0.0
            n_fde_5 = 0

            k_ade_sum = 0.0
            n_k_ade = 0
            k_fde_1_sum = 0.0
            n_k_fde_1 = 0
            k_fde_3_sum = 0.0
            n_k_fde_3 = 0
            k_fde_5_sum = 0.0
            n_k_fde_5 = 0

            eval_time = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            i = 0

            for data in tqdm(test_loader, desc=f"Eval"):
                data = data.to(device, non_blocking=True)

                ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

                start_event.record()
                mdn_out = model(data, scene).view(B, T, 3, 5)

                pi = torch.softmax(mdn_out[..., 0], dim=-1)
                mu = mdn_out[..., 1:3]

                exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)  # [B, T, 2]
                pred_abs_pos = torch.cumsum(exp_rel, dim=1) * DE_NORMALIZE + data.x_pos[ego_idx, -1:, :]

                mu_k = mu.permute(0, 2, 1, 3)
                pred_abs_pos_k = torch.cumsum(mu_k, dim=2) * DE_NORMALIZE + data.x_pos[ego_idx, -1:, :].unsqueeze(1)

                end_event.record()
                torch.cuda.synchronize()
                step_time_ms = start_event.elapsed_time(end_event)
                eval_time += (step_time_ms / 1000.0)

                #### plotting
                if i < 10:
                    i += 1
                    agent_idx = 0  # ego already indexed
                    x_mask = data.x_mask[agent_idx].bool().cpu().numpy()
                    y_mask = data.y_mask[agent_idx].bool().cpu().numpy()

                    obs_pos = data.x_pos[agent_idx].cpu().numpy()[x_mask]
                    fut_pos = data.y_pos[agent_idx].cpu().numpy()[y_mask]
                    pred_k = pred_abs_pos_k[agent_idx].cpu().numpy()
                    pred_k = pred_k[:, y_mask, :]

                    last_valid_t = y_mask.sum() - 1
                    pi_k = pi[agent_idx, last_valid_t].cpu().numpy()

                    plot_mdn_k_trajectories(
                        file_name=f"epoch_{region}_{ship_group}_{i}_mdn_k_traj",
                        obs_pos=obs_pos,
                        fut_pos=fut_pos,
                        pred_abs_k=pred_k,
                        pi_k=pi_k,
                        region = region,
                    )
                else:
                    break

                ### end plotting

                ade = ade_per_agent(pred_abs_pos, data)
                ade_sum += ade.sum().item()
                n_ade += ade.numel()
                fde_1 = fde_per_agent(pred_abs_pos, data, 6)
                fde_1_sum += fde_1.sum().item()
                n_fde_1 += fde_1.numel()
                fde_3 = fde_per_agent(pred_abs_pos, data, 18)
                fde_3_sum += fde_3.sum().item()
                n_fde_3 += fde_3.numel()
                fde_5 = fde_per_agent(pred_abs_pos, data, 30)
                fde_5_sum += fde_5.sum().item()
                n_fde_5 += fde_5.numel()

                k_ade = k_ade_per_agent(pred_abs_pos_k, data)
                k_ade_sum += k_ade.sum().item()
                n_k_ade += k_ade.numel()
                k_fde_1 = k_fde_per_agent(pred_abs_pos_k, data, 6)
                k_fde_1_sum += k_fde_1.sum().item()
                n_k_fde_1 += k_fde_1.numel()
                k_fde_3 = k_fde_per_agent(pred_abs_pos_k, data, 18)
                k_fde_3_sum += k_fde_3.sum().item()
                n_k_fde_3 += k_fde_3.numel()
                k_fde_5 = k_fde_per_agent(pred_abs_pos_k, data, 30)
                k_fde_5_sum += k_fde_5.sum().item()
                n_k_fde_5 += k_fde_5.numel()

                (
                    pred_risk,
                    min_pred_dist,
                    high_risk_count,
                    close_dist_count,
                    collision_count,
                    count
                ) = compute_batch_collision_risk(data, pred_abs_pos)

                total_pred_risk += pred_risk
                total_min_pred_dist += min_pred_dist
                total_high_risk += high_risk_count
                total_close_dist += close_dist_count
                total_collision_count += collision_count
                total_graphs += count

            logging.info(f"Total Graphs: {total_graphs}")

            logging.info(f"Mean Pred Risk: {(total_pred_risk / total_graphs).item()}")
            logging.info(f"Mean Min Pred Distance: {(total_min_pred_dist / total_graphs).item()}")
            logging.info(f"High Risk Ratio: {(total_high_risk / total_graphs).item() * 100}")
            logging.info(f"Close Distance Ratio: {(total_close_dist / total_graphs).item() * 100}")
            logging.info(f"Collision Ratio: {(total_collision_count / total_graphs).item() * 100}")

            logging.info(f"ADE: {ade_sum / n_ade}")
            logging.info(f"FDE_1: {fde_1_sum / n_fde_1}")
            logging.info(f"FDE_2: {fde_3_sum / n_fde_3}")
            logging.info(f"FDE_3: {fde_5_sum / n_fde_5}")

            logging.info(f"min_ADE@3: {k_ade_sum / n_k_ade}")
            logging.info(f"min_FDE_1@3: {k_fde_1_sum / n_k_fde_1}")
            logging.info(f"min_FDE_2@3: {k_fde_3_sum / n_k_fde_3}")
            logging.info(f"min_FDE_3@3: {k_fde_5_sum / n_k_fde_5}")
            logging.info(f"eval_time {eval_time / 60:.2f} minutes / {len(test_loader)}")

def get_absolute(data, pred_rel_pos):
    ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
    last_pos = data.x_pos[ego_idx, -1, :].unsqueeze(1)
    pred_abs = torch.cumsum(pred_rel_pos, dim=1) * DE_NORMALIZE + last_pos
    return pred_abs


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    ### submodules:

    """
    best_ckpt_path = Path("checkpoints/traisformer/traisformer_path_best.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    prior_module = TrAISformer(ckpt["config"])
    prior_module.load_state_dict(ckpt["model_state_dict"])
    prior_module.eval()
    prior_module.requires_grad_(False)


    ### main module

    best_ckpt_path = Path(f"data/gmm/cluster_16/ais_gmm.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)

    trais = TrAISformer(ckpt["prior_config"]).to(device)
    trais.load_state_dict(ckpt["prior_state_dict"])
    trais.eval()
    trais.requires_grad_(False)

    ais_gmm = AIS_GMM(trais, n_clusters=ckpt["n_clusters"])
    ais_gmm.gmm = ckpt["gmm"]

    path = Path("data/gmm")
    sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))

    cluster_contiguous = np.ascontiguousarray(sl.load_cluster(path, 16))
    cluster_maps = torch.from_numpy(cluster_contiguous).to(device).to(torch.float32)
    prior_module = MAP_GMM(ais_gmm, cluster_maps)
    """
    regions = {
        "kiel": [10.12, 54.31, 10.33, 54.46],
        "aarhus": [10.21, 56.04, 10.47, 56.17],
        "odense": [10.42, 55.42, 10.68, 55.55],
        "little_belt": [9.64, 55.25, 9.90, 55.37],
    }

    best_ckpt_path = Path("checkpoints/nereus/nereus_map_best.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    cfg = ckpt["config"]

    for region, bbox in regions.items():

        path = DATA_FOLDER_PATH / f"maps/2_standardized/dma_10/{region}/"
        rasterizer = Rasterizer(bbox)
        sl = SceneLoader(rasterizer)
        density_contiguous = np.ascontiguousarray(sl.load_density(path))
        density_maps = torch.from_numpy(density_contiguous).to(device).to(torch.float32)

        model = NEREUS(
            config = cfg,
            static_module = True,
            social_module = None, # GAT(cfg), # GAT(cfg) EgoSocialPooling(cfg)
            map_module = True, # True
            prior_module =  None, # DensityIntent(density_maps), #DensityIntent(density_maps), # prior_module DensityIntent(density_maps)
            map_atte_module = False,
        )
        model.load_state_dict(ckpt["model_state_dict"])

        model.rasterizer = rasterizer
        if hasattr(model, "map_cnn") and model.map_cnn is not None:
            model.map_cnn.rasterizer = rasterizer
        if hasattr(model, "prior_cnn") and model.prior_cnn is not None:
            model.prior_cnn.rasterizer = rasterizer

        model.eval()

        logger(file_prefix=f"eval_dma_{region}_{best_ckpt_path.name}")
        logging.info(best_ckpt_path)

        data_folder = DATA_FOLDER_PATH / f"ais/4_features/dma_10/{region}"
        full_eval(data_folder, model, region, bbox)

"""

CUDA_VISIBLE_DEVICES=0 python -u src/t_eval/full_eval_nereus.py

"""