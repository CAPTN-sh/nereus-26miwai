import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from functools import lru_cache


def mse(pred_rel, batch, config = None):
    *_, fut_rel, fut_mask, _ = batch
    mse = (pred_rel - fut_rel).pow(2).sum(dim=-1)[fut_mask].mean()

    return mse, {"mse": mse}

def fde_per_agent(pred_abs, fut_pos, fut_mask):
    B, T, _ = pred_abs.shape

    dist = torch.norm(pred_abs - fut_pos, dim=-1)  # [B, T]
    full_traj_mask = fut_mask.bool().all(dim=1)

    fde = dist[:, -1][full_traj_mask]
    return fde

def ade_per_agent(pred_abs, fut_pos, fut_mask):
    dist = torch.norm(pred_abs - fut_pos, dim=-1)     # [B, T]
    mask = fut_mask.float()

    dist_sum = (dist * mask).sum(dim=1)
    ade_agent = dist_sum / mask.sum(dim=1)

    return ade_agent

def eval_lstm(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    trial_number = 0,
    config=None,
):
    """
    Evaluate the TrAISformer intent head by KL loss and save one qualitative heat map.
    - Computes KL div vs. rasterized future bag-of-cells target (see loss.py).
    - Plots a single sample: predicted heat map + outline of GT future cells.
    """
    model.eval()

    num_batches = 0
    ade_sum = 0.0
    n_ade = 0
    fde_sum = 0.0
    n_fde = 0

    plotted = True
    n_plots = 3

    n_eval_batches = int(len(eval_loader) // 10)
    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches >= n_eval_batches:
                break

            num_batches += 1

            batch = [t.to(device, non_blocking=True) for t in batch]
            pred_rel = model(batch, scene)

            _, obs_pos, _, obs_mask, fut_pos, _, fut_mask, _ = batch
            pred_abs = torch.cumsum(pred_rel, dim=1) + obs_pos[:, -1:, :] 

            ade = ade_per_agent(pred_abs, fut_pos, fut_mask)
            ade_sum += ade.sum().item()
            n_ade += ade.numel()

            fde = fde_per_agent(pred_abs, fut_pos, fut_mask)
            fde_sum += fde.sum().item()
            n_fde += fde.numel()


            if not plotted:
                plotted = True 
                plot_traj(
                    f"lstm_trial_{trial_number}",
                    obs_pos, fut_pos, pred_abs,
                    n_plots,
                    obs_mask=obs_mask,
                    fut_mask=fut_mask,
                    epoch = epoch,
                )

    ade_avg = ade_sum / n_ade
    fde_avg = fde_sum / n_fde

    logging.info(f"[Eval] epoch={epoch} ade={ade_avg:.6f} fde={fde_avg:.6f}")
    return ade_avg

@lru_cache(maxsize=1)
def _load_background():
    map_path = Path("data/maps/2_standardized/fhkiel_train/kiel/land.geojson")
    background = gpd.read_file(map_path).to_crs("EPSG:25832")
    return background


def _to_xy(traj_i: torch.Tensor):
    """traj_i: [T, 2] on any device"""
    traj_i = traj_i.detach().cpu().float()
    xs = traj_i[:, 0].numpy()
    ys = traj_i[:, 1].numpy()
    return xs, ys


def _apply_mask(traj_i: torch.Tensor, mask_i: torch.Tensor | None):
    """Keep only valid timesteps if mask is provided."""
    if mask_i is None:
        return traj_i
    mask_i = mask_i.detach().cpu().bool()
    return traj_i.detach().cpu()[mask_i]


def plot_traj(
    file_name: str,
    obs_pos: torch.Tensor,   # [B, Tobs, 2]
    fut_pos: torch.Tensor,   # [B, Tfut, 2]
    pred_pos: torch.Tensor,  # [B, Tfut, 2]
    n_plots: int,
    obs_mask: torch.Tensor | None = None,  # [B, Tobs]
    fut_mask: torch.Tensor | None = None,  # [B, Tfut]
    epoch = 0,
):
    os.makedirs("images", exist_ok=True)

    background = _load_background()

    fig, ax = plt.subplots(figsize=(12, 12))  # much more reasonable
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    def plot_one(ax, traj, i, color, label=None, mask=None, draw_line=True):
        traj_i = _apply_mask(traj[i], None if mask is None else mask[i])
        xs, ys = _to_xy(traj_i)
        ax.scatter(xs, ys, color=color, alpha=0.5, s=12)

    n = min(n_plots, obs_pos.shape[0])
    for i in range(n):
        plot_one(ax, obs_pos, i, color="blue",  mask=obs_mask)
        plot_one(ax, fut_pos, i, color="green", mask=fut_mask)
        # predicted has no mask: assume full length
        plot_one(ax, pred_pos, i, color="red",  mask=fut_mask)

    # Legend that matches scatter/line visuals
    legend_elements = [
        Line2D([0], [0], color="blue",  marker="o", linestyle="-", label="Observed"),
        Line2D([0], [0], color="green", marker="o", linestyle="-", label="Ground Truth"),
        Line2D([0], [0], color="red",   marker="o", linestyle="-", label="Predicted"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    minx, miny, maxx, maxy = background.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"AIS Positions epoch:{epoch}")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    out_path = Path("images") / f"{file_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)