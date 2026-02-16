import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
import geopandas as gpd

def loss(pred_rel, data, config = None):
    #Data(x, x_pos, obs_mask, edge_index, edge_attr, y, y_pos, fut_mask)
    fut_rel = data.y_rel_pos
    fut_mask = data.y_mask

    err = (pred_rel - fut_rel).pow(2).sum(dim=-1)
    err = err * fut_mask

    mse = err.sum() / fut_mask.sum().clamp_min(1)

    # per agent
    # per_traj_mse = err.sum(dim=1) / fut_mask.sum(dim=1).clamp_min(1)
    # mse = per_traj_mse.mean()

    return mse, {"mse": mse}

def mdn_loss(mdn_out, data, config=None):
    """
    mdn_out: [B, T, K*5]
    data.y_rel_pos: [B, T, 2]
    data.y_mask: [B, T]
    """

    fut_rel = data.y_rel_pos          # [B, T, 2]
    fut_mask = data.y_mask            # [B, T]

    B, T, _ = mdn_out.shape
    K = config.mdn_modes

    # ---- unpack MDN parameters ----
    mdn_out = mdn_out.view(B, T, K, 5)

    pi = mdn_out[..., 0]              # [B, T, K]
    mu = mdn_out[..., 1:3]            # [B, T, K, 2]
    sigma = mdn_out[..., 3:5]         # [B, T, K, 2]

    pi = torch.softmax(pi, dim=-1)
    sigma = torch.exp(sigma) + 1e-6   # ensure positivity

    # ---- compute log-likelihood ----
    fut_rel = fut_rel.unsqueeze(2)    # [B, T, 1, 2]

    dist = torch.distributions.Normal(mu, sigma)
    log_prob = dist.log_prob(fut_rel).sum(dim=-1)  # [B, T, K]

    log_pi = torch.log(pi + 1e-6)
    log_likelihood = torch.logsumexp(log_pi + log_prob, dim=-1)  # [B, T]

    # ---- apply mask (same as MSE) ----
    nll = -log_likelihood * fut_mask

    loss = nll.sum() / fut_mask.sum().clamp_min(1)

    return loss, {"mdn_nll": loss}


def ade_per_agent(pred_abs, data):
    #Data(x, x_pos, obs_mask, edge_index, edge_attr, y, y_pos, fut_mask)
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    dist = dist * data.y_mask

    ade_per_agent = dist.sum(dim=1) / data.y_mask.sum(dim=1).clamp_min(1)

    return ade_per_agent

def fde_per_agent(pred_abs, data):
    full_traj_mask = data.y_mask.bool().all(dim=1)
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    fde = dist[:, -1][full_traj_mask]
    return fde

def eval(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    trial_number = 0,
    config=None,
):

    model.eval()

    num_batches = 0
    ade_sum = 0.0
    n_ade = 0
    fde_sum = 0.0
    n_fde = 0

    with torch.inference_mode():
        for data in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches >= 1000:
                break

            num_batches += 1

            data = data.to(device, non_blocking=True)
            pred_rel = model(data, scene)

            revert_norm = 100.0
            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
            pred_abs = torch.cumsum(pred_rel, dim=1) * revert_norm + data.x_pos[ego_idx, -1:, :]

            ade = ade_per_agent(pred_abs, data)
            ade_sum += ade.sum().item()
            n_ade += ade.numel()

            fde = fde_per_agent(pred_abs, data)
            fde_sum +=  fde.sum().item()
            n_fde += fde.numel()

    ade_avg = ade_sum / n_ade
    fde_avg = fde_sum / n_fde

    logging.info(f"[Eval] epoch={epoch} ade={ade_avg:.6f} fde={fde_avg:.6f}")
    return ade_avg



def unpack_mdn_eval(mdn_out, num_modes):
    """
    mdn_out: [B, T, K*5]
    returns:
        pi: [B, T, K]
        mu: [B, T, K, 2]
    """
    B, T, _ = mdn_out.shape
    K = num_modes

    mdn_out = mdn_out.view(B, T, K, 5)

    pi = torch.softmax(mdn_out[..., 0], dim=-1)
    mu = mdn_out[..., 1:3]

    return pi, mu


def k_ade_fde_per_agent(pred_abs_k, data):
    """
    pred_abs_k: [B, K, T, 2]
    returns:
        k_ade_min: [B]
        k_fde_min: [B_full]
    """
    B, K, T, _ = pred_abs_k.shape

    # ADE
    gt = data.y_pos.unsqueeze(1)               # [B, 1, T, 2]
    dist = torch.norm(pred_abs_k - gt, dim=-1) # [B, K, T]
    dist = dist * data.y_mask.unsqueeze(1)

    ade_k = dist.sum(dim=2) / data.y_mask.sum(dim=1, keepdim=True).clamp_min(1)
    k_ade_min = ade_k.min(dim=1).values

    # FDE
    full_traj_mask = data.y_mask.bool().all(dim=1)
    fde_k = dist[:, :, -1]                      # [B, K]
    k_fde_min = fde_k[full_traj_mask].min(dim=1).values

    return k_ade_min, k_fde_min


def eval_mdn(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    trial_number=0,
    config=None,
):

    model.eval()

    ade_sum = fde_sum = 0.0
    k_ade_sum = k_fde_sum = 0.0
    n_ade = n_fde = 0
    n_k_ade = n_k_fde = 0

    mdn_loss_sum = 0.0
    n_mdn = 0

    revert_norm = 100.0

    with torch.inference_mode():
        for i, data in enumerate(tqdm(eval_loader, desc="Evaluating (MDN)")):
            if i >= 1000:
                break

            data = data.to(device, non_blocking=True)
            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

            # ---- forward ----
            mdn_out = model(data, scene)  # [B, T, K*5]

            # ---- MDN loss (TUNING METRIC) ----
            loss_val, _ = mdn_loss(mdn_out, data, config)
            mdn_loss_sum += loss_val.item()
            n_mdn += 1

            # ---- unpack MDN for displacement metrics ----
            pi, mu = unpack_mdn_eval(mdn_out, config.mdn_modes)

            # ---- expected trajectory ----
            exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)  # [B, T, 2]
            pred_abs = (
                torch.cumsum(exp_rel, dim=1) * revert_norm
                + data.x_pos[ego_idx, -1:, :]
            )

            ade = ade_per_agent(pred_abs, data)
            ade_sum += ade.sum().item()
            n_ade += ade.numel()

            fde = fde_per_agent(pred_abs, data)
            fde_sum += fde.sum().item()
            n_fde += fde.numel()

            # ---- K trajectories (min over modes) ----
            mu_k = mu.permute(0, 2, 1, 3)

            pred_abs_k = (
                torch.cumsum(mu_k, dim=2) * revert_norm
                + data.x_pos[ego_idx, -1:, :].unsqueeze(1)
            )

            k_ade, k_fde = k_ade_fde_per_agent(pred_abs_k, data)

            k_ade_sum += k_ade.sum().item()
            n_k_ade += k_ade.numel()

            k_fde_sum += k_fde.sum().item()
            n_k_fde += k_fde.numel()

            if i % 50 == 0:
                agent_idx = 0  # ego already indexed

                # find best mode by FDE
                gt_final = data.y_pos[agent_idx, -1]                 # [2]
                pred_final_k = pred_abs_k[agent_idx, :, -1]          # [K, 2]
                fde_k = torch.norm(pred_final_k - gt_final, dim=-1)
                best_k = torch.argmin(fde_k).item()

                obs_pos = data.x_pos[agent_idx].cpu().numpy()        # [T_obs, 2]
                fut_pos = data.y_pos[agent_idx].cpu().numpy()        # [T_fut, 2]
                pred_k = pred_abs_k[agent_idx].cpu().numpy()         # [K, T_fut, 2]

                plot_mdn_k_trajectories(
                    file_name=f"epoch_{epoch}_{i}_mdn_k_traj",
                    obs_pos=obs_pos,
                    fut_pos=fut_pos,
                    pred_abs_k=pred_k,
                    best_k=best_k,
                )

    # ---- averages ----
    ade_avg = ade_sum / n_ade
    fde_avg = fde_sum / n_fde
    k_ade_avg = k_ade_sum / n_k_ade
    k_fde_avg = k_fde_sum / n_k_fde
    mdn_loss_avg = mdn_loss_sum / n_mdn

    logging.info(
        f"[Eval] epoch={epoch} "
        f"NLL={mdn_loss_avg:.4f} "
        f"ADE={ade_avg:.4f} FDE={fde_avg:.4f} "
        f"K-ADE={k_ade_avg:.4f} K-FDE={k_fde_avg:.4f}"
    )

    # ---- return tuning metric FIRST ----
    return mdn_loss_avg


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os


def plot_mdn_k_trajectories(
    file_name,
    obs_pos,        # [T_obs, 2]
    fut_pos,        # [T_fut, 2]
    pred_abs_k,     # [K, T_fut, 2]
    best_k: int,    # index of best mode
):
    fig, ax = plt.subplots(figsize=(20, 20))

    # ---- background map ----
    map_path = "/home/bbi/nereus/assets/maps/2_standardized/fh_10/kiel/land.geojson"
    background = gpd.read_file(map_path).to_crs("EPSG:25832")
    background.plot(ax=ax, facecolor="lightgray", edgecolor="black", alpha=0.5)

    # ---- observed ----
    ax.scatter(
        obs_pos[:, 0],
        obs_pos[:, 1],
        color="blue",
        marker="o",
        s=3,
        alpha=0.6,
        label="Observed",
    )

    # ---- ground truth future ----
    ax.scatter(
        fut_pos[:, 0],
        fut_pos[:, 1],
        color="green",
        marker="o",
        s=3,
        alpha=0.6,
        label="Ground Truth",
    )

    # ---- K predicted trajectories ----
    K = pred_abs_k.shape[0]
    for k in range(K):
        traj = pred_abs_k[k]
        if k == best_k:
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                color="red",
                marker="o",
                s=4,
                alpha=0.6,
                label="Best MDN Mode",
                zorder=5,
            )
        else:
            ax.scatter(
                traj[:, 0],
                traj[:, 1],
                color="orange",
                s=3,
                alpha=0.3,
                zorder=2,
            )

    # ---- styling ----
    minx, miny, maxx, maxy = background.total_bounds
    pad_x = (maxx - minx) * 0.02
    pad_y = (maxy - miny) * 0.02
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title("MDN K-Trajectory Prediction")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    ax.legend()
    os.makedirs("images", exist_ok=True)
    plt.savefig(f"images/{file_name}.png", dpi=200)
    plt.close()
