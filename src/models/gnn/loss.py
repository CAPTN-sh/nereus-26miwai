import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
import geopandas as gpd

def loss(pred_rel, batch, config = None):
    #Data(x, x_pos, obs_mask, edge_index, edge_attr, y, y_pos, fut_mask)
    ego_mask = batch.is_ego

    pred_ego = pred_rel[ego_mask]
    gt_ego = batch.y[ego_mask][:, :, :2]
    fut_mask = batch.fut_mask[ego_mask]

    err = (pred_ego - gt_ego).pow(2).sum(dim=-1)
    err = err * fut_mask

    mse = err.sum() / fut_mask.sum().clamp_min(1)

    # per agent
    # per_traj_mse = err.sum(dim=1) / fut_mask.sum(dim=1).clamp_min(1)
    # mse = per_traj_mse.mean()

    return mse, {"mse": mse}

def ade_per_agent(pred_abs, batch):
    #Data(x, x_pos, obs_mask, edge_index, edge_attr, y, y_pos, fut_mask)
    ego_mask = batch.is_ego

    pred_ego = pred_abs[ego_mask]
    gt_ego = batch.y_pos[ego_mask]
    fut_mask = batch.fut_mask[ego_mask]

    dist = torch.norm(pred_ego - gt_ego, dim=-1)
    dist = dist * fut_mask

    ade_per_agent = dist.sum(dim=1) / fut_mask.sum(dim=1).clamp_min(1)

    return ade_per_agent

def fde_per_agent(pred_abs, batch):
    ego_mask = batch.is_ego

    pred_ego = pred_abs[ego_mask]
    gt_ego = batch.y_pos[ego_mask]
    fut_mask = batch.fut_mask[ego_mask]
    full_traj_mask = fut_mask.bool().all(dim=1)

    dist = torch.norm(pred_ego - gt_ego, dim=-1)

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
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches >= 500:
                break

            num_batches += 1

            batch = batch.to(device, non_blocking=True)
            pred_rel = model(batch, scene)

            revert_norm = 100.0
            pred_abs = torch.cumsum(pred_rel, dim=1) * revert_norm + batch.x_pos[:, -1:, :]

            ade = ade_per_agent(pred_abs, batch)
            ade_sum += ade.sum().item()
            n_ade += ade.numel()

            fde = fde_per_agent(pred_abs, batch)
            fde_sum +=  fde.sum().item()
            n_fde += fde.numel()

    ade_avg = ade_sum / n_ade
    fde_avg = fde_sum / n_fde

    logging.info(f"[Eval] epoch={epoch} ade={ade_avg:.6f} fde={fde_avg:.6f}")
    return ade_avg

def _apply_mask(traj_i: torch.Tensor, mask_i: torch.Tensor | None):
    """Keep only valid timesteps if mask is provided."""
    if mask_i is None:
        return traj_i
    mask_i = mask_i.detach().cpu().bool()
    return traj_i.detach().cpu()[mask_i]
