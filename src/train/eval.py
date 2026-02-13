import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DE_NORMALIZE = 100.0

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

def eval(epoch, model: nn.Module, eval_loader: DataLoader, device, scene, trial_number, config):
    model.eval()

    n_batches = 0
    n_ade = 0
    n_fde = 0
    n_k_ade = 0
    n_k_fde = 0

    ade_sum = 0.0
    fde_sum = 0.0

    k_ade_sum = 0.0
    k_fde_sum = 0.0

    with torch.inference_mode():
        for data in tqdm(eval_loader, desc="Evaluating"):
            if n_batches >= 500:
                break
            data = data.to(device)
            n_batches += 1

            best_pred_pos_rel, k_pred_pos_rel = model.inference(data, scene)

            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
            last_pos = data.x_pos[ego_idx, -1:, :]
            pred_abs = torch.cumsum(best_pred_pos_rel, dim=1) * DE_NORMALIZE + last_pos

            ade = ade_per_agent(pred_abs, data)
            ade_sum += ade.sum().item()
            n_ade += ade.numel()

            fde = fde_per_agent(pred_abs, data)
            fde_sum +=  fde.sum().item()
            n_fde += fde.numel()

            pred_abs_k = torch.cumsum(k_pred_pos_rel, dim=2) * DE_NORMALIZE + last_pos.unsqueeze(1)
            k_ade, k_fde = k_ade_fde_per_agent(pred_abs_k, data)

            k_ade_sum += k_ade.sum().item()
            n_k_ade += k_ade.numel()

            k_fde_sum += k_fde.sum().item()
            n_k_fde += k_fde.numel()

    ade = ade_sum / n_ade
    fde = fde_sum / n_fde

    ade_k = k_ade_sum / n_k_ade
    fde_k = k_fde_sum / n_k_fde

    logging.info(f"[Eval] ADE: {ade:.4f}, FDE: {fde:.4f}, ADE_k: {ade_k:.4f}, FDE_k: {fde_k:.4f}")

    model.train()

    return ade