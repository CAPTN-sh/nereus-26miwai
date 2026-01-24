import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from train.plot import plot_traj
DE_NORMALIZE = 100.0

def eval_oracle_k(k_pred_pos_rel, batch):
    *_, fut_pos_rel, fut_mask, _ = batch

    diff = (
        k_pred_pos_rel.cumsum(3)
        - fut_pos_rel.unsqueeze(1).cumsum(3)
    )
    l2 = torch.norm(diff, dim=2) * DE_NORMALIZE

    mask = fut_mask.squeeze(1).float()
    mask = mask.unsqueeze(1).expand_as(l2)

    valid_ade  = mask.sum(2) >= 12
    ade = torch.zeros_like(valid_ade, dtype=l2.dtype)
    ade[valid_ade] = (l2 * mask).sum(2)[valid_ade] / mask.sum(2)[valid_ade]
    ade_k = ade.min(1).values

    valid_fde = mask[:, :, -1] > 0
    fde = l2[:, :, -1]
    fde_k = fde.min(1).values[valid_fde.any(dim=1)]

    return ade_k, fde_k


def eval_loss(pred_pos_rel, batch, epoch=None):
    *_, fut_pos_rel, fut_mask, _ = batch

    diff = pred_pos_rel.cumsum(-1) - fut_pos_rel.cumsum(-1)
    l2 = torch.norm(diff, dim=1) * DE_NORMALIZE

    mask = fut_mask.squeeze(-1).float().view_as(l2)

    valid_ade = mask.sum(1) >= 12
    valid_fde = mask[:, -1] > 0

    ade = (l2 * mask).sum(1)[valid_ade] / mask.sum(1)[valid_ade]
    fde = l2[:, -1][valid_fde]

    return ade, fde

def eval(epoch, model: nn.Module, eval_loader: DataLoader, device, scene, trial_number, config):
    model.eval()

    n_batches = 0
    n_ade = 0
    n_fde = 0

    ade_sum = 0.0
    fde_sum = 0.0

    ade_k_sum = 0.0
    fde_k_sum = 0.0

    plot_cached = None
    plot = False

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            if n_batches >= 500:
                break
            batch = [t.to(device) for t in batch]
            n_batches += 1

            best_pred_pos_rel, k_pred_pos_rel = model.inference(batch, scene)
            ade, fde = eval_loss(best_pred_pos_rel, batch)

            ade_sum += ade.sum().item()
            n_ade += ade.numel()
            fde_sum += fde.sum().item()
            n_fde += ade.numel()

            ade_k, fde_k = eval_oracle_k(k_pred_pos_rel, batch)
            ade_k_sum += ade_k.sum().item()
            fde_k_sum += fde_k.sum().item()

            if plot and (plot_cached is None):
                obs_feat, obs_pos, obs_pos_rel, obs_mask, fut_pos, fut_pos_rel, fut_mask, seq_start_end = batch
                pred_pos = obs_pos[:, :, -1].unsqueeze(2) + best_pred_pos_rel.cumsum(dim=2) * DE_NORMALIZE
                plot_cached = [
                    obs_pos.detach().cpu(),
                    fut_pos.detach().cpu(),
                    pred_pos.detach().cpu(),
                    seq_start_end.detach().cpu(),
                ]

    ade = ade_sum / n_ade
    fde = fde_sum / n_fde

    ade_k = ade_k_sum / n_ade
    fde_k = fde_k_sum / n_fde

    logging.info(
        f"[Eval] ADE: {ade:.4f}, FDE: {fde:.4f}, ADE_k: {ade_k:.4f}, FDE_k: {fde_k:.4f}"
    )
    if plot:
        plot_traj(f"eval_{model.__class__.__name__}_{epoch}_{trial_number}", *plot_cached)

    model.train()

    return ade