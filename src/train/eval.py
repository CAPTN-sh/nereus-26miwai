import logging

import torch
import torch.distributed as dist
from torch import amp, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.plot import plot_traj
DE_NORMALIZE = 100.0

def eval_oracle_k(k_pred_pos_rel, batch):
    *_, fut_pos_rel, fut_mask, _ = batch

    diff = (
        k_pred_pos_rel.cumsum(3)
        - fut_pos_rel.unsqueeze(1).cumsum(3)
    )
    l2 = torch.norm(diff, dim=2) * DE_NORMALIZE      # [B, K, T]

    mask = fut_mask.squeeze(1).float()               # [B, T]
    mask = mask.unsqueeze(1).expand_as(l2)           # [B, K, T]


    valid_ade  = mask.sum(2) > 0                           # [B, K]
    ade = torch.zeros_like(valid_ade, dtype=l2.dtype)
    ade[valid_ade] = (l2 * mask).sum(2)[valid_ade] / mask.sum(2)[valid_ade]
    ade_k = ade.min(1).values.mean()

    valid_fde = mask[:, :, -1] > 0
    fde = l2[:, :, -1]
    fde_k = fde.min(1).values[valid_fde.any(dim=1)].mean()

    l2max = l2.masked_fill(mask == 0, -float("inf")).max(2).values
    l2max_k = l2max.min(1).values[valid_ade.any(dim=1)].mean()

    loss_k = (ade_k + fde_k + l2max_k) / 3

    return loss_k, {"ade_k": ade_k, "fde_k": fde_k, "l2max_k": l2max_k}


def eval_loss(pred_pos_rel, batch, epoch=None):
    *_, fut_pos_rel, fut_mask, _ = batch

    diff = pred_pos_rel.cumsum(-1) - fut_pos_rel.cumsum(-1)
    l2 = torch.norm(diff, dim=1) * DE_NORMALIZE   # [B, T]

    mask = fut_mask.squeeze(-1).float().view_as(l2)

    valid_ade = mask.sum(1) > 0
    valid_fde = mask[:, -1] > 0

    # ADE (agent-level)
    ade_per_agent = torch.zeros_like(valid_ade, dtype=l2.dtype)
    ade_per_agent[valid_ade] = (
        (l2 * mask).sum(1)[valid_ade] / mask.sum(1)[valid_ade]
    )
    ade = ade_per_agent[valid_ade].mean()

    # FDE (agent-level)
    fde_per_agent = l2[:, -1]
    fde_per_agent[~valid_fde] = float("inf")
    fde = fde_per_agent[valid_fde].mean()

    # L2 max (agent-level, masked)
    l2max_per_agent = l2.masked_fill(mask == 0, -float("inf")).max(1).values
    l2max = l2max_per_agent[valid_ade].mean()

    loss = (ade + fde + l2max) / 3

    return loss, {"ade": ade, "fde": fde, "l2max": l2max}

def eval(epoch, model: nn.Module, eval_loader: DataLoader, device, scene, trial_number, config):
    dist_is_init = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_is_init else 0

    model.eval()

    num_batches = torch.tensor(0.0, device=device)

    loss_sum = torch.tensor(0.0, device=device)
    ade_sum = torch.tensor(0.0, device=device)
    fde_sum = torch.tensor(0.0, device=device)
    l2max_sum = torch.tensor(0.0, device=device)

    loss_k_sum = torch.tensor(0.0, device=device)
    ade_k_sum = torch.tensor(0.0, device=device)
    fde_k_sum = torch.tensor(0.0, device=device)
    l2max_k_sum = torch.tensor(0.0, device=device)

    plot_cached = None
    eval_oracle = False
    plot = False

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            if num_batches >= 500:
                break
            batch = [t.to(device) for t in batch]

            best_pred_pos_rel, k_pred_pos_rel = model.inference(batch, scene)
            loss, loss_dict = eval_loss(best_pred_pos_rel, batch)
            eval_oracle = k_pred_pos_rel is not None
            if eval_oracle:
                loss_k, loss_dict_k = eval_oracle_k(k_pred_pos_rel, batch)

            num_batches += 1

            loss_sum += loss.item()
            ade_sum += loss_dict["ade"].item()
            fde_sum += loss_dict["fde"].item()
            l2max_sum += loss_dict["l2max"].item()

            if eval_oracle:
                loss_k_sum += loss_k.item()
                ade_k_sum += loss_dict_k["ade_k"].item()
                fde_k_sum += loss_dict_k["fde_k"].item()
                l2max_k_sum += loss_dict_k["l2max_k"].item()

            if plot and (plot_cached is None) and (rank == 0):
                obs_feat, obs_pos, obs_pos_rel, obs_mask, fut_pos, fut_pos_rel, fut_mask, seq_start_end = batch
                pred_pos = obs_pos[:, :, -1].unsqueeze(2) + best_pred_pos_rel.cumsum(dim=2) * DE_NORMALIZE
                plot_cached = [
                    obs_pos.detach().cpu(),
                    fut_pos.detach().cpu(),
                    pred_pos.detach().cpu(),
                    seq_start_end.detach().cpu(),
                ]

    if dist_is_init:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(ade_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(fde_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(l2max_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

        if eval_oracle:
            dist.all_reduce(loss_k_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(ade_k_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(fde_k_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(l2max_k_sum, op=dist.ReduceOp.SUM)

    loss = loss_sum / num_batches
    ade = ade_sum / num_batches
    fde = fde_sum / num_batches
    l2max = l2max_sum / num_batches

    if eval_oracle:
        loss_k = loss_k_sum / num_batches
        ade_k = ade_k_sum / num_batches
        fde_k = fde_k_sum / num_batches
        l2max_k = l2max_k_sum / num_batches

    if rank == 0:
        logging.info(
            f"[Eval] loss: {loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, l2_max: {l2max:.4f}"
        )
        if eval_oracle:
            logging.info(
                f"[Eval] loss_k: {loss_k:.4f}, ADE_k: {ade_k:.4f}, FDE_k: {fde_k:.4f}, l2_max_k: {l2max_k:.4f}"
            )
        if plot:
            plot_traj(f"eval_{model.__class__.__name__}_{epoch}_{trial_number}", *plot_cached)

    model.train()

    return loss.item()
