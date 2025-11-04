import logging

import torch
import torch.distributed as dist
from torch import amp, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.plot import plot_traj


def eval_oracle_k(k_pred_pos_rel, batch):
    _, _, _, _, fut_pos_rel, seq_start_end = batch
    ego = seq_start_end[:, 0]
    l2 = torch.norm(k_pred_pos_rel[ego] - fut_pos_rel[ego].unsqueeze(1), dim=2)

    ade_k = l2.mean(dim=2).min(dim=1).values.mean()
    fde_k = l2[:, :, -1].min(dim=1).values.mean()
    l2max_k = l2.max(dim=2).values.min(dim=1).values.mean()
    loss_k = (ade_k + fde_k + l2max_k) / 3

    return loss_k, {"ade_k": ade_k, "fde_k": fde_k, "l2max_k": l2max_k}


def eval_loss(pred_pos_rel, batch, epoch=None):
    _, _, _, _, fut_pos_rel, seq_start_end = batch
    ego = seq_start_end[:, 0]
    l2 = torch.norm(pred_pos_rel[ego] - fut_pos_rel[ego], dim=1)

    ade = l2.mean()
    fde = l2[:, -1].mean()
    l2max = l2.max(dim=1).values.mean()
    loss = (ade + fde + l2max) / 3

    return loss, {"ade": ade, "fde": fde, "l2max": l2max}


def eval(epoch, model: nn.Module, eval_loader: DataLoader, device, scene, scene_meta):
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

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            if num_batches > 100:
                break
            batch = [t.to(device) for t in batch]

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                best_pred_pos_rel, k_pred_pos_rel = model.inference(
                    batch, scene, scene_meta
                )
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

            if plot_cached is None and rank == 0:
                _, obs_pos, _, fut_pos, _, seq_start_end = batch
                pred_pos = obs_pos[:, :, -1].unsqueeze(2) + best_pred_pos_rel.cumsum(
                    dim=2
                )
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
        plot_traj(f"eval_{model.__class__.__name__}_{epoch}", *plot_cached)

    model.train()
