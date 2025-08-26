import logging

import torch
import torch.distributed as dist
from torch import amp, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.plot import plot_traj


def eval_loss(pred_pos_rel, batch):
    _, _, _, _, fut_pos_rel, seq_start_end = batch
    ego_vessels = seq_start_end[:, 0]
    diff = (pred_pos_rel - fut_pos_rel)[ego_vessels]
    l2_per_t = torch.norm(diff, dim=1)

    ade = l2_per_t.mean()
    fde = l2_per_t[:, -1].mean()
    l2max = l2_per_t.max(dim=1).values.mean()
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

    plot_cached = None

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = [t.to(device) for t in batch]

            with amp.autocast(device_type="cuda"):
                pred_pos_rel = model.inference(batch, scene, scene_meta)
                loss, loss_dict = eval_loss(pred_pos_rel, batch)

            num_batches += 1
            loss_sum += loss.item()
            ade_sum += loss_dict["ade"].item()
            fde_sum += loss_dict["fde"].item()
            l2max_sum += loss_dict["l2max"].item()

            if plot_cached is None and rank == 0:
                _, obs_pos, _, fut_pos, _, seq_start_end = batch
                pred_pos = obs_pos[:, :, -1].unsqueeze(2) + pred_pos_rel.cumsum(dim=2)
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

    loss = loss_sum / num_batches
    ade = ade_sum / num_batches
    fde = fde_sum / num_batches
    l2max = l2max_sum / num_batches

    if rank == 0:
        logging.info(
            f"[Eval] loss: {loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, l2_max: {l2max:.4f}"
        )
        plot_traj(f"eval_{model.__class__.__name__}_{epoch}", *plot_cached)

    model.train()
