import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from train.plot import plot_traj


def eval(plot_name, model: nn.Module, eval_loader: DataLoader, device, scene, scene_meta):
    dist_is_init = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_is_init else 0

    model.eval()

    num_traj = 0
    ade_sum = 0.0
    fde_sum = 0.0
    l2max_sum = 0.0
    
    plot_cached = None

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end = [
                tensor.to(device) for tensor in batch
            ]
            obs_pos_last = obs_pos[:, :, -1]

            #with amp.autocast(device_type="cuda"):
            pred_pos_rel_best = model.inference(
                obs_feat, obs_pos_last, obs_pos_rel, seq_start_end, scene, scene_meta
            )
            pred_pos_rel_best = pred_pos_rel_best.squeeze(1)

            diff = pred_pos_rel_best - fut_pos_rel
            l2_per_t = torch.norm(diff, dim=1)

            num_traj += l2_per_t.size(0)

            ade_sum += l2_per_t.sum().item()
            fde_sum += l2_per_t[:, -1].sum().item()
            l2max_sum += l2_per_t.max(dim=1).values.sum().item()

            if plot_cached is None and rank == 0:
                plot_cached = (
                    obs_pos.detach().cpu(),
                    fut_pos.detach().cpu(),
                    pred_pos_rel_best.detach().cpu(),
                    seq_start_end.detach().cpu(),
                )

    if dist_is_init:
        dist.all_reduce(ade_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(fde_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(l2max_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_traj, op=dist.ReduceOp.SUM)


    ade = ade_sum / num_traj * 12
    fde = fde_sum / num_traj
    l2_maxT = l2max_sum / num_traj

    if rank == 0:
        print(f"[Eval] ADE: {ade:.4f}, FDE: {fde:.4f}, l2_maxT: {l2_maxT:.4f}")

        obs_pos, fut_pos, pred_pos_rel_best, seq_start_end = plot_cached
        plot_traj(plot_name, obs_pos, fut_pos, pred_pos_rel_best, seq_start_end)

    model.train()