import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging

def rec_loss(output, data, config = None):
    #Data(x, x_pos, obs_mask, edge_index, edge_attr, y, y_pos, fut_mask)
    rec, _ = output

    ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
    traj_feat = data.x[ego_idx, :, :]
    mask = data.x_mask[ego_idx, :]

    fut_pos_rel = data.y_rel_pos.unsqueeze(1)
    fut_mask = data.y_mask

    err = (traj_feat - rec).pow(2).mean(dim=-1) 
    err = err * mask

    mse = err.sum() / mask.sum().clamp_min(1)

    return mse, {"mse": mse}

def ade_per_agent(output, data):
    rec, _ = output

    ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
    traj_feat = data.x[ego_idx, :, :2]
    mask = data.x_mask[ego_idx, :]

    revert_norm = 100.0
    dist = torch.norm(rec[:, :, :2] - traj_feat, dim=-1) * revert_norm
    dist = dist * mask

    ade_per_agent = dist.sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    return ade_per_agent

def rec_eval(
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
    mse_sum = 0.0
    n_mse = 0
    ade_sum = 0.0
    n_ade = 0.0
    n_eval_batches = int(len(eval_loader) // 10)
    with torch.inference_mode():
        for data in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches >= n_eval_batches:
                break

            num_batches += 1

            data = data.to(device, non_blocking=True)
            output = model(data, scene)

            mse, _ = rec_loss(output, data)
            mse_sum += mse.item()
            n_mse += 1

            ade = ade_per_agent(output, data)
            ade_sum += ade.sum().item()
            n_ade += ade.numel()

    mse_avg = mse_sum / n_mse
    ade_avg = ade_sum / n_ade

    logging.info(f"[Eval] epoch={epoch} mse={mse_avg:.6f} ade={ade_avg:.6f}")
    return mse_avg
