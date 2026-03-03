import torch

from utils.config import STEPS_PER_MINUTE

def ade_per_agent(pred_abs, data):
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    dist = dist * data.y_mask

    ade_per_agent = dist.sum(dim=1) / data.y_mask.sum(dim=1).clamp_min(1)

    return ade_per_agent

def fde_per_agent(pred_abs, data, t=5*STEPS_PER_MINUTE):
    full_traj_mask = data.y_mask.sum(dim=1) >= t
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    
    fde = dist[:, t-1][full_traj_mask]
    return fde

def k_ade_per_agent(pred_abs_k, data):
    gt = data.y_pos.unsqueeze(1)              
    dist = torch.norm(pred_abs_k - gt, dim=-1)
    dist = dist * data.y_mask.unsqueeze(1)

    ade_k = dist.sum(dim=2) / data.y_mask.sum(dim=1, keepdim=True).clamp_min(1)
    k_ade_min = ade_k.min(dim=1).values

    return k_ade_min

def k_fde_per_agent(pred_abs_k, data, t=5*STEPS_PER_MINUTE):
    gt = data.y_pos.unsqueeze(1)              
    dist = torch.norm(pred_abs_k - gt, dim=-1)
    dist = dist * data.y_mask.unsqueeze(1)

    # FDE
    full_traj_mask = data.y_mask.sum(dim=1) >= t
    fde_k = dist[:, :, t-1]
    k_fde_min = fde_k[full_traj_mask].min(dim=1).values

    return k_fde_min