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


def de_series_sums(pred_abs, pred_abs_k, data):
    """Per-timestep displacement-error sums over a batch, for the DE series.

    Computes, for every prediction timestep at once (rather than only the
    1/3/5-minute FDE snapshots), the masked-summed error and valid count so
    batches can be accumulated and then divided:

      * ``de_sum[T]``   – summed L2 error of the expected trajectory
      * ``k_de_sum[T]`` – summed min-over-K L2 error
      * ``count[T]``    – number of valid (masked) agents at each step

    Accumulate ``*_sum`` and ``count`` across the loader, then ``de_sum / count``
    gives the per-step DE (a.k.a. FDE-at-t) series and ``de_series.mean()`` the
    ADE. Replaces the per-metric MetricAccumulators.

    Args:
        pred_abs:   [B, T, 2] expected absolute positions.
        pred_abs_k: [B, K, T, 2] per-mode absolute positions.
        data:       batch with ``y_pos`` [B, T, 2] and ``y_mask`` [B, T].

    Returns:
        (de_sum, k_de_sum, count), each a [T] tensor.
    """
    mask = data.y_mask.float()                                        # [B, T]
    de = torch.norm(pred_abs - data.y_pos, dim=-1)                    # [B, T]
    de_k = torch.norm(pred_abs_k - data.y_pos.unsqueeze(1), dim=-1)   # [B, K, T]
    k_de = de_k.min(dim=1).values                                    # [B, T]

    de_sum = (de * mask).sum(dim=0)                                  # [T]
    k_de_sum = (k_de * mask).sum(dim=0)                              # [T]
    count = mask.sum(dim=0)                                          # [T]
    return de_sum, k_de_sum, count
