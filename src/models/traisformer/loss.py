from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from models.traisformer.params import TraisformerParams
from models.utils.maps.rasterize import Rasterizer
from torchvision.transforms.functional import gaussian_blur

RASTER = Rasterizer(TraisformerParams().bbox)

def loss_intent_heatmap2(
    output: Dict[str, torch.Tensor],
    batch,
    config = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    logits = output["intent_logits"]
    log_probs = F.log_softmax(logits.flatten(1), dim=1)

    *_, fut_pos, fut_mask, fin_pos = batch
    # target = rasterize_heatmap(fut_pos, fut_mask)
    target = rasterize_destination2(fin_pos)

    ce_loss = torch.sum(-target * log_probs, dim=1).mean()
    # nn.KLDivLoss

    # https://arxiv.org/pdf/1701.06548
    n_cells = RASTER.x_size * RASTER.y_size
    entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
    entropy = entropy / np.log(n_cells)

    eps = 1e-2
    loss = ce_loss - eps * entropy
    return loss, {"ce": ce_loss, "entropy": entropy}


def rasterize_destination2(fin_pos):
    B, *_ = fin_pos.shape
    x_bins, y_bins, *_ = RASTER.get_total_grid_sizes()

    x_idx = torch.floor((fin_pos[:, 0] - RASTER.x_min) / RASTER.pos_res).to(torch.int64)
    y_idx = torch.floor((fin_pos[:, 1] - RASTER.y_min) / RASTER.pos_res).to(torch.int64)

    grid = torch.zeros((B, x_bins, y_bins), device=fin_pos.device)
    indices = torch.stack([torch.arange(B, device=fin_pos.device), x_idx, y_idx], dim=0)
    grid[indices[0], indices[1], indices[2]] = 1.0

    grid = gaussian_blur(grid, [13, 13], [2.0, 2.0])
    grid = grid / (grid.sum(dim=(1, 2), keepdim=True) + 1e-8)

    # https://arxiv.org/pdf/1701.06548
    # eps = 0.1
    #grid = (1 - eps) * grid + eps / (x_bins * y_bins)

    return grid.view(B, -1)



def rasterize_heatmap2(fut_pos, fut_mask):
    device = fut_pos.device
    x_bins, y_bins, *_ = RASTER.get_total_grid_sizes()

    R = int(2.0 * 4)

    d = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
    oi, oj = torch.meshgrid(d, d, indexing="ij")
    off_i = oi.reshape(-1)
    off_j = oj.reshape(-1)

    sigma_m = 75.0
    inv_2sigma2 = torch.tensor(1.0 / (2.0 * sigma_m * sigma_m),
                               device=device, dtype=torch.float32)

    B, T, _ = fut_pos.shape
    K = off_i.numel()

    flat_pos = fut_pos[fut_mask]  # (M,2)
    b_idx = torch.arange(B, device=device)[:, None].expand(B, T)[fut_mask]  # (M,)

    # make sure these are plain floats (or tensors on device)
    x_min = float(RASTER.x_min)
    y_min = float(RASTER.y_min)
    res   = float(RASTER.pos_res)

    ci = torch.floor((flat_pos[:, 0] - x_min) / res).to(torch.int64)
    cj = torch.floor((flat_pos[:, 1] - y_min) / res).to(torch.int64)

    cx = x_min + (ci.to(torch.float32) + 0.5) * res
    cy = y_min + (cj.to(torch.float32) + 0.5) * res
    fx = flat_pos[:, 0] - cx
    fy = flat_pos[:, 1] - cy

    ni = ci[:, None] + off_i[None, :].to(torch.int64)
    nj = cj[:, None] + off_j[None, :].to(torch.int64)

    dx = fx[:, None] - off_i[None, :] * res
    dy = fy[:, None] - off_j[None, :] * res
    d2 = dx * dx + dy * dy
    val = torch.exp(-d2 * inv_2sigma2)  # (M,K)

    inb = (ni >= 0) & (ni < x_bins) & (nj >= 0) & (nj < y_bins)
    ni = ni[inb]; nj = nj[inb]; val = val[inb]

    cells_per = x_bins * y_bins
    flat_idx = (ni * y_bins + nj) + b_idx.repeat_interleave(K)[inb.view(-1)] * cells_per

    out = torch.zeros(B * cells_per, device=device, dtype=torch.float32)
    out.scatter_reduce_(0, flat_idx, val, reduce="amax", include_self=True)

    grid = out.view(B, x_bins, y_bins)
    grid = grid / (grid.sum(dim=(1, 2), keepdim=True) + 1e-8)

    # https://arxiv.org/pdf/1701.06548
    #eps = 0.1
    #grid = (1 - eps) * grid + eps / (x_bins * y_bins)

    return grid.view(B, -1)