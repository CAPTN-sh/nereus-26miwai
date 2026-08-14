
import torch
import torch.nn.functional as F

from data.map.rasterize import Rasterizer
from models.traisformer.params import TraisformerParams
from utils.config import TRAIN_BBOX

RASTER = Rasterizer(TRAIN_BBOX)

def loss_heatmap(output, data, config: TraisformerParams):
    if config.pred_scope == "path":
        return loss_occupancy_heatmap(output, data, config)
    if config.pred_scope == "destination":
        return loss_intent_heatmap(output, data, config)

def loss_intent_heatmap(
    output,
    data,
    config=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, z = output
    logits = logits[data.fin_pos_mask]

    assert logits.shape[1] == 1
    logits = logits.squeeze(1)
    B, H, W = logits.shape

    # 1. Wahrscheinlichkeiten berechnen
    probs_fine = F.softmax(logits.view(B, -1), dim=1) # (B, H*W)
    log_probs_fine = F.log_softmax(logits.view(B, -1), dim=1)

    # 2. Target Erstellung (Ground Truth)
    fin_pos = data.fin_pos[data.fin_pos_mask]
    target_fine = rasterize_destination(fin_pos) # (B, H, W)
    target_fine_flat = target_fine.view(B, -1)

    # Wo liegt das Schiff wirklich? (Index des Pixels)
    target_indices = torch.argmax(target_fine_flat, dim=1)

    # 3. Fine-Resolution Loss
    ce_fine = torch.sum(-target_fine_flat * log_probs_fine, dim=1).mean()

    # 4. Multi-Resolution / Coarse Loss (10x10 Pooling)
    pool_size = config.coarse_loss_pool_size
    # Wir bringen probs_fine zurück in 2D für das Pooling
    probs_fine_2d = probs_fine.view(B, 1, H, W)
    probs_coarse = F.avg_pool2d(probs_fine_2d, kernel_size=pool_size) * (pool_size**2)
    target_coarse = F.avg_pool2d(target_fine.unsqueeze(1), kernel_size=pool_size) * (pool_size**2)

    ce_coarse = torch.sum(-target_coarse.view(B, -1) * torch.log(probs_coarse.view(B, -1) + 1e-8), dim=1).mean()

    # 5. Wissenschaftliche Metriken (ohne Gradientenberechnung)
    with torch.no_grad():
        # --- Hit Rates ---

        top1_indices = torch.argmax(probs_fine, dim=1)
        hit1 = (top1_indices == target_indices).float().mean()

        _, top5_indices = torch.topk(probs_fine, k=5, dim=1)
        hit5 = (top5_indices == target_indices.unsqueeze(1)).any(dim=1).float().mean()

        # --- Mean Target Probability (MTP) ---
        p_gt = probs_fine.gather(1, target_indices.unsqueeze(1)).mean()

    # 6. Gesamt-Loss
    loss = ce_fine + config.coarse_loss_beta * ce_coarse

    return loss, {
        "ce_fine": ce_fine,
        "ce_coarse": ce_coarse,
        "p_gt": p_gt,
        "hit1": hit1,
        "hit5": hit5,
    }

def rasterize_destination(fin_pos):
    B = fin_pos.shape[0]
    x_bins, y_bins, *_ = RASTER.get_total_grid_sizes()

    x_idx = torch.floor((fin_pos[:, 0] - RASTER.x_min) / RASTER.pos_res).to(torch.int64)
    y_idx = torch.floor((fin_pos[:, 1] - RASTER.y_min) / RASTER.pos_res).to(torch.int64)

    x_idx = x_idx.clamp(0, x_bins - 1)
    y_idx = y_idx.clamp(0, y_bins - 1)

    grid = torch.zeros((B, x_bins, y_bins), device=fin_pos.device)
    grid[torch.arange(B), x_idx, y_idx] = 1.0

    return grid

def loss_occupancy_heatmap(
    output,
    data,
    config=None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, z = output
    assert logits.shape[1] == 1
    logits = logits.squeeze(1)

    B, H, W = logits.shape

    # 1. Probabilities (Softmax over the entire spatial grid)
    probs_fine = F.softmax(logits.view(B, -1), dim=1) # (B, H*W)
    log_probs_fine = F.log_softmax(logits.view(B, -1), dim=1)

    # 2. Ground Truth & Normalization
    target_fine = data.y_heatmap.view(B, 1, H, W)
    target_flat = target_fine.view(B, -1)

    # 3. Fine-Resolution Loss (Categorical Cross Entropy)
    ce_fine = torch.sum(-target_flat * log_probs_fine, dim=1).mean()

    # 4. Multi-Resolution / Coarse Loss (3x3 Pooling)
    pool_size = config.coarse_loss_pool_size
    probs_fine_2d = probs_fine.view(B, 1, H, W)
    # Average pooling * pool_size^2 keeps the sum of the map at 1.0
    probs_coarse = F.avg_pool2d(probs_fine_2d, kernel_size=pool_size) * (pool_size**2)

    # Pool binary target and normalize into a coarse PDF
    target_coarse = F.avg_pool2d(target_fine, kernel_size=pool_size) * (pool_size**2)
    target_coarse_flat = target_coarse.view(B, -1)

    # Coarse Loss using log of pooled probabilities
    ce_coarse = torch.sum(-target_coarse_flat * torch.log(probs_coarse.view(B, -1) + 1e-8), dim=1).mean()

    # 5. Thesis Metrics (Probability Allocation)
    with torch.no_grad():
        # Probability Mass Coverage / path coverage
        pmc = torch.sum(probs_fine * (target_flat > 0), dim=1).mean()

        # Distribution Overlap
        overlap = torch.sum(torch.min(target_flat, probs_fine), dim=1).mean()

    # 6. Total Loss
    loss = ce_fine + config.coarse_loss_beta * ce_coarse

    return loss, {
        "ce_fine": ce_fine,
        "ce_coarse": ce_coarse,
        "pmc": pmc,
        "overlap": overlap,
    }

def rasterize_occupancy(fut_pos, fut_mask):
    """Renders multiple future positions into a single occupancy grid.
    """
    B, T, _ = fut_pos.shape
    x_bins, y_bins, *_ = RASTER.get_total_grid_sizes()
    device = fut_pos.device

    # Flatten B and T to process all points, then filter by mask
    flat_pos = fut_pos.view(-1, 2)
    flat_mask = fut_mask.view(-1)

    valid_pos = flat_pos[flat_mask]
    # Calculate batch indices for the valid positions
    batch_indices = torch.arange(B, device=device).view(B, 1).expand(B, T).reshape(-1)
    valid_batch_indices = batch_indices[flat_mask]

    x_idx = torch.floor((valid_pos[:, 0] - RASTER.x_min) / RASTER.pos_res).to(torch.int64)
    y_idx = torch.floor((valid_pos[:, 1] - RASTER.y_min) / RASTER.pos_res).to(torch.int64)

    x_idx = x_idx.clamp(0, x_bins - 1)
    y_idx = y_idx.clamp(0, y_bins - 1)

    grid = torch.zeros((B, x_bins, y_bins), device=device)

    # Use index_put_ to set 1.0 at occupied coordinates
    # We use .unique() logic if you want to avoid multiple increments,
    # but for a binary mask, simple assignment is fine.
    grid[valid_batch_indices, x_idx, y_idx] = 1.0
    grid = grid / (grid.sum(dim=(1, 2), keepdim=True) + 1e-8)

    # https://arxiv.org/pdf/1701.06548
    # eps = 0.1
    # grid = (1 - eps) * grid + eps / (x_bins * y_bins)
    return grid
