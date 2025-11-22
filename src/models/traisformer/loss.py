from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from .params import TraisformerParams
from .rasterize import Rasterizer

def fut_to_heatmap(batch, device, dtype):
    config = TraisformerParams()
    _, _, _, fut_pos, _, _ = batch
    B = fut_pos.size(0)

    raster = Rasterizer(config)
    x_idx, y_idx = raster.pos_to_index(fut_pos.to(device))
    x_bins, y_bins, _, _ = raster.get_total_gird_sizes()

    target = torch.zeros(B, x_bins * y_bins, device=device, dtype=dtype)
    lin = (x_idx * y_bins + y_idx).clamp_(0, x_bins * y_bins - 1)
    ones = torch.ones_like(lin, dtype=dtype)
    target.scatter_add_(dim=1, index=lin.long(), src=ones)
    target = target.clip(0, 1)

    # add blur gaussian aprox
    kernel = torch.tensor(
        [[1, 1, 1],
         [1, 4, 1],
         [1, 1, 1]],
        device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    #kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)

    img = target.view(B, 1, x_bins, y_bins)
    img = F.pad(img, (1, 1, 1, 1), mode="reflect")
    blurred = F.conv2d(img, kernel)

    target = blurred.view(B, -1).clip(0, 4)
    target = target/target.sum(dim=1, keepdim=True)

    return target


def loss_intent_heatmap(
    output: Dict[str, torch.Tensor],
    batch,
    epoch: int | None = None,
    config: TraisformerParams | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if config is None:
        config = TraisformerParams()

    logits = output["intent_logits"]
    log_probs = F.log_softmax(logits.flatten(1), dim=1)

    target = fut_to_heatmap(batch, logits.device, logits.dtype)

    ce_loss = torch.sum(-target * log_probs, dim=1).mean()
    kl_loss = F.kl_div(log_probs, target, reduction="batchmean")
    mse = F.mse_loss(log_probs, target)

    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1).mean()

    loss = ce_loss - 1e-3 * entropy

    return loss, {"ce": ce_loss, "kl_div": kl_loss, "mse": mse}
