import logging
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# KL loss over rasterized future positions
from models.traisformer.loss import fut_to_heatmap, loss_intent_heatmap
from train.plot_heatmap import plot_heatmap


def eval_heatmap(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    scene_meta,
):
    """
    Evaluate the TrAISformer intent head by KL loss and save one qualitative heat map.
    - Computes KL div vs. rasterized future bag-of-cells target (see loss.py).
    - Plots a single sample: predicted heat map + outline of GT future cells.
    """
    model.eval()

    num_batches = 0
    kl_sum = 0.0

    # cache for plotting only once
    plotted = 0

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches > 0:
                break

            batch = [t.to(device) for t in batch]

            output = model(batch, scene, scene_meta)
            loss, _ = loss_intent_heatmap(output, batch, epoch=epoch, config=None)
            kl_sum += float(loss.item())
            num_batches += 1

            # Make and save ONE qualitative plot per epoch
            if plotted < 2:
                plotted += 1

                B, _, x_bins, y_bins = output["intent_logits"].shape

                targets = fut_to_heatmap(batch, device, output["intent_logits"].dtype)
                target_heatmap = targets.view(B, x_bins, y_bins).contiguous()
                target_heatmap = target_heatmap[0].detach().cpu().numpy()
                plot_heatmap(f"target_heatmap_{epoch}", target_heatmap)

                logits = output["intent_logits"].flatten(1)
                probs = F.softmax(logits, dim=1)
                pred_heatmap = probs.view(B, x_bins, y_bins).contiguous()
                pred_heatmap = pred_heatmap[0].detach().cpu().numpy()
                plot_heatmap(f"pred_heatmap_{epoch}_{plotted}", pred_heatmap)

    # Average KL over batches
    if num_batches > 0:
        kl_avg = kl_sum / num_batches
        logging.info(f"[Eval HeatMap] epoch={epoch} KL_div={kl_avg:.6f}")
    else:
        logging.info("[Eval HeatMap] No eval batches processed.")

    model.train()
