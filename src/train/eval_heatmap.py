import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# KL loss over rasterized future positions
from models.traisformer.hierarchical_loss import (
    loss_intent_heatmap,
    rasterize_destination,
    loss_occupancy_heatmap,
    rasterize_occupancy,
)
from train.plot_heatmap import plot_heatmap


def eval_heatmap(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    trial_number = 0,
    config=None,
):
    model.eval()

    if config.pred_scope not in ["path", "destination"]:
        raise KeyError(f"Unknown prediction scope: {config.pred_scope}")

    num_batches = 0
    ce_fine = 0.0

    p_gt, hit1, hit5 = 0.0, 0.0, 0.0
    pmc, overlap = 0.0, 0.0

    plotted = True
    n_plots = 5

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches >= 500:
                break

            batch = batch.to(device, non_blocking=True)
            output = model(batch, scene)
            
            if config.pred_scope == "path":
                loss, loss_dict = loss_occupancy_heatmap(output, batch, config=config)
                pmc += float(loss_dict["pmc"])
                overlap += float(loss_dict["overlap"])
            if config.pred_scope == "destination":
                loss, loss_dict = loss_intent_heatmap(output, batch, config=config)
                p_gt += float(loss_dict["p_gt"])
                hit1 += float(loss_dict["hit1"])
                hit5 += float(loss_dict["hit5"])

            # Summiere alle Werte auf
            ce_fine += float(loss_dict["ce_fine"])
            num_batches += 1

            # Qualitative Plots (Bleibt gleich)
            if not plotted:
                plotted = True
                logits = output
                B, _, x_bins, y_bins = logits.shape

                if config.pred_scope == "path":
                    target = rasterize_occupancy(batch.y_pos, batch.y_mask)
                if config.pred_scope == "destination":
                    fin_pos = batch.fin_pos[batch.fin_pos_mask]
                    target = rasterize_destination(fin_pos)
                    logits = logits[batch.fin_pos_mask]
                probs = F.softmax(logits.flatten(1), dim=1)

                for i in range(min(n_plots, B)):
                    p_hmap = probs.view(B, x_bins, y_bins)[i].detach().cpu().numpy()
                    plot_heatmap(f"pred_heatmap_{config.pred_scope[:4]}_tn{trial_number}_{i}", p_hmap, epoch=epoch)

                    if epoch == 1:
                        t_hmap = target.view(B, x_bins, y_bins)[i].detach().cpu().numpy()
                        plot_heatmap(f"target_heatmap_{config.pred_scope[:4]}_{i}", t_hmap)

    ce_fine /= num_batches
    
    if config.pred_scope == "path":
        pmc /= num_batches
        overlap /= num_batches
        logging.info(f"[Eval HeatMap] Epoch {epoch} - ce_fine: {ce_fine:.6f}, pmc: {pmc:.2%}, overlap: {overlap:.2%}")
    
    if config.pred_scope == "destination":
        hit1 = hit1 / num_batches
        hit5 = hit5 / num_batches
        p_gt = p_gt / num_batches
        logging.info(f"[Eval HeatMap] Epoch {epoch} - ce_fine: {ce_fine:.6f}, p_gt: {p_gt:.2%}, hit@1: {hit1:.2%}, hit@5: {hit5:.2%}")
        
    return ce_fine