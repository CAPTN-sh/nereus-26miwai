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
    loss_sum = 0.0
    nll_gt_sum = 0.0

    p_gt_sum = 0.0
    hit1_sum = 0.0
    hit5_sum = 0.0

    pmc_sum = 0.0
    overlap_sum = 0.0

    plotted = False
    n_plots = 5

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):

            if num_batches >= 500:
                break

            batch = [t.to(device) for t in batch]

            output = model(batch, scene)
            
            if config.pred_scope == "path":
                loss, loss_dict = loss_occupancy_heatmap(output, batch, config=config)
                pmc_sum += float(loss_dict["pmc"])
                overlap_sum += float(loss_dict["overlap"])
            if config.pred_scope == "destination":
                loss, loss_dict = loss_intent_heatmap(output, batch, config=config)
                p_gt_sum += float(loss_dict["p_gt"])
                hit1_sum += float(loss_dict["hit1"])
                hit5_sum += float(loss_dict["hit5"])

            # Summiere alle Werte auf
            loss_sum += float(loss.item())
            nll_gt_sum += float(loss_dict["nll_gt"])
            
            num_batches += 1

            # Qualitative Plots (Bleibt gleich)
            if not plotted:
                plotted = True
                logits = output["intent_logits"]
                B, _, x_bins, y_bins = logits.shape
                _, obs_pos, _, obs_mask, fut_pos, _, fut_mask, fin_pos = batch

                if config.pred_scope == "path":
                    target = rasterize_occupancy(fut_pos, fut_mask)
                if config.pred_scope == "destination":
                    target = rasterize_destination(fin_pos)
                probs = F.softmax(logits.flatten(1), dim=1)

                for i in range(min(n_plots, B)):
                    obs_i = obs_pos[i][obs_mask[i]]
                    p_hmap = probs.view(B, x_bins, y_bins)[i].detach().cpu().numpy()
                    plot_heatmap(f"pred_heatmap_{config.pred_scope[:4]}_tn{trial_number}_{i}", p_hmap, epoch=epoch)

                    if epoch == 1:
                        t_hmap = target.view(B, x_bins, y_bins)[i].detach().cpu().numpy()
                        plot_heatmap(f"target_heatmap_{config.pred_scope[:4]}_{i}", t_hmap)

    nll_gt_avg = nll_gt_sum / num_batches
    
    if config.pred_scope == "path":
        pmc_avg = pmc_sum / num_batches
        overlap_avg = overlap_sum / num_batches
        logging.info(f"[Eval HeatMap] Epoch {epoch} - nll_gt: {nll_gt_avg:.6f}, pmc: {pmc_avg:.2%}, overlap: {overlap_avg:.2%}")
    
        return nll_gt_avg
    
    if config.pred_scope == "destination":
        hit1_avg = hit1_sum / num_batches
        hit5_avg = hit5_sum / num_batches
        p_gt_avg = p_gt_sum / num_batches
        logging.info(f"[Eval HeatMap] Epoch {epoch} - nll_gt: {nll_gt_avg:.6f}, p_gt: {p_gt_avg:.2%}, hit@1: {hit1_avg:.2%}, hit@5: {hit5_avg:.2%}")
        
        return nll_gt_avg