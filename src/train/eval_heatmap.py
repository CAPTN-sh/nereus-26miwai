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
    ce_fine_sum = 0.0
    ce_coarse_sum = 0.0
    
    # Neue Metriken Akkumulatoren

    hit1_sum = 0.0
    hit5_sum = 0.0
    mde_sum = 0.0
    mtp_sum = 0.0

    precision_sum = 0.0
    recall_sum = 0.0

    plotted = True
    n_plots = 5

    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):

            if num_batches >= 500:
                break

            batch = [t.to(device) for t in batch]

            output = model(batch, scene)
            
            if config.pred_scope == "path":
                loss, loss_dict = loss_occupancy_heatmap(output, batch, config=config)
                precision_sum += float(loss_dict["precision"])
                recall_sum += float(loss_dict["recall"])
            if config.pred_scope == "destination":
                loss, loss_dict = loss_intent_heatmap(output, batch, config=config)
                hit1_sum += float(loss_dict["hit1"])
                hit5_sum += float(loss_dict["hit5"])
                mde_sum += float(loss_dict["mde_meters"])
                mtp_sum += float(loss_dict["mean_target_prob"])

            # Summiere alle Werte auf
            loss_sum += float(loss.item())
            ce_fine_sum += float(loss_dict["ce_fine"])
            ce_coarse_sum += float(loss_dict["ce_coarse"])
            
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
                    plot_heatmap(f"pred_heatmap_tn{trial_number}_p{i}", p_hmap, epoch=epoch)

                    if epoch == 1:
                        t_hmap = target.view(B, x_bins, y_bins)[i].detach().cpu().numpy()
                        plot_heatmap(f"target_heatmap_p{i}", t_hmap)

    # Durchschnittsberechnung
    loss_avg = loss_sum / num_batches
    ce_fine_avg = ce_fine_sum / num_batches
    ce_coarse_avg = ce_coarse_sum / num_batches
    
    if config.pred_scope == "path":
        precision_avg = precision_sum / num_batches
        recall_avg = recall_sum / num_batches
        logging.info(f"[Eval HeatMap] Epoch {epoch} - Loss Total: {loss_avg:.6f} ce_fine: {ce_fine_avg:.4f}, ce_coarse: {ce_coarse_avg:.4f}, precision_avg: {precision_avg:.2%}, recall_avg: {recall_avg:.2%}")
    if config.pred_scope == "destination":
        hit1_avg = hit1_sum / num_batches
        hit5_avg = hit5_sum / num_batches
        mde_avg = mde_sum / num_batches
        mtp_avg = mtp_sum / num_batches
        logging.info(f"[Eval HeatMap] Epoch {epoch} - Loss Total: {loss_avg:.6f} ce_fine: {ce_fine_avg:.4f}, ce_coarse: {ce_coarse_avg:.4f}, hit@1: {hit1_avg:.2%}, hit@5: {hit5_avg:.2%}, MDE: {mde_avg:.2f} m, MTP: {mtp_avg:.2%}")
    model.train()
    
    return ce_fine_avg