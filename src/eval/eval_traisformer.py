import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.traisformer.loss import loss_intent_heatmap, loss_occupancy_heatmap


def eval_traisformer(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    config=None,
):
    """
    Eval function for traisormer returning "overlap" or "hit1 (depending on pred_scope) 
    for hyper parameter tuning.
    """  
    model.eval()

    if config.pred_scope not in ["path", "destination"]:
        raise KeyError(f"Unknown prediction scope: {config.pred_scope}")

    num_batches = 0
    num_samples = 0
    ce_fine = 0.0

    p_gt, hit1, hit5 = 0.0, 0.0, 0.0
    pmc, overlap = 0.0, 0.0

    n_eval_batches = int(len(eval_loader) // 10)
    with torch.inference_mode():
        for batch in tqdm(eval_loader, desc="Evaluating (heat maps)"):
            if num_batches >= n_eval_batches:
                break

            batch = batch.to(device, non_blocking=True)
            output = model(batch, scene)
            
            if config.pred_scope == "path":
                loss, loss_dict = loss_occupancy_heatmap(output, batch, config=config)
                ce_fine += float(loss_dict["ce_fine"])
                pmc += float(loss_dict["pmc"])
                overlap += float(loss_dict["overlap"])
                num_samples += 1
                
            if config.pred_scope == "destination":
                B = batch.fin_pos_mask.sum().item()
                loss, loss_dict = loss_intent_heatmap(output, batch, config=config)
                ce_fine += float(loss_dict["ce_fine"]) * B
                p_gt += float(loss_dict["p_gt"]) * B
                hit1 += float(loss_dict["hit1"]) * B
                hit5 += float(loss_dict["hit5"]) * B
                num_samples += B

            num_batches += 1

    ce_fine /= num_samples
    
    if config.pred_scope == "path":
        pmc /= num_samples
        overlap /= num_samples
        logging.info(f"[Eval HeatMap] Epoch {epoch} - ce_fine: {ce_fine:.6f}, pmc: {pmc:.2%}, overlap: {overlap:.2%}")
        return -overlap
    
    if config.pred_scope == "destination":
        hit1 = hit1 / num_samples
        hit5 = hit5 / num_samples
        p_gt = p_gt / num_samples
        logging.info(f"[Eval HeatMap] Epoch {epoch} - ce_fine: {ce_fine:.6f}, p_gt: {p_gt:.2%}, hit@1: {hit1:.2%}, hit@5: {hit5:.2%}")
        return -hit1