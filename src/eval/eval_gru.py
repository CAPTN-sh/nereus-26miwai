import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval.metrics.displacement import ade_per_agent, fde_per_agent
from eval.metrics.accumulator import MetricAccumulator

from utils.config import STEPS_PER_MINUTE

DE_NORMALIZE = 100.0

def eval_gru(epoch, model: nn.Module, eval_loader: DataLoader, device, scene, config):
    """
    Eval function for GRU_RNN returning "ade" for hyper parameter tuning.
    """  
    model.eval()

    names = ["ade", "fde_5"]
    metrics = {k: MetricAccumulator() for k in names}

    n_eval_batches = int(len(eval_loader) // 10)
    with torch.inference_mode():
        for n_batches, data in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if n_batches >= n_eval_batches:
                break

            data = data.to(device)
            best_pred_pos_rel = model.inference(data, scene)

            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
            last_pos = data.x_pos[ego_idx, -1:, :]
            pred_abs_pos = torch.cumsum(best_pred_pos_rel, dim=1) * DE_NORMALIZE + last_pos

            metrics["ade"].update(ade_per_agent(pred_abs_pos, data))
            metrics["fde_5"].update(fde_per_agent(pred_abs_pos, data, 5 * STEPS_PER_MINUTE))

    log_str = f"[Eval] Epoch {epoch} -"
    for name, accumulator in metrics.items():
        log_str += f"{name}: {accumulator.compute():.4f},"
    logging.info(log_str)

    model.train()

    return metrics["ade"].compute()