import torch.nn as nn
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import logging
from models.nereus.loss import mdn_loss
from eval.metrics.displacement import ade_per_agent, fde_per_agent, k_ade_per_agent, k_fde_per_agent
from eval.metrics.accumulator import MetricAccumulator
from utils.config import STEPS_PER_MINUTE

def unpack_mdn_eval(mdn_out, num_modes):
    B, T, _ = mdn_out.shape
    K = num_modes

    mdn_out = mdn_out.view(B, T, K, 5)

    pi = torch.softmax(mdn_out[..., 0], dim=-1)
    mu = mdn_out[..., 1:3]

    return pi, mu

def eval_nereus(
    epoch: int,
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    scene,
    config=None,
):

    model.eval()

    names = ["ade", "fde_5", "k_ade", "k_fde_5"]
    metrics = {k: MetricAccumulator() for k in names}

    mdn_loss_sum = 0.0
    n_mdn = 0

    revert_norm = 100.0

    n_eval_batches = int(len(eval_loader) // 10)
    with torch.inference_mode():
        for n_batches, data in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if n_batches >= n_eval_batches:
                break

            data = data.to(device, non_blocking=True)
            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

            # ---- forward ----
            mdn_out = model(data, scene)  # [B, T, K*5]

            # ---- MDN loss (TUNING METRIC) ----
            loss_val, _ = mdn_loss(mdn_out, data, config)
            mdn_loss_sum += loss_val.item()
            n_mdn += 1

            # ---- unpack MDN for displacement metrics ----
            pi, mu = unpack_mdn_eval(mdn_out, config.mdn_modes)

            # ---- expected trajectory ----
            exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)  # [B, T, 2]
            pred_abs_pos = (
                torch.cumsum(exp_rel, dim=1) * revert_norm
                + data.x_pos[ego_idx, -1:, :]
            )
            # ---- K trajectories (min over modes) ----
            mu_k = mu.permute(0, 2, 1, 3)

            pred_abs_pos_k = (
                torch.cumsum(mu_k, dim=2) * revert_norm
                + data.x_pos[ego_idx, -1:, :].unsqueeze(1)
            )

            metrics["ade"].update(ade_per_agent(pred_abs_pos, data))
            metrics["fde_5"].update(fde_per_agent(pred_abs_pos, data, 5 * STEPS_PER_MINUTE))

            metrics["k_ade"].update(k_ade_per_agent(pred_abs_pos_k, data))
            metrics["k_fde_5"].update(k_fde_per_agent(pred_abs_pos_k, data, 5 * STEPS_PER_MINUTE))

    # ---- averages ----

    mdn_loss_avg = mdn_loss_sum / n_mdn
    log_str = f"[Eval] Epoch {epoch} - nll: {mdn_loss_avg:.4f},"
    for name, accumulator in metrics.items():
        log_str += f"{name}: {accumulator.compute():.4f},"
    logging.info(log_str)

    # ---- return tuning metric FIRST ----
    return mdn_loss_avg