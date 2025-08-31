import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator
from tqdm import tqdm

from lazy_loader.loader import lazy_loader
from train.eval import eval_loss
from train.plot import plot_traj
from utils.logger import logger


def eval_spline(data_folder: Path):

    logging.info("[Eval] model: PchipInterpolator")

    eval_dset, eval_sampler, eval_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-06-01"),
        max_date=pd.Timestamp("2022-06-01"),
        world_size=1,
        rank=0,
        batch_size=256,
        pin_memory=False,
        max_neighbors=0,
    )

    num_batches = 0
    loss_sum = 0.0
    ade_sum = 0.0
    fde_sum = 0.0
    l2max_sum = 0.0

    for batch in tqdm(eval_loader, desc="Evaluating Spline"):
        obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end = batch

        obs_len = obs_pos_rel.shape[2]
        fut_len = fut_pos_rel.shape[2]
        t_x = np.arange(obs_len)
        t_y = np.arange(obs_len, obs_len + fut_len)

        spline = PchipInterpolator(t_x, obs_pos_rel, axis=2, extrapolate=True)
        pred_pos_rel = torch.as_tensor(spline(t_y), dtype=fut_pos.dtype)

        loss, loss_dict = eval_loss(pred_pos_rel, batch)

        num_batches += 1
        loss_sum += loss.item()
        ade_sum += loss_dict["ade"].item()
        fde_sum += loss_dict["fde"].item()
        l2max_sum += loss_dict["l2max"].item()

    loss = loss_sum / num_batches
    ade = ade_sum / num_batches
    fde = fde_sum / num_batches
    l2max = l2max_sum / num_batches

    logging.info(
        f"[Eval] loss: {loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, l2_max: {l2max:.4f}"
    )
    pred_pos = obs_pos[:, :, -1].unsqueeze(2) + pred_pos_rel.cumsum(dim=2)
    plot_traj("prediction_spline", obs_pos, fut_pos, pred_pos, seq_start_end)


if __name__ == "__main__":
    logger(file_prefix="spline_server")
    data_folder = Path("/home/bbiesenbach/data/kiel/ais/3_features")
    eval_spline(data_folder)
