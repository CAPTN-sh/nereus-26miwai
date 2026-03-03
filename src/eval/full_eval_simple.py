import logging
import os
from pathlib import Path
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.logger import logger
from data.graph.build_dataloader import graph_loader

from scipy.interpolate import PchipInterpolator
from models.gru.model import GRU_RNN

from utils.config import AIS_FOLDER_PATH, STEPS_PER_MINUTE
from eval.metrics.cpa import compute_batch_collision_risk
from eval.metrics.displacement import ade_per_agent, fde_per_agent
from eval.metrics.accumulator import MetricAccumulator

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Scale factor to convert normalized relative displacement back to meters
DE_NORMALIZE = 100


def full_eval(data_folder, model_name, model):
    """
    Full extensive eval script for GRU_RNN and Spline per ship_group.
    """
    for ship_group in ["all", "sailing", "cargo", "passenger", "other"]:
        logging.info("#"*20)
        logging.info(f"[SHIP GROUP]: {ship_group}")

        test_loader, _ = graph_loader(
            data_folder=data_folder,
            flag="val",
            min_date=pd.Timestamp("2022-01-01"),
            max_date=pd.Timestamp("2023-01-01"),
            batch_size=512,
            pin_memory=True,
            pred_len=30,
            obs_len=60,
            max_edge_dist=500,
            shuffle=False,
            ship_group = ship_group,
        )

        if model_name == "gru":
            model = model.to(device)
            model.eval()

        with torch.inference_mode():
            total_pred_risk = 0.0
            total_min_pred_dist = 0.0
            total_collision_count = 0.0
            total_graphs = 0

            names = ["ade", "fde_1", "fde_3", "fde_5"]
            metrics = {k: MetricAccumulator() for k in names}

            # Measure pure model inference time (excluding dataloading)
            eval_time = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for data in tqdm(test_loader, desc=f"Eval"):
                data = data.to(device, non_blocking=True)

                start_event.record()

                ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

                if model_name == "gru":
                    pred_rel_pos = model(data, None)

                    # Convert predicted relative displacements to absolute positions
                    last_pos = data.x_pos[ego_idx, -1, :].unsqueeze(1)
                    pred_abs_pos = torch.cumsum(pred_rel_pos, dim=1) * DE_NORMALIZE + last_pos

                if model_name == "spline":
                    obs_pos = data.x[ego_idx, : , :2]
                    t_obs  = np.arange(60)
                    t_pred = np.arange(60, 90)

                    spline = PchipInterpolator(t_obs, obs_pos.cpu().numpy(), axis=1, extrapolate=True)
                    pred_rel_pos = spline(t_pred)
                    pred_rel_pos = torch.from_numpy(pred_rel_pos).to(obs_pos.device).to(obs_pos.dtype)

                    # Convert predicted relative displacements to absolute positions
                    last_pos = data.x_pos[ego_idx, -1, :].unsqueeze(1)
                    pred_abs_pos = torch.cumsum(pred_rel_pos, dim=1) * DE_NORMALIZE + last_pos

                if model_name == "ground_truth":
                    pred_abs_pos = data.y_all[ego_idx, 0]

                end_event.record()
                torch.cuda.synchronize()

                step_time_ms = start_event.elapsed_time(end_event)
                eval_time += (step_time_ms / 1000.0)

                metrics["ade"].update(ade_per_agent(pred_abs_pos, data))
                metrics["fde_1"].update(fde_per_agent(pred_abs_pos, data, 1 * STEPS_PER_MINUTE))
                metrics["fde_3"].update(fde_per_agent(pred_abs_pos, data, 3 * STEPS_PER_MINUTE))
                metrics["fde_5"].update(fde_per_agent(pred_abs_pos, data, 5 * STEPS_PER_MINUTE))

                pred_risk, min_pred_dist, collision_count, count = compute_batch_collision_risk(data, pred_abs_pos)

                total_pred_risk += pred_risk
                total_min_pred_dist += min_pred_dist
                total_collision_count += collision_count
                total_graphs += count

            logging.info(f"Total Graphs: {total_graphs}")

            logging.info(f"Mean Pred Risk: {(total_pred_risk / total_graphs).item()}")
            logging.info(f"Mean Min Pred Distance: {(total_min_pred_dist / total_graphs).item()}")
            logging.info(f"Collision Ratio: {(total_collision_count / total_graphs).item() * 100}")

            for name, accumulator in metrics.items():
                logging.info(f"{name}: {accumulator.compute()}")

            logging.info(f"eval_time {eval_time / 60:.2f} minutes / {len(test_loader)}")


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model_name = "ground_truth" # ["ground_truth", "gru", "spline"]

    if model_name == "gru":
        best_ckpt_path = Path("checkpoints/gru/gru_best.pt")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model = GRU_RNN(ckpt["config"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        logger(file_prefix=f"eval_{best_ckpt_path.name}")
        logging.info(best_ckpt_path)
    else:
        model = None
        logger(file_prefix=f"eval_{model_name}")
        logging.info(model_name)

    full_eval(AIS_FOLDER_PATH, model_name, model)

"""
run in bash with:

CUDA_VISIBLE_DEVICES=0 python -u src/eval/full_eval_simple.py
"""