import logging
import os
from pathlib import Path
import torch

from utils.logger import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

from data.graph.build_dataloader import graph_loader
from data.map.scene_gernerator import SceneLoader
from data.map.rasterize import Rasterizer

from models.desire.model import DESIRE

from utils.config import AIS_FOLDER_PATH, MAP_FOLDER_PATH, STEPS_PER_MINUTE, TRAIN_BBOX
from eval.metrics.cpa import compute_batch_collision_risk
from eval.metrics.displacement import ade_per_agent, fde_per_agent, k_ade_per_agent, k_fde_per_agent
from eval.metrics.accumulator import MetricAccumulator

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Scale factor to convert normalized relative displacement back to meters
DE_NORMALIZE = 100

def full_eval(data_folder, model):
    """
    Full extensive eval script for DESIRE per ship_group.
    """  
        
    for ship_group in ["all", "sailing", "cargo", "passenger", "other"]:
        logging.info("#"*20)
        logging.info(f"[SHIP GROUP]: {ship_group}")

        test_loader, _ = graph_loader(
            data_folder=data_folder,
            flag="test",
            min_date=pd.Timestamp("2022-01-01"),
            max_date=pd.Timestamp("2024-01-01"),
            batch_size=512,
            pin_memory=True,
            pred_len=30,
            obs_len=60,
            max_edge_dist=500,
            shuffle=False,
            ship_group = ship_group,
        )

        sl = SceneLoader(Rasterizer(TRAIN_BBOX))
        scene_contiguous = np.ascontiguousarray(sl.load_scene(MAP_FOLDER_PATH))
        scene = torch.from_numpy(scene_contiguous).to(device).to(torch.float32)

        model = model.to(device)
        model.eval()

        with torch.inference_mode():
            total_pred_risk = 0.0
            total_min_pred_dist = 0.0
            total_collision_count = 0.0
            total_graphs = 0

            names = ["ade", "fde_1", "fde_3", "fde_5", "k_ade", "k_fde_1", "k_fde_3", "k_fde_5"]
            metrics = {k: MetricAccumulator() for k in names}

            eval_time = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for data in tqdm(test_loader, desc=f"Eval"):
                data = data.to(device, non_blocking=True)

                start_event.record()
                best_pred_pos_rel, k_pred_pos_rel = model.inference(data, scene)

                # Convert predicted relative displacements to absolute positions
                ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
                last_pos = data.x_pos[ego_idx, -1, :].unsqueeze(1)
                pred_abs_pos = torch.cumsum(best_pred_pos_rel, dim=1) * DE_NORMALIZE + last_pos
                pred_abs_pos_k = torch.cumsum(k_pred_pos_rel, dim=2) * DE_NORMALIZE + last_pos.unsqueeze(1)

                end_event.record()
                torch.cuda.synchronize()

                step_time_ms = start_event.elapsed_time(end_event)
                eval_time += (step_time_ms / 1000.0)

                metrics["ade"].update(ade_per_agent(pred_abs_pos, data))
                metrics["fde_1"].update(fde_per_agent(pred_abs_pos, data, 1 * STEPS_PER_MINUTE))
                metrics["fde_3"].update(fde_per_agent(pred_abs_pos, data, 3 * STEPS_PER_MINUTE))
                metrics["fde_5"].update(fde_per_agent(pred_abs_pos, data, 5 * STEPS_PER_MINUTE))

                metrics["k_ade"].update(k_ade_per_agent(pred_abs_pos_k, data))
                metrics["k_fde_1"].update(k_fde_per_agent(pred_abs_pos_k, data, 1 * STEPS_PER_MINUTE))
                metrics["k_fde_3"].update(k_fde_per_agent(pred_abs_pos_k, data, 3 * STEPS_PER_MINUTE))
                metrics["k_fde_5"].update(k_fde_per_agent(pred_abs_pos_k, data, 5 * STEPS_PER_MINUTE))

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

    best_ckpt_path = Path("checkpoints/desire/desire_best.pt")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model = DESIRE(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger(file_prefix=f"eval_{best_ckpt_path.name}")
    logging.info(best_ckpt_path)

    full_eval(AIS_FOLDER_PATH, model)

"""
run in bash with:

CUDA_VISIBLE_DEVICES=1 python -u src/eval/full_eval_desire.py
"""