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

from models.nereus.init import init_nereus
from eval.metrics.displacement import ade_per_agent, fde_per_agent, k_ade_per_agent, k_fde_per_agent
from eval.metrics.accumulator import MetricAccumulator

from utils.config import DATA_FOLDER_PATH, STEPS_PER_MINUTE, AIS_SOURCE
from eval.metrics.cpa import compute_batch_collision_risk

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Scale factor to convert normalized relative displacement back to meters
DE_NORMALIZE = 100

def full_eval(model, region, bbox):
    """
    Full extensive eval script for NEREUS per ship_group.
    """
    for ship_group in ["all", "sailing", "cargo", "passenger", "other"]:
        logging.info("#"*20)
        logging.info(f"[SHIP GROUP]: {ship_group}")

        data_folder = DATA_FOLDER_PATH / f"ais/4_features/{AIS_SOURCE}_10/{region}"
        map_folder = DATA_FOLDER_PATH / f"maps/2_standardized/{AIS_SOURCE}_10/{region}"

        B, T = 512, 5 * STEPS_PER_MINUTE

        test_loader, _ = graph_loader(
            data_folder=data_folder,
            flag="test",
            min_date=pd.Timestamp("2022-01-01"),
            max_date=pd.Timestamp("2024-01-01"),
            batch_size=B,
            pin_memory=True,
            pred_len=T,
            obs_len=60,
            max_edge_dist=500,
            shuffle=True,
            ship_group = ship_group,
        )

        sl = SceneLoader(Rasterizer(bbox))

        scene_contiguous = np.ascontiguousarray(sl.load_scene(map_folder))
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

            # Measure pure model inference time (excluding dataloading)
            eval_time = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for data in tqdm(test_loader, desc=f"Eval"):
                data = data.to(device, non_blocking=True)

                ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

                start_event.record()
                mdn_out = model(data, scene).view(B, T, 3, 5)

                # Convert mdn to absolute positions
                pi = torch.softmax(mdn_out[..., 0], dim=-1)
                mu = mdn_out[..., 1:3]
                exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)  # [B, T, 2]
                pred_abs_pos = torch.cumsum(exp_rel, dim=1) * DE_NORMALIZE + data.x_pos[ego_idx, -1:, :]
                mu_k = mu.permute(0, 2, 1, 3)
                pred_abs_pos_k = torch.cumsum(mu_k, dim=2) * DE_NORMALIZE + data.x_pos[ego_idx, -1:, :].unsqueeze(1)

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

def eval_model(best_ckpt_path, device, regions):
    ckpt = torch.load(best_ckpt_path, map_location=device)
    cfg = ckpt["config"]

    for region, bbox in regions.items():
        logging.info(region)

        rasterizer = Rasterizer(bbox)

        model = init_nereus(best_ckpt_path.name, cfg, device, rasterizer=rasterizer)
        model.load_state_dict(ckpt["model_state_dict"])

        model.rasterizer = rasterizer
        if hasattr(model, "map_cnn") and model.map_cnn is not None:
            model.map_cnn.rasterizer = rasterizer
        if hasattr(model, "prior_cnn") and model.prior_cnn is not None:
            model.prior_cnn.rasterizer = rasterizer

        model.eval()

        full_eval(model, region, bbox)

if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # for other regions set AIS_SOURCE = "dma" in utils/config.py as "fh" is not available
    regions = {
        "kiel": [10.12, 54.31, 10.33, 54.46],
        #"aarhus": [10.21, 56.04, 10.47, 56.17],
        #"odense": [10.42, 55.42, 10.68, 55.55],
        #"little_belt": [9.64, 55.25, 9.90, 55.37],
    }

    best_ckpt_path = Path("checkpoints/nereus/nereus_density_cnn_gat_best.pt")
    logger(file_prefix=f"eval_{AIS_SOURCE}_{best_ckpt_path.name}")
    logging.info(best_ckpt_path)

    eval_model(best_ckpt_path, device, regions)

"""
run in bash with:

CUDA_VISIBLE_DEVICES=0 python -u src/eval/full_eval_nereus.py
"""