import logging
import os
from pathlib import Path
import torch

from utils.logger import logger
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from loaders.graph_loader.loader import graph_loader
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer

from scipy.interpolate import PchipInterpolator

from models.rnn.model import RNN

from utils.config import DATA_FOLDER_PATH
from eval.cpa import compute_batch_collision_risk

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

DE_NORMALIZE = 100


def ade_per_agent(pred_abs, data):
    #Data(x, x_pos, obs_mask, edge_index, edge_attr, y, y_pos, fut_mask)
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    dist = dist * data.y_mask

    ade_per_agent = dist.sum(dim=1) / data.y_mask.sum(dim=1).clamp_min(1)

    return ade_per_agent

def fde_per_agent(pred_abs, data, t):
    full_traj_mask = data.y_mask.sum(dim=1) >= t
    dist = torch.norm(pred_abs - data.y_pos, dim=-1)
    fde = dist[:, t-1][full_traj_mask]
    return fde

def full_eval(data_folder, model):
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

        #path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
        #sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))

        path = DATA_FOLDER_PATH / "maps/2_standardized/dma_10/aarhus/"
        sl = SceneLoader(Rasterizer([10.21, 56.04, 10.47, 56.17]))

        scene_contiguous = np.ascontiguousarray(sl.load_scene(path))
        scene = torch.from_numpy(scene_contiguous).to(device).to(torch.float32)

        model = model.to(device)
        model.eval()

        with torch.inference_mode():
            total_pred_risk = 0.0
            total_min_pred_dist = 0.0
            total_high_risk = 0.0
            total_close_dist = 0.0
            total_collision_count = 0.0
            total_graphs = 0

            ade_sum = 0.0
            n_ade = 0
            fde_1_sum = 0.0
            n_fde_1 = 0
            fde_3_sum = 0.0
            n_fde_3 = 0
            fde_5_sum = 0.0
            n_fde_5 = 0
            eval_time = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for data in tqdm(test_loader, desc=f"Eval"):
                data = data.to(device, non_blocking=True)

                start_event.record()
                pred_rel_pos = model(data, None)
                end_event.record()
                torch.cuda.synchronize()

                step_time_ms = start_event.elapsed_time(end_event)
                eval_time += (step_time_ms / 1000.0)

                ego_idx = data.is_ego.nonzero(as_tuple=True)[0]

                ### spline
                #obs_pos = data.x[ego_idx, : , :2]
                #t_obs  = np.arange(60)
                #t_pred = np.arange(60, 90)

                #spline = PchipInterpolator(t_obs, obs_pos.cpu().numpy(), axis=1, extrapolate=True)
                #pred_rel_pos = spline(t_pred)  # (N_ego, 30, 2)
                #pred_rel_pos = torch.from_numpy(pred_rel_pos).to(obs_pos.device).to(obs_pos.dtype)

                pred_abs_pos = get_absolute(data, pred_rel_pos)

                ### ground truth
                #pred_abs_pos = data.y_all[ego_idx, 0]

                ade = ade_per_agent(pred_abs_pos, data)
                ade_sum += ade.sum().item()
                n_ade += ade.numel()
                fde_1 = fde_per_agent(pred_abs_pos, data, 6)
                fde_1_sum += fde_1.sum().item()
                n_fde_1 += fde_1.numel()
                fde_3 = fde_per_agent(pred_abs_pos, data, 18)
                fde_3_sum += fde_3.sum().item()
                n_fde_3 += fde_3.numel()
                fde_5 = fde_per_agent(pred_abs_pos, data, 30)
                fde_5_sum += fde_5.sum().item()
                n_fde_5 += fde_5.numel()

                (
                    pred_risk,
                    min_pred_dist,
                    high_risk_count,
                    close_dist_count,
                    collision_count,
                    count
                ) = compute_batch_collision_risk(data, pred_abs_pos)

                total_pred_risk += pred_risk
                total_min_pred_dist += min_pred_dist
                total_high_risk += high_risk_count
                total_close_dist += close_dist_count
                total_collision_count += collision_count
                total_graphs += count

            logging.info(f"Total Graphs: {total_graphs}")

            logging.info(f"Mean Pred Risk: {(total_pred_risk / total_graphs).item()}")
            logging.info(f"Mean Min Pred Distance: {(total_min_pred_dist / total_graphs).item()}")
            logging.info(f"High Risk Ratio: {(total_high_risk / total_graphs).item() * 100}")
            logging.info(f"Close Distance Ratio: {(total_close_dist / total_graphs).item() * 100}")
            logging.info(f"Collision Ratio: {(total_collision_count / total_graphs).item() * 100}")

            logging.info(f"ADE: {ade_sum / n_ade}")
            logging.info(f"FDE_1: {fde_1_sum / n_fde_1}")
            logging.info(f"FDE_2: {fde_3_sum / n_fde_3}")
            logging.info(f"FDE_3: {fde_5_sum / n_fde_5}")
            logging.info(f"eval_time {eval_time / 60:.2f} minutes / {len(test_loader)}")

def get_absolute(data, pred_rel_pos):
    ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
    last_pos = data.x_pos[ego_idx, -1, :].unsqueeze(1)
    pred_abs = torch.cumsum(pred_rel_pos, dim=1) * DE_NORMALIZE + last_pos
    return pred_abs


if __name__ == "__main__":
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    best_ckpt_path = Path("checkpoints/rnn") / f"rnn_256_best.pt"
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model = RNN(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger(file_prefix=f"eval_{best_ckpt_path.name}")
    logging.info(best_ckpt_path)
    #logger(file_prefix=f"eval_ground_truth")
    #logging.info("ground_truth")

    data_folder = DATA_FOLDER_PATH / "ais/4_features/dma_10/aarhus"
    full_eval(data_folder, model)