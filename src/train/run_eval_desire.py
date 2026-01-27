import torch
import pandas as pd
from pathlib import Path

from loaders.scene_loader.loader import scene_loader
from models.desire.model import DESIRE
from models.desire.utils.params import DESIREParams
from train.eval import eval  # or eval_lstm / eval_heatmap
from models.utils.maps.scene_gernerator import process_maps
from utils.config import DATA_FOLDER_PATH
import numpy as np


@torch.no_grad()
def main():
    # ---- config ----
    #/home/bbi/nereus/nereus/checkpoints/k/trial_1_best.pt
    ckpt_path = Path("checkpoints/k/trial_1_best.pt")
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
    batch_size = 64

    # ---- model ----
    cfg = DESIREParams()
    #'hidden_size': 256, 'out_channels': 32, 'latent_size_factor': 4, 'num_samples': 5}
    cfg.hidden_size = 256
    cfg.out_channels = 32
    cfg.latent_size = 256//4
    cfg.num_samples = 5
    cfg.num_refine_iters = 2
    
    device = torch.device("cuda:0")
    model = DESIRE(cfg).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(model.SGM.num_samples)

    # ---- data ----
    feat_cols = [
        "speed", "course", "acc", "angular_difference",
        "length", "width", "ship_group", "hour_of_day"
    ]

    _, _, eval_loader = scene_loader(
        data_folder=data_folder,
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=feat_cols,
        pred_len=cfg.pred_len,
        obs_len=cfg.obs_len,
    )

    # ---- scene (if needed) ----
    if hasattr(model, "rasterizer"):
        path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
        scene = torch.from_numpy(
            np.ascontiguousarray(
                process_maps(model.rasterizer, path), dtype=np.float32
            )
        ).unsqueeze(0).to(device)
    else:
        scene = None

    # ---- eval ----
    metric = eval(
        epoch=0,
        model=model,
        eval_loader=eval_loader,
        device=device,
        scene=scene,
        trial_number=0,
        config=cfg,
    )

    print(f"Eval metric: {float(metric):.6f}")


if __name__ == "__main__":
    CUDA_VISIBLE_DEVICES=1
    main()