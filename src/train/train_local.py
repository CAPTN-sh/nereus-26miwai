import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lazy_loader.loader import lazy_loader
from models.desire.desire import DESIRE
from models.desire.nn.loss import k_total_loss
from models.desire.utils.params import DESIREParams
from train.eval import eval


def train_cpu(
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    batch_size: int = 128,
    num_epochs: int = 30,
    norm_clip_value: float = 1.0,
    lr: float = 4e-3,
):
    # --- CPU device ---
    device = torch.device("cpu")
    torch.set_num_threads(
        max(1, torch.get_num_threads())
    )  # let PyTorch use available CPU threads
    print(f"Starting training on {device}")

    # --- DataLoaders (single-process, no DDP) ---
    train_dset, train_sampler, train_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2023-05-03"),
        max_date=pd.Timestamp("2023-05-03"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=False,
    )
    eval_dset, eval_sampler, eval_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2023-05-05"),
        max_date=pd.Timestamp("2023-05-05"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=False,
    )

    print("additional features:", train_dset.feature_cols)
    print(f"There are {len(train_dset)} traj loaded for training")
    print(f"There are {len(eval_dset)} traj loaded for evaluation")

    # --- Scene tensor ---
    npz = np.load(scene_path)
    scene = torch.from_numpy(npz["I"]).unsqueeze(0).to(device)
    scene_meta = json.load(open(scene_meta_path))

    # --- Model ---
    params = DESIREParams()
    model = DESIRE(params).to(device)

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs // 4, gamma=0.5)

    # --- Train ---
    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0.0
        num_batches = 0

        # some loaders require an epoch set even in single-process; ignore if not needed
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()

            obs_feat, obs_pos, obs_pos_rel, fut_pos, fut_pos_rel, seq_start_end = [
                tensor.to(device) for tensor in batch
            ]
            obs_pos_last = obs_pos[:, :, -1]

            # Forward
            pred_pos_rel, pred_pos_rel_refined, mean, log_var, scores = model(
                obs_feat,
                obs_pos_last,
                obs_pos_rel,
                fut_pos_rel,
                seq_start_end,
                scene,
                scene_meta,
            )

            tloss, (l_reg, l_kld, l_ioc, l_ref) = k_total_loss(
                pred_pos_rel, pred_pos_rel_refined, fut_pos_rel, mean, log_var, scores
            )
            final_loss = tloss

            # Backprop
            final_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), norm_clip_value)
            optimizer.step()

            sum_loss += final_loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = sum_loss / max(1, num_batches)
        print(
            f"[Epoch {epoch}] total_loss={avg_loss:.6f}"
            f"l_reg={l_reg.item():.6f}  l_kld={l_kld.item():.6f}  l_ioc={l_ioc.item():.6f}   l_ref={l_ref.item():.6f}"
        )

        eval(f"eval_plot_{epoch}", model, eval_loader, device, scene, scene_meta)


if __name__ == "__main__":
    data_folder = Path("data/kiel/ais/3_features")
    scene_path = Path("data/kiel/scenes/bev.npz")
    scene_meta_path = Path("data/kiel/scenes/bev_meta.json")

    train_cpu(
        data_folder=data_folder,
        scene_path=scene_path,
        scene_meta_path=scene_meta_path,
        batch_size=4,
        num_epochs=10,
    )
