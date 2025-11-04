import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lazy_loader.loader import lazy_loader
from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from models.lstm.model import LSTMModel
from models.traisformer.loss import loss_intent_heatmap
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from train.eval import eval, eval_loss
from train.eval_heatmap import eval_heatmap
from utils.logger import logger


def train_cpu(
    model,
    loss_fn,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    eval_fn=eval,
    max_neighbors: int = 10,
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
        min_date=pd.Timestamp("2022-05-03"),
        max_date=pd.Timestamp("2024-05-03"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=False,
        max_neighbors=max_neighbors,
        feat_cols=["speed", "course"],
    )
    eval_dset, eval_sampler, eval_loader = lazy_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-05-03"),
        max_date=pd.Timestamp("2024-05-03"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=False,
        max_neighbors=max_neighbors,
        feat_cols=["speed", "course"],
    )

    print("additional features:", train_dset.feature_cols)
    print(f"There are {len(train_dset)} traj loaded for training")
    print(f"There are {len(eval_dset)} traj loaded for evaluation")

    # --- Scene tensor ---
    npz = np.load(scene_path)
    scene = torch.from_numpy(npz["I"]).unsqueeze(0).to(device)
    scene_meta = json.load(open(scene_meta_path))

    scene_meta["world_to_bev"] = torch.as_tensor(
        scene_meta["world_to_bev"], device=device, dtype=torch.float32
    )

    # --- Model ---
    model = model.to(device)

    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=num_epochs // 4, gamma=0.5
    )

    # --- Train ---
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        loss_sum_dict = {}
        loss_sum = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # if batch_idx > 50:
            #    break
            optimizer.zero_grad()
            batch = [t.to(device) for t in batch]

            output = model(batch, scene, scene_meta)
            loss, loss_dict = loss_fn(output, batch, epoch)

            # Backprop
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), norm_clip_value)
            # TODO loss clippen
            optimizer.step()

            num_batches += 1
            loss_sum += loss.item()
            for loss_name, loss_val in loss_dict.items():
                loss_sum_dict[loss_name] = (
                    loss_sum_dict.get(loss_name, 0.0) + loss_val.item()
                )

        scheduler.step()

        loss_sum /= num_batches
        loss_print = f"[Epoch {epoch}] train_loss={loss_sum:.4f}"
        for loss_name, loss_val in loss_sum_dict.items():
            loss_val /= num_batches
            loss_print += f" {loss_name}={loss_val:.4f}"
        logging.info(loss_print)

        eval_fn(epoch, model, eval_loader, device, scene, scene_meta)


if __name__ == "__main__":
    data_folder = Path("data/ais/4_features/fhkiel_train/kiel/")
    scene_path = Path("data/scenes/fhkiel_train/kiel/bev.npz")
    scene_meta_path = Path("data/scenes/fhkiel_train/kiel/bev_meta.json")

    model_options = ["DESIRE", "LSTM", "TRAISFORMER"]
    model_choise = model_options[2]

    logger(file_prefix=f"train_local_{model_choise}")

    if model_choise == "DESIRE":
        model = DESIRE(DESIREParams())
        loss_fn = loss_desire
        eval_fn = eval
        max_neighbors = 5

    if model_choise == "LSTM":
        model = LSTMModel(pred_len=36)
        loss_fn = eval_loss
        eval_fn = eval
        max_neighbors = 0

    if model_choise == "TRAISFORMER":
        cfg = TraisformerParams()
        model = TrAISformer(cfg)
        loss_fn = loss_intent_heatmap
        eval_fn = eval_heatmap
        max_neighbors = 0

    train_cpu(
        model=model,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        data_folder=data_folder,
        scene_path=scene_path,
        scene_meta_path=scene_meta_path,
        max_neighbors=max_neighbors,
        batch_size=16,
        num_epochs=10,
    )
