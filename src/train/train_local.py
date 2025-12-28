import json
import logging
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from sceen_loader.loader import sceen_loader
from train.eval import eval


def tune_cpu(
    trial: optuna.trial.Trial,
    model,
    model_params,
    model_hyper_params,
    loss_fn,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    num_epochs: int = 10,
    eval_fn=eval,
):
    # --- Hyperparameters to be tuned by Optuna ---
    for param, values in model_hyper_params.items():
        setattr(model_params, param, trial.suggest_categorical(param, values))

    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    return train_cpu(
        model(model_params),
        loss_fn,
        data_folder,
        scene_path,
        scene_meta_path,
        eval_fn=eval_fn,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
    )


def train_cpu(
    model,
    loss_fn,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    eval_fn=eval,
    num_epochs: int = 10,
    batch_size: int = 16,
    lr: float = 4e-3,
):

    # --- CPU device ---
    device = torch.device("cpu")
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    print(f"Starting training on {device}")

    # --- DataLoaders (single-process, no DDP) ---
    train_dset, train_sampler, train_loader = sceen_loader(
        data_folder=data_folder / "fhkiel_train/kiel",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2022-05-16"), # hyper: 2022-05-16, full: 2024-01-01
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=False,
        feat_cols=["speed", "course"],
    )

    eval_dset, eval_sampler, eval_loader = sceen_loader(
        data_folder=data_folder / "fhkiel_val/kiel",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2023-03-25"), # hyper: 2023-03-25, full: 2024-01-01 
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=False,
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
            nn.utils.clip_grad_norm_(model.parameters(), 1)
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

        optuna_loss = eval_fn(epoch, model, eval_loader, device, scene, scene_meta)
    return optuna_loss
