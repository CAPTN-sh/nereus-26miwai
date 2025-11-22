import json
import logging
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.optim as optim
from torch import amp
from tqdm import tqdm

from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from models.lstm.model import LSTMModel
from models.lstm.params import LSTMParams
from sceen_loader.loader import sceen_loader
from train.eval import eval, eval_loss
from utils.logger import logger

os.environ["OMP_NUM_THREADS"] = "4"


def tune_gpu(
    trial: optuna.trial.Trial,
    gpu_id: int,
    args,
    model,
    loss_fn,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    batch_size: int = 128,
    num_epochs: int = 30,
    norm_clip_value: float = 1.0,
    lr: float = 4e-3,
    eval_fn=eval,
):
    # --- Hyperparameters to be tuned by Optuna ---
    model_params = args["model_params"]
    for param, values in args["model_hyper_params"].items():
        setattr(model_params, param, trial.suggest_categorical(param, values))

    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    ### --- GPU Setup --- ###
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    logging.info(f"[Trial {trial.number}] Starting training on {device}")

    ### --- DataLoader --- ###
    train_dset, train_sampler, train_loader = sceen_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-04-14"),
        max_date=pd.Timestamp("2022-04-16"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=["speed", "course"],
    )

    eval_dset, eval_sampler, eval_loader = sceen_loader(
        data_folder=data_folder,
        min_date=pd.Timestamp("2022-06-01"),
        max_date=pd.Timestamp("2022-06-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=["speed", "course"],
    )

    logging.info(f"[Train] model: {model.__name__}")
    logging.info(f"There are {len(train_dset)} traj loaded for training")
    logging.info(f"There are {len(eval_dset)} traj loaded for evaluation")

    ### --- Model --- ###
    model = model(model_params).to(device)
    npz = np.load(scene_path)
    scene = torch.from_numpy(npz["I"]).unsqueeze(0).to(device)  # TODO unsqueeze?
    scene_meta = json.load(open(scene_meta_path))

    scene_meta["world_to_bev"] = torch.as_tensor(
        scene_meta["world_to_bev"], device=device, dtype=torch.float32
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=num_epochs // 4, gamma=0.5
    )

    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)

        loss_sum_dict = {}
        loss_sum = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            batch = [t.to(device) for t in batch]

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch, scene, scene_meta)
                loss, loss_dict = loss_fn(output, batch, epoch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clip_value)
            scaler.step(optimizer)
            scaler.update()

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

        if (epoch + 1) % 4 == 0:
            optuna_loss = eval_fn(epoch, model, eval_loader, device, scene, scene_meta)
            trial.report(optuna_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return optuna_loss


def objective(trial: optuna.trial.Trial, gpu_id: int, args):
    return tune_gpu(trial, gpu_id, args, **args["train_kwargs"])


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model_options = ["DESIRE", "LSTM"]
    model_choise = model_options[1]

    logger(file_prefix=f"train_server_{model_choise}_rank_{local_rank}")

    if model_choise == "DESIRE":
        model = DESIRE
        model_params = DESIREParams()
        model_hyper_params = {}
        loss_fn = loss_desire
        eval_fn = eval

    if model_choise == "LSTM":
        model = LSTMModel
        model_params = LSTMParams()
        model_hyper_params = {"hidden_size": [64, 128, 256]}
        loss_fn = eval_loss
        eval_fn = eval

    args = {
        "model_params": model_params,
        "model_hyper_params": model_hyper_params,
        "train_kwargs": {
            "model": model,
            "loss_fn": loss_fn,
            "data_folder": Path("/home/bbiesenbach/data/kiel/ais/3_features"),
            "scene_path": Path("data/kiel/scenes/bev.npz"),
            "scene_meta_path": Path("data/kiel/scenes/bev_meta.json"),
            "num_epochs": 64,
            "eval_fn": eval_fn,
        },
    }

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner()
    )
    study = optuna.create_study(
        study_name=f"study_{model_choise}",
        storage="sqlite:///db.sqlite3",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, local_rank, args),
        n_trials=20 // int(os.environ.get("WORLD_SIZE", 1)),
    )
    if local_rank == 0:
        logging.info(f"Best trial: {study.best_trial.value}")
        logging.info(f"Best params: {study.best_trial.params}")
