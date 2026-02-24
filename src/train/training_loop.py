import json
import logging
import os
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
import optuna

from loaders.graph_loader.loader import graph_loader
from train.early_stopper import EarlyStopper
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer

from utils.config import DATA_FOLDER_PATH

HOUR = 3600

def train_single_gpu(
    model,
    cfg,
    loss_fn,
    eval_fn,
    data_folder: Path,
    trial: optuna.Trial,
    lr: float,
    weight_decay: float = 1e-5,
    batch_size: int = 512,
    best_ckpt_path: Path = None,
):
    # One trial = one GPU (the worker process sets CUDA_VISIBLE_DEVICES)
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    print(
        "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"),
        " torch.cuda.current_device()=", torch.cuda.current_device(),
        " name=", torch.cuda.get_device_name(0),
        flush=True
    )

    logging.info("#" * 20)
    logging.info(f"[Trial] number={trial.number}")
    trial_settings = {k: round(v, 6) if isinstance(v, float) else v for k, v in trial.params.items()}
    logging.info(f"[Trial {trial.number}] {trial_settings}")
    
    path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
    sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))

    scene_contiguous = np.ascontiguousarray(sl.load_scene(path))
    scene = torch.from_numpy(scene_contiguous).to(device).to(torch.float32)

    model = model.to(device)
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("n_model_params", n_model_params)
    logging.info(f"Trainable parameters: {n_model_params}")

    if n_model_params > 5_000_000:
        logging.info(f"[TrialPruned] Parameter budget exceeded: {n_model_params:,}")
        raise optuna.exceptions.TrialPruned()

    train_loader, _ = graph_loader(
        data_folder=data_folder,
        flag="train",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        batch_size=batch_size,
        pin_memory=True,
        pred_len=cfg.pred_len,
        obs_len=cfg.obs_len,
        max_edge_dist = cfg.max_dist,
    )

    eval_loader, _ = graph_loader(
        data_folder=data_folder,
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        batch_size=batch_size,
        pin_memory=True,
        pred_len=cfg.pred_len,
        obs_len=cfg.obs_len,
        max_edge_dist = cfg.max_dist,
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_batches = 0
    num_batches = 0
    loss_sum = 0.0

    is_tuning = best_ckpt_path is None
    batches_per_eval = int(len(train_loader) // 10)

    if is_tuning:
        max_epochs = 1
        max_seconds = HOUR * 1
        warmup_batches = batches_per_eval
    else:
        max_epochs = 10
        max_seconds = HOUR * 10
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        warmup_batches = batches_per_eval * 3
    
    warmup_lambda = lambda step: min(1.0, (step + 1) / warmup_batches)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=5,
        cooldown=2,
        min_lr=1e-6,
    )

    stopper = EarlyStopper(patience=15, min_delta=1e-4)
    
    best_metric = float("inf")

    start_time = time.perf_counter()
    training_time = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for epoch in range(max_epochs):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)

            start_event.record()

            batch = batch.to(device, non_blocking=True)
            output = model(batch, scene)
            loss, _loss_dict = loss_fn(output, batch, config=cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            end_event.record()
            torch.cuda.synchronize()

            step_time_ms = start_event.elapsed_time(end_event)
            training_time += (step_time_ms / 1000.0)

            num_batches += 1
            total_batches += 1
            loss_sum += float(loss.item())

            if total_batches <= warmup_batches:
                warmup_scheduler.step()

            time_exeeded = time.perf_counter() - start_time > max_seconds
            if (total_batches % batches_per_eval == 0) or time_exeeded:
                eval_step = int(total_batches//batches_per_eval) + int(time_exeeded)

                train_loss = loss_sum / max(1, num_batches)
                logging.info(f"[Eval Step {eval_step}] train_loss={train_loss:.6f} train_time={training_time / 60:.2f} minutes")

                num_batches = 0
                loss_sum = 0.0

                with torch.no_grad():
                    metric = eval_fn(
                        eval_step, 
                        model, 
                        eval_loader,
                        device,
                        scene,
                        trial_number=trial.number,
                        config=cfg,
                    )
                model.train()

                metric = float(metric)
                if metric < best_metric:
                    best_metric = metric
                    if not is_tuning:
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "metric": best_metric,
                                "eval_step": eval_step,
                                "config": cfg,
                            },
                            best_ckpt_path,
                        )
                        logging.info(
                            f"[Eval Step {eval_step}] New best metric={best_metric:.6f} → saved model"
                        )

                if not is_tuning:
                    scheduler.step(metric)

                # report to Optuna (so pruning can work)
                trial.report(metric, step=eval_step)
                trial.set_user_attr("epochs_ran", eval_step)
                if trial.should_prune():
                    logging.info(f"[TrialPruned] PercentilePruner")
                    raise optuna.TrialPruned()
                
                if (total_batches > warmup_batches):
                    if stopper.step(metric):
                        logging.info(f"[Eval Step {eval_step}] Early stopping : best={stopper.best:.6f}")
                        break

                if time_exeeded:
                    logging.info(f"[Eval Step {eval_step}] Time budget exceeded: best={best_metric:.6f}")
                    break

        if stopper.stop or time_exeeded:
            break
    return best_metric