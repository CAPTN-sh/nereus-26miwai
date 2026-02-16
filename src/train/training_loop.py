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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

HOUR = 3600

# ---------- JSONL logging (multi-process safe via file lock) ----------
def _append_jsonl_atomic(jsonl_path: Path, record: dict):
    """
    Append one JSON object per line to jsonl_path.
    Uses an advisory file lock (fcntl) so multiple GPU workers can write safely.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # fcntl exists on Linux (your setup)
    import fcntl  # noqa: WPS433

    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def trial_jsonl_callback(jsonl_path: Path):
    """
    Optuna callback factory: called after each trial finishes (success/pruned/fail).
    Appends a JSON record to a shared JSONL.
    """
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        formatted_params = {k: round(v, 7) if isinstance(v, float) else v for k, v in trial.params.items()}
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial_number": trial.number,
            "epochs_ran": trial.user_attrs.get("epochs_ran", None),
            "state": trial.state.name,
            "n_model_params": trial.user_attrs.get("n_model_params", None),
            "value": trial.value,
            "params": formatted_params,
            "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        }
        _append_jsonl_atomic(jsonl_path, record)

    return _cb

def train_single_gpu(
    model,
    cfg,
    loss_fn,
    eval_fn,
    data_folder: Path,
    trial: optuna.Trial,
    lr: float,
    weight_decay: float = 1e-5,
    batch_size: int = 256,
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

    if n_model_params > 10_000_000:
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
    batches_per_eval = 5_000
    max_epochs = 4

    if is_tuning:
        max_batches = 50_000
        max_seconds = HOUR * 1
    else:
        max_batches = 999_999
        max_seconds = HOUR * 16
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            total_steps=max_epochs * len(train_loader),
            pct_start=0.2,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
    
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

            if not is_tuning:
                scheduler.step()

            num_batches += 1
            total_batches += 1
            loss_sum += float(loss.item())

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
                # report to Optuna (so pruning can work)
                trial.report(metric, step=eval_step)
                trial.set_user_attr("epochs_ran", eval_step)
                if trial.should_prune():
                    logging.info(f"[TrialPruned] PercentilePruner")
                    raise optuna.TrialPruned()

                if (total_batches >= max_batches):
                    logging.info(f"[Eval Step {eval_step}] Trajectory budget exceeded: best={best_metric:.6f}")
                    break

                if time_exeeded:
                    logging.info(f"[Eval Step {eval_step}] Time budget exceeded: best={best_metric:.6f}")
                    break

        if (total_batches >= max_batches) or time_exeeded:
            break
    return best_metric