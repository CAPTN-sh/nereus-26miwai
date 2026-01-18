import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import amp
from tqdm import tqdm
import optuna

from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from models.lstm.loss import mse, eval_lstm
from models.lstm.model import LSTMModel
from models.lstm.params import LSTMParams
from models.traisformer.hierarchical_loss import loss_intent_heatmap, loss_occupancy_heatmap
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from train.eval_heatmap import eval_heatmap
from scene_loader.loader import scene_loader
from train.eval import eval, eval_loss
from utils.logger import logger
from train.early_stopper import EarlyStopper
from models.utils.maps.scene_gernerator import process_maps

from utils.config import DATA_FOLDER_PATH

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
            "value": trial.value,
            "params": formatted_params,
            "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", None),
        }
        _append_jsonl_atomic(jsonl_path, record)

    return _cb


def train_single_gpu(
    model_cls,
    cfg,
    loss_fn,
    eval_fn,
    data_folder: Path,
    batch_size: int,
    num_epochs: int,
    lr:float,
    weight_decay:float,
    trial: optuna.Trial | None = None,
):
    # One trial = one GPU (the worker process sets CUDA_VISIBLE_DEVICES)
    assert torch.cuda.is_available()

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

    feat_cols = ["speed", "course", "acc", "angular_difference", "length", "width", "ship_group"]
    train_dset, _, train_loader = scene_loader(
        data_folder=data_folder,
        flag="train",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=feat_cols,
    )

    eval_dset, _, eval_loader = scene_loader(
        data_folder=data_folder,
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=feat_cols,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = model_cls(cfg).to(device)

    if hasattr(model, "rasterizer"):
        print("loading scene layers")
        path = DATA_FOLDER_PATH / "maps/2_standardized/fh/kiel/" #TODO select scene depending on model
        scene_contiguous = np.ascontiguousarray(process_maps(model.rasterizer, path), dtype=np.float32)
        scene = torch.from_numpy(scene_contiguous).unsqueeze(0).to(device)
    else:
        scene = None

    total_batches = 0
    num_batches = 0
    eval_step = 0
    loss_sum = 0.0
    max_batches = 40_000        # 40_000 bei batchsize 64
    batches_per_eval = 1_000    # 1_000
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_batches = batches_per_eval
    warmup_lambda = lambda step: min(1.0, (step + 1) / warmup_batches)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=0.1, 
        patience=4,
        cooldown=2,
        min_lr=1e-6,
    )
    """

    scaler = amp.GradScaler()
    stopper = EarlyStopper(patience=5, min_delta=1e-4)
    best_metric = float("inf")

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)

            batch = [t.to(device, non_blocking=True) for t in batch]
            output = model(batch, scene)
            loss, _loss_dict = loss_fn(output, batch, eval_step, config=cfg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            num_batches += 1
            total_batches += 1
            loss_sum += float(loss.item())

            if total_batches <= warmup_batches:
                warmup_scheduler.step()

            if total_batches % batches_per_eval == 0:
                eval_step = int(total_batches//batches_per_eval)

                train_loss = loss_sum / max(1, num_batches)
                logging.info(f"[Eval Step {eval_step}] train_loss={train_loss:.4f}")

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
                best_metric = min(best_metric, metric)
                # scheduler.step(metric)
                
                # report to Optuna (so pruning can work)
                trial.report(best_metric, step=eval_step)
                trial.set_user_attr("epochs_ran", eval_step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                # local early stopping based on *current val metric*
                if (total_batches > warmup_batches):
                    if stopper.step(metric):
                        logging.info(f"[Eval Step {eval_step}] Early stopping: best={stopper.best:.6f}")
                        break
                if (total_batches >= max_batches):
                    logging.info(f"[Eval Step {eval_step}] Late stopping: best={stopper.best:.6f}")
                    break

        if stopper.stop or (total_batches >= max_batches):
            break
    return best_metric

def make_objective(
    model_choice: str,
    data_folder: Path,
    pred_scope: str,
):
    # pick model
    if model_choice == "DESIRE":
        model_cls, cfg, loss_fn, eval_fn = DESIRE, DESIREParams(), loss_desire, eval
    elif model_choice == "LSTM":
        model_cls, cfg, loss_fn, eval_fn = LSTMModel, LSTMParams(), mse, eval_lstm
    elif model_choice == "TRAISFORMER":
        model_cls, cfg, (TrAISformer, TraisformerParams())
        loss_fn, eval_fn = (loss_occupancy_heatmap if pred_scope == "path" else loss_intent_heatmap, eval_heatmap)
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    def objective(trial: optuna.Trial):
        # --- general params ---
        batch_size = trial.suggest_categorical("batch_size", [64])
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay",  1e-6, 1e-3, log=True)

        # --- traisformer params ---
        if (model_choice == "DESIRE"):
            cfg.hidden_size = trial.suggest_categorical("hidden_size", [32, 48, 64, 80])
            cfg.out_channels = trial.suggest_categorical("out_channels", [16, 32])
            cfg.latent_size = trial.suggest_categorical("latent_size", [16, 32, 48])
            cfg.num_samples = trial.suggest_categorical("num_samples", [1, 2, 3])
        try:
            metric = train_single_gpu(
                model_cls=model_cls,
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                batch_size=batch_size,
                num_epochs=1,
                lr=lr,
                weight_decay=weight_decay,
                trial=trial,
            )
            return metric
        except torch.cuda.OutOfMemoryError:
            logging.info("[ERROR] Out of memory!")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    return objective

def run_worker():
    """
    One worker process. It sees exactly ONE GPU because launcher sets CUDA_VISIBLE_DEVICES.
    All workers share the same Optuna storage to coordinate trials.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_choice = os.environ.get("MODEL_CHOICE", "TRAISFORMER")
    storage = os.environ.get(
        "OPTUNA_STORAGE", f"sqlite:///optuna_{model_choice.lower()}.db"
    )
    study_name = os.environ.get("OPTUNA_STUDY", f"hpo_{model_choice.lower()}")

    # Shared JSONL path (same for all workers)
    jsonl_path = Path(os.environ.get("OPTUNA_JSONL", f"optuna_{study_name}.jsonl"))

    # your paths
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh/kiel"

    logger(file_prefix=f"optuna_worker_{model_choice}")
    logging.info(study_name)

    sampler = optuna.samplers.TPESampler(
        multivariate=True, 
        constant_liar=True,
    )

    pruner = optuna.pruners.PercentilePruner(
        percentile=10.0,
        n_startup_trials=10,
        n_warmup_steps=8,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        model_choice=model_choice,
        data_folder=data_folder,
        pred_scope = "path", # "path" "destination"
    )

    cb = trial_jsonl_callback(jsonl_path)

    study.optimize(objective, n_trials=17, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    run_worker()

"""

CUDA_VISIBLE_DEVICES=0 MODEL_CHOICE=DESIRE OPTUNA_STORAGE="sqlite:///desire_rel.db" OPTUNA_STUDY="desire_rel" OPTUNA_JSONL="desire_rel.jsonl" python -u src/train/train_tune_desire.py

"""