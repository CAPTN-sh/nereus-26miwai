import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
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
from loader_heatmap.loader import loader_heatmap
from train.eval import eval, eval_loss
from utils.logger import logger
from train.early_stopper import EarlyStopper
from models.utils.maps.scene_gernerator import process_maps
from models.utils.maps.rasterize import Rasterizer

from utils.config import DATA_FOLDER_PATH, STEPS_PER_MINUTE

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

    feat_cols = ["speed", "course"] #, "acc", "angular_difference", "length", "width", "ship_group"]
    train_dset, _, train_loader = loader_heatmap(
        data_folder=data_folder,
        flag="train",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=feat_cols,
        fut_len=cfg.pred_len,
        obs_len=cfg.obs_len,
    )

    eval_dset, _, eval_loader = loader_heatmap(
        data_folder=data_folder,
        flag="val",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=feat_cols,
        fut_len=cfg.pred_len,
        obs_len=cfg.obs_len,
    )

    model = model_cls(cfg).to(device)

    path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
    my_rasterizer = Rasterizer([10.12, 54.31, 10.33, 54.46])

    scene_contiguous = np.ascontiguousarray(process_maps(my_rasterizer, path))
    scene = torch.from_numpy(scene_contiguous).unsqueeze(0).to(device).to(torch.float32)

    total_batches = 0
    num_batches = 0
    loss_sum = 0.0
    max_batches = 20_000
    batches_per_eval = 1_000
    
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

    stopper = EarlyStopper(patience=5, min_delta=1e-4)
    best_metric = float("inf")

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)

            batch = [t.to(device, non_blocking=True) for t in batch]

            output = model(batch, scene)
            loss, _loss_dict = loss_fn(output, batch, config=cfg)

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
        model_cls, cfg, loss_fn, eval_fn = (
            TrAISformer,
            TraisformerParams(),
            loss_occupancy_heatmap if pred_scope == "path" else loss_intent_heatmap,
            eval_heatmap,
        )
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    def objective(trial: optuna.Trial):
        # --- general params ---
        batch_size = trial.suggest_categorical("batch_size", [512])
        cfg.obs_len = 10 * STEPS_PER_MINUTE

        if (model_choice == "LSTM"):
            lr = trial.suggest_categorical("lr",  [1e-4, 5e-4, 1e-3, 3e-3])
            weight_decay = trial.suggest_categorical("weight_decay",  [1e-6, 1e-5, 1e-4])
            hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 1024])
            cfg.enc_hidden_size = hidden_size
            cfg.dec_hidden_size = hidden_size

        # --- traisformer params ---
        if (model_choice == "TRAISFORMER"):
            lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
            weight_decay = trial.suggest_float("weight_decay",  1e-6, 1e-4, log=True)

            cfg.pred_scope = pred_scope
            logging.info(f"pred_scope={pred_scope}")

            if pred_scope == "path":
                cfg.pred_len = 20 * STEPS_PER_MINUTE

            cfg.n_layer = trial.suggest_int("n_layer", 4, 8, step=2)
            cfg.n_head = trial.suggest_categorical("n_head", [4, 8])
            cfg.n_embd = trial.suggest_categorical("n_embd", [256, 512, 1024])

            cat_meta = [("spatial", [4], 2), ("kinematic", [2], 2)]
            cat_meta += [("dynamic", [0,1,2], 2), ("terrain", [0,1,2], 1), ("vessel", [0,1,2], 1)]

            names, _, costs = zip(*cat_meta)
            splits = [trial.suggest_categorical(f"n_{n}", r) for n, r, c in cat_meta]
            budget = (cfg.n_embd) // 8

            embd = np.array([int(s * budget / sum(splits)) for s in splits])
            remainders = np.array(splits) * budget / sum(splits) % 1
            embd[np.argsort(-remainders, kind="stable")[:budget - embd.sum()]] += 1

            print_dim = "embd split:"
            for name, cost, value in zip(names, costs, embd):
                print_dim += f" {name}:{value * 8}"
                setattr(cfg, f"n_{name}_embd", int(value * 8 // cost))
            logging.info(print_dim)

            cfg.dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
            cfg.attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.2, step=0.05)

            cfg.coarse_loss_beta = trial.suggest_categorical("coarse_loss_beta", [0.0, 0.5, 1.0])
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
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"

    logger(file_prefix=f"optuna_worker_{model_choice}")
    logging.info(study_name)

    if False:
            #"obs_minutes": [1, 5, 10, 15, 20],
        sampler = optuna.samplers.GridSampler({
            "lr":  [1e-4, 5e-4, 1e-3, 3e-3],
            "weight_decay":  [1e-6, 1e-5, 1e-4],
            "hidden_size": [256, 512, 1024],
        })
        pruner = optuna.pruners.NopPruner()
    else:
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            constant_liar=True,
        )

        pruner = optuna.pruners.PercentilePruner(
            percentile=75.0,
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

    study.optimize(objective, n_trials=30, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")

if __name__ == "__main__":
    run_worker()

"""
[Experiment 1] Best observation length for short and long term
CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///obs_trais_dest_256.db" OPTUNA_STUDY="obs_trais_dest_256" OPTUNA_JSONL="obs_trais_dest_256.jsonl" python -u src/train/train_tune.py
CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///obs_trais_dest_256.db" OPTUNA_STUDY="obs_trais_dest_256" OPTUNA_JSONL="obs_trais_dest_256.jsonl" python -u src/train/train_tune.py
CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=LSTM OPTUNA_STORAGE="sqlite:///obs_lstm.db" OPTUNA_STUDY="obs_lstm" OPTUNA_JSONL="obs_lstm.jsonl" python -u src/train/train_tune.py

[Experiment 2] LSTM:
CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=LSTM OPTUNA_STORAGE="sqlite:///lstm_rel.db" OPTUNA_STUDY="lstm_rel" OPTUNA_JSONL="lstm_rel.jsonl" python -u src/train/train_tune.py

[Experiment 3] TrAISfromer:
CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///path_traisformer.db" OPTUNA_STUDY="path_traisformer" OPTUNA_JSONL="path_traisformer.jsonl" python -u src/train/train_tune.py


CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///path_traisformer.db" OPTUNA_STUDY="path_traisformer" OPTUNA_JSONL="path_traisformer.jsonl" python -u src/train/train_tune.py
CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///dest_traisformer.db" OPTUNA_STUDY="dest_traisformer" OPTUNA_JSONL="dest_traisformer.jsonl" python -u src/train/train_tune.py

CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=LSTM OPTUNA_STORAGE="sqlite:///obs_len_lstm_512.db" OPTUNA_STUDY="obs_len_lstm_512" OPTUNA_JSONL="obs_len_lstm_512.jsonl" python -u src/train/train_tune.py
"""