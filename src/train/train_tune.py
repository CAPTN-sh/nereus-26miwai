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
from loader_heatmap.loader import loader_heatmap
from train.eval import eval, eval_loss
from utils.logger import logger
from train.early_stopper import EarlyStopper
from models.traisformer.scene_gernerator import process_maps
from models.traisformer.rasterize import Rasterizer

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
    scene_path: Path,
    scene_meta_path: Path,
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
    formatted_params = {k: round(v, 7) if isinstance(v, float) else v for k, v in trial.params.items()}
    logging.info(f"[Trial] params={formatted_params}")

    train_dset, _, train_loader = loader_heatmap(
        data_folder=data_folder,
        flag="train",
        min_date=pd.Timestamp("2022-01-01"),
        max_date=pd.Timestamp("2024-01-01"),
        world_size=1,
        rank=0,
        batch_size=batch_size,
        pin_memory=True,
        feat_cols=["speed", "course"],
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
        batch_size=128,
        pin_memory=True,
        feat_cols=["speed", "course"],
        fut_len=cfg.pred_len,
        obs_len=cfg.obs_len,
    )

    model = model_cls(cfg).to(device)

    if model == "DESIRE":
        npz = np.load(scene_path)
        scene = torch.from_numpy(npz["I"]).unsqueeze(0).to(device)
        scene_meta = json.load(open(scene_meta_path))
        scene_meta["world_to_bev"] = torch.as_tensor(
            scene_meta["world_to_bev"], device=device, dtype=torch.float32
        )
    #TODO select scene depending on model
    path = "/data/projects/ship_tracker/assets/maps/2_standardized/fh/kiel/"
    my_rasterizer = Rasterizer([10.12, 54.31, 10.33, 54.46])

    scene_contiguous = np.ascontiguousarray(process_maps(my_rasterizer, path))
    scene = torch.from_numpy(scene_contiguous).unsqueeze(0).to(device)
    scene_meta = None

    total_batches = 0
    num_batches = 0
    loss_sum = 0.0
    SAMPLES_PER_EVAL = 256_000 # 20_783_360 total samples
    batches_per_eval = SAMPLES_PER_EVAL // batch_size
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    warmup_batches = batches_per_eval
    warmup_lambda = lambda step: min(1.0, (step + 1) / warmup_batches)
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        factor=0.5, 
        patience=4,
        min_lr = 1e-7,
    )

    scaler = amp.GradScaler()
    stopper = EarlyStopper(patience=15, min_delta=1e-4)
    best_metric = float("inf")

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)

            batch = [t.to(device, non_blocking=True) for t in batch]

            with amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(batch, scene, scene_meta)
                loss, _loss_dict = loss_fn(output, batch, config=cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

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
                        scene_meta, 
                        trial_number=trial.number,
                        config=cfg,
                    )
                model.train()

                metric = float(metric)
                best_metric = min(best_metric, metric)
                scheduler.step(metric)
                
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

        if stopper.stop:
            break
    return best_metric

def make_objective(
    model_choice: str,
    data_folder: Path,
    scene_path: Path,
    scene_meta_path: Path,
    pred_scope: str,
    full_feat_set: bool,
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
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        base_lr = trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
        lr = base_lr * (batch_size / 64) ** 0.5
        weight_decay = trial.suggest_float("weight_decay",  1e-5, 1e-3, log=True)

        # --- traisformer params ---
        if (model_choice == "TRAISFORMER"):
            cfg.pred_scope = pred_scope
            cfg.full_feat_set = full_feat_set

            if pred_scope == "path":
                cfg.pred_len = 45 * 12

            cfg.n_layer = trial.suggest_int("n_layer", 2, 8)
            cfg.n_head = trial.suggest_categorical("n_head", [4, 8])
            head_dim = trial.suggest_int("head_dim", 32, 96, step=16)
            cfg.n_embd = cfg.n_head * head_dim

            cat_meta = [("spatial", 1, 2), ("kinematic", 0, 2)]
            if cfg.full_feat_set:
                cat_meta += [("dynamic", 0, 2), ("terrain", 0, 1), ("vessel", 0, 1)]

            names, starts, costs = zip(*cat_meta)
            splits = [trial.suggest_int(f"n_{name}", start, 4) for name, start, _ in cat_meta]

            embd = np.full(len(names), 8)
            budget = (cfg.n_embd - 8*sum(costs)) // 8
            while budget > 0:
                for i in range(4):
                    for k, cost in enumerate(costs):
                        if budget >= cost and splits[k] > i:
                            embd[k] += 8
                            budget -= 8 * cost

            for name, value in zip(names, embd):
                setattr(cfg, f"n_{name}_embd", 8 + value)

            cfg.dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
            cfg.attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.25, step=0.05)

            cfg.coarse_loss_beta = trial.suggest_categorical("coarse_loss_beta", [0.0, 0.5, 1.0, 2.0])
        try:
            metric = train_single_gpu(
                model_cls=model_cls,
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                scene_path=scene_path,
                scene_meta_path=scene_meta_path,
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
    data_folder = Path("/data/projects/ship_tracker/assets/ais/4_features/fh/kiel")
    scene_path = Path("data/kiel/scenes/bev.npz")
    scene_meta_path = Path("data/kiel/scenes/bev_meta.json")

    logger(file_prefix=f"optuna_worker_{model_choice}")
    logging.info(study_name)

    sampler = optuna.samplers.TPESampler(
        multivariate=True, 
        constant_liar=True,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=15,
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
        scene_path=scene_path,
        scene_meta_path=scene_meta_path,
        pred_scope = "destination", # "path" "destination"
        full_feat_set= True,
    )

    cb = trial_jsonl_callback(jsonl_path)

    study.optimize(objective, n_trials=50, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")

if __name__ == "__main__":
    run_worker()

"""
CUDA_VISIBLE_DEVICES=0 MODEL_CHOICE=LSTM OPTUNA_STORAGE="sqlite:///obs_len_lstm.db" OPTUNA_STUDY="obs_len_lstm" OPTUNA_JSONL="obs_len_lstm.jsonl" python -u src/train/train_tune.py

CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///traisformer_dest_light.db" OPTUNA_STUDY="traisformer_dest_light" OPTUNA_JSONL="traisformer_dest_light.jsonl" python -u src/train/train_tune.py
CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///traisformer_dest_full.db" OPTUNA_STUDY="traisformer_dest_full" OPTUNA_JSONL="traisformer_dest_full.jsonl" python -u src/train/train_tune.py

CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///traisformer_path_light.db" OPTUNA_STUDY="traisformer_path_light" OPTUNA_JSONL="traisformer_path_light.jsonl" python -u src/train/train_tune.py
CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///traisformer_path_full.db" OPTUNA_STUDY="traisformer_path_full" OPTUNA_JSONL="traisformer_path_full.jsonl" python -u src/train/train_tune.py
"""