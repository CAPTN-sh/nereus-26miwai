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

from models.nereus.params import NEREUSParams
from loaders.graph_loader.loader import graph_loader
from utils.logger import logger
from train.early_stopper import EarlyStopper
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer

from models.traisformer.hierarchical_loss import loss_intent_heatmap, loss_occupancy_heatmap
from train.eval_heatmap import eval_heatmap
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams

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
            "n_model_params": trial.user_attrs.get("n_model_params", None),
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

    ckpt_dir = Path("checkpoints/traisformer")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"traisformer_{cfg.pred_scope}_best.pt"

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

    model = model_cls(cfg).to(device)
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr("n_model_params", n_model_params)
    logging.info(f"Trainable parameters: {n_model_params:,}")

    if n_model_params > 3_100_000:
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
        max_edge_dist = 0,
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
        max_edge_dist = 0,
    )

    total_batches = 0
    num_batches = 0
    loss_sum = 0.0
    max_batches = 25_000
    batches_per_eval = 1_000
    max_seconds = 60 * 60 * 1
    
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

    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        model.train()
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad(set_to_none=True)

            batch = batch.to(device, non_blocking=True)
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
                if metric < best_metric:
                    best_metric = metric
                    """
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
                scheduler.step(metric)
                """
                
                # report to Optuna (so pruning can work)
                trial.report(best_metric, step=eval_step)
                trial.set_user_attr("epochs_ran", eval_step)
                if trial.should_prune():
                    logging.info(f"[TrialPruned] PercentilePruner")
                    raise optuna.TrialPruned()

                if (total_batches > warmup_batches):
                    if stopper.step(metric):
                        logging.info(f"[Eval Step {eval_step}] Early stopping : best={stopper.best:.6f}")
                        break
                if (total_batches >= max_batches):
                    logging.info(f"[Eval Step {eval_step}] Trajectory budget exceeded: best={stopper.best:.6f}")
                    break

                if time.perf_counter() - start_time > max_seconds:
                    logging.info(f"[Eval Step {eval_step}] Time budget exceeded: best={stopper.best:.6f}")
                    break

        if stopper.stop or (total_batches >= max_batches) or (time.perf_counter() - start_time > max_seconds):
            break
    return best_metric

def make_objective(
    data_folder: Path,
):

    def objective(trial: optuna.Trial):
        model_cls, cfg, loss_fn, eval_fn = (
            TrAISformer,
            TraisformerParams(),
            loss_occupancy_heatmap,
            eval_heatmap,
        )
        #cfg.intent_head = trial.suggest_categorical("intent_head", ["factorized"])



        # "factorized" -> 30, "cnn" -> 24, "mixture" -> 26, "lowrank" -> 25, upsample 
        cfg.intent_head = trial.suggest_categorical("intent_head", ["factorized", "cnn", "mixture", "lowrank"])
        cfg.k_rank = trial.suggest_categorical("k_rank", [i for i in range(50)])

        lr = trial.suggest_categorical("lr", [3e-4])
        weight_decay = 1e-4
        batch_size = 512

        cfg.n_embd = trial.suggest_categorical("n_embd", [128])
        cfg.n_layer = trial.suggest_categorical("n_layer", [3])
        cfg.n_head = trial.suggest_categorical("n_head", [4])

        try:
            metric = train_single_gpu(
                model_cls=model_cls,
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                batch_size=batch_size,
                num_epochs=10,
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
    
    # 1,243,712
    # "factorized" -> 30, "cnn" -> 45, "mixture" -> 26, "lowrank" -> 25  -> 3 / 1.8
    # "factorized" -> 18, "cnn" -> 35, "mixture" -> 16, "lowrank" -> 15  -> 2.10 / 0.9
    # "factorized" -> 12, "cnn" -> 29, "mixture" -> 11, "lowrank" -> 10  -> 1.65 / 0.45
    
    grid = {"intent_head": ["factorized"], "k_rank": [30, 18, 12]}
    grid = {"intent_head": ["cnn"], "k_rank": [45, 35, 29]}
    #grid = {"intent_head": ["mixture"], "k_rank": [26, 16, 11]}
    #grid = {"intent_head": ["lowrank"], "k_rank": [25, 15, 10]}


    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_choice = os.environ.get("MODEL_CHOICE")
    storage = os.environ.get("OPTUNA_STORAGE", f"sqlite:///optuna_{model_choice.lower()}.db")
    study_name = os.environ.get("OPTUNA_STUDY", f"hpo_{model_choice.lower()}")
    jsonl_path = Path(os.environ.get("OPTUNA_JSONL", f"optuna_{study_name}.jsonl"))

    # your paths
    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"

    logger(file_prefix=f"optuna_worker_{model_choice}")
    logging.info(study_name)

    if True:
        sampler = optuna.samplers.GridSampler(grid)
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

    objective = make_objective(data_folder)

    cb = trial_jsonl_callback(jsonl_path)

    study.optimize(objective, n_trials=50, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")

if __name__ == "__main__":
    run_worker()

"""
[Experiment 1] Best observation length for short and long term

# traisformer
CUDA_VISIBLE_DEVICES=0 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_head.db" OPTUNA_STUDY="trais_head" OPTUNA_JSONL="trais_head.jsonl" python -u src/train/train_tune_traisformer.py

CUDA_VISIBLE_DEVICES=1 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_head.db" OPTUNA_STUDY="trais_head" OPTUNA_JSONL="trais_head.jsonl" python -u src/train/train_tune_traisformer.py

CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_head.db" OPTUNA_STUDY="trais_head" OPTUNA_JSONL="trais_head.jsonl" python -u src/train/train_tune_traisformer.py

CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_head.db" OPTUNA_STUDY="trais_head" OPTUNA_JSONL="trais_head.jsonl" python -u src/train/train_tune_traisformer.py

"""