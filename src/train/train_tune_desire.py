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

from models.desire.model import DESIRE
from models.desire.nn.loss import loss_desire
from models.desire.utils.params import DESIREParams
from loaders.graph_loader.loader import graph_loader
from utils.logger import logger
from train.eval import eval
from train.early_stopper import EarlyStopper
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer
from train.training_loop import train_single_gpu, trial_jsonl_callback

from utils.config import DATA_FOLDER_PATH, STEPS_PER_MINUTE

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def make_objective(data_folder: Path):

    def objective(trial: optuna.Trial):
        model_cls, cfg, loss_fn, eval_fn = DESIRE, DESIREParams(), loss_desire, eval

        lr = trial.suggest_categorical("lr", [1e-3])

        cfg.hidden_size = trial.suggest_categorical("hidden_size", [256])
        cfg.intermediate_size = trial.suggest_categorical("intermediate_size", [16, 32, 64])
        cfg.latent_size = trial.suggest_categorical("latent_size", [16])
        cfg.out_channels = trial.suggest_categorical("out_channels", [16])
        cfg.max_dist = trial.suggest_categorical("max_dist", [1000]) #[10, 500, 1000, 2000])
        cfg.num_refine_iters = trial.suggest_categorical("num_refine_iters", [1])
        cfg.num_samples = trial.suggest_categorical("num_samples", [4])

        try:
            metric = train_single_gpu(
                model_cls=model_cls,
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                trial=trial,
                lr=lr,
                best_ckpt_path = None,
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
    grid = {
        "intermediate_size": [16, 32, 64]
    }

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

    objective = make_objective(data_folder=data_folder)

    cb = trial_jsonl_callback(jsonl_path)

    study.optimize(objective, n_trials=50, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")

if __name__ == "__main__":
    run_worker()

"""
[Experiment 1] Best observation length for short and long term

# encoder decoder
CUDA_VISIBLE_DEVICES=1 MODEL_CHOICE=DESIRE OPTUNA_STORAGE="sqlite:///desire_inter.db" OPTUNA_STUDY="desire_inter" OPTUNA_JSONL="desire_inter.jsonl" python -u src/train/train_tune_desire.py

"""