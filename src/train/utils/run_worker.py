import logging
import os
from pathlib import Path

import optuna
import torch

from utils.logger import logger, trial_jsonl_callback

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def run_worker(objective, grid = None):
    """One worker process. It sees exactly ONE GPU because launcher sets CUDA_VISIBLE_DEVICES.
    All workers share the same Optuna storage to coordinate trials.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_choice = os.environ.get("MODEL_CHOICE")
    storage = os.environ.get("OPTUNA_STORAGE", f"sqlite:///optuna_{model_choice.lower()}.db")
    study_name = os.environ.get("OPTUNA_STUDY", f"hpo_{model_choice.lower()}")
    jsonl_path = Path(os.environ.get("OPTUNA_JSONL", f"optuna_{study_name}.jsonl"))

    logger(file_prefix=f"optuna_worker_{model_choice}")
    logging.info(study_name)

    if not grid:
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            constant_liar=True,
        )
        pruner = optuna.pruners.PercentilePruner(
            percentile=75.0,
            n_startup_trials=10,
            n_warmup_steps=8,
        )
    else:
        sampler = optuna.samplers.GridSampler(grid)
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    cb = trial_jsonl_callback(jsonl_path)
    study.optimize(objective, n_trials=50, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")
