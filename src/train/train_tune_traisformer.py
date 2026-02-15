import logging
import os
from pathlib import Path
import torch
import optuna

from utils.logger import logger
from models.traisformer.hierarchical_loss import loss_intent_heatmap, loss_occupancy_heatmap
from train.eval_heatmap import eval_heatmap
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from train.training_loop import train_single_gpu, trial_jsonl_callback

from utils.config import DATA_FOLDER_PATH, STEPS_PER_MINUTE

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
        # cfg.intent_head = trial.suggest_categorical("intent_head", ["factorized", "cnn", "mixture", "lowrank"])
        # cfg.k_rank = trial.suggest_categorical("k_rank", [i for i in range(50)])

        lr = trial.suggest_categorical("lr", [1e-4, 3e-4, 1e-3])
        cfg.n_embd = trial.suggest_categorical("n_embd", [128])
        cfg.n_layer = trial.suggest_categorical("n_layer", [2, 3, 4, 6, 8])
        #cfg.n_head = trial.suggest_categorical("n_head", [4, 8])

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
        "lr": [3e-4],
        "n_embd": [128],
        "n_layer": [3],
        "n_head": [4]
        }
    
    # "factorized" -> 30, "cnn" -> 45, "mixture" -> 26, "lowrank" -> 25  -> 2337480 (3037144)
    # "factorized" -> 15, "cnn" -> 35, "mixture" -> 16, "lowrank" -> 15  -> 1168740 / 1184240
    # "factorized" -> 7, "cnn" -> 29, "mixture" -> 11, "lowrank" -> 10  -> 545412 / 553688

    # grid = {"intent_head": ["factorized"], "k_rank": [30, 15, 7]}
    # grid = {"intent_head": ["cnn"], "k_rank": [45, 32, 22]}
    # grid = {"intent_head": ["mixture"], "k_rank": [26, 13, 6]}
    # grid = {"intent_head": ["lowrank"], "k_rank": [25, 12, 5]}

    # grid = {"intent_head": ["factorized"], "k_rank": [15]}# [30, 18, 12]}
    # grid = {"intent_head": ["cnn"], "k_rank": [22]} #[45, 35, 29]}
    # grid = {"intent_head": ["mixture"], "k_rank": [7]} #[26, 16, 11]}
    # grid = {"intent_head": ["lowrank"], "k_rank": [5]} #[25, 15, 10]}


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
CUDA_VISIBLE_DEVICES=0 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_fact.db" OPTUNA_STUDY="trais_fact" OPTUNA_JSONL="trais_fact.jsonl" python -u src/train/train_tune_traisformer.py

CUDA_VISIBLE_DEVICES=1 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_fact2.db" OPTUNA_STUDY="trais_fact" OPTUNA_JSONL="trais_fact.jsonl" python -u src/train/train_tune_traisformer.py

CUDA_VISIBLE_DEVICES=2 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///sdfdfsd.db" OPTUNA_STUDY="trais_head" OPTUNA_JSONL="trais_head.jsonl" python -u src/train/train_tune_traisformer.py

CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_head.db" OPTUNA_STUDY="trais_head" OPTUNA_JSONL="trais_head.jsonl" python -u src/train/train_tune_traisformer.py

"""