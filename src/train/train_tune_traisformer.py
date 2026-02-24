import logging
from pathlib import Path
import torch
import optuna

from models.traisformer.hierarchical_loss import loss_occupancy_heatmap, loss_intent_heatmap
from train.eval_heatmap import eval_heatmap
from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from train.training_loop import train_single_gpu
from train.run_worker import run_worker

from utils.config import DATA_FOLDER_PATH

def make_objective(
    data_folder: Path,
):
    def objective(trial: optuna.Trial):
        model_cls, cfg,  = TrAISformer, TraisformerParams()

        ### define search space
        lr = trial.suggest_categorical("lr", [1e-3])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5])
        cfg.n_embd = trial.suggest_categorical("n_embd", [128])
        cfg.n_layer = trial.suggest_categorical("n_layer", [4])
        cfg.n_head = trial.suggest_categorical("n_head", [4])

        cfg.pred_scope = trial.suggest_categorical("pred_scope", ["path", "destination"])

        if cfg.pred_scope == "path":
            loss_fn, eval_fn = loss_occupancy_heatmap, eval_heatmap
            cfg.pred_len = 0
        if cfg.pred_scope == "destination":
            loss_fn, eval_fn = loss_intent_heatmap, eval_heatmap
            cfg.pred_len = 1

        try:
            metric = train_single_gpu(
                model=model_cls(cfg),
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                trial=trial,
                weight_decay=weight_decay,
                lr=lr,
                best_ckpt_path = Path("checkpoints/traisformer/traisformer_dest_best.pt"),
            )
            return metric
        except torch.cuda.OutOfMemoryError:
            logging.info("[ERROR] Out of memory!")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    return objective

def train():
    """
    One worker process. It sees exactly ONE GPU because launcher sets CUDA_VISIBLE_DEVICES.
    All workers share the same Optuna storage to coordinate trials.
    """

    grid = {
        "pred_scope": ["path", "destination"]
    }

    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
    objective = make_objective(data_folder, grid)

    run_worker(objective, grid)

if __name__ == "__main__":
    train()

"""
CUDA_VISIBLE_DEVICES=0 MODEL_CHOICE=TRAISFORMER OPTUNA_STORAGE="sqlite:///trais_dest_full.db" OPTUNA_STUDY="trais_dest_full" OPTUNA_JSONL="trais_dest_full.jsonl" python -u src/train/train_tune_traisformer.py
"""