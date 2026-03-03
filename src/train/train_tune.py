import logging
import os
import torch
import optuna
from pathlib import Path

from models.gru.model import GRU_RNN
from models.nereus.params import NEREUSParams
from models.gru.loss import mse_loss
from eval.eval_gru import eval_gru

from models.desire.model import DESIRE
from models.desire.params import DESIREParams
from models.desire.loss import loss_desire
from eval.eval_desire import eval_desire

from models.traisformer.model import TrAISformer
from models.traisformer.params import TraisformerParams
from models.traisformer.loss import loss_heatmap
from eval.eval_traisformer import eval_traisformer

from models.nereus.init import init_nereus
from models.nereus.loss import mdn_loss
from eval.eval_nereus import eval_nereus

from train.utils.run_worker import run_worker
from train.utils.training_loop import train_single_gpu
from utils.config import AIS_FOLDER_PATH

def make_objective(data_folder: Path, mode = "train"):

    def objective(trial: optuna.Trial):
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    
        model_choice = os.environ.get("MODEL_CHOICE")
        study_name = os.environ.get("OPTUNA_STUDY")

        if model_choice == "DESIRE":
            cfg = DESIREParams()
        elif model_choice == "TRAISFORMER":
            cfg = TraisformerParams()
        else:
            cfg = NEREUSParams()

        # tune params
        lr = trial.suggest_categorical("lr", [1e-3, 3e-4, 1e-4])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5])
        cfg.hidden_size = trial.suggest_categorical("hidden_size", [256])

        if model_choice == "GRU_RNN":
            loss_fn, eval_fn = mse_loss, eval_gru
            cfg.max_dist = None
            model = GRU_RNN(cfg)
            
        if model_choice == "DESIRE":
            loss_fn, eval_fn = loss_desire, eval_desire
            model = DESIRE(cfg)

        if model_choice == "TRAISFORMER":
            loss_fn, eval_fn = loss_heatmap, eval_traisformer
            cfg.pred_scope = ["path"] # ["path", "destination"]
            cfg.pred_len = int(cfg.pred_scope == "destination")
            model = TrAISformer(cfg)

        if model_choice == "NEREUS":
            loss_fn, eval_fn = mdn_loss, eval_nereus
            model = init_nereus(study_name, cfg, device)

        checkpoint_path = None
        if mode == "train":
            checkpoint_path = Path(f"checkpoints/{model_choice.lower()}/{study_name}_best.pt")

        try:
            metric = train_single_gpu(
                model = model,
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                trial=trial,
                lr=lr,
                weight_decay=weight_decay,
                best_ckpt_path = checkpoint_path,
            )
            return metric
        except torch.cuda.OutOfMemoryError:
            logging.info("[ERROR] Out of memory!")
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

    return objective

def train():
    grid = {
        "lr": [1e-3, 3e-4, 1e-4],
    }
    objective = make_objective(AIS_FOLDER_PATH, mode = "train") # ["train", "tune"]
    run_worker(objective, grid)

if __name__ == "__main__":
    train()

"""
Example run:

CUDA_VISIBLE_DEVICES=0 MODEL_CHOICE=NEREUS OPTUNA_STORAGE="sqlite:///nereus.db" OPTUNA_STUDY="nereus_map_gat_path" OPTUNA_JSONL="nereus.jsonl" python -u src/train/train_tune.py

-   seclect model: MODEL_CHOICE = [GRU_RNN, DESIRE, TRAISFORMER, NEREUS]
    if NEREUS, select modules through study name:
        "cnn_gat" -> ScenePoolingCNN + GAT
        for full view of option see models/nereus/init.py
-   select grid for GridSampler, leave grid empty for TPESampler + PercentilePruner
-   mode:
        "tune", to run trials exploring the hyper parameter search space
        "train", full training with checkpoint saving
"""