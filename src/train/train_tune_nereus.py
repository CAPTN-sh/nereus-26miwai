import logging
import os
from pathlib import Path
import numpy as np
import torch
import optuna

from models.nereus.social import GAT, EgoSocialPooling
from models.nereus.model import NEREUS
from models.nereus.intent import DensityIntent
from models.nereus.loss import mdn_loss, eval_mdn
from models.nereus.params import NEREUSParams
from utils.logger import logger, trial_jsonl_callback
from models.utils.maps.scene_gernerator import SceneLoader
from models.utils.maps.rasterize import Rasterizer
from models.gmm.model import AIS_GMM, MAP_GMM

from models.traisformer.model import TrAISformer

from utils.config import DATA_FOLDER_PATH
from train.training_loop import train_single_gpu

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def make_objective(data_folder: Path):

    def objective(trial: optuna.Trial):
        model_cls, cfg, loss_fn, eval_fn = NEREUS, NEREUSParams(), mdn_loss, eval_mdn

        # general params
        lr = trial.suggest_categorical("lr", [1e-3])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-5])

        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        
        path = DATA_FOLDER_PATH / "maps/2_standardized/fh_10/kiel/"
        sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))

        density_contiguous = np.ascontiguousarray(sl.load_density(path))
        density_maps = torch.from_numpy(density_contiguous).to(device).to(torch.float32)

        """
        best_ckpt_path = Path("checkpoints/traisformer/traisformer_path_best.pt")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        prior_module = TrAISformer(ckpt["config"])
        prior_module.load_state_dict(ckpt["model_state_dict"])
        prior_module.eval()
        prior_module.requires_grad_(False)
        

        best_ckpt_path = Path(f"data/gmm/cluster_{n_cluster}/ais_gmm.pt")
        ckpt = torch.load(best_ckpt_path, map_location=device)

        trais = TrAISformer(ckpt["prior_config"]).to(device)
        trais.load_state_dict(ckpt["prior_state_dict"])
        trais.eval()
        trais.requires_grad_(False)

        ais_gmm = AIS_GMM(trais, n_clusters=ckpt["n_clusters"])
        ais_gmm.gmm = ckpt["gmm"]

        path = Path("data/gmm")
        sl = SceneLoader(Rasterizer([10.12, 54.31, 10.33, 54.46]))

        cluster_contiguous = np.ascontiguousarray(sl.load_cluster(path, n_cluster))
        cluster_maps = torch.from_numpy(cluster_contiguous).to(device).to(torch.float32)
        prior_module = MAP_GMM(ais_gmm, cluster_maps)

        """

        model = model_cls(
            config = cfg,
            static_module = True,
            social_module = GAT(cfg), # GAT(cfg), #EgoSocialPooling(cfg), #GAT(cfg), EgoSocialPooling
            map_module = True, # True
            prior_module = DensityIntent(density_maps), # DensityIntent(density_maps), # DensityIntent(density_maps), # prior_module, # DensityIntent(density_maps)
            map_atte_module = False,
        )

        try:
            metric = train_single_gpu(
                model=model,
                cfg=cfg,
                loss_fn=loss_fn,
                eval_fn=eval_fn,
                data_folder=data_folder,
                trial=trial,
                weight_decay=weight_decay,
                lr=lr,
                best_ckpt_path = Path("checkpoints/nereus/nereus_all_best.pt"),
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
        "lr": [1e-3],
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

    study.optimize(objective, n_trials=3, gc_after_trial=True, callbacks=[cb])

    if study.best_trial is not None:
        logging.info(f"BEST value={study.best_value}")
        logging.info(f"BEST params={study.best_params}")

if __name__ == "__main__":
    run_worker()

"""
[Experiment 1] Best observation length for short and long term

# encoder decoder
CUDA_VISIBLE_DEVICES=1 MODEL_CHOICE=NEREUS OPTUNA_STORAGE="sqlite:///nereus_all_full.db" OPTUNA_STUDY="nereus_all_full" OPTUNA_JSONL="nereus_all_full.jsonl" python -u src/train/train_tune_nereus.py

CUDA_VISIBLE_DEVICES=3 MODEL_CHOICE=NEREUS OPTUNA_STORAGE="sqlite:///nereus_map_res.db" OPTUNA_STUDY="nereus_map_res" OPTUNA_JSONL="nereus_map_res.jsonl" python -u src/train/train_tune_nereus.py

"""