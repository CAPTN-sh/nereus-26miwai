"""YAML-configured training script with hyperparameter sweep support.

Usage:
    python -u src/train/train_yaml.py config/nereus_cnn_gat.yaml
    python -u src/train/train_yaml.py config/nereus_cnn_gat.yaml --skip 5
"""
import argparse
import copy
import itertools
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.loggers import TensorBoardLogger

from models.nereus.params import NEREUSParams
from data.graph.build_dataloader import AISDataModule
from train.pl_modules import (
    DESIREModule,
    GRUModule,
    ISSTGCNNModule,
    NereusModule,
    TrAISformerARModule,
    TrAISformerHeatmapModule,
)
from train.utils.callbacks import make_yaml_callbacks
from utils.config import AIS_FOLDER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
torch.set_float32_matmul_precision('high')

# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def _set_nested(d: dict, path: str, value):
    """Write ``value`` at dot-notation ``path`` inside ``d`` in-place."""
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _apply_overrides(base: dict, overrides: dict) -> dict:
    """Return a deep-copy of *base* with dot-notation *overrides* applied."""
    config = copy.deepcopy(base)
    for path, value in overrides.items():
        _set_nested(config, path, value)
    return config


def _expand_combinations(defaults: dict, combinations: list) -> list[tuple[dict, dict]]:
    """Expand a ``combinations`` list into ``(config, overrides)`` pairs.

    Each element of *combinations* is a dict mapping dot-notation param paths
    to *lists* of values.  All keys within a single element are cross-producted;
    results from every element are concatenated.

    Example::

        combinations:
          - nereus_modules.social: [gat, pool]
            nereus_modules.map: [cnn, null]   # 2×2 = 4 runs
          - training.lr: [1e-3, 3e-4]
            cfg.rnn_hidden_size: [128, 256]   # 2×2 = 4 runs  →  total 8
    """
    all_runs: list[tuple[dict, dict]] = []
    for group in combinations:
        keys = list(group.keys())
        value_lists = [group[k] for k in keys]
        for combo in itertools.product(*value_lists):
            overrides = dict(zip(keys, combo))
            config = _apply_overrides(defaults, overrides)
            all_runs.append((config, overrides))
    return all_runs


def _run_name(base_experiment: str, overrides: dict) -> str:
    if not overrides:
        return base_experiment
    parts = [base_experiment]
    for path, value in overrides.items():
        short_key = path.split(".")[-1]
        parts.append(f"{short_key}_{value}")
    return "_".join(str(p) for p in parts)


# ---------------------------------------------------------------------------
# Module / param builders
# ---------------------------------------------------------------------------

def _apply_cfg_overrides(cfg, overrides: dict):
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            logging.warning("Config key %r not found on %s, skipping.", key, type(cfg).__name__)


def _build_nereus_params(cfg_dict: dict) -> NEREUSParams:
    cfg = NEREUSParams()
    _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
    return cfg


def _build_data_cfg(cfg_dict: dict):
    """Build the params object used only to configure the datamodule.

    The datamodule reads ``pred_len``/``obs_len``/``max_dist``, all present on
    every model's params class. Using the class matching the model lets its
    ``cfg`` overrides apply without spurious "key not found" warnings.
    """
    model = cfg_dict["model"].upper()
    if model == "DESIRE":
        from models.desire.params import DESIREParams
        cfg = DESIREParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return cfg
    if model == "IS_STGCNN":
        from models.isstgcnn.params import ISSTGCNNParams
        cfg = ISSTGCNNParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return cfg
    if model == "TRAISFORMER_AR":
        from models.traisformer.params_ar import TraisformerARParams
        cfg = TraisformerARParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return cfg
    return _build_nereus_params(cfg_dict)


def _build_module(cfg_dict: dict, datamodule: AISDataModule) -> pl.LightningModule:
    model = cfg_dict["model"].upper()
    train = cfg_dict.get("training", {})

    batches_per_eval = max(1, len(datamodule._train_loader) // 10)
    warmup_batches = int(train.get("warmup_batches", batches_per_eval))

    common = dict(
        lr=float(train["lr"]),
        weight_decay=float(train.get("weight_decay", 1e-5)),
        warmup_batches=warmup_batches,
        batches_per_eval=batches_per_eval,
        is_tuning=False,
        extra_hparams={
            "training.batch_size": int(train.get("batch_size", 512)),
            "training.max_epochs": int(train.get("max_epochs", 10)),
            "training.early_stopping_patience": int(train.get("early_stopping_patience", 10)),
            "training.max_seconds": train.get("max_seconds"),
        },
    )

    if model == "NEREUS":
        from models.nereus.loss import mdn_loss
        cfg = _build_nereus_params(cfg_dict)
        modules = cfg_dict.get("nereus_modules", {})
        return NereusModule(
            cfg=cfg, loss_fn=mdn_loss,
            social=modules.get("social"),
            map_module=modules.get("map"),
            prior=modules.get("prior"),
            **common,
        )

    if model == "GRU_RNN":
        from models.gru.loss import mse_loss
        cfg = _build_nereus_params(cfg_dict)
        cfg.max_dist = None
        return GRUModule(cfg=cfg, loss_fn=mse_loss, **common)

    if model == "DESIRE":
        from models.desire.loss import loss_desire
        from models.desire.params import DESIREParams
        cfg = DESIREParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return DESIREModule(cfg=cfg, loss_fn=loss_desire, **common)

    if model == "IS_STGCNN":
        from models.isstgcnn.loss import loss_isstgcnn
        from models.isstgcnn.params import ISSTGCNNParams
        cfg = ISSTGCNNParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return ISSTGCNNModule(cfg=cfg, loss_fn=loss_isstgcnn, **common)

    if model == "TRAISFORMER":
        from models.traisformer.loss import loss_heatmap
        from models.traisformer.params import TraisformerParams
        cfg = TraisformerParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return TrAISformerHeatmapModule(cfg=cfg, loss_fn=loss_heatmap, **common)

    if model == "TRAISFORMER_AR":
        from models.traisformer.loss_ar import loss_traisformer_ar
        from models.traisformer.params_ar import TraisformerARParams
        cfg = TraisformerARParams()
        _apply_cfg_overrides(cfg, cfg_dict.get("cfg", {}))
        return TrAISformerARModule(cfg=cfg, loss_fn=loss_traisformer_ar, **common)

    raise ValueError(
        f"Unknown model: {model!r}. Choose NEREUS, GRU_RNN, DESIRE, TRAISFORMER, "
        f"TRAISFORMER_AR or IS_STGCNN."
    )


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def _train_one(cfg_dict: dict, run_name: str, experiment: str) -> float:
    train = cfg_dict.get("training", {})
    device_id = int(train.get("device_id", 0))
    batch_size = int(train.get("batch_size", 512))
    max_epochs = int(train.get("max_epochs", 10))
    data_folder = Path(cfg_dict.get("data_folder", AIS_FOLDER_PATH))

    assert torch.cuda.is_available(), "GPU required"
    torch.cuda.set_device(device_id)

    tmp_cfg = _build_data_cfg(cfg_dict)
    datamodule = AISDataModule(data_folder=data_folder, cfg=tmp_cfg, batch_size=batch_size)
    datamodule.setup()

    module = _build_module(cfg_dict, datamodule)

    n_params = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    logging.info("[%s] Trainable parameters: %s", run_name, f"{n_params:,}")

    tb_logger = TensorBoardLogger(save_dir="lightning_logs", name=experiment)

    max_seconds = train.get("max_seconds")
    callbacks = make_yaml_callbacks(
        checkpoint_dir=Path(tb_logger.log_dir),
        early_stopping_patience=int(train.get("early_stopping_patience", 10)),
        max_seconds=float(max_seconds) if max_seconds is not None else None,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[device_id],
        gradient_clip_val=1.0,
        val_check_interval=module.batches_per_eval,
        limit_val_batches=0.1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=tb_logger,
    )

    trainer.fit(model=module, datamodule=datamodule)
    logging.info("[%s] Best val_metric: %.6f", run_name, module.best_metric)
    return module.best_metric


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument(
        "--skip", type=int, default=0,
        help="Skip the first N experiments (useful for resuming an interrupted sweep)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f)

    # Support both flat config (single run) and defaults+combinations (sweep)
    if "defaults" in raw:
        defaults = raw["defaults"]
        combinations = raw.get("combinations", [])
    else:
        defaults = raw
        combinations = []

    if combinations:
        runs = _expand_combinations(defaults, combinations)
    else:
        runs = [(defaults, {})]

    base_experiment = defaults.get("experiment", "run")
    total = len(runs)
    logging.info("Total experiments: %d  (skipping first %d)", total, args.skip)

    results: list[tuple[str, float]] = []
    for i, (cfg_dict, overrides) in enumerate(runs):
        if i < args.skip:
            logging.info("[%d/%d] Skipping %s", i + 1, total, overrides)
            continue
        run_name = _run_name(base_experiment, overrides)
        logging.info("[%d/%d] Starting %s | overrides: %s", i + 1, total, run_name, overrides)

        metric = _train_one(cfg_dict, run_name, experiment=base_experiment)#, version=i)
        results.append((run_name, float(metric)))


    if len(results) > 1:
        logging.info("--- Sweep results ---")
        for name, metric in sorted(results, key=lambda x: x[1]):
            logging.info("  %.6f  %s", metric, name)


if __name__ == "__main__":
    main()
