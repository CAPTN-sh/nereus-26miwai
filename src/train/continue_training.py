"""Continue training from Lightning checkpoints until convergence.

Resumes full trainer state (optimizer, LR scheduler, epoch/step counters,
early-stopping state) from a ``best.ckpt`` and trains with no time budget:
EarlyStopping on ``val_metric`` is the only terminator (plus an optional
``--max-epochs`` safety cap).

Results go to a sibling experiment folder so the original checkpoint is never
overwritten, e.g. ``lightning_logs/nereus_ablation/version_3/best.ckpt``
continues into ``lightning_logs/nereus_ablation_continued/version_3/``.

Usage:
    python -u src/train/continue_training.py lightning_logs/nereus_ablation/version_0/best.ckpt
    python -u src/train/continue_training.py "lightning_logs/nereus_ablation/*/best.ckpt" --device 1
    python -u src/train/continue_training.py <ckpt...> --patience 15 --max-epochs 100
"""
import argparse
import dataclasses
import glob
import logging
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data.graph.build_dataloader import AISDataModule
from models.nereus.params import NEREUSParams
from train.pl_modules import (
    DESIREModule,
    GRUModule,
    ISSTGCNNModule,
    NereusModule,
    TrAISformerARModule,
    TrAISformerHeatmapModule,
)
from utils.config import AIS_FOLDER_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
torch.set_float32_matmul_precision('high')

_MODULE_CLASSES = {
    "NereusModule": NereusModule,
    "GRUModule": GRUModule,
    "TrAISformerHeatmapModule": TrAISformerHeatmapModule,
    "TrAISformerARModule": TrAISformerARModule,
    # legacy name, kept so checkpoints trained before the heatmap/AR split still load
    "TrAISformerModule": TrAISformerHeatmapModule,
    "DESIREModule": DESIREModule,
    "ISSTGCNNModule": ISSTGCNNModule,
}


def _datamodule_cfg(hparams: dict) -> NEREUSParams:
    """Rebuild the datamodule cfg from a checkpoint's flat ``cfg.*`` hparams.

    The datamodule always uses NEREUSParams (as in train_yaml._train_one),
    regardless of model type, so foreign fields are dropped.
    """
    valid = {f.name for f in dataclasses.fields(NEREUSParams)}
    fields = {
        k[len("cfg."):]: v for k, v in hparams.items()
        if k.startswith("cfg.") and k[len("cfg."):] in valid
    }
    # GRU checkpoints store max_dist=None on the module cfg, but the loader
    # still needs a real edge distance.
    if fields.get("max_dist") is None:
        fields["max_dist"] = NEREUSParams.max_dist
    return NEREUSParams(**fields)


def _continued_logger(ckpt_path: Path) -> TensorBoardLogger:
    """lightning_logs/<exp>/<version>/best.ckpt -> lightning_logs/<exp>_continued/<version>."""
    version_dir = ckpt_path.parent
    experiment = version_dir.parent.name
    return TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{experiment}_continued",
        version=version_dir.name,
    )


def _continue_one(ckpt_path: Path, device_id: int, patience: int | None, max_epochs: int, data_folder: Path):
    from train.utils.callbacks import make_yaml_callbacks

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt["hyper_parameters"])
    class_name = hparams.get("model_class") or hparams.get("model")
    module_cls = _MODULE_CLASSES.get(class_name)
    if module_cls is None:
        raise ValueError(f"Unknown model_class {class_name!r} in {ckpt_path}")

    batch_size = int(hparams.get("training.batch_size", 512))
    if patience is None:
        patience = int(hparams.get("training.early_stopping_patience", 10))

    logging.info(
        "Resuming %s from %s (epoch %s, global_step %s) | batch_size=%d patience=%d",
        class_name, ckpt_path, ckpt.get("epoch"), ckpt.get("global_step"), batch_size, patience,
    )

    torch.cuda.set_device(device_id)
    module = module_cls.load_from_checkpoint(str(ckpt_path), map_location=f"cuda:{device_id}")

    datamodule = AISDataModule(data_folder=data_folder, cfg=_datamodule_cfg(hparams), batch_size=batch_size)
    datamodule.setup()

    tb_logger = _continued_logger(ckpt_path)
    callbacks = make_yaml_callbacks(
        checkpoint_dir=Path(tb_logger.log_dir),
        early_stopping_patience=patience,
        max_seconds=None,
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

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=str(ckpt_path))
    logging.info("[%s] Best val_metric: %.6f", tb_logger.log_dir, module.best_metric)
    return module.best_metric


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "checkpoints", nargs="+",
        help="Lightning .ckpt paths (glob patterns allowed, e.g. 'lightning_logs/exp/*/best.ckpt')",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--patience", type=int, default=None,
        help="EarlyStopping patience (default: the checkpoint's training.early_stopping_patience)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=-1,
        help="Safety cap on total epochs, counted from epoch 0 of the original run (-1 = unlimited)",
    )
    parser.add_argument("--data-folder", type=Path, default=AIS_FOLDER_PATH)
    args = parser.parse_args()

    ckpt_paths: list[Path] = []
    for pattern in args.checkpoints:
        matches = sorted(glob.glob(pattern))
        if matches:
            ckpt_paths.extend(Path(m) for m in matches)
        else:
            ckpt_paths.append(Path(pattern))

    missing = [p for p in ckpt_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Checkpoints not found: {missing}")

    assert torch.cuda.is_available(), "GPU required"

    logging.info("Continuing %d checkpoint(s)", len(ckpt_paths))
    results: list[tuple[str, float]] = []
    for i, ckpt_path in enumerate(ckpt_paths):
        logging.info("[%d/%d] %s", i + 1, len(ckpt_paths), ckpt_path)
        metric = _continue_one(
            ckpt_path,
            device_id=args.device,
            patience=args.patience,
            max_epochs=args.max_epochs,
            data_folder=args.data_folder,
        )
        results.append((str(ckpt_path), float(metric)))

    if len(results) > 1:
        logging.info("--- Continuation results ---")
        for name, metric in sorted(results, key=lambda x: x[1]):
            logging.info("  %.6f  %s", metric, name)


if __name__ == "__main__":
    main()
