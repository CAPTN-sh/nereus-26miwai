import logging
import os
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from data.graph.build_dataloader import AISDataModule  # noqa: F401  (re-exported for callers)
from train.pl_modules import _MODEL_REGISTRY, NereusModule  # noqa: F401  (re-exported for callers)
from train.utils.callbacks import make_tune_callbacks

HOUR = 3600


def train_single_gpu(
    cfg,
    model_choice: str,
    data_folder: Path,
    trial: optuna.Trial,
    lr: float,
    weight_decay: float = 1e-5,
    batch_size: int = 512,
    best_ckpt_path: Path | None = None,
):
    assert torch.cuda.is_available()
    device_id = 2
    device = torch.device(f"cuda:{device_id}")  # TODO: parameter for device
    torch.cuda.set_device(device)
    print(
        "CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"),
        " torch.cuda.current_device()=", torch.cuda.current_device(),
        " name=", torch.cuda.get_device_name(0),
        flush=True,
    )

    logging.info("#" * 20)
    logging.info(f"[Trial] number={trial.number}")
    trial_settings = {k: round(v, 6) if isinstance(v, float) else v for k, v in trial.params.items()}
    logging.info(f"[Trial {trial.number}] {trial_settings}")

    datamodule = AISDataModule(cfg=cfg, batch_size=batch_size, data_folder=data_folder)
    datamodule.setup()
    assert datamodule._train_loader is not None

    is_tuning = best_ckpt_path is None
    batches_per_eval = max(1, len(datamodule._train_loader) // 10)

    warmup_batches = batches_per_eval if is_tuning else batches_per_eval * 3
    max_epochs = 1 if is_tuning else 10
    max_seconds = HOUR if is_tuning else HOUR * 10

    loss_fn, module_cls = _MODEL_REGISTRY[model_choice]
    module = module_cls(
        cfg=cfg, loss_fn=loss_fn,
        lr=lr, weight_decay=weight_decay,
        warmup_batches=warmup_batches, batches_per_eval=batches_per_eval,
        is_tuning=is_tuning,
    )

    n_model_params = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    trial.set_user_attr("n_model_params", n_model_params)
    logging.info(f"Trainable parameters: {n_model_params}")
    if n_model_params > 5_000_000:
        logging.info(f"[TrialPruned] Parameter budget exceeded: {n_model_params:,}")
        raise optuna.exceptions.TrialPruned()

    callbacks = make_tune_callbacks(trial, max_seconds, checkpoint_dir=best_ckpt_path)

    study_name = os.environ.get("OPTUNA_STUDY", "default")
    tb_logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=model_choice.lower(),
        version=f"{study_name}_trial{trial.number}",
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[device_id],
        gradient_clip_val=1.0,
        val_check_interval=batches_per_eval,
        limit_val_batches=0.1,
        callbacks=callbacks,
        enable_progress_bar=True,
        logger=tb_logger,
    )

    trainer.fit(model=module, datamodule=datamodule)

    return module.best_metric
