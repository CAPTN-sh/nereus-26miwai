import logging
import time

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

__all__ = [
    "TimeBudgetCallback",
    "OptunaPruningCallback",
    "EarlyStopping",
    "ModelCheckpoint",
    "make_yaml_callbacks",
    "make_tune_callbacks",
]


class TimeBudgetCallback(pl.Callback):
    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self._start = 0.0

    def on_train_start(self, trainer, pl_module):
        self._start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if time.perf_counter() - self._start > self.max_seconds:
            trainer.should_stop = True


class OptunaPruningCallback(pl.Callback):
    """Reports val_metric to Optuna and raises TrialPruned when pruner fires."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get("val_metric")
        if metric is None:
            return
        step = trainer.global_step
        self.trial.report(float(metric), step=step)
        self.trial.set_user_attr("epochs_ran", step)
        if self.trial.should_prune():
            logging.info("[TrialPruned] PercentilePruner")
            raise optuna.TrialPruned()


def make_yaml_callbacks(
    checkpoint_dir,
    early_stopping_patience: int = 10,
    max_seconds: float | None = None,
) -> list[pl.Callback]:
    """Build the standard callback list for a YAML-driven training run."""
    callbacks: list[pl.Callback] = [
        EarlyStopping(
            monitor="val_metric",
            patience=early_stopping_patience,
            min_delta=1e-4,
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val_metric",
            save_top_k=1,
            mode="min",
            save_weights_only=False,
        ),
    ]
    if max_seconds is not None:
        callbacks.append(TimeBudgetCallback(max_seconds))
    return callbacks


def make_tune_callbacks(
    trial: optuna.Trial,
    max_seconds: float,
    checkpoint_dir=None,
) -> list[pl.Callback]:
    """Build the callback list for an Optuna tuning trial."""
    callbacks: list[pl.Callback] = [
        TimeBudgetCallback(max_seconds),
        OptunaPruningCallback(trial),
        EarlyStopping(monitor="val_metric", patience=15, min_delta=1e-4, mode="min"),
    ]
    if checkpoint_dir is not None:
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir.parent),
                filename=checkpoint_dir.stem,
                monitor="val_metric",
                save_top_k=1,
                mode="min",
                save_weights_only=False,
            )
        )
    return callbacks
