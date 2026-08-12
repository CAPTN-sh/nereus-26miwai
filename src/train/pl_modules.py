import dataclasses
import logging
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
from tqdm import tqdm

from data.map.rasterize import Rasterizer
from data.map.scene_gernerator import SceneLoader
from eval.metrics.accumulator import MetricAccumulator
from eval.metrics.displacement import ade_per_agent, fde_per_agent, k_ade_per_agent, k_fde_per_agent
from models.desire.loss import loss_desire
from models.desire.model import DESIRE
from models.desire.params import DESIREParams
from models.gru.loss import mse_loss
from models.gru.model import GRU_RNN
from models.isstgcnn.loss import loss_isstgcnn
from models.isstgcnn.model import ISSTGCNN
from models.isstgcnn.params import ISSTGCNNParams
from models.nereus.init import build_nereus, init_nereus
from models.nereus.loss import mdn_loss
from models.nereus.params import NEREUSParams
from models.traisformer.loss import loss_heatmap
from models.traisformer.loss_ar import loss_traisformer_ar
from models.traisformer.model_ar import TrAISformerAR
from models.traisformer.model_heatmap import TrAISformerHeatmap
from models.traisformer.params import TraisformerParams
from models.traisformer.params_ar import TraisformerARParams
from utils.config import MAP_FOLDER_PATH, STEPS_PER_MINUTE, TRAIN_BBOX


def _unpack_mdn(mdn_out, num_modes):
    B, T, _ = mdn_out.shape
    mdn_out = mdn_out.view(B, T, num_modes, 5)
    pi = torch.softmax(mdn_out[..., 0], dim=-1)
    mu = mdn_out[..., 1:3]
    return pi, mu


def _restore_checkpoint_kwargs(kwargs: dict, cfg_cls) -> None:
    """Normalise a module's ``kwargs`` so both construction paths look identical.

    On ``load_from_checkpoint`` Lightning replays saved hparams flat — e.g.
    ``cfg.pred_len``, ``training.batch_size``, ``nereus_modules.social`` plus a
    ``model_class`` tag — none of which the ctors accept as-is. This rebuilds the
    structured kwargs the ctors expect (a no-op on normal construction, where
    ``cfg`` is already a live object and no flattened keys are present):

    * ``model``/``model_class`` tags are dropped.
    * ``cfg.<field>`` entries are rebuilt into a ``cfg_cls`` under ``cfg``.
    * ``training.*`` entries are run metadata only and are dropped.
    * ``nereus_modules.*`` entries carry architecture choices and are kept,
      merged into ``extra_hparams``.

    Mutates ``kwargs`` in place.
    """
    kwargs.pop("model", None)        # legacy key from older checkpoints
    kwargs.pop("model_class", None)

    cfg_fields = {k[len("cfg."):]: kwargs.pop(k) for k in list(kwargs) if k.startswith("cfg.")}
    if "cfg" not in kwargs and cfg_fields:
        valid = {f.name for f in dataclasses.fields(cfg_cls)}
        kwargs["cfg"] = cfg_cls(**{k: v for k, v in cfg_fields.items() if k in valid})

    for k in [k for k in kwargs if k.startswith("training.")]:
        kwargs.pop(k)

    module_hparams = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith("nereus_modules.")}
    if module_hparams:
        extra = kwargs.get("extra_hparams") or {}
        extra.update(module_hparams)
        kwargs["extra_hparams"] = extra


class _BaseModule(pl.LightningModule):
    def __init__(
        self,
        cfg,
        loss_fn,
        lr: float,
        weight_decay: float,
        warmup_batches: int,
        batches_per_eval: int,
        is_tuning: bool,
        extra_hparams: dict | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_batches = warmup_batches
        self.batches_per_eval = batches_per_eval
        self.is_tuning = is_tuning
        self.best_metric: float = float("inf")
        hparams = {
            "model_class": type(self).__name__,
            "lr": lr,
            "weight_decay": weight_decay,
            "warmup_batches": warmup_batches,
            "batches_per_eval": batches_per_eval,
            "is_tuning": is_tuning,
        }
        cfg_fields = dataclasses.fields(cfg) if dataclasses.is_dataclass(cfg) else []
        for f in cfg_fields:
            hparams[f"cfg.{f.name}"] = getattr(cfg, f.name)
        if extra_hparams:
            hparams.update(extra_hparams)
        self.save_hyperparameters(hparams)
        scene = torch.from_numpy(
            np.ascontiguousarray(SceneLoader(Rasterizer(TRAIN_BBOX)).load_scene(MAP_FOLDER_PATH))
        ).float()
        self.register_buffer("scene", scene)
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def forward(self, batch):
        return self.model(batch, self.scene)

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss, _ = self.loss_fn(output, batch, config=self.cfg)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        step = self.trainer.global_step
        if step < self.warmup_batches:
            scale = (step + 1) / self.warmup_batches
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr * scale

    def configure_optimizers(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.is_tuning:
            return optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=5, cooldown=2, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_metric", "interval": "epoch"},
        }

    def _run_series_loop(self, loader) -> dict[str, torch.Tensor]:
        """Return {metric_name: [T] tensor} of per-timestep series over loader."""
        return {}

    def on_train_end(self):
        import gc

        # AdamW keeps m+v buffers (~2× model size) in GPU memory.
        # Free them before running the full-dataset series loop.
        for opt in self.trainer.optimizers:
            opt.zero_grad(set_to_none=True)
            opt.state.clear()
        gc.collect()
        torch.cuda.empty_cache()

        dm = self.trainer.datamodule
        loaders = {"val": dm.val_dataloader(), "test": dm.test_dataloader()}
        self.model.eval()
        with torch.inference_mode():
            for split, loader in loaders.items():
                metrics = self._run_series_loop(loader)
                gc.collect()
                torch.cuda.empty_cache()
                for name, vals in metrics.items():
                    if vals.numel() == 1:
                        v = vals.item()
                        logging.info(f"[{split}] {name}: {v:.4f}")
                        self.logger.experiment.add_scalar(f"{name}/{split}", v)
                    else:
                        logging.info(f"[{split}] {name}: {','.join(f'{v:.4f}' for v in vals.tolist())}")
                        for t, v in enumerate(vals.tolist()):
                            self.logger.experiment.add_scalar(f"{name}/{split}", v, global_step=t)


class NereusModule(_BaseModule):
    def __init__(
        self,
        social: str | None = None,
        map_module: str | None = None,
        prior: str | None = None,
        **kwargs,
    ):
        _restore_checkpoint_kwargs(kwargs, NEREUSParams)  # rebuild structured kwargs on ckpt load
        kwargs.setdefault("loss_fn", mdn_loss)  # not saved as an hparam; fixed per module type
        extra = kwargs.pop("extra_hparams", None) or {}
        # On checkpoint reload the arch choice arrives flattened in extra, not as args.
        social = social if social is not None else extra.get("nereus_modules.social")
        map_module = map_module if map_module is not None else extra.get("nereus_modules.map")
        prior = prior if prior is not None else extra.get("nereus_modules.prior")
        # Store before super().__init__ because _build_model is called inside it
        self._social_choice = social
        self._map_choice = map_module
        self._prior_choice = prior
        extra.update({
            "nereus_modules.social": social,
            "nereus_modules.map": map_module,
            "nereus_modules.prior": prior,
        })
        super().__init__(extra_hparams=extra, **kwargs)
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}
        self._val_loss_sum = 0.0
        self._val_n = 0
        self._val_cpa_risk = 0.0
        self._val_cpa_dist = 0.0
        self._val_cpa_collisions = 0.0
        self._val_cpa_n = 0

    def _build_model(self):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if self._social_choice is not None or self._map_choice is not None or self._prior_choice is not None:
            return build_nereus(self.cfg, device, social=self._social_choice, map_mod=self._map_choice, prior=self._prior_choice)
        # Fallback: derive architecture from OPTUNA_STUDY env var (Optuna path)
        study_name = os.environ.get("OPTUNA_STUDY", "")
        return init_nereus(study_name, self.cfg, device)

    def on_validation_epoch_start(self):
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}
        self._val_loss_sum = 0.0
        self._val_n = 0

    def validation_step(self, batch, batch_idx):
        ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
        mdn_out = self(batch)
        loss_val, _ = mdn_loss(mdn_out, batch, self.cfg)
        self._val_loss_sum += loss_val.item()
        self._val_n += 1
        pi, mu = _unpack_mdn(mdn_out, self.cfg.mdn_modes)
        exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)
        pred_abs_pos = torch.cumsum(exp_rel, dim=1) * 100.0 + batch.x_pos[ego_idx, -1:, :]
        mu_k = mu.permute(0, 2, 1, 3)
        pred_abs_pos_k = torch.cumsum(mu_k, dim=2) * 100.0 + batch.x_pos[ego_idx, -1:, :].unsqueeze(1)
        self._val_metrics["ade"].update(ade_per_agent(pred_abs_pos, batch))
        self._val_metrics["fde_5"].update(fde_per_agent(pred_abs_pos, batch, 5 * STEPS_PER_MINUTE))
        self._val_metrics["k_ade"].update(k_ade_per_agent(pred_abs_pos_k, batch))
        self._val_metrics["k_fde_5"].update(k_fde_per_agent(pred_abs_pos_k, batch, 5 * STEPS_PER_MINUTE))

    def on_validation_epoch_end(self):
        if self._val_n == 0:
            return
        nll = self._val_loss_sum / self._val_n
        self.log("val_metric", nll, prog_bar=True)
        for name, acc in self._val_metrics.items():
            self.log(f"val_{name}", float(acc.compute()))
        if nll < self.best_metric:
            self.best_metric = float(nll)
        eval_step = self.trainer.global_step // max(1, self.batches_per_eval)
        log_str = f"[Eval] Epoch {eval_step} - nll: {nll:.4f}, " + ", ".join(
            f"{k}: {acc.compute():.4f}" for k, acc in self._val_metrics.items()
        )
        logging.info(log_str)

    def _run_series_loop(self, loader) -> dict[str, torch.Tensor]:
        T = self.cfg.pred_len
        nll_sum  = torch.zeros(T, device=self.device)
        de_sum   = torch.zeros(T, device=self.device)
        k_de_sum = torch.zeros(T, device=self.device)
        sq_sum   = torch.zeros(T, device=self.device)
        step_n   = torch.zeros(T, device=self.device)
        n = 0
        for i, data in enumerate(tqdm(loader, desc="series loop")):
            data = data.to(self.device)
            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
            mdn_out = self.model(data, self.scene)
            _, loss_dict = mdn_loss(mdn_out, data, self.cfg)
            nll_sum += loss_dict["mdn_nll_series"]
            n += 1
            pi, mu = _unpack_mdn(mdn_out, self.cfg.mdn_modes)
            exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)
            pred = torch.cumsum(exp_rel, dim=1) * 100.0 + data.x_pos[ego_idx, -1:, :]
            mask = data.y_mask.float()
            diff = pred - data.y_pos
            err  = torch.norm(diff, dim=-1)
            de_sum += (err * mask).sum(dim=0)
            sq_sum += (diff.pow(2).sum(dim=-1) * mask).sum(dim=0)
            mu_k   = mu.permute(0, 2, 1, 3)
            pred_k = torch.cumsum(mu_k, dim=2) * 100.0 + data.x_pos[ego_idx, -1:, :].unsqueeze(1)
            err_k  = torch.norm(pred_k - data.y_pos.unsqueeze(1), dim=-1)
            k_de_sum += (err_k.min(dim=1).values * mask).sum(dim=0)
            step_n += mask.sum(dim=0)
            del data, mdn_out, loss_dict, pi, mu, exp_rel, pred, mask, diff, err, mu_k, pred_k, err_k#vibe coded attempt to stop oom kill
            if i % 200 == 199:
                torch.cuda.empty_cache()

        n_clamp  = step_n.clamp(min=1)
        step_de   = (de_sum   / n_clamp).cpu()
        step_k_de = (k_de_sum / n_clamp).cpu()
        mse       = (sq_sum   / n_clamp).cpu()

        out = {
            "step_nll":   (nll_sum / max(1, n)).cpu(),
            "step_de":    step_de,
            "step_k_de":  step_k_de,
            "step_mse":   mse,
            "step_rmse":  mse.sqrt(),
            "ade":        step_de.mean().unsqueeze(0),
            "k_ade":      step_k_de.mean().unsqueeze(0),
        }
        for mins in (1, 3, 5):
            idx = mins * STEPS_PER_MINUTE - 1
            if idx < T:
                out[f"fde_{mins}min"]   = step_de[idx].unsqueeze(0)
                out[f"k_fde_{mins}min"] = step_k_de[idx].unsqueeze(0)
        return out


class GRUModule(_BaseModule):
    def __init__(self, **kwargs):
        _restore_checkpoint_kwargs(kwargs, NEREUSParams)
        kwargs.setdefault("loss_fn", mse_loss)  # not saved as an hparam; fixed per module type
        super().__init__(**kwargs)
        self._val_ade = MetricAccumulator()
        self._val_fde5 = MetricAccumulator()

    def _build_model(self):
        return GRU_RNN(self.cfg)

    def on_validation_epoch_start(self):
        self._val_ade = MetricAccumulator()
        self._val_fde5 = MetricAccumulator()

    def validation_step(self, batch, batch_idx):
        ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
        pred_rel = self.model.inference(batch, self.scene)
        pred_abs_pos = torch.cumsum(pred_rel, dim=1) * 100.0 + batch.x_pos[ego_idx, -1:, :]
        self._val_ade.update(ade_per_agent(pred_abs_pos, batch))
        self._val_fde5.update(fde_per_agent(pred_abs_pos, batch, 5 * STEPS_PER_MINUTE))

    def on_validation_epoch_end(self):
        ade = float(self._val_ade.compute())
        fde5 = float(self._val_fde5.compute())
        self.log("val_metric", ade, prog_bar=True)
        self.log("val_fde_5", fde5)
        if ade < self.best_metric:
            self.best_metric = ade
        eval_step = self.trainer.global_step // max(1, self.batches_per_eval)
        logging.info(f"[Eval] Epoch {eval_step} - ade: {ade:.4f}, fde_5: {fde5:.4f}")

    def _run_series_loop(self, loader) -> dict[str, torch.Tensor]:
        T = self.cfg.pred_len
        de_sum = torch.zeros(T, device=self.device)
        sq_sum = torch.zeros(T, device=self.device)
        step_n = torch.zeros(T, device=self.device)
        for i, data in enumerate(tqdm(loader, desc="series loop")):
            data = data.to(self.device)
            ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
            pred_rel = self.model.inference(data, self.scene)
            pred = torch.cumsum(pred_rel, dim=1) * 100.0 + data.x_pos[ego_idx, -1:, :]
            mask = data.y_mask.float()
            diff = pred - data.y_pos
            err  = torch.norm(diff, dim=-1)
            de_sum += (err * mask).sum(dim=0)
            sq_sum += (diff.pow(2).sum(dim=-1) * mask).sum(dim=0)
            step_n += mask.sum(dim=0)
            del data, pred_rel, pred, mask, diff, err
            if i % 200 == 199:
                torch.cuda.empty_cache()
        n_clamp  = step_n.clamp(min=1)
        step_de  = (de_sum / n_clamp).cpu()
        mse      = (sq_sum / n_clamp).cpu()
        out = {
            "step_de":   step_de,
            "step_mse":  mse,
            "step_rmse": mse.sqrt(),
            "ade":       step_de.mean().unsqueeze(0),
        }
        for mins in (1, 3, 5):
            idx = mins * STEPS_PER_MINUTE - 1
            if idx < T:
                out[f"fde_{mins}min"] = step_de[idx].unsqueeze(0)
        return out


class TrAISformerHeatmapModule(_BaseModule):
    """Single-shot heatmap variant: predicts path occupancy or a destination cell.

    Has no time axis, so it reports overlap/PMC or hit@k rather than ADE/FDE. For
    displacement metrics use :class:`TrAISformerARModule`.
    """

    def __init__(self, **kwargs):
        _restore_checkpoint_kwargs(kwargs, TraisformerParams)
        kwargs.setdefault("loss_fn", loss_heatmap)  # not saved as an hparam; fixed per module type
        super().__init__(**kwargs)
        self._val_ce_fine = 0.0
        self._val_secondary = 0.0
        self._val_extra: dict[str, float] = {}
        self._val_n = 0

    def _build_model(self):
        return TrAISformerHeatmap(self.cfg)

    def on_validation_epoch_start(self):
        self._val_ce_fine = 0.0
        self._val_secondary = 0.0
        self._val_extra = {}
        self._val_n = 0

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        _, loss_dict = self.loss_fn(output, batch, config=self.cfg)
        if self.cfg.pred_scope == "path":
            self._val_ce_fine += float(loss_dict["ce_fine"])
            self._val_secondary += float(loss_dict["overlap"])
            for k in ("pmc",):
                self._val_extra[k] = self._val_extra.get(k, 0.0) + float(loss_dict[k])
            self._val_n += 1
        else:
            B = int(batch.fin_pos_mask.sum().item())
            self._val_ce_fine += float(loss_dict["ce_fine"]) * B
            self._val_secondary += float(loss_dict["hit1"]) * B
            for k in ("hit5", "p_gt"):
                self._val_extra[k] = self._val_extra.get(k, 0.0) + float(loss_dict[k]) * B
            self._val_n += B

    def on_validation_epoch_end(self):
        if self._val_n == 0:
            return
        ce_fine = self._val_ce_fine / self._val_n
        secondary = self._val_secondary / self._val_n
        metric = -secondary  # lower is better for EarlyStopping mode="min"
        self.log("val_metric", metric, prog_bar=True)
        self.log("val_ce_fine", ce_fine)
        for k, v in self._val_extra.items():
            self.log(f"val_{k}", v / self._val_n)
        if metric < self.best_metric:
            self.best_metric = float(metric)
        eval_step = self.trainer.global_step // max(1, self.batches_per_eval)
        if self.cfg.pred_scope == "path":
            logging.info(f"[Eval HeatMap] Epoch {eval_step} - ce_fine: {ce_fine:.6f}, overlap: {secondary:.2%}")
        else:
            logging.info(f"[Eval HeatMap] Epoch {eval_step} - ce_fine: {ce_fine:.6f}, hit@1: {secondary:.2%}")


class DESIREModule(_BaseModule):
    def __init__(self, **kwargs):
        _restore_checkpoint_kwargs(kwargs, DESIREParams)
        kwargs.setdefault("loss_fn", loss_desire)  # not saved as an hparam; fixed per module type
        super().__init__(**kwargs)
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}

    def _build_model(self):
        return DESIRE(self.cfg)

    def on_validation_epoch_start(self):
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}

    def _predict_abs(self, batch):
        """Run DESIRE inference and lift relative displacements to absolute positions.

        Returns (pred_abs_pos [B, T, 2], pred_abs_pos_k [B, K, T, 2]) for the ego
        agents. Mirrors the conversion in eval/full_eval_nereus.py's _predict_abs.
        """
        ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
        best_rel, k_rel = self.model.inference(batch, self.scene)
        last_pos = batch.x_pos[ego_idx, -1:, :]
        pred_abs_pos = torch.cumsum(best_rel, dim=1) * 100.0 + last_pos
        pred_abs_pos_k = torch.cumsum(k_rel, dim=2) * 100.0 + last_pos.unsqueeze(1)
        return pred_abs_pos, pred_abs_pos_k

    def validation_step(self, batch, batch_idx):
        pred_abs_pos, pred_abs_pos_k = self._predict_abs(batch)
        self._val_metrics["ade"].update(ade_per_agent(pred_abs_pos, batch))
        self._val_metrics["fde_5"].update(fde_per_agent(pred_abs_pos, batch, 5 * STEPS_PER_MINUTE))
        self._val_metrics["k_ade"].update(k_ade_per_agent(pred_abs_pos_k, batch))
        self._val_metrics["k_fde_5"].update(k_fde_per_agent(pred_abs_pos_k, batch, 5 * STEPS_PER_MINUTE))

    def on_validation_epoch_end(self):
        ade = float(self._val_metrics["ade"].compute())
        self.log("val_metric", ade, prog_bar=True)
        for name, acc in self._val_metrics.items():
            self.log(f"val_{name}", float(acc.compute()))
        if ade < self.best_metric:
            self.best_metric = ade
        eval_step = self.trainer.global_step // max(1, self.batches_per_eval)
        log_str = f"[Eval] Epoch {eval_step} - " + ", ".join(
            f"{k}: {acc.compute():.4f}" for k, acc in self._val_metrics.items()
        )
        logging.info(log_str)

    def _run_series_loop(self, loader) -> dict[str, torch.Tensor]:
        T = self.cfg.pred_len
        de_sum   = torch.zeros(T, device=self.device)
        k_de_sum = torch.zeros(T, device=self.device)
        sq_sum   = torch.zeros(T, device=self.device)
        step_n   = torch.zeros(T, device=self.device)
        for i, data in enumerate(tqdm(loader, desc="series loop")):
            data = data.to(self.device)
            pred, pred_k = self._predict_abs(data)
            mask = data.y_mask.float()
            diff = pred - data.y_pos
            err  = torch.norm(diff, dim=-1)
            de_sum += (err * mask).sum(dim=0)
            sq_sum += (diff.pow(2).sum(dim=-1) * mask).sum(dim=0)
            err_k = torch.norm(pred_k - data.y_pos.unsqueeze(1), dim=-1)
            k_de_sum += (err_k.min(dim=1).values * mask).sum(dim=0)
            step_n += mask.sum(dim=0)
            del data, pred, pred_k, mask, diff, err, err_k
            if i % 200 == 199:
                torch.cuda.empty_cache()
        n_clamp   = step_n.clamp(min=1)
        step_de   = (de_sum   / n_clamp).cpu()
        step_k_de = (k_de_sum / n_clamp).cpu()
        mse       = (sq_sum   / n_clamp).cpu()
        out = {
            "step_de":   step_de,
            "step_k_de": step_k_de,
            "step_mse":  mse,
            "step_rmse": mse.sqrt(),
            "ade":       step_de.mean().unsqueeze(0),
            "k_ade":     step_k_de.mean().unsqueeze(0),
        }
        for mins in (1, 3, 5):
            idx = mins * STEPS_PER_MINUTE - 1
            if idx < T:
                out[f"fde_{mins}min"]   = step_de[idx].unsqueeze(0)
                out[f"k_fde_{mins}min"] = step_k_de[idx].unsqueeze(0)
        return out


class ISSTGCNNModule(_BaseModule):
    """IS-STGCNN / Social-STGCNN baseline.

    Training supervises every node in the scene (Social-STGCNN's convention, switchable
    via ``cfg.supervise``); validation and eval report ego-only metrics so the numbers
    stay comparable with the other baselines.
    """

    def __init__(self, **kwargs):
        _restore_checkpoint_kwargs(kwargs, ISSTGCNNParams)
        kwargs.setdefault("loss_fn", loss_isstgcnn)  # not saved as an hparam; fixed per module type
        super().__init__(**kwargs)
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}

    def _build_model(self):
        return ISSTGCNN(self.cfg)

    def on_validation_epoch_start(self):
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}

    def _predict_abs(self, batch):
        ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
        best_rel, k_rel = self.model.inference(batch, self.scene)
        last_pos = batch.x_pos[ego_idx, -1:, :]
        pred_abs_pos = torch.cumsum(best_rel, dim=1) * 100.0 + last_pos
        pred_abs_pos_k = torch.cumsum(k_rel, dim=2) * 100.0 + last_pos.unsqueeze(1)
        return pred_abs_pos, pred_abs_pos_k

    def validation_step(self, batch, batch_idx):
        pred_abs_pos, pred_abs_pos_k = self._predict_abs(batch)
        self._val_metrics["ade"].update(ade_per_agent(pred_abs_pos, batch))
        self._val_metrics["fde_5"].update(fde_per_agent(pred_abs_pos, batch, 5 * STEPS_PER_MINUTE))
        self._val_metrics["k_ade"].update(k_ade_per_agent(pred_abs_pos_k, batch))
        self._val_metrics["k_fde_5"].update(k_fde_per_agent(pred_abs_pos_k, batch, 5 * STEPS_PER_MINUTE))

    def on_validation_epoch_end(self):
        ade = float(self._val_metrics["ade"].compute())
        self.log("val_metric", ade, prog_bar=True)
        for name, acc in self._val_metrics.items():
            self.log(f"val_{name}", float(acc.compute()))
        if ade < self.best_metric:
            self.best_metric = ade
        eval_step = self.trainer.global_step // max(1, self.batches_per_eval)
        log_str = f"[Eval] Epoch {eval_step} - " + ", ".join(
            f"{k}: {acc.compute():.4f}" for k, acc in self._val_metrics.items()
        )
        logging.info(log_str)

    def _run_series_loop(self, loader) -> dict[str, torch.Tensor]:
        T = self.cfg.pred_len
        nll_sum  = torch.zeros(T, device=self.device)
        de_sum   = torch.zeros(T, device=self.device)
        k_de_sum = torch.zeros(T, device=self.device)
        sq_sum   = torch.zeros(T, device=self.device)
        step_n   = torch.zeros(T, device=self.device)
        n = 0
        for i, data in enumerate(tqdm(loader, desc="series loop")):
            data = data.to(self.device)
            _, loss_dict = loss_isstgcnn(self.model(data, self.scene), data, self.cfg)
            nll_sum += loss_dict["nll_series"]
            n += 1
            pred, pred_k = self._predict_abs(data)
            mask = data.y_mask.float()
            diff = pred - data.y_pos
            err  = torch.norm(diff, dim=-1)
            de_sum += (err * mask).sum(dim=0)
            sq_sum += (diff.pow(2).sum(dim=-1) * mask).sum(dim=0)
            err_k = torch.norm(pred_k - data.y_pos.unsqueeze(1), dim=-1)
            k_de_sum += (err_k.min(dim=1).values * mask).sum(dim=0)
            step_n += mask.sum(dim=0)
            del data, loss_dict, pred, pred_k, mask, diff, err, err_k
            if i % 200 == 199:
                torch.cuda.empty_cache()
        n_clamp   = step_n.clamp(min=1)
        step_de   = (de_sum   / n_clamp).cpu()
        step_k_de = (k_de_sum / n_clamp).cpu()
        mse       = (sq_sum   / n_clamp).cpu()
        out = {
            "step_nll":  (nll_sum / max(1, n)).cpu(),
            "step_de":   step_de,
            "step_k_de": step_k_de,
            "step_mse":  mse,
            "step_rmse": mse.sqrt(),
            "ade":       step_de.mean().unsqueeze(0),
            "k_ade":     step_k_de.mean().unsqueeze(0),
        }
        for mins in (1, 3, 5):
            idx = mins * STEPS_PER_MINUTE - 1
            if idx < T:
                out[f"fde_{mins}min"]   = step_de[idx].unsqueeze(0)
                out[f"k_fde_{mins}min"] = step_k_de[idx].unsqueeze(0)
        return out


class TrAISformerARModule(_BaseModule):
    """Autoregressive TrAISformer as published: next-token CE, rollout at eval.

    Trains on cross-entropy but selects on ego ADE, so ``val_metric`` is directly
    comparable with the other trajectory baselines. Note the rollout is the expensive
    part -- one forward pass per predicted step per hypothesis -- so validation is
    slower than the loss alone would suggest.
    """

    def __init__(self, **kwargs):
        _restore_checkpoint_kwargs(kwargs, TraisformerARParams)
        kwargs.setdefault("loss_fn", loss_traisformer_ar)  # not saved as an hparam
        super().__init__(**kwargs)
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}

    def _build_model(self):
        return TrAISformerAR(self.cfg)

    def on_validation_epoch_start(self):
        self._val_metrics = {k: MetricAccumulator() for k in ["ade", "fde_5", "k_ade", "k_fde_5"]}

    def _predict_abs(self, batch):
        ego_idx = batch.is_ego.nonzero(as_tuple=True)[0]
        best_rel, k_rel = self.model.inference(batch, self.scene)
        last_pos = batch.x_pos[ego_idx, -1:, :]
        pred_abs_pos = torch.cumsum(best_rel, dim=1) * 100.0 + last_pos
        pred_abs_pos_k = torch.cumsum(k_rel, dim=2) * 100.0 + last_pos.unsqueeze(1)
        return pred_abs_pos, pred_abs_pos_k

    def validation_step(self, batch, batch_idx):
        pred_abs_pos, pred_abs_pos_k = self._predict_abs(batch)
        self._val_metrics["ade"].update(ade_per_agent(pred_abs_pos, batch))
        self._val_metrics["fde_5"].update(fde_per_agent(pred_abs_pos, batch, 5 * STEPS_PER_MINUTE))
        self._val_metrics["k_ade"].update(k_ade_per_agent(pred_abs_pos_k, batch))
        self._val_metrics["k_fde_5"].update(k_fde_per_agent(pred_abs_pos_k, batch, 5 * STEPS_PER_MINUTE))

    def on_validation_epoch_end(self):
        ade = float(self._val_metrics["ade"].compute())
        self.log("val_metric", ade, prog_bar=True)
        for name, acc in self._val_metrics.items():
            self.log(f"val_{name}", float(acc.compute()))
        if ade < self.best_metric:
            self.best_metric = ade
        eval_step = self.trainer.global_step // max(1, self.batches_per_eval)
        log_str = f"[Eval] Epoch {eval_step} - " + ", ".join(
            f"{k}: {acc.compute():.4f}" for k, acc in self._val_metrics.items()
        )
        logging.info(log_str)

    def _run_series_loop(self, loader) -> dict[str, torch.Tensor]:
        T = self.cfg.pred_len
        ce_sum   = torch.zeros(T, device=self.device)
        de_sum   = torch.zeros(T, device=self.device)
        k_de_sum = torch.zeros(T, device=self.device)
        sq_sum   = torch.zeros(T, device=self.device)
        step_n   = torch.zeros(T, device=self.device)
        n = 0
        for i, data in enumerate(tqdm(loader, desc="series loop")):
            data = data.to(self.device)
            _, loss_dict = loss_traisformer_ar(self.model(data, self.scene), data, self.cfg)
            ce_sum += loss_dict["ce_series"]
            n += 1
            pred, pred_k = self._predict_abs(data)
            mask = data.y_mask.float()
            diff = pred - data.y_pos
            err  = torch.norm(diff, dim=-1)
            de_sum += (err * mask).sum(dim=0)
            sq_sum += (diff.pow(2).sum(dim=-1) * mask).sum(dim=0)
            err_k = torch.norm(pred_k - data.y_pos.unsqueeze(1), dim=-1)
            k_de_sum += (err_k.min(dim=1).values * mask).sum(dim=0)
            step_n += mask.sum(dim=0)
            del data, loss_dict, pred, pred_k, mask, diff, err, err_k
            if i % 200 == 199:
                torch.cuda.empty_cache()
        n_clamp   = step_n.clamp(min=1)
        step_de   = (de_sum   / n_clamp).cpu()
        step_k_de = (k_de_sum / n_clamp).cpu()
        mse       = (sq_sum   / n_clamp).cpu()
        out = {
            "step_ce":   (ce_sum / max(1, n)).cpu(),
            "step_de":   step_de,
            "step_k_de": step_k_de,
            "step_mse":  mse,
            "step_rmse": mse.sqrt(),
            "ade":       step_de.mean().unsqueeze(0),
            "k_ade":     step_k_de.mean().unsqueeze(0),
        }
        for mins in (1, 3, 5):
            idx = mins * STEPS_PER_MINUTE - 1
            if idx < T:
                out[f"fde_{mins}min"]   = step_de[idx].unsqueeze(0)
                out[f"k_fde_{mins}min"] = step_k_de[idx].unsqueeze(0)
        return out


_MODEL_REGISTRY: dict[str, tuple] = {
    "GRU_RNN":        (mse_loss,             GRUModule),
    "NEREUS":         (mdn_loss,             NereusModule),
    "TRAISFORMER":    (loss_heatmap,         TrAISformerHeatmapModule),
    "TRAISFORMER_AR": (loss_traisformer_ar,  TrAISformerARModule),
    "DESIRE":         (loss_desire,          DESIREModule),
    "IS_STGCNN":      (loss_isstgcnn,        ISSTGCNNModule),
}
