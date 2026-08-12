import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from data.graph.build_dataloader import graph_loader
from data.map.rasterize import Rasterizer
from data.map.scene_gernerator import SceneLoader
from eval.metrics.accumulator import MetricAccumulator
from eval.metrics.cpa import compute_batch_cpa_stats
from eval.metrics.displacement import ade_per_agent, de_series_sums, k_ade_per_agent
from models.nereus.init import init_nereus
from utils.config import AIS_SOURCE, DATA_FOLDER_PATH, STEPS_PER_MINUTE
from utils.logger import logger

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Scale factor to convert normalized relative displacement back to meters
DE_NORMALIZE = 100

# Quantile levels reported for the per-graph critical-encounter metrics (min hull
# distance over the horizon, DCPA, TCPA) -- covers both tails so rare close/imminent
# encounters show up, not just the mean.
CPA_QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def _quantile_columns(name: str, values: torch.Tensor) -> dict:
    """{f"{name}_p01": ..., f"{name}_p05": ..., ...} from a 1-D tensor, ignoring inf/nan
    (graphs with no valid critical-neighbor pair, e.g. all-ego graphs)."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {f"{name}_p{int(q * 100):02d}": float("nan") for q in CPA_QUANTILES}
    qs = torch.quantile(finite.double(), torch.tensor(CPA_QUANTILES, dtype=torch.float64))
    return {f"{name}_p{int(q * 100):02d}": v.item() for q, v in zip(CPA_QUANTILES, qs)}

def _append_group_csv(csv_path, region, ship_group, de_series, k_de_series, count, ade, k_ade, collision, eval_time):
    """Append one (region, ship_group) block of per-timestep rows to the model CSV.

    Written incrementally after every ship group so partial progress survives a
    crash. Per-step columns are ``fde``/``k_fde`` (the DE-at-t series); group-level
    scalars (``ade``, collision stats, ...) are repeated across the block's rows so
    the file stays a single tidy table per model.
    """
    T = de_series.numel()
    steps = np.arange(1, T + 1)
    df = pd.DataFrame({
        "region": region,
        "ship_group": ship_group,
        "step": steps,
        "minute": steps / STEPS_PER_MINUTE,
        "fde": de_series.cpu().numpy(),
        "k_fde": k_de_series.cpu().numpy(),
        "n_step": count.cpu().numpy(),
        "ade": ade,
        "k_ade": k_ade,
        "eval_time_min": eval_time / 60,
        # collision/CPA dict already contains every scalar + quantile column; each
        # value is repeated across the block's rows like ade/k_ade above.
        **collision,
    })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def _predict_abs(model, model_kind, data, scene, B, T):
    """Return (pred_abs_pos [B, T, 2], pred_abs_pos_k [B, K, T, 2]) for the ego agents.

    Dispatches on ``model_kind``:

    * ``nereus`` decodes the MDN mixture to an expected trajectory plus per-mode
      trajectories (K = number of mixture modes).
    * ``desire`` takes its best hypothesis plus the K refined samples; ``isstgcnn``
      (Gaussian mean plus K draws) and ``traisformer_ar`` (greedy rollout plus K
      sampled rollouts) match that contract.
    * ``gru`` is deterministic: a single trajectory reused as the sole (K=1)
      sample, so the k-metrics collapse to the point metrics.

    All are lifted from relative displacements to absolute positions the same way
    (cumsum, denormalise, add last observed position).
    """
    ego_idx = data.is_ego.nonzero(as_tuple=True)[0]
    last_pos = data.x_pos[ego_idx, -1:, :]
    if model_kind in ("desire", "isstgcnn", "traisformer_ar"):
        best_rel, k_rel = model.inference(data, scene)
        pred_abs_pos = torch.cumsum(best_rel, dim=1) * DE_NORMALIZE + last_pos
        pred_abs_pos_k = torch.cumsum(k_rel, dim=2) * DE_NORMALIZE + last_pos.unsqueeze(1)
        return pred_abs_pos, pred_abs_pos_k

    if model_kind == "gru":
        pred_rel = model.inference(data, scene)
        pred_abs_pos = torch.cumsum(pred_rel, dim=1) * DE_NORMALIZE + last_pos
        return pred_abs_pos, pred_abs_pos.unsqueeze(1)

    mdn_out = model(data, scene).view(B, T, 3, 5)
    pi = torch.softmax(mdn_out[..., 0], dim=-1)
    mu = mdn_out[..., 1:3]
    exp_rel = torch.sum(pi.unsqueeze(-1) * mu, dim=2)  # [B, T, 2]
    pred_abs_pos = torch.cumsum(exp_rel, dim=1) * DE_NORMALIZE + last_pos
    mu_k = mu.permute(0, 2, 1, 3)
    pred_abs_pos_k = torch.cumsum(mu_k, dim=2) * DE_NORMALIZE + last_pos.unsqueeze(1)
    return pred_abs_pos, pred_abs_pos_k


def full_eval(model, region, bbox, device, csv_path: Path, source: str = AIS_SOURCE, model_kind: str = "nereus"):
    """Full extensive eval script per ship_group.

    Args:
        model:      trajectory model (rasterizer already swapped for this region).
        region:     Region name, e.g. "kiel", "aarhus".
        bbox:       Geographic bounding box [lon_min, lat_min, lon_max, lat_max].
        device:     torch.device to run inference on.
        csv_path:   Per-model CSV; results are appended after each ship group.
        source:     AIS source tag ("fh" or "dma") — controls data folder and filename prefix.
        model_kind: "nereus" (MDN decode) or "desire" (best + K-sample inference).
    """
    for ship_group in ["all", "sailing", "cargo", "passenger", "other"]:
        logging.info("#"*20)
        logging.info(f"[SHIP GROUP]: {ship_group}")

        data_folder = DATA_FOLDER_PATH / f"ais/4_features/{source}_10/{region}"
        map_folder = DATA_FOLDER_PATH / f"maps/2_standardized/{source}_10/{region}"

        B, T = 512, 5 * STEPS_PER_MINUTE

        test_loader, _ = graph_loader(
            data_folder=data_folder,
            flag="test",
            min_date=pd.Timestamp("2022-01-01"),
            max_date=pd.Timestamp("2024-01-01"),
            batch_size=B,
            pin_memory=True,
            pred_len=T,
            obs_len=60,
            max_edge_dist=500,
            shuffle=True,
            ship_group = ship_group,
        )

        sl = SceneLoader(Rasterizer(bbox))

        scene_contiguous = np.ascontiguousarray(sl.load_scene(map_folder))
        scene = torch.from_numpy(scene_contiguous).to(device, torch.float32)

        model = model.to(device)
        model.eval()

        with torch.inference_mode():
            # Per-graph CPA stats, kept on CPU and concatenated after the loop so we can
            # report quantiles (not just the mean) over the whole eval run.
            cpa_min_dist_chunks = []
            cpa_dcpa_chunks = []
            cpa_tcpa_chunks = []
            cpa_risk_chunks = []
            cpa_collision_chunks = []

            # Per-timestep DE series accumulators (for the FDE-at-t columns).
            de_sum = torch.zeros(T, device=device)
            k_de_sum = torch.zeros(T, device=device)
            step_count = torch.zeros(T, device=device)

            # Scalar ADE via the original per-agent definition (mean over agents
            # of each agent's mean DE) — not the same as de_series.mean().
            ade_acc = MetricAccumulator()
            k_ade_acc = MetricAccumulator()

            # Measure pure model inference time (excluding dataloading)
            eval_time = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for data in tqdm(test_loader, desc="Eval"):
                data = data.to(device, non_blocking=True)

                start_event.record()
                pred_abs_pos, pred_abs_pos_k = _predict_abs(model, model_kind, data, scene, B, T)
                end_event.record()
                torch.cuda.synchronize()
                step_time_ms = start_event.elapsed_time(end_event)
                eval_time += (step_time_ms / 1000.0)

                batch_de, batch_k_de, batch_count = de_series_sums(pred_abs_pos, pred_abs_pos_k, data)
                de_sum += batch_de
                k_de_sum += batch_k_de
                step_count += batch_count

                ade_acc.update(ade_per_agent(pred_abs_pos, data))
                k_ade_acc.update(k_ade_per_agent(pred_abs_pos_k, data))

                cpa_stats = compute_batch_cpa_stats(data, pred_abs_pos)
                cpa_min_dist_chunks.append(cpa_stats["min_dist"].cpu())
                cpa_dcpa_chunks.append(cpa_stats["dcpa"].cpu())
                cpa_tcpa_chunks.append(cpa_stats["tcpa"].cpu())
                cpa_risk_chunks.append(cpa_stats["risk"].cpu())
                cpa_collision_chunks.append(cpa_stats["collision"].cpu())

            de_series = (de_sum / step_count.clamp_min(1)).cpu()
            k_de_series = (k_de_sum / step_count.clamp_min(1)).cpu()
            ade = ade_acc.compute()
            k_ade = k_ade_acc.compute()

            cpa_min_dist = torch.cat(cpa_min_dist_chunks) if cpa_min_dist_chunks else torch.empty(0)
            cpa_dcpa = torch.cat(cpa_dcpa_chunks) if cpa_dcpa_chunks else torch.empty(0)
            cpa_tcpa = torch.cat(cpa_tcpa_chunks) if cpa_tcpa_chunks else torch.empty(0)
            cpa_risk = torch.cat(cpa_risk_chunks) if cpa_risk_chunks else torch.empty(0)
            cpa_collision = torch.cat(cpa_collision_chunks) if cpa_collision_chunks else torch.empty(0, dtype=torch.bool)
            total_graphs = cpa_min_dist.numel()
            graphs = max(total_graphs, 1)

            collision = {
                "pred_risk": float(cpa_risk.sum() / graphs) if total_graphs else 0.0,
                "min_pred_dist": float(cpa_min_dist[torch.isfinite(cpa_min_dist)].mean()) if total_graphs else 0.0,
                "dcpa_mean": float(cpa_dcpa[torch.isfinite(cpa_dcpa)].mean()) if total_graphs else 0.0,
                "tcpa_mean_s": float(cpa_tcpa[torch.isfinite(cpa_tcpa)].mean()) if total_graphs else 0.0,
                "collision_ratio": float(cpa_collision.float().mean()) * 100 if total_graphs else 0.0,
                "n_graphs": total_graphs,
                **_quantile_columns("min_pred_dist", cpa_min_dist),
                **_quantile_columns("dcpa", cpa_dcpa),
                **_quantile_columns("tcpa_s", cpa_tcpa),
            }

            logging.info(f"Total Graphs: {total_graphs}")
            logging.info(f"Mean Pred Risk: {collision['pred_risk']}")
            logging.info(f"Mean Min Pred Distance: {collision['min_pred_dist']}")
            logging.info(f"Mean DCPA: {collision['dcpa_mean']}  Mean TCPA: {collision['tcpa_mean_s']}s")
            logging.info(f"Collision Ratio: {collision['collision_ratio']}")
            logging.info(
                "Min-dist quantiles (m): "
                + ", ".join(f"p{int(q*100):02d}={collision[f'min_pred_dist_p{int(q*100):02d}']:.1f}" for q in CPA_QUANTILES)
            )
            logging.info(f"ade: {ade}  k_ade: {k_ade}")
            for mins in (1, 3, 5):
                idx = mins * STEPS_PER_MINUTE - 1
                if idx < T:
                    logging.info(f"fde_{mins}: {de_series[idx].item()}  k_fde_{mins}: {k_de_series[idx].item()}")
            logging.info(f"eval_time {eval_time / 60:.2f} minutes / {len(test_loader)}")

            _append_group_csv(
                csv_path, region, ship_group, de_series, k_de_series,
                step_count.cpu(), ade, k_ade, collision, eval_time,
            )

def swap_rasterizer(model, bbox):
    """Replace the rasterizer on the model and all sub-modules that hold one.

    Reuses the model's own pos_res when it already has a rasterizer, instead of
    silently falling back to Rasterizer's default. Several models are built at a
    non-default resolution (DESIRE's SCF, TrAISformerAR's position-cell embeddings)
    -- dropping that on a region swap desyncs the grid size from what the model's
    embeddings/lookup tables were actually sized for.
    """
    pos_res = model.rasterizer.pos_res if hasattr(model, "rasterizer") else None
    kwargs = {"pos_res": pos_res} if pos_res is not None else {}

    # DESIRE keeps its rasterizer on the model and its SCF sub-module.
    if hasattr(model, "IOC") and hasattr(model.IOC, "scf"):
        rasterizer = Rasterizer(bbox, **kwargs)
        model.rasterizer = rasterizer
        model.IOC.scf.rasterizer = rasterizer
        return rasterizer

    rasterizer = Rasterizer(bbox, **kwargs)
    model.rasterizer = rasterizer
    if hasattr(model, "map_cnn") and model.map_cnn is not None:
        model.map_cnn.rasterizer = rasterizer
    if hasattr(model, "prior_cnn") and model.prior_cnn is not None:
        model.prior_cnn.rasterizer = rasterizer
    return rasterizer


def _model_csv_path(ckpt_path: Path, source: str, suffix: str = "") -> Path:
    """Per-model CSV in the checkpoint's parent folder, wiped so each run starts fresh.

    ``suffix`` (e.g. "_stochastic") keeps a second eval of the same checkpoint --
    under different ``model_overrides`` -- from clobbering the first one's CSV.
    """
    csv_path = ckpt_path.parent / f"eval_{source}{suffix}.csv"
    csv_path.unlink(missing_ok=True)
    logging.info("Writing results to %s", csv_path)
    return csv_path


def eval_model(best_ckpt_path, device, regions, source: str = AIS_SOURCE):
    ckpt = torch.load(best_ckpt_path, map_location=device)
    cfg = ckpt["config"]
    csv_path = _model_csv_path(best_ckpt_path, source)

    for region, bbox in regions.items():
        logging.info(region)
        rasterizer = Rasterizer(bbox)
        model = init_nereus(best_ckpt_path.name, cfg, device, rasterizer=rasterizer)
        model.load_state_dict(ckpt["model_state_dict"])
        swap_rasterizer(model, bbox)
        model.eval()
        full_eval(model, region, bbox, device, csv_path, source=source)


def eval_lightning_ckpt(
    ckpt_path: Path, device, regions, source: str = AIS_SOURCE,
    csv_suffix: str = "", model_overrides: dict | None = None,
):
    """Load a Lightning .ckpt and evaluate across multiple regions.

    ``model_overrides`` sets attributes directly on ``model.cfg`` after loading (e.g.
    ``{"greedy_best": False}`` for TrAISformerAR) -- eval-time decoding knobs only,
    never learned weights, so this is safe to flip without retraining. Combine with
    ``csv_suffix`` so the alternate run's CSV doesn't overwrite the default one.
    """
    from train.pl_modules import (
        DESIREModule,
        GRUModule,
        ISSTGCNNModule,
        NereusModule,
        TrAISformerARModule,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    class_name = ckpt["hyper_parameters"].get("model_class") or ckpt["hyper_parameters"].get("model")
    module_cls, model_kind = {
        "DESIREModule": (DESIREModule, "desire"),
        "GRUModule": (GRUModule, "gru"),
        "ISSTGCNNModule": (ISSTGCNNModule, "isstgcnn"),
        "TrAISformerARModule": (TrAISformerARModule, "traisformer_ar"),
    }.get(class_name, (NereusModule, "nereus"))

    pl_module = module_cls.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = pl_module.model.to(device)
    model.eval()
    for key, value in (model_overrides or {}).items():
        setattr(model.cfg, key, value)
        logging.info("model_overrides: cfg.%s = %r", key, value)
    csv_path = _model_csv_path(ckpt_path, source, suffix=csv_suffix)
    for region, bbox in regions.items():
        logging.info(region)
        swap_rasterizer(model, bbox)
        full_eval(model, region, bbox, device, csv_path, source=source, model_kind=model_kind)


def eval_from_yaml(config_path: Path):
    """Evaluate multiple checkpoints from a YAML config, sharing dataset cache across runs.

    YAML structure::

        source: fh          # "fh" | "dma"
        device: 0           # GPU index
        regions:
          kiel: [10.12, 54.31, 10.33, 54.46]
          # aarhus: [10.21, 56.04, 10.47, 56.17]

        # One or both of:
        checkpoints:
          - lightning_logs/nereus_ablation/version_0/checkpoints/best.ckpt
          - lightning_logs/nereus_ablation/version_1/checkpoints/best.ckpt
        checkpoint_glob: "lightning_logs/nereus_ablation/*/checkpoints/best.ckpt"

        # Optional -- Lightning .ckpt only. Sets attributes on model.cfg after loading
        # (eval-time decoding knobs, not learned weights) and suffixes the CSV so this
        # doesn't overwrite a default-settings eval of the same checkpoint:
        csv_suffix: "_stochastic"
        model_overrides:
          greedy_best: false
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    source = cfg.get("source", AIS_SOURCE)
    device_id = int(cfg.get("device", 0))
    regions = cfg.get("regions", {"kiel": [10.12, 54.31, 10.33, 54.46]})
    csv_suffix = cfg.get("csv_suffix", "")
    model_overrides = cfg.get("model_overrides")

    assert torch.cuda.is_available(), "GPU required"
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    ckpt_paths: list[Path] = []
    for p in cfg.get("checkpoints", []):
        ckpt_paths.append(Path(p))
    if "checkpoint_glob" in cfg:
        ckpt_paths.extend(Path(p) for p in sorted(glob.glob(cfg["checkpoint_glob"])))

    if not ckpt_paths:
        raise ValueError("No checkpoints found. Specify 'checkpoints' or 'checkpoint_glob' in the YAML.")

    logger(file_prefix=f"eval_{source}_{config_path.stem}")
    logging.info("Evaluating %d checkpoints | regions: %s", len(ckpt_paths), list(regions.keys()))

    for ckpt_path in ckpt_paths:
        logging.info("=" * 60)
        logging.info("Checkpoint: %s", ckpt_path)
        if ckpt_path.suffix == ".ckpt":
            eval_lightning_ckpt(
                ckpt_path, device, regions, source=source,
                csv_suffix=csv_suffix, model_overrides=model_overrides,
            )
        else:
            eval_model(ckpt_path, device, regions, source=source)


if __name__ == "__main__":
    import argparse

    ALL_REGIONS = {
        "kiel":         [10.12, 54.31, 10.33, 54.46],
        "aarhus":       [10.21, 56.04, 10.47, 56.17],
        "odense":       [10.42, 55.42, 10.68, 55.55],
        "little_belt":  [9.64,  55.25,  9.90, 55.37],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input", type=Path,
        help=".yaml eval config (multiple checkpoints) OR a single .ckpt/.pt checkpoint",
    )
    # Single-checkpoint flags (ignored when input is a YAML file):
    parser.add_argument("--source", default=AIS_SOURCE, choices=["fh", "dma"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--regions", nargs="+",
        choices=list(ALL_REGIONS), default=["kiel"],
    )
    args = parser.parse_args()

    if args.input.suffix in (".yaml", ".yml"):
        eval_from_yaml(args.input)
    else:
        # Single-checkpoint mode
        regions = {r: ALL_REGIONS[r] for r in args.regions}
        assert torch.cuda.is_available()
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
        logger(file_prefix=f"eval_{args.source}_{args.input.stem}")
        logging.info("checkpoint: %s  source: %s  regions: %s", args.input, args.source, args.regions)
        if args.input.suffix == ".ckpt":
            eval_lightning_ckpt(args.input, device, regions, source=args.source)
        else:
            eval_model(args.input, device, regions, source=args.source)

