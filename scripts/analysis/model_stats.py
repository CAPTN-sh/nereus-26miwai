"""One-off script: training steps, param counts, FLOPs, and inference latency for one
or more checkpoints.

Not part of the eval pipeline -- run manually. Defaults to the full nereus_ablation
sweep (original use case); pass explicit checkpoint paths to cover other models
(GRU/DESIRE/TrAISformer-AR/IS-STGCNN), which dispatch through model.inference(...)
(matching how full_eval_nereus.py actually calls them) rather than NEREUS's raw
model(batch, scene) forward.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from data.graph.build_dataloader import graph_loader
from data.map.rasterize import Rasterizer
from data.map.scene_gernerator import SceneLoader
from utils.config import DATA_FOLDER_PATH, MAP_FOLDER_PATH, TRAIN_BBOX

device = torch.device("cuda:0")
torch.cuda.set_device(device)

DEFAULT_GLOB = "lightning_logs/nereus_ablation/version_*/best.ckpt"
N_WARMUP = 10
N_TIMED = 50

# model_class (hparams.yaml) -> (PL module class, call convention). "forward" matches
# how full_eval_nereus.py evaluates NEREUS (model(batch, scene) directly); every other
# model's actual eval-time behavior goes through .inference(batch, scene) instead
# (sampling/decoding logic that .forward() doesn't reproduce).
def _registry():
    from train.pl_modules import DESIREModule, GRUModule, ISSTGCNNModule, NereusModule, TrAISformerARModule
    return {
        "NereusModule": (NereusModule, "forward"),
        "GRUModule": (GRUModule, "inference"),
        "DESIREModule": (DESIREModule, "inference"),
        "TrAISformerARModule": (TrAISformerARModule, "inference"),
        "ISSTGCNNModule": (ISSTGCNNModule, "inference"),
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_submodule = {}
    for name, sub in model.named_children():
        n = sum(p.numel() for p in sub.parameters())
        if n:
            by_submodule[name] = n
    return total, trainable, by_submodule


def _call(model, call_kind, batch, scene):
    return model(batch, scene) if call_kind == "forward" else model.inference(batch, scene)


def time_inference(model, call_kind, batch, scene):
    """Average wall-clock ms per forward pass (bs=32) via CUDA events, GPU-synced."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_TIMED)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_TIMED)]
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            _call(model, call_kind, batch, scene)
        torch.cuda.synchronize()
        for i in range(N_TIMED):
            starts[i].record()
            _call(model, call_kind, batch, scene)
            ends[i].record()
        torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times_ms) / len(times_ms)


def process_checkpoint(ckpt_path: Path, batch, scene):
    version = ckpt_path.parent.name
    experiment = ckpt_path.parent.parent.name
    print(f"=== {experiment}/{version} ===")

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    global_step = raw_ckpt["global_step"]
    epoch = raw_ckpt["epoch"]
    class_name = raw_ckpt["hyper_parameters"].get("model_class") or raw_ckpt["hyper_parameters"].get("model")
    module_cls, call_kind = _registry().get(class_name, (None, None))
    if module_cls is None:
        print(f"  skip: unrecognized model_class {class_name!r}")
        return None

    pl_module = module_cls.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = pl_module.model.to(device).eval()

    social = pl_module.hparams.get("nereus_modules.social")
    map_mod = pl_module.hparams.get("nereus_modules.map")
    prior = pl_module.hparams.get("nereus_modules.prior")

    total_params, trainable_params, by_submodule = count_params(model)

    flop_counter = FlopCounterMode(display=False)
    try:
        with torch.inference_mode(), flop_counter:
            _call(model, call_kind, batch, scene)
        flops = flop_counter.get_total_flops()
    except Exception as e:
        print(f"  FLOP count failed: {e}")
        flops = None

    avg_ms_bs32 = time_inference(model, call_kind, batch, scene)

    row = {
        "experiment": experiment,
        "version": version,
        "model_class": class_name,
        "social": social,
        "map": map_mod,
        "prior": prior,
        "global_step": global_step,
        "epoch": epoch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops_per_fwd_bs32": flops,
        "avg_inference_ms_bs32": avg_ms_bs32,
        "avg_inference_ms_per_sample": avg_ms_bs32 / batch.y_pos.shape[0],
        "params_by_submodule": by_submodule,
    }
    print(json.dumps(row, indent=2))

    del pl_module, model
    torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoints", nargs="*",
        help=f"Explicit checkpoint paths. Defaults to {DEFAULT_GLOB!r} if omitted.",
    )
    parser.add_argument("--out", default=None, help="Output CSV/JSON basename (without extension)")
    args = parser.parse_args()

    ckpt_paths = (
        [Path(p) for p in args.checkpoints] if args.checkpoints
        else sorted((Path(p) for p in glob.glob(DEFAULT_GLOB)),
                    key=lambda p: int(p.parent.name.split("_")[1]))
    )
    out_base = args.out or ("lightning_logs/nereus_ablation/model_stats" if not args.checkpoints
                             else "lightning_logs/model_stats")

    data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
    loader, _ = graph_loader(
        data_folder=data_folder, flag="test",
        min_date=pd.Timestamp("2022-01-01"), max_date=pd.Timestamp("2024-01-01"),
        batch_size=32, pin_memory=False, pred_len=30, obs_len=60,
        max_edge_dist=500, shuffle=False, ship_group="all",
    )
    batch = next(iter(loader)).to(device)
    scene = torch.from_numpy(
        np.ascontiguousarray(SceneLoader(Rasterizer(TRAIN_BBOX)).load_scene(MAP_FOLDER_PATH))
    ).to(device, torch.float32)

    results = [r for r in (process_checkpoint(p, batch, scene) for p in ckpt_paths) if r is not None]

    Path(f"{out_base}.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_base}.json")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "params_by_submodule"} for r in results])
    df.to_csv(f"{out_base}.csv", index=False)
    print(f"Wrote {out_base}.csv")
    print(df.to_string())


if __name__ == "__main__":
    main()
