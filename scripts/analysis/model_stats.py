"""One-off script: training steps, param counts, and FLOPs for each nereus_ablation checkpoint.

Not part of the eval pipeline -- run manually against lightning_logs/nereus_ablation.
"""
import glob
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.flop_counter import FlopCounterMode

from data.graph.build_dataloader import graph_loader
from data.map.rasterize import Rasterizer
from data.map.scene_gernerator import SceneLoader
from train.pl_modules import NereusModule
from utils.config import DATA_FOLDER_PATH, MAP_FOLDER_PATH, TRAIN_BBOX

device = torch.device("cuda:0")
torch.cuda.set_device(device)

VERSIONS_GLOB = "lightning_logs/nereus_ablation/version_*/best.ckpt"

# One shared real batch (kiel/test) reused as dummy input for every model variant.
data_folder = DATA_FOLDER_PATH / "ais/4_features/fh_10/kiel"
loader, _ = graph_loader(
    data_folder=data_folder,
    flag="test",
    min_date=pd.Timestamp("2022-01-01"),
    max_date=pd.Timestamp("2024-01-01"),
    batch_size=32,
    pin_memory=False,
    pred_len=30,
    obs_len=60,
    max_edge_dist=500,
    shuffle=False,
    ship_group="all",
)
batch = next(iter(loader)).to(device)

scene = torch.from_numpy(
    __import__("numpy").ascontiguousarray(SceneLoader(Rasterizer(TRAIN_BBOX)).load_scene(MAP_FOLDER_PATH))
).to(device, torch.float32)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_submodule = {}
    for name, sub in model.named_children():
        n = sum(p.numel() for p in sub.parameters())
        if n:
            by_submodule[name] = n
    return total, trainable, by_submodule


N_WARMUP = 10
N_TIMED = 50


def time_inference(model):
    """Average wall-clock ms per forward pass (bs=32) via CUDA events, GPU-synced."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_TIMED)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_TIMED)]
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            model(batch, scene)
        torch.cuda.synchronize()
        for i in range(N_TIMED):
            starts[i].record()
            model(batch, scene)
            ends[i].record()
        torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times_ms) / len(times_ms)


results = []
for ckpt_path in sorted(glob.glob(VERSIONS_GLOB), key=lambda p: int(Path(p).parent.name.split("_")[1])):
    ckpt_path = Path(ckpt_path)
    version = ckpt_path.parent.name
    print(f"=== {version} ===")

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    global_step = raw_ckpt["global_step"]
    epoch = raw_ckpt["epoch"]

    pl_module = NereusModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    model = pl_module.model.to(device).eval()

    social = pl_module.hparams.get("nereus_modules.social")
    map_mod = pl_module.hparams.get("nereus_modules.map")
    prior = pl_module.hparams.get("nereus_modules.prior")

    total_params, trainable_params, by_submodule = count_params(model)

    flop_counter = FlopCounterMode(display=False)
    try:
        with torch.inference_mode(), flop_counter:
            model(batch, scene)
        flops = flop_counter.get_total_flops()
    except Exception as e:
        print(f"  FLOP count failed: {e}")
        flops = None

    avg_ms_bs32 = time_inference(model)

    row = {
        "version": version,
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
    results.append(row)
    print(json.dumps(row, indent=2))

    del pl_module, model
    torch.cuda.empty_cache()

out_path = Path("lightning_logs/nereus_ablation/model_stats.json")
out_path.write_text(json.dumps(results, indent=2))
print(f"\nWrote {out_path}")

df = pd.DataFrame([{k: v for k, v in r.items() if k != "params_by_submodule"} for r in results])
df.to_csv("lightning_logs/nereus_ablation/model_stats.csv", index=False)
print(df.to_string())
