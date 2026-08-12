"""Extract per-run training duration/step counts and train/val loss curves from
TensorBoard event logs under lightning_logs/.

Raw tfevents binaries aren't committed to the public repo, but these derived CSVs are
small and worth keeping -- this is what backs the README's training-time notes. Not
part of the training/eval pipeline itself; a reporting utility.

Usage:
    uv run python scripts/analysis/training_stats.py                          # all experiments
    uv run python scripts/analysis/training_stats.py "lightning_logs/nereus_ablation/version_*"
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
import yaml
from tbparse import SummaryReader

CURVE_TAGS = ("train_loss", "val_metric")


def _max_seconds(run_dir: Path) -> float | None:
    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        return None
    hparams = yaml.safe_load(hparams_path.read_text()) or {}
    return hparams.get("training.max_seconds")


def _version_sort_key(run_dir: Path):
    experiment = run_dir.parent.name
    tail = run_dir.name.rsplit("_", 1)[-1]
    version_num = int(tail) if tail.isdigit() else run_dir.name
    return (experiment, version_num)


def process_run(run_dir: Path) -> dict | None:
    """Write <run_dir>/loss_curve.csv and return a one-row summary dict, or None if
    the run has no usable train_loss scalar (e.g. an empty/failed run dir)."""
    try:
        df = SummaryReader(str(run_dir), extra_columns={"wall_time"}).scalars
    except Exception as e:
        print(f"skip (unreadable tfevents): {run_dir}  ({e})")
        return None

    if df.empty:
        return None

    curve = df[df.tag.isin(CURVE_TAGS)][["step", "tag", "value"]].sort_values(["tag", "step"])
    if not curve.empty:
        curve.to_csv(run_dir / "loss_curve.csv", index=False)

    tl = df[df.tag == "train_loss"]
    if tl.empty:
        return None
    ep = df[df.tag == "epoch"]

    final_step = int(tl["step"].max())
    final_epoch = float(ep["value"].max()) if len(ep) else None
    elapsed_s = float(tl["wall_time"].max() - tl["wall_time"].min())
    max_seconds = _max_seconds(run_dir)

    return {
        "experiment": run_dir.parent.name,
        "version": run_dir.name,
        "final_step": final_step,
        "final_epoch": final_epoch,
        "elapsed_hours": round(elapsed_s / 3600, 3),
        "max_seconds": max_seconds,
        "hit_time_budget": (elapsed_s >= max_seconds - 1000) if max_seconds else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "glob_pattern", nargs="?", default="lightning_logs/*/version_*",
        help="Glob over run directories (default: every experiment, every version)",
    )
    parser.add_argument(
        "--out", default="lightning_logs/training_stats.csv",
        help="Where to write the combined summary CSV",
    )
    args = parser.parse_args()

    run_dirs = sorted(
        (Path(p) for p in glob.glob(args.glob_pattern) if Path(p).is_dir()),
        key=_version_sort_key,
    )

    results = []
    for run_dir in run_dirs:
        row = process_run(run_dir)
        if row is None:
            print(f"skip (no train_loss found): {run_dir}")
            continue
        results.append(row)
        print(row)

    if not results:
        print("No runs with train_loss found.")
        return

    df_out = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(df_out.to_string())


if __name__ == "__main__":
    main()
