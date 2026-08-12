"""Aggregate every full_eval_nereus.py eval_<source>*.csv found under lightning_logs/
into two tables: a cross-model comparison and a critical-encounter-quantiles table.

See results/METHODOLOGY.md for how each metric is computed. A scriptable replacement
for the notebook-only load_eval_csvs()/groupby/pivot logic in eval.ipynb -- this is
what backs the README's results table. Not part of the training/eval pipeline itself;
a reporting utility.

Tolerates the mixed CSV schema across old and newly-regenerated runs: dcpa_mean,
tcpa_mean_s, and the *_p01..p99 quantile columns only exist in eval CSVs written after
full_eval_nereus.py added CPA/TCPA/DCPA reporting -- older files simply don't have them
and are aggregated over whatever columns they do have.

Also picks up "variant" CSVs of the same checkpoint (e.g. eval_fh_stochastic.csv from
full_eval_nereus.py's csv_suffix option -- see TraisformerARParams.greedy_best) as
separate rows tagged "run [variant]", not merged into the base run.

Usage:
    uv run python scripts/analysis/summarize_results.py
    uv run python scripts/analysis/summarize_results.py --region kiel --ship-group all
    uv run python scripts/analysis/summarize_results.py > results/summary.md
"""
import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

SCALAR_COLS = [
    "ade", "k_ade", "pred_risk", "min_pred_dist",
    "dcpa_mean", "tcpa_mean_s", "collision_ratio",
]
QUANTILE_METRICS = ["min_pred_dist", "dcpa", "tcpa_s"]
QUANTILE_LEVELS = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
FDE_MINUTES = [1, 3, 5]
STEPS_PER_MINUTE = 6

# Display names for non-NEREUS model classes (hparams.yaml's model_class is a Lightning
# module name, not meant for humans).
MODEL_CLASS_LABELS = {
    "GRUModule": "GRU",
    "DESIREModule": "DESIRE",
    "TrAISformerARModule": "TrAISformer-AR",
    "ISSTGCNNModule": "IS-STGCNN",
}
# lightning_logs/ experiment dir -> display name, for the plot legend.
FAMILY_LABELS = {
    "nereus_ablation": "NEREUS",
    "baselines": "GRU",
    "desire": "DESIRE",
    "traisformer_ar": "TrAISformer-AR",
    "isstgcnn": "IS-STGCNN",
}


def load_eval_csvs(glob_pattern: str, source: str) -> pd.DataFrame:
    frames = []
    prefix = f"eval_{source}"
    for csv_path in sorted(Path(p) for p in glob.glob(glob_pattern, recursive=True)):
        df = pd.read_csv(csv_path)
        df["experiment"] = csv_path.parent.parent.name
        df["version"] = csv_path.parent.name
        # "eval_fh.csv" -> variant ""; "eval_fh_stochastic.csv" -> variant "stochastic"
        df["variant"] = csv_path.stem[len(prefix):].lstrip("_")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No eval CSVs matched {glob_pattern!r}")
    long_df = pd.concat(frames, ignore_index=True)
    long_df["run"] = (
        long_df["experiment"] + "/" + long_df["version"]
        + long_df["variant"].apply(lambda v: f" [{v}]" if v else "")
    )
    return long_df


def _read_hparams(run_dir: Path) -> dict:
    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.exists():
        return {}
    return yaml.safe_load(hparams_path.read_text()) or {}


def _model_class(hp: dict, run_dir: Path) -> str:
    # "model_class" is the current hparam key; "model" is a legacy key some older
    # checkpoints still carry (see _restore_checkpoint_kwargs in pl_modules.py).
    return hp.get("model_class") or hp.get("model") or run_dir.parent.name


def _nereus_fields(hp: dict) -> tuple[str, str, str]:
    return (
        hp.get("nereus_modules.social") or "-",
        hp.get("nereus_modules.map") or "-",
        hp.get("nereus_modules.prior") or "-",
    )


def model_label(run_dir: Path) -> str:
    """Best-effort human label from hparams.yaml -- model class, plus (for NEREUS) the
    social/map/prior module choice. Falls back to the run directory name."""
    hp = _read_hparams(run_dir)
    if not hp:
        return run_dir.name
    model = _model_class(hp, run_dir)
    if model == "NereusModule":
        social, map_, prior = _nereus_fields(hp)
        return f"NEREUS(social={social}, map={map_}, prior={prior})"
    return MODEL_CLASS_LABELS.get(model, model)


def _display_label(run: str, model: str) -> str:
    """model_label() collapses to the same string for a checkpoint's variant runs
    (e.g. TrAISformer-AR greedy vs [stochastic] share one hparams.yaml) -- append the
    run's own "[variant]" tag, if any, so plots/labels stay disambiguated."""
    if "[" in run:
        return f"{model} [{run.split('[', 1)[1]}"
    return model


def _run_dirs(scalar_index, lightning_logs_dir: str) -> list[Path]:
    """Map a "experiment/version [variant]" run id back to its on-disk directory."""
    return [Path(lightning_logs_dir) / run.split(" [")[0] for run in scalar_index]


def summarize(long_df: pd.DataFrame, region: str, ship_group: str, lightning_logs_dir: str) -> pd.DataFrame:
    df = long_df[(long_df["region"] == region) & (long_df["ship_group"] == ship_group)]
    if df.empty:
        raise ValueError(f"No rows for region={region!r} ship_group={ship_group!r}")

    present_scalars = [c for c in SCALAR_COLS if c in df.columns]
    scalar = df.groupby("run")[present_scalars].mean()

    fde = df.pivot_table(index="run", columns="step", values="fde", aggfunc="mean")
    k_fde = df.pivot_table(index="run", columns="step", values="k_fde", aggfunc="mean")
    for m in FDE_MINUTES:
        step = m * STEPS_PER_MINUTE
        if step in fde.columns:
            scalar[f"fde_{m}min"] = fde[step]
        if step in k_fde.columns:
            scalar[f"k_fde_{m}min"] = k_fde[step]

    scalar.insert(0, "model", [model_label(d) for d in _run_dirs(scalar.index, lightning_logs_dir)])
    return scalar.sort_values("ade")


def summarize_quantiles(long_df: pd.DataFrame, region: str, ship_group: str, lightning_logs_dir: str) -> pd.DataFrame | None:
    """Per-run p01..p99 quantiles of min_pred_dist / dcpa / tcpa_s, if the CSVs have them."""
    df = long_df[(long_df["region"] == region) & (long_df["ship_group"] == ship_group)]
    cols = [f"{metric}_{level}" for metric in QUANTILE_METRICS for level in QUANTILE_LEVELS]
    present = [c for c in cols if c in df.columns]
    if not present:
        return None

    table = df.groupby("run")[present].mean()  # constant per run block; mean == first
    table.insert(0, "model", [model_label(d) for d in _run_dirs(table.index, lightning_logs_dir)])
    return table.reindex(columns=["model"] + present)


# Always present regardless of model type (saved by _BaseModule.__init__ in
# pl_modules.py); everything else is a "cfg.*"/"nereus_modules.*" field that only
# some model types have.
COMMON_HP_KEYS = ["lr", "weight_decay", "warmup_batches", "batches_per_eval"]


def summarize_hyperparams(long_df: pd.DataFrame, lightning_logs_dir: str) -> dict[str, pd.DataFrame]:
    """One table per experiment family (nereus_ablation/baselines/desire/...), one row
    per unique checkpoint within it (not per eval variant -- hyperparameters are a
    property of training, shared by every eval-time variant of the same checkpoint;
    e.g. TrAISformer-AR's greedy/[stochastic]/[n16] runs all reuse one hparams.yaml,
    since num_samples/greedy_best there are eval-time overrides, not trained values).

    Split by family (rather than one big union-of-all-fields table) because different
    model types share almost no cfg.* fields -- a single table is mostly empty cells.
    Columns within a family are the union of that family's hparams.yaml keys, still
    NaN only where one run in the family genuinely differs from another (e.g. a
    resumed run with a tweaked LR).
    """
    checkpoints = long_df[["experiment", "version"]].drop_duplicates()
    by_family: dict[str, dict[str, dict]] = {}
    keys_by_family: dict[str, set[str]] = {}
    for _, r in checkpoints.iterrows():
        family, version = r["experiment"], r["version"]
        ckpt_id = f"{family}/{version}"
        hparams_path = Path(lightning_logs_dir) / ckpt_id / "hparams.yaml"
        if not hparams_path.exists():
            continue
        hp = yaml.safe_load(hparams_path.read_text()) or {}
        flat = {
            k: v for k, v in hp.items()
            if k in COMMON_HP_KEYS or k.startswith("cfg.") or k.startswith("nereus_modules.")
        }
        by_family.setdefault(family, {})[ckpt_id] = flat
        keys_by_family.setdefault(family, set()).update(flat.keys())

    tables = {}
    for family, hp_by_ckpt in by_family.items():
        all_keys = keys_by_family[family]
        col_order = [k for k in COMMON_HP_KEYS if k in all_keys] + sorted(all_keys - set(COMMON_HP_KEYS))
        table = pd.DataFrame({
            ckpt_id: {c: flat.get(c) for c in col_order} for ckpt_id, flat in hp_by_ckpt.items()
        }).T
        table.insert(0, "model", [model_label(Path(lightning_logs_dir) / ckpt_id) for ckpt_id in table.index])
        tables[family] = table.sort_index()
    return tables


FAMILY_COLORS = {
    "nereus_ablation": "#4C72B0",
    "baselines": "#DD8452",
    "desire": "#55A868",
    "traisformer_ar": "#C44E52",
    "isstgcnn": "#8172B2",
}


def _run_families(long_df: pd.DataFrame, runs) -> list[str]:
    experiment_of = long_df.groupby("run")["experiment"].first()
    return [experiment_of.get(r, "other") for r in runs]


def _aligned_labels(runs, models, lightning_logs_dir: str) -> list[str]:
    """Build y-tick labels; NEREUS rows get their social/map/prior fields padded to a
    common per-column width, so (rendered in a monospace font) the fields visually
    line up across rows instead of each label being a different length."""
    dirs = _run_dirs(runs, lightning_logs_dir)
    fields = [_nereus_fields(_read_hparams(d)) if m.startswith("NEREUS") else None for d, m in zip(dirs, models)]

    widths = [0, 0, 0]
    for f in fields:
        if f:
            widths = [max(w, len(v)) for w, v in zip(widths, f)]

    labels = []
    for run, model, f in zip(runs, models, fields):
        if f:
            social, map_, prior = (v.ljust(w) for v, w in zip(f, widths))
            labels.append(f"NEREUS(social={social}, map={map_}, prior={prior})")
        else:
            labels.append(_display_label(run, model))
    return labels


def plot_ade_ranking(table: pd.DataFrame, families: list[str], lightning_logs_dir: str, out_path: Path) -> None:
    """Horizontal bar chart of ade/k_ade for every run, sorted (table is pre-sorted by
    ade), colored by experiment family."""
    fig, ax = plt.subplots(figsize=(9.5, max(3, 0.3 * len(table))))
    y = list(range(len(table)))
    colors = [FAMILY_COLORS.get(f, "#999999") for f in families]
    ax.barh(y, table["ade"], color=colors)
    if "k_ade" in table.columns:
        ax.scatter(table["k_ade"], y, marker="|", s=120, color="black", linewidths=1.5, zorder=3)
    ax.set_yticks(y)
    labels = _aligned_labels(table.index, table["model"], lightning_logs_dir)
    ax.set_yticklabels(labels, fontsize=7, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Displacement error (m)")
    ax.set_title("ADE (bar) / minADE best-of-K (black tick) by run, sorted by ADE")
    # Only legend families actually present in `families` -- FAMILY_COLORS is a fixed
    # superset (e.g. excluded experiments like isstgcnn would otherwise still show up).
    present = [f for f in FAMILY_COLORS if f in families]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in present]
    handles.append(plt.Line2D([0], [0], marker="|", color="black", linestyle="", markersize=10))
    legend_labels = [FAMILY_LABELS.get(f, f) for f in present] + ["minADE"]
    ax.legend(handles, legend_labels, title="model", loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_quantile_curves(
    quantiles: pd.DataFrame, out_path: Path, metric: str = "min_pred_dist",
    levels: list[str] | None = None,
) -> None:
    """Per-run quantile curve for one metric -- shows the tail shape, not just the mean.

    Plots every run passed in (caller picks the subset, e.g. just the headline models,
    since 30+ overlapping lines is unreadable). Defaults to p05..p50: p01 is 0 for
    every run (uninformative) and p75+ is dominated by "no nearby vessel" graphs,
    swamping the scale and hiding exactly the close-encounter differences this plot
    exists to show.
    """
    levels = levels or ["p05", "p10", "p25", "p50"]
    cols = [f"{metric}_{level}" for level in levels]
    cols = [c for c in cols if c in quantiles.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(cols))
    for run, row in quantiles.iterrows():
        label = _display_label(run, row.get("model", run))
        ax.plot(x, row[cols].values, marker="o", markersize=3, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.rsplit("_", 1)[-1] for c in cols])
    ax.set_xlabel("quantile")
    ax.set_ylabel(f"{metric} (m)" if metric != "tcpa_s" else f"{metric} (s)")
    ax.set_title(f"{metric} quantiles by run")
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids adding a tabulate dependency)."""
    cols = ["run"] + list(df.columns)
    rows = [[idx] + [row[c] for c in df.columns] for idx, row in df.iterrows()]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightning-logs-dir", default="lightning_logs")
    parser.add_argument("--source", default="fh")
    parser.add_argument("--region", default="kiel")
    parser.add_argument("--ship-group", default="all")
    parser.add_argument(
        "--exclude-experiments", nargs="*", default=[],
        help="Top-level lightning_logs/ subdirs to drop, e.g. nereus_ablation_continued "
             "desire_continued (resumed/superseded training runs, not final results)",
    )
    parser.add_argument(
        "--plots-dir", default="results",
        help="Where to write plot_*.png (referenced by relative path in the printed markdown)",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    pattern = f"{args.lightning_logs_dir}/**/version_*/eval_{args.source}*.csv"
    long_df = load_eval_csvs(pattern, args.source)
    print(f"Loaded {long_df['run'].nunique()} runs from {pattern}", file=sys.stderr)

    if args.exclude_experiments:
        before = long_df["run"].nunique()
        long_df = long_df[~long_df["experiment"].isin(args.exclude_experiments)]
        print(
            f"Excluded {args.exclude_experiments} -> {before - long_df['run'].nunique()} runs dropped",
            file=sys.stderr,
        )

    table = summarize(long_df, args.region, args.ship_group, args.lightning_logs_dir).round(3)
    quantiles = summarize_quantiles(long_df, args.region, args.ship_group, args.lightning_logs_dir)

    plots_dir = Path(args.plots_dir)
    ade_plot = quantile_plot = None
    if not args.no_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
        families = _run_families(long_df, table.index)
        ade_plot = plots_dir / "plot_ade_ranking.png"
        plot_ade_ranking(table, families, args.lightning_logs_dir, ade_plot)
        print(f"Wrote {ade_plot}", file=sys.stderr)

        if quantiles is not None:
            # Full 30-config ablation is unreadable as overlapping lines -- plot only
            # the cross-model headline runs (everything outside nereus_ablation) plus
            # the paper's NEREUS_worst/NEREUS_best (see REPRODUCE.md for those two IDs).
            headline_mask = (~quantiles.index.str.startswith("nereus_ablation/")) | quantiles.index.str.contains(
                r"nereus_ablation/version_(?:8|15)\b", regex=True
            )
            quantile_plot = plots_dir / "plot_min_pred_dist_quantiles.png"
            plot_quantile_curves(quantiles[headline_mask], quantile_plot, metric="min_pred_dist")
            print(f"Wrote {quantile_plot}", file=sys.stderr)

    print(f"See METHODOLOGY.md for how these metrics are computed.\n")
    print("## Cross-model comparison\n")
    if ade_plot is not None:
        print(f"![ADE/minADE ranking]({ade_plot.name})\n")
    print(to_markdown(table))
    if quantiles is not None:
        print(
            "\n## Critical-encounter quantiles\n\n"
            "p01/p05/.../p99 of the per-graph `min_pred_dist`, `dcpa`, and `tcpa_s` -- "
            "see METHODOLOGY.md. Only runs whose CSV has these columns are listed "
            "(regenerated after full_eval_nereus.py added CPA/TCPA/DCPA reporting).\n"
        )
        if quantile_plot is not None:
            print(f"![min_pred_dist quantiles, headline runs]({quantile_plot.name})\n")
        print(to_markdown(quantiles.round(2)))
    else:
        print("\n(No runs with quantile columns found.)")

    print(
        "\n## Hyperparameters\n\n"
        "One row per checkpoint (eval-time decoding variants like [stochastic]/[n16] "
        "share their base checkpoint's row -- those overrides aren't trained "
        "hyperparameters). One table per model family: different model types share "
        "almost no cfg.* fields, so a single combined table is mostly empty cells.\n"
    )
    hyperparams_by_family = summarize_hyperparams(long_df, args.lightning_logs_dir)
    for family, hp_table in hyperparams_by_family.items():
        print(f"\n### {FAMILY_LABELS.get(family, family)}\n")
        print(to_markdown(hp_table))


if __name__ == "__main__":
    main()
