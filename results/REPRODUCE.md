# Reproducing the results

Every number in `paper_table.csv` / `summary.md` comes from a real `full_eval_nereus.py`
run against a real trained checkpoint — nothing here is hand-computed. This file records
the exact commands, so results can be regenerated (or checked) end to end.

## 1. Training

| Model | Command | Output |
|---|---|---|
| NEREUS (30-way module ablation) | `uv run python src/train/train_yaml.py config/nereus_all.yaml` | `lightning_logs/nereus_ablation/version_0..29/` |
| GRU baseline | `uv run python src/train/train_yaml.py config/train_gru.yaml` | `lightning_logs/baselines/version_4/` |
| DESIRE baseline | `uv run python src/train/train_yaml.py config/train_desire.yaml` | `lightning_logs/desire/version_0/` |
| TrAISformer-AR baseline | `uv run python src/train/train_yaml.py config/train_traisformer_ar.yaml` | `lightning_logs/traisformer_ar/version_0/` |
| IS-STGCNN baseline | `uv run python src/train/train_yaml.py config/train_isstgcnn.yaml` | `lightning_logs/isstgcnn/version_0..5/` |

NEREUS additionally requires the GMM prior fitted first: `uv run python src/train/fit_gmm.py`.

## 2. Evaluation

Each model was evaluated with `full_eval_nereus.py` in YAML mode, which globs every
checkpoint under the matching experiment dir and writes one `eval_fh.csv` per run:

```
uv run python src/eval/full_eval_nereus.py config/eval_nereus.yaml            # 30 NEREUS ablation configs
uv run python src/eval/full_eval_nereus.py config/eval_desire.yaml            # DESIRE
uv run python src/eval/full_eval_nereus.py config/eval_traisformer_ar.yaml    # TrAISformer-AR
uv run python src/eval/full_eval_nereus.py config/eval_isstgcnn.yaml          # IS-STGCNN (excluded from results, see README)
```

GRU was evaluated directly against its single checkpoint (no sweep needed):

```
uv run python src/eval/full_eval_nereus.py lightning_logs/baselines/version_4/best.ckpt --regions kiel
```

All eval runs use `--regions kiel` (the training region) and the `fh` (Kiel Fjord) AIS
source, matching the paper.

**Note on TrAISformer-AR:** it can only be evaluated on its training region — its
position embeddings are absolute grid cells, not a swappable rasterizer (see
`src/models/traisformer/model_ar.py`'s docstring / `_check_grid`). `swap_rasterizer` in
`full_eval_nereus.py` used to silently drop the model's own `pos_res` on any region swap
(defaulting to 50 instead of TrAISformer-AR's trained 25), which broke evaluation even on
the *same* region — fixed in this cleanup pass.

**TrAISformer-AR decoding comparison** (`results/traisformer_ar_decoding.csv`, see the
README caveat): the same checkpoint evaluated twice, differing only in how the row-0
("best") rollout is decoded —

```
uv run python src/eval/full_eval_nereus.py config/eval_traisformer_ar.yaml             # greedy (default) -> eval_fh.csv
uv run python src/eval/full_eval_nereus.py config/eval_traisformer_ar_stochastic.yaml  # single stochastic sample -> eval_fh_stochastic.csv
```

## 3. Aggregation

```
uv run python scripts/analysis/training_stats.py                  # per-run step/epoch/wall-time + loss_curve.csv
uv run python scripts/analysis/summarize_results.py \
    --exclude-experiments nereus_ablation_continued desire_continued isstgcnn \
    > results/summary.md                                          # full comparison table (all runs)
```

`nereus_ablation_continued` and `desire_continued` are earlier/resumed training runs
superseded by the fresh `nereus_ablation` and `desire` runs above — excluded so the
summary reflects final results only.

`results/paper_table.csv` is the subset of `summary.md` matching `main.tex`'s
`tab:nereus-metrics` (GRU, DESIRE, NEREUS_worst = all modules off, NEREUS_best = GAT +
attention-map + density-prior), plus a TrAISformer-AR row the paper table didn't include.
IS-STGCNN is excluded — see the README caveat.

NEREUS_worst/NEREUS_best identification: `nereus_ablation/version_8` is
`social=-, map=-, prior=-` (paper's "all modules off"); `nereus_ablation/version_15` is
`social=gat, map=atte, prior=density` (paper's best combination) — both confirmed via
`hparams.yaml` in their respective run dirs, and version_15 also happens to be the
empirically lowest-ADE config across all 30 ablations.
