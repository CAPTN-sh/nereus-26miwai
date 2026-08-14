# Reproducing the results
This file records the exact commands, so results can be regenerated (or checked) end to end.
Some models are inherently stochastic, so exact metrics may not match with repeated experiments, unless strictly seeded.


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
    --exclude-experiments isstgcnn \
    > results/summary.md                                          # full comparison table (all runs)
```

`results/paper_table.csv` is the subset of `summary.md` matching `main.tex`'s
`tab:nereus-metrics` (GRU, DESIRE, NEREUS_worst = all modules off, NEREUS_best = GAT +
attention-map + density-prior), plus a TrAISformer-AR row the paper table didn't include.
IS-STGCNN is excluded — see the README caveat.

NEREUS_worst/NEREUS_best identification: `nereus_ablation/version_8` is
`social=-, map=-, prior=-` (paper's "all modules off"); `nereus_ablation/version_15` is
`social=gat, map=atte, prior=density`.

## 4. Model size / training cost / inference latency

`results/model_stats_summary.csv` (params, FLOPs, GPU-timed inference latency, training
steps/time for the 5 README headline models) was built from:

```
uv run python scripts/analysis/model_stats.py \
    lightning_logs/baselines/version_4/best.ckpt \
    lightning_logs/desire/version_0/best.ckpt \
    lightning_logs/traisformer_ar/version_0/best.ckpt \
    --out lightning_logs/model_stats                              # GRU/DESIRE/TrAISformer-AR only
uv run python scripts/analysis/training_stats.py                  # (already run above; reused here)
```

NEREUS_worst/NEREUS_best params/FLOPs/latency reuse `lightning_logs/nereus_ablation/model_stats.csv`
(the full 30-config ablation run, generated the same way against `config/eval_nereus.yaml`'s
checkpoints). `model_stats.py` calls each model's actual `.inference()` method (not `.forward()`)
for every model except NEREUS, matching how `full_eval_nereus.py` evaluates them.

`results/plot_budget_capped_val_loss.png` plots `val_metric` from each budget-capped
config's `loss_curve.csv` (the 9 `nereus_ablation` configs with `hit_time_budget=True`
in `lightning_logs/training_stats.csv`, all `social=pool`) — shows all 9 still
decreasing at cutoff, i.e. genuinely budget-limited rather than converged.

`results/model_stats_all.csv` is the same params/FLOPs/latency/training-cost table as
above, but for every checkpoint in the repo (all 30 NEREUS ablation configs plus
GRU/DESIRE/TrAISformer-AR — 33 rows).