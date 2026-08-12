# NEREUS

**N**autical **E**nvironment-aware **R**oute **E**stimation **U**nder uncertainty for **S**urface vessels — a modular probabilistic framework for context-aware ship trajectory forecasting.

Code accompanying our MIWAI 2026 submission (citation to follow once the paper is public).

## Setup

```bash
uv sync
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.11.0%2Bcu130.html
```

## Data

Models are trained on AIS trajectories from the Kiel Fjord (`fh_10` source) and evaluated
on `fh_10`/`dma_10`. Raw AIS data is preprocessed by a separate, currently-internal
pipeline; it is not included in this repo. Get in touch with the authors for access.

## Repository structure

```
src/
  data/    AIS/map dataloaders (GraphTrajectoryDataset, SceneLoader, density/GMM maps)
  models/  one directory per model — desire, gmm, gru, isstgcnn, nereus, traisformer
  train/   training entry points (train_yaml.py, fit_gmm.py, continue_training.py, train_tune.py)
  eval/    full_eval_nereus.py — the single evaluation entry point for every model below
  utils/   config and logging
config/    one train_*.yaml / eval_*.yaml pair per model
scripts/analysis/   reporting utilities (training-time stats, results aggregation) — not
                     part of the training/eval pipeline itself
results/   the numbers below, plus REPRODUCE.md documenting the exact commands used
lightning_logs/     per-run hparams.yaml, eval_fh.csv, and loss_curve.csv (train/val loss
                     extracted from the raw TensorBoard logs). Checkpoints and the raw
                     TensorBoard event files are not committed — retrain to regenerate them.
```

## Training

```bash
uv run python src/train/fit_gmm.py                                  # NEREUS prior-module prerequisite
uv run python src/train/train_yaml.py config/nereus_all.yaml         # NEREUS, 30-config module ablation
uv run python src/train/train_yaml.py config/train_gru.yaml          # GRU baseline
uv run python src/train/train_yaml.py config/train_desire.yaml       # DESIRE baseline
uv run python src/train/train_yaml.py config/train_traisformer_ar.yaml  # TrAISformer-AR baseline
uv run python src/train/train_yaml.py config/train_isstgcnn.yaml     # IS-STGCNN baseline
```

## Evaluation

`src/eval/full_eval_nereus.py` is the canonical evaluation entry point for every model
in this repo (it dispatches on the checkpoint's model class); a YAML config sweeps every
checkpoint in an experiment directory, and a single checkpoint path can also be passed
directly:

```bash
uv run python src/eval/full_eval_nereus.py config/eval_nereus.yaml
uv run python src/eval/full_eval_nereus.py config/eval_desire.yaml
uv run python src/eval/full_eval_nereus.py config/eval_traisformer_ar.yaml
uv run python src/eval/full_eval_nereus.py lightning_logs/baselines/version_4/best.ckpt --regions kiel
```

See `results/REPRODUCE.md` for the full command reference, including how each
`lightning_logs/` experiment directory maps to its config.

## Results

ADE / minADE and FDE / minFDE (m) at 1/3/5 minutes, Kiel Fjord test set, all ship types
(from `results/paper_table.csv`; **NEREUS_worst** = all optional modules off,
**NEREUS_best** = GAT social module + attention map module + density-based prior):

| Model | ADE | minADE | FDE@1min | minFDE@1min | FDE@3min | minFDE@3min | FDE@5min | minFDE@5min |
|---|---|---|---|---|---|---|---|---|
| GRU | 65.54 | 65.54 | 16.73 | 16.73 | 78.87 | 78.87 | 160.87 | 160.87 |
| DESIRE | 66.22 | 66.00 | 17.05 | 16.94 | 80.02 | 79.73 | 160.18 | 159.58 |
| TrAISformer-AR (K=4) | 77.55 | 55.07 | 23.29 | 18.03 | 91.20 | 61.34 | 187.01 | 116.28 |
| TrAISformer-AR (K=16) | 77.55 | 34.73 | 23.29 | 12.42 | 91.20 | 32.87 | 187.00 | **60.24** |
| NEREUS_worst | 61.45 | 46.88 | 13.67 | 10.76 | 74.22 | 54.08 | 154.56 | 109.16 |
| **NEREUS_best** | **46.85** | **35.26** | **11.15** | **8.26** | **56.03** | **39.84** | **118.58** | 82.35 |

The full 30-configuration module-ablation comparison is in `results/summary.md`, along
with per-run TCPA/DCPA/collision-risk statistics.

**TrAISformer-AR's minADE/minFDE at matched sample count (K=16, matching the paper's
own best-of-16 evaluation — see `results/METHODOLOGY.md`) is competitive with, and at
longer horizons beats, NEREUS_best:** minADE 34.73 vs. 35.26, minFDE@3min 32.87 vs.
39.84, and minFDE@5min 60.24 vs. 82.35. NEREUS_best only wins at the 1-minute horizon
(minFDE@1min 8.26 vs. 12.42). `ade`/`fde` (the single-hypothesis columns) are unaffected
by K — only `k_ade`/`k_fde` (best-of-K) improve with more samples, per
`config/eval_traisformer_ar_n16.yaml`. This tracks with the paper's own thesis: sampled
multimodal decoding pays off increasingly at longer horizons, where route-branching
uncertainty dominates.

**IS-STGCNN is excluded from the results above.** Its implementation is likely faulty:
it uses first-order Nomoto steering rather than the full MMG maneuvering model, and the
social-sampling module's implementation details had to be guessed from the paper (see
`src/models/isstgcnn/`). Numbers exist in `lightning_logs/isstgcnn/` for reference but
should not be treated as a fair baseline comparison.

**On TrAISformer-AR's ADE/minADE gap:** its `ade`/`fde` columns use greedy (argmax)
decoding for the single reported trajectory, which is exactly the paper's own
`TrAISformer_No-Stoch` ablation — the paper reports a ~3x degradation from this same
greedy-vs-stochastic switch (Table I: 0.94 nmi stochastic best-of-16 vs. 2.88 nmi
greedy, 2h horizon), so a worse `ade` than the sampled `k_ade` here is expected
behavior for this architecture, not a bug. Two eval-time-only overrides (no retraining
-- see `results/REPRODUCE.md`) let us probe this further on the same checkpoint
(`results/traisformer_ar_decoding.csv`):

| Decoding | ADE | minADE |
|---|---|---|
| Greedy (default, used above), K=4 | 77.55 | 55.07 |
| Single stochastic sample, K=4 | 85.38 | 55.06 |
| Greedy, K=16 (`config/eval_traisformer_ar_n16.yaml`) | 77.55 | **34.73** |

Two findings:
- **Single stochastic sample vs. greedy (both K=4):** greedy wins. A lone random draw
  can land on any mode of the learned distribution, while greedy always takes the
  highest-probability (evidently better-calibrated) one. `ADE` is worse for the
  stochastic row (noisier single trajectory); `minADE` is essentially unchanged either
  way, as expected -- it's always best-of-`num_samples`, sampled the same way
  regardless of `greedy_best`.
- **Sample count (K=4 vs. K=16, both greedy):** `ADE` is unaffected (same greedy row
  0 regardless of K), but `minADE` improves substantially with more samples to pick the
  best from -- see the Results section above for how this closes the gap with NEREUS.

Our training loss also differs slightly from the paper: `loss_ar.py` sums the four
per-attribute (lat/lon/SOG/COG) cross-entropies at the trained resolution only. The
paper additionally adds a second cross-entropy term at a 3x coarser resolution
(`β · CE(h'_t, l'_t)`, Algorithm 3), which they report gives a marginal improvement —
not implemented here.

## Attribution

- DESIRE implementation adapted from [AkashGanesan/desire-pytorch](https://github.com/AkashGanesan/desire-pytorch).
- TrAISformer implementation adapted from [CIA-Oceanix/TrAISformer](https://github.com/CIA-Oceanix/TrAISformer).

## License

MIT — see [LICENSE](LICENSE).
