# How these numbers are computed

Every row in `summary.md`/`paper_table.csv`/`traisformer_ar_decoding.csv` is produced by
`src/eval/full_eval_nereus.py` running a trained checkpoint against the real Kiel Fjord
test split — nothing here is hand-computed or estimated. Full per-metric definitions are
in `src/eval/metrics/displacement.py` and `src/eval/metrics/cpa.py`; summary:

- **`ade`** — mean displacement error (metres) of the model's single reported
  trajectory (the mixture-mean for NEREUS/GRU, the greedy-decoded rollout for
  TrAISformer-AR, etc.) against ground truth, averaged over all valid future timesteps
  per agent, then averaged over agents (`ade_per_agent`).
- **`k_ade` (minADE)** — same, but best-of-K over the model's K sampled/mixture-mode
  trajectories: the per-agent minimum error over K, then averaged over agents
  (`k_ade_per_agent`). K is the model's number of modes/samples (3 for NEREUS's MDN,
  `num_samples + 1` for TrAISformer-AR, etc).

  **This is a minimum, not a mean, over the K samples — matching how the TrAISformer
  paper (Nguyen & Fablet, arXiv:2109.03958) defines its own headline metric, not a
  softer "average-case" number.** The paper states this explicitly, twice:
  > "We used a best-of-N criterion, i.e. for each model, we sampled N predictions for
  > each target trajectory and reported **the best result**. In this paper, N = 16."
  > (§IV-A, Evaluation criteria)

  > "we report among 16 sampled trajectories **the one closest to the real
  > trajectory**." (§IV-B, discussing Fig. 7)

  So our `k_ade`/`k_fde` is the correct analog of their reported "TrAISformer" numbers
  — except at K=4 (`num_samples=3`) rather than their N=16 (see
  `config/eval_traisformer_ar_n16.yaml` to reproduce at matched N).
- **`fde_<m>min` / `k_fde_<m>min`** — same as `ade`/`k_ade` but at a single timestep
  (m minutes ahead) instead of averaged over the whole horizon.
- **`pred_risk`**, **`min_pred_dist`**, **`dcpa_mean`**, **`tcpa_mean_s`**,
  **`collision_ratio`** — all from `compute_batch_cpa_stats`
  (`src/eval/metrics/cpa.py`), evaluated between the model's *predicted* ego
  trajectory and every neighbouring vessel's *ground-truth* future:
  - `min_pred_dist`: the true minimum hull-to-hull distance over the whole predicted
    horizon, to the closest-approaching neighbour, averaged across all evaluated
    graphs (a graph = one prediction instance with ≥1 neighbour).
  - `dcpa_mean` / `tcpa_mean_s`: for the single most critical neighbour per graph (the
    one with the smallest hull distance at its own estimated closest point of
    approach, via linear extrapolation), the mean distance at that point (`dcpa_mean`,
    metres) and mean time to reach it (`tcpa_mean_s`, seconds).
  - `pred_risk`: a combined TCPA/DCPA risk score in [0, 1] for that same critical
    neighbour (`shape_aware_cpa_and_min_dist`), 1 = imminent and close.
  - `collision_ratio`: percentage of graphs where `min_pred_dist <= 0` at some point in
    the horizon, i.e. the model's predicted ego hull and a neighbour's ground-truth
    hull physically overlap.
- **Quantile columns** (`<metric>_p01` .. `<metric>_p99`) — the p01/p05/p10/p25/p50/
  p75/p90/p95/p99 percentiles of the *per-graph* values behind `min_pred_dist`,
  `dcpa_mean`, and `tcpa_mean_s` above, taken over every graph in the eval run (not
  just their means) — this is what shows the tail of critical/close encounters rather
  than only the average case.

All numbers in `summary.md` are for `region=kiel`, `ship_group=all` unless noted
otherwise (`scripts/analysis/summarize_results.py --region`/`--ship-group`).
