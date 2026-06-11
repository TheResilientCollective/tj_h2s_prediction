# feature_trim_berry

**Date:** 2026-06-10
**Status:** Phase 1 done; Phase 2 promotion shipped in PR #28
**Author:** session-handoff

## Question

Two motivations, one experiment:

1. **Overdetermination.** The deployed XGBoost model fits the training data well but degrades on out-of-window data — the textbook signature of a feature set too large for the signal the data carries. The calibration arc in [tj_calibration/tijuana-dispersion-experiments](../../../tj_calibration/tijuana-dispersion-experiments/docs/calibration_status.md) empirically dismissed 9 of the 44 production features (all 6 SBIWTP + 3 flow magnitude derivatives), plus identified several tree-redundant wind aggregates. Does removing them actually improve held-out performance?

2. **Complaint-rate visibility.** Monitor logs show H₂S complaints rising in the **5–10 ppb band**, with **8 ppb** a known complaint-trigger level. Any feature change has to be evaluated not just at the alert thresholds (30, 100 ppb) but also at the lower thresholds that drive public-facing complaints. Does any trim silently degrade complaint-rate detection?

## Bottom line

Trimming to 33 features (`MODEL_FEATURES_EVIDENCE`) **improves alert recall and ties complaint-rate recall** — no trade-off. Deeper trims (19, 11) fail both bars. See [RESULTS.md](RESULTS.md) for numbers; recommendation is a Phase 2 PR promoting `MODEL_FEATURES_EVIDENCE` to be the production default.

## Approach

Train a production-config XGBoost regressor on Berry (NESTOR-BES) for each of four candidate feature sets and score all four against the calibration-aligned harness:

| Set | Features | Composition |
|---|---|---|
| **D - Baseline** | 44 | current `MODEL_FEATURES` (control) |
| **C - Evidence-only** | 33 | D minus 6 SBIWTP + 3 flow derivatives + 2 wind 4h rolling |
| **B - Lean** | 19 | C minus derived interactions + remaining wind rolling + lower-importance weather + `hour_sin`/`hour_cos` |
| **A - Minimal** | 11 | calibration's load-bearing core only — `is_night` carries hour-of-day signal |

Each set is defined as a module-level constant in [h2s.constants](../../projects/h2s/src/h2s/constants.py) and pinned by [tests/test_constants.py](../../projects/h2s/tests/test_constants.py) (14 tests).

The script calls `get_xgb_regressor()` directly (bypassing `train_and_select`) so the comparison isolates **feature-set effects, not selector effects**. Chronological 70/30 split per [`calibration_eval.chronological_split`](../../projects/h2s/src/h2s/training/calibration_eval.py) — same split boundary used by the [PR #26 retrain](../2026-05-20_calm_night_feature_refresh/RESULTS.md).

### Scoring

Each candidate evaluated at **all four categorical boundaries**:

| Threshold | Boundary | What it measures |
|---|---|---|
| **5 ppb** | green / yellow_low | Complaint-rate floor — any caution-level event |
| **10 ppb** | yellow_low / yellow_high | Complaint-rate band — includes the 8 ppb known trigger |
| **30 ppb** | yellow_high / orange | Operational watch alert |
| **100 ppb** | extreme | Critical alert (calibration's headline threshold) |

Reported per slice: overall, calm (`stable_atm=1`), windy (`stable_atm=0`). Persistence floor (`h2s_lag_1h → ppb`) included as reference.

### Acceptance criteria

A candidate X < D wins if all three hold versus baseline:

- `recall@30(X) ≥ recall@30(D) − 0.02` (≤ 2 pp drop at the watch threshold)
- `recall@100(X) ≥ recall@100(D) − 0.05` (≤ 5 pp drop at the critical threshold)
- `Spearman(X) ≥ Spearman(D) − 0.02`

Recall@5 and recall@10 are **reported but advisory** — they let the reviewer double-check that no candidate silently degrades complaint-rate skill, but don't gate the decision. Decision rule: **smallest winning candidate ships**; if none wins, keep `MODEL_FEATURES` (D).

## How to reproduce

```bash
cd projects/h2s

# 1. Tests for the candidate-set definitions
uv run pytest tests/test_constants.py -v

# 2. Eval-harness regression guard (PR #26's tests)
uv run pytest tests/test_calibration_eval.py tests/test_feature_builder.py tests/test_train_and_select.py -q

# 3. Run the ablation (trains 4 XGBoost regressors, ~30s on a laptop)
uv run python ../../experiments/2026-06-10_feature_trim_berry/feature_ablation.py \
  --out ../../experiments/2026-06-10_feature_trim_berry/output/ablation.json

# 4. Dagster definitions still load (production code path untouched)
uv run dg check defs
```

Sanity check: the baseline (D) numbers should reproduce PR #26's retrain — Spearman ≈ 0.78, recall@30 ≈ 0.80, recall@100 ≈ 0.64 on Berry, n_test = 3559.

## Dependencies

- **Data**: `data/modeldata_h2s_nofill.parquet` (NESTOR-BES rows, ~11.9k after filtering)
- **Library**: [`h2s.training.calibration_eval`](../../projects/h2s/src/h2s/training/calibration_eval.py) and [`h2s.training.multi_station_trainer`](../../projects/h2s/src/h2s/training/multi_station_trainer.py) (both shipped in PR #26; this experiment extends `eval_regressor` and `train_and_select` to also handle recall@5 and recall@10)
- **No new external dependencies.**

## Phase boundaries

- **Phase 1 (this experiment): ablate + recommend.** Done. Shipped in [PR #27](https://github.com/TheResilientCollective/tj_h2s_prediction/pull/27).
- **Phase 2 (PR #28): promote `MODEL_FEATURES_EVIDENCE` to the production default.** Done. `MODEL_FEATURES` is now 33 features; the old 44-feature list is preserved as `MODEL_FEATURES_LEGACY` for deployed-model backward compatibility. `feature_builder.ensure_base_features` was left alone (its idempotent branches still compute flow_log/SBIWTP defaults, which legacy preprocessing paths still read). The `multi_station_training_job` → `station_deployment_job` cycle was not part of PR #28 — that's a separate operational step.
- **Phase 3 (later): generalize.** Run the same ablation on IB-Civic-Ctr and San-Ysidro (current test slice has too few positives@100 there for reliable evidence); run it on the 3-class classifier path that the production hourly pipeline uses.

## Files

### Scripts

- `feature_ablation.py` — trains all four candidates, scores via `calibration_report`, prints the scoreboard, persists `ablation.json`
- `forecast_candidates.py` — trains the Evidence and Lean candidates, evaluates each on the same held-out 30 %, then applies each to the latest 15-min forecast input to produce a preview forecast. Writes per-candidate CSV + a self-describing `run_manifest.json`. Intended for dashboards that need to display the forecast alongside model provenance.

### Outputs (`output/`, tracked in git)

The output directory is **committed** so every code change carries a snapshot of what it produces. When the model code shifts, re-run and commit again; the git diff shows what moved.

- `ablation.json` — full per-candidate evaluation payload from `feature_ablation.py` (features list, fit metrics, calibration report, feature importance, acceptance verdict)
- `forecast_evidence.csv` — Evidence-only (33 feat) preview forecast: `time, predicted_ppb, predicted_category` per 15-min timestep
- `forecast_lean.csv` — Lean (19 feat) preview forecast, same schema
- `run_manifest.json` — self-describing snapshot ([schema](#run-manifest-schema)): git SHA + branch + dirty flag, input/output paths and time ranges, training/test split metadata, XGBoost hyperparameters, per-candidate feature lists and holdout metrics, forecast summary (max/mean ppb, category counts)

### Run manifest schema

The manifest is the contract with downstream consumers (e.g. a forecast dashboard). It answers "what produced this snapshot?" without requiring the consumer to read the source. Top-level shape:

```
{
  schema_version: "1.0",
  run: { produced_at, git_sha, git_branch, git_dirty, station },
  inputs: { training_parquet, forecast_input },
  split: { method, train_fraction, train_n, test_n, test_time_range },
  categorical_thresholds_ppb: { yellow_low_min, yellow_high_min, orange_min, critical_min },
  eval_thresholds_ppb: [5, 10, 30, 100],
  candidates: [
    {
      code, name, n_features, features: [...],
      hyperparameters: { ...XGBoost params... },
      holdout_metrics: { mae, rmse, r2, recall_5/10/30/100, n_positives_5/10/30/100, n_test },
      forecast_output: "forecast_<slug>.csv",
      forecast_summary: { n_predictions, max_predicted_ppb, mean_predicted_ppb, category_counts }
    }, ...
  ]
}
```

The `git_dirty` flag tells the consumer whether the producing code was committed when the snapshot was taken (`false` = exactly the state at `git_sha`; `true` = the producer had uncommitted edits and the snapshot may not be reproducible from `git_sha` alone).
