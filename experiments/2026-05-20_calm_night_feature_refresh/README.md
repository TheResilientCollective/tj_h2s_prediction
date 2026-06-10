# calm_night_feature_refresh

**Date:** 2026-05-20
**Status:** **partial** — Phase A evaluation rails landed, full plan not executed
**Author:** session-handoff to reviewer

## Question

Do the calibration findings from `tj_calibration` (temperature as the dominant exogenous driver of calm-night H₂S, autoregressive dominance of magnitude, persistence ≈ 0.70 Spearman at Berry, exogenous recall@100 ≈ 0) translate into measurable improvements when folded into the production XGBoost classifier here? Specifically, can the production model beat persistence at the alert thresholds (30 ppb watch, 100 ppb critical), and does adding regime-conditional features lift calm-night recall without inflating overall false-alarm rate?

## Approach

Two-step research arc, each a separate artifact in this folder:

1. **Persistence baseline** ([persistence_baseline.py](persistence_baseline.py)) — establishes the autoregressive floor. Chronological 70/30 split per site. `h2s_lag_1h → ppb` used directly as the prediction. Outputs Spearman, recall@30, recall@100 per site and per regime (`stable_atm` ∈ {0, 1}). This is the number every trained model must beat.
2. **Berry XGB head-to-head** ([retrain_compare_berry.py](retrain_compare_berry.py)) — single-site, regression-only retrain on `NESTOR - BES` with two feature sets (43 baseline, 44 with `wind_x_stable_atm`), forced through the production-config XGBoost regressor. Compared side-by-side against persistence using the same evaluation harness.

Both scripts use the new [`h2s.training.calibration_eval`](../../projects/h2s/src/h2s/training/calibration_eval.py) module — Spearman, recall@threshold, regime-stratified `CalibrationReport`, and `persistence_prediction`. Tests in [tests/test_calibration_eval.py](../../projects/h2s/tests/test_calibration_eval.py).

## How to reproduce

```bash
cd projects/h2s

# Persistence floor — all 3 sites, regime-stratified
uv run python ../../experiments/2026-05-20_calm_night_feature_refresh/persistence_baseline.py \
    --out ../../experiments/2026-05-20_calm_night_feature_refresh/output/persistence.json

# Berry XGB head-to-head — auto mode picks XGB now that the selector is recall-aware
uv run python ../../experiments/2026-05-20_calm_night_feature_refresh/retrain_compare_berry.py \
    --mode auto \
    --out ../../experiments/2026-05-20_calm_night_feature_refresh/output/berry_compare.json

# Test suite — 31 new tests, all green
uv run pytest tests/test_calibration_eval.py tests/test_feature_builder.py tests/test_train_and_select.py -v
```

## Dependencies

- Real data: `data/modeldata_h2s_nofill.parquet` (43,127 rows, Nov 2023 → Apr 2026, 3 sites).
- Library: the new `h2s.training.calibration_eval` module (added this session).
- Did **not** add `tijuana-dispersion @ v0.4.0` — the plan's hybrid physics features were not implemented.

## Deviations from the plan

This experiment **does not match** the original plan's full Phase A scope. The deltas are documented in detail in [RESULTS.md](RESULTS.md) — read that before reviewing. Headline gaps: only Berry tested (not 3 stations), regression-only (not 3-class XGBClassifier), no S3 baseline-from-prod comparison, no candidate-model S3 upload, no `tijuana_dispersion` hybrid features, no upstream-parquet issue opened. **A different single feature was added (`wind_x_stable_atm`) where the plan called for five** (three calibration-aware + two physics).

In addition, two production code paths were modified outside Phase A's "don't touch production" boundary: the selector default in [`train_and_select`](../../projects/h2s/src/h2s/training/multi_station_trainer.py) and the addition of `wind_x_stable_atm` to `CORE_FEATURES` in [`constants.py`](../../projects/h2s/src/h2s/constants.py). Both have empirical justification — see RESULTS.md.

## Notes

The persistence baseline ([calibration_eval.persistence_prediction](../../projects/h2s/src/h2s/training/calibration_eval.py)) and the calibration-aligned scoreboard are general-purpose infrastructure. They're located in `experiments/` for the historical record of this experiment, but the underlying module belongs at the library layer and is already there.
