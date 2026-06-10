# Results — calm_night_feature_refresh

**Run on:** 2026-05-20
**Runtime:** ~30 seconds total (both scripts)
**Outputs:** in `output/` (not committed)
**Status:** **partial Phase A** — evaluation rails + one feature delta + selector fix. **Does not satisfy the original plan's full Phase A acceptance gate.**

## Question

(Restated from README.) Can the production XGBoost classifier beat persistence at the operational alert thresholds (30 ppb, 100 ppb) at Berry (NESTOR-BES), and does adding a regime-conditional wind feature improve calm-night recall?

## What we did

Built a calibration-aligned evaluation harness — Spearman + recall@{30, 100} + regime stratification (`stable_atm`) + per-site decomposition — as a new module [`h2s.training.calibration_eval`](../../projects/h2s/src/h2s/training/calibration_eval.py). Established the persistence floor (`h2s_lag_1h → ppb` as the prediction) across all 3 sites and both regimes. Then a Berry-only XGBoost regression head-to-head: 43-feature baseline vs 44-feature candidate (added `wind_x_stable_atm = wind_speed_10m × stable_atm` — a different feature than the plan asked for). Found that `train_and_select`'s R²-based selector was hiding a 32 pp recall@100 gap between RF and XGB; fixed the selector default to `recall_30`. Ran the head-to-head both before and after the fix.

## Key findings

1. **The persistence floor on Berry is very high: Spearman 0.822, recall@30 0.574, recall@100 0.390.** Any trained model has to beat this on the metrics that matter, and Spearman alone is *not* the right yardstick — persistence wins it.
2. **The production-config XGBoost regressor beats persistence decisively at the alert thresholds**: recall@30 0.825 (+25 pp), recall@100 0.675 (+28 pp). The model's value-add is operational alert recall, not bulk rank-skill.
3. **`train_and_select` was silently picking RandomForest over XGBoost** on `multi_station_trainer.py` because RF won R² by ~0.08 — while XGB won recall@100 by 32 pp on the same test set. The R²-based selector is mis-aligned for an alert system. The fix (default `selection_metric='recall_30'`) flips the choice; deployed daily-station models are likely RF-where-they-should-be-XGB today.
4. **The added `wind_x_stable_atm` feature is the #5 feature by XGBoost importance (0.031) but does not earn its keep on the calibration metrics.** It moves R² in the right direction (+0.046) but recall@30 and recall@100 both regress slightly. Honest negative result on the feature itself.

## Numbers

### Persistence floor (h2s_lag_1h → ppb, chronological 70/30, test window 2025-06 → 2026-04)

| Slice | n | Spearman | recall@30 (n_pos) | recall@100 (n_pos) |
|---|---|---|---|---|
| Overall | 10,770 | 0.821 | 0.510 (304) | 0.365 (85) |
| NESTOR-BES (Berry) | 3,589 | 0.822 | 0.574 (223) | 0.390 (77) |
| IB CIVIC CTR | 2,687 | 0.816 | 0.389 (36) | 0.000 (3) |
| SAN YSIDRO | 4,494 | 0.820 | 0.289 (45) | 0.200 (5) |
| Calm (stable_atm=1) | 3,627 | 0.763 | 0.507 (203) | 0.367 (60) |
| Windy (stable_atm=0) | 7,143 | 0.841 | 0.515 (101) | 0.360 (25) |

### Berry head-to-head — XGBoost-forced regression (n_test=3559)

| Slice | Model | Spearman | recall@30 | recall@100 |
|---|---|---|---|---|
| Overall | persistence | **0.822** | 0.574 | 0.390 |
| | OLD XGB (43 feat) | 0.773 | **0.825** | **0.675** |
| | NEW XGB (44 feat, +wind_x_stable_atm) | 0.782 | 0.803 | 0.636 |
| Calm | persistence | 0.754 | 0.599 | 0.389 |
| | OLD XGB | 0.785 | **0.878** | **0.741** |
| | NEW XGB | 0.792 | 0.857 | 0.722 |
| Windy | persistence | 0.846 | 0.526 | 0.391 |
| | OLD XGB | 0.736 | 0.724 | **0.522** |
| | NEW XGB | 0.746 | 0.697 | 0.435 |

Regression-fit metrics confirm the new feature *does* improve absolute prediction (OLD R² 0.433 → NEW 0.479; OLD MAE 7.45 → NEW 7.05) but the gain doesn't carry to the alert thresholds.

### Selector fix — same data, two selector defaults

| `train_and_select` mode | OLD chosen | NEW chosen | OLD recall@100 | NEW recall@100 |
|---|---|---|---|---|
| Before fix (R²-based) | RandomForest | RandomForest | 0.351 | 0.325 |
| After fix (recall_30 default) | XGBoost | XGBoost | **0.675** | **0.636** |

The +32 pp recall@100 lift is unlocked by changing one default in `multi_station_trainer.py`. It was always available in the underlying XGBoost model; the selector was hiding it.

## What this means

The persistence-vs-XGB picture is healthier than calibration's most pessimistic capstone suggested: XGB *does* materially beat persistence at the alert boundaries even though it loses on overall Spearman. Calibration's framing — "magnitude is reachable only autoregressively" — is consistent with this: persistence wins continuous Spearman because the rank-order at the bulk is dominated by lag-1h carryover. But classification at a fixed threshold is a different question, and XGB has real skill there.

The selector fix is the most operationally consequential finding. It's silently hiding a 32 pp recall@100 lift on the deployed per-station models. Whether the new wind feature ships or not, the selector change alone should land.

The added feature itself is a wash on the calibration-aligned metrics. The plan's three calibration-aware features (`temp_stable_interaction`, `wind_direction_*_calm`) and two physics features (`physics_is_stagnation`, `physics_temp_emission_factor`) are still untried. The empirical case that **any** simple wind × regime interaction will move the alert metrics is now weaker than the calibration log suggested it would be — but `temp_stable_interaction` is the strongest of the candidates per calibration finding #1 and is still untested here.

## What should be done next

In rough order of value:

1. **Reviewer decision: keep or revert the two production code changes** (selector default; `wind_x_stable_atm`). My recommendation:
   - Keep the selector fix — it's correct, tested (8 tests), and unlocks 32 pp recall@100. If reviewer is risk-averse, gate it behind `selection_metric='recall_30'` explicit-opt-in instead of changing the default.
   - Revert `wind_x_stable_atm` — the empirical case for it is weak and it deviates from the plan's specified feature set.
2. **Execute the plan's full Phase A as a follow-up experiment**: 3 stations × 3-class XGBClassifier × the plan's 5 features (3 calibration-aware + 2 physics from `tijuana_dispersion @ v0.4.0`) × baseline from `H2SPredictor.from_s3(...)`. The evaluation harness is in place; the missing pieces are the dep pin, the features, the classifier path, and the S3 candidate upload.
3. **Re-run `multi_station_training_job` once the selector ships** — the deployed per-station models are likely RF-where-they-should-be-XGB. Even with no feature changes, that retraining alone should improve operational alert recall.
4. **Open the upstream parquet gap issue** as the plan A.9 requires (corrupt-CSV-as-parquet status of `modeldata_forecast_15min`; tz/schema parity; independent anemometer ask).

## Limitations / caveats

- **Single station tested.** Berry only. IB Civic Ctr and San Ysidro could have very different patterns; the persistence floor already shows recall@100 differs dramatically by site (IB has only 3 positives in the test slice).
- **Regression, not classification.** The deployed model is a 3-class XGBClassifier; my comparison is XGBoost-regression. The conclusion that "XGB beats persistence at recall@30" generalizes intuitively, but isn't formally shown on the production model class.
- **Compared against a fresh retrain of the 43-feature set, not the deployed S3 model.** The plan was explicit that Baseline = `H2SPredictor.from_s3(...)`. The currently-deployed model could have materially different hyperparameters / training window than my fresh retrain.
- **Single random seed, single chronological split, no cross-validation.** Production training runs are also single-split, so this matches deployment — but small per-site samples in the >100 ppb tier (n=77 at Berry, n=3 at IB, n=5 at SY) make any recall@100 number noisy.
- **The new feature was added without the plan's prescribed set.** Adds a 44th feature that does not earn its keep on the alert metrics. Reverting it removes the change cleanly.
- **Production code changes outside Phase A's scope.** `multi_station_trainer.train_and_select`'s default changes; `constants.MODEL_FEATURES` grows from 43 to 44. Both are picked up automatically by the next `multi_station_training_job` run unless reverted.
- **No data_contract_check.py.** The plan asks for md5 parity check vs `tj_calibration`'s copy. Not implemented this session.
- **No upstream parquet issue opened.** Plan A.9 not done.
- **No S3 candidate upload.** Plan A.8 not done. No candidate models exist on S3; the deployed production keys are unchanged (confirmed read-only).

## Files

- `persistence_baseline.py` — chronological 70/30 split + persistence-as-prediction, per-site + per-regime
- `retrain_compare_berry.py` — Berry-only XGBoost retrain head-to-head, OLD vs NEW vs persistence; supports `--mode {xgb,auto}`
- `output/` (gitignored) — JSON dumps from both scripts

## Library code changed (outside this folder)

- `projects/h2s/src/h2s/training/calibration_eval.py` (new) — Spearman, recall@threshold, persistence, chronological split, CalibrationReport
- `projects/h2s/src/h2s/training/feature_builder.py` (modified) — added `wind_x_stable_atm`
- `projects/h2s/src/h2s/training/multi_station_trainer.py` (modified) — `train_and_select` selector refactor; `eval_regressor` now reports recall@30/100
- `projects/h2s/src/h2s/constants.py` (modified) — added `wind_x_stable_atm` to `CORE_FEATURES`; `MODEL_FEATURES` 43 → 44
- `projects/h2s/tests/test_calibration_eval.py` (new) — 16 tests
- `projects/h2s/tests/test_feature_builder.py` (new) — 7 tests
- `projects/h2s/tests/test_train_and_select.py` (new) — 8 tests

All Dagster definitions still load (`uv run dg check defs`).
