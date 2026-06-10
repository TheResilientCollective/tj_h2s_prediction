# feature_trim_berry

**Date:** 2026-06-10
**Status:** done
**Author:** session-handoff

## Question

The deployed XGBoost model fits the training data well but degrades on out-of-window data — the textbook signature of an overdetermined model with too many features for the signal the data carries. The calibration arc in
[tj_calibration/tijuana-dispersion-experiments](../../../tj_calibration/tijuana-dispersion-experiments/docs/calibration_status.md)
empirically dismissed several feature families (all 6 SBIWTP features, flow magnitude derivatives, wind² volatilization-style terms). Does removing those features improve the model on the calibration-aligned alert metrics (Spearman + recall@{30, 100}), or are they providing some hidden signal we'd lose?

## Approach

Train a production-config XGBoost regressor on Berry (NESTOR-BES) for four candidate feature sets and score all four against the calibration-aligned harness from PR #26:

| Set | Features | Composition |
|---|---|---|
| **D - Baseline** | 44 | current `MODEL_FEATURES` (control) |
| **C - Evidence-only** | 33 | D minus 6 SBIWTP + 3 flow derivatives + 2 wind 4h rolling |
| **B - Lean** | 19 | C minus all derived interactions + remaining wind rolling + lower-importance weather + `hour_sin`/`hour_cos` |
| **A - Minimal** | 11 | calibration's load-bearing core only — `is_night` carries hour-of-day signal |

Each set is defined as a module-level constant in
[h2s.constants](../../projects/h2s/src/h2s/constants.py) and verified by
[tests/test_constants.py](../../projects/h2s/tests/test_constants.py) (14 tests).

The script bypasses `train_and_select` (calling `get_xgb_regressor()` directly)
so the comparison isolates feature-set effects, not selector effects.

Chronological 70/30 split per `calibration_eval.chronological_split` —
same boundary used by the
[PR #26 retrain experiment](../2026-05-20_calm_night_feature_refresh/RESULTS.md).

### Scoring

Each candidate scored at **all four categorical boundaries** — recall at 5 ppb
(green/yellow_low), 10 ppb (yellow_low/yellow_high), 30 ppb (watch), and
100 ppb (critical). The 5 / 10 ppb cuts surface complaint-rate behavior:
monitor logs show complaints rising in the 5-10 ppb band, with 8 ppb known
as a complaint-trigger level. Recall@30 and recall@100 gate acceptance;
recall@5 and recall@10 are reported but advisory.

### Acceptance criteria

For each X < D, X "wins" if all three hold:
- `recall@30(X) ≥ recall@30(D) − 0.02` (≤ 2 pp drop at the watch threshold)
- `recall@100(X) ≥ recall@100(D) − 0.05` (≤ 5 pp drop at the critical threshold)
- `Spearman(X) ≥ Spearman(D) − 0.02`

Decision: **smallest winning set ships.** If none win, keep `MODEL_FEATURES` (D).

## How to reproduce

```bash
cd projects/h2s

# Tests for the candidate set definitions
uv run pytest tests/test_constants.py -v

# Regression guard: PR #26's eval harness still green
uv run pytest tests/test_calibration_eval.py tests/test_feature_builder.py tests/test_train_and_select.py -q

# Run the ablation
uv run python ../../experiments/2026-06-10_feature_trim_berry/feature_ablation.py \
  --out ../../experiments/2026-06-10_feature_trim_berry/output/ablation.json

# Sanity: baseline (D) should reproduce ~Spearman 0.78 / recall@30 0.80 /
# recall@100 0.64 from PR #26's RESULTS.md.

# Dagster definitions still load (no production code path touched)
uv run dg check defs
```

## Dependencies

- Real data: `data/modeldata_h2s_nofill.parquet`
- Library: `h2s.training.calibration_eval`, `h2s.training.multi_station_trainer`
  (both shipped in PR #26)
- No external dependencies added.

## Notes

This is **Phase 1** of the feature reduction. Phase 2 (promote the winner to
`MODEL_FEATURES` and retrain production per-station models) is a separate
decision and PR. Phase 3 (extend to IB-Civic-Ctr, San-Ysidro, and the 3-class
classifier path) is a follow-up. See [RESULTS.md](RESULTS.md) for the actual
numbers and recommendations.
