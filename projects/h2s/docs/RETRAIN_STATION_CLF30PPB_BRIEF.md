# Handoff Brief: Retrain SAN_YSIDRO & IB_CIVIC_CTR clf_30ppb

**Status:** Plan approved, ready to execute in cloud Claude Code session.
**Date:** 2026-06-19
**Working branch start point:** `master` @ commit `a0fc80a`
**Suggested worktree:** `feature/retrain-station-clf30ppb`

---

## 1. Background — Why This Work Exists

The daily station forecast pipeline uses per-station classifiers for the 30 ppb
orange threshold (`clf_30ppb_evidence.pkl`). Three sites are deployed:

| Station | clf_30ppb backtest recall | Status |
|---------|--------------------------|--------|
| NESTOR__BES | **81%** | ✓ Working |
| IB_CIVIC_CTR | **50%** | ⚠ Degraded from 74% in-sample |
| SAN_YSIDRO | **0%** | ❌ Broken |

NESTOR is the main operational site and is fine. SAN_YSIDRO and IB_CIVIC_CTR
need either retraining, threshold tuning, or to be dropped from the 3-class
logic (currently they fall back to `prob_10 > 0.5 → orange`).

Backtest source: `s3://test/tijuana/forecast/backtest/2026-05/backtest_results.json`
Training reports: `s3://test/tijuana/forecast/models/stations/{STATION}/training_report.json`

### What we already verified is NOT the problem
- Models load correctly from S3 (evidence + lean variants both present)
- Features match training (33 evidence, 19 lean) — confirmed via features_evidence.json
- The backfill script bug (lag feature corruption) is **fixed** in commits below
- Threshold sweep (`scripts/sweep_prob_30_threshold.py`) shows NESTOR is great at any threshold

### Recent fixes already on master
1. `4e3feb8` — Fix `monthly_performance_viz` to skip days with no matches
2. `2eeb457` — Load `clf_30ppb` in backfill hindcast
3. `9cf9f97` — Lower `PROB_30_ALERT` from 0.35 → 0.25
4. `32ea847` — Disable broken `clf_30ppb` for non-NESTOR (uses 2-class fallback)
5. `f25785c` — Use real observations (not interpolated) in backfill
6. `d64199e` — Preserve real H2S lag features in backfill (BA jumped 0.35 → 0.77)
7. `a0fc80a` — Add `sweep_prob_30_threshold.py` for tuning

---

## 2. Goal & Acceptance Gate

**Goal:** Get usable 50-70% orange recall for SAN_YSIDRO and IB_CIVIC_CTR at
deployment threshold `PROB_30_ALERT = 0.25`.

**Acceptance gate (per station):**
```
recall@(threshold=0.25) ≥ 0.50   AND   FAR ≤ 0.10
```
Measured on the monthly walk-forward backtest (the same harness that produces
`backtest_results.json`).

**Out of scope:** Per-station PROB_30_ALERT (would require refactor of
`classify_risk` to accept station-aware thresholds — defer to follow-up).

---

## 3. Approach: Targeted SMOTE on clf_30ppb

The user picked **Option A: SMOTE oversampling on the 30 ppb minority class
only**. Keep `regression`, `clf_5ppb`, `clf_10ppb` unchanged.

### Why SMOTE
- SAN_YSIDRO clf_30ppb has AUC 0.959 (excellent discrimination) but recall 0.339
  at threshold=0.5 → calibration / class-imbalance problem, not capacity
- Only ~60 positive samples per non-NESTOR station vs ~340 for NESTOR
- `model_trainer.py` already has `apply_smote()` (BorderlineSMOTE with k_neighbors
  auto-fallback) but it's NOT plumbed through `train_and_select()` used by the
  station training pipeline

### Cheap pre-check FIRST: Experiment C (no training)
Before doing any retraining, run the existing
`scripts/sweep_prob_30_threshold.py` BUT modify it to:
1. Report metrics per-station (not aggregated)
2. Use the SAN_YSIDRO clf_30ppb model (currently disabled for non-NESTOR in
   `backfill_validation.py` line ~373)

If SAN_YSIDRO's 0.96 AUC means a lower threshold (e.g., 0.10) recovers recall
to ≥ 0.50, no retraining needed for that station — just document the
per-station threshold as a known limitation.

**Run this first.** If it works, you skip 80% of the work.

---

## 4. Files to Modify (if retraining is needed)

### 4a. Plumb SMOTE through the trainer

**`projects/h2s/src/h2s/training/multi_station_trainer.py`** — function
`train_and_select` (line 256)
- Add parameter: `use_smote_on_minority: bool = False`
- Inside the classifier branch (line 331 onward), if True call
  `model_trainer.apply_smote(X_train_df, y_train_series)` before fitting
- Note: `train_and_select` currently takes numpy arrays — wrap in pd.DataFrame
  for SMOTE then unwrap, or refactor to accept DataFrames

**`projects/h2s/src/h2s/training/model_trainer.py`** — `apply_smote` already exists
- Verify k_neighbors fallback works for very small minority (n_pos < 6) — code
  reads `k = min(5, n_minority - 1)` so should be OK
- Confirm SMOTE is applied per-fold inside CV, not globally — re-read function
  carefully; current behavior is likely "before CV" which leaks. Fix to apply
  inside fold loop (see `model_trainer.py` ~line 161 for existing fold pattern)

### 4b. Wire the flag through the orchestration

**`projects/h2s/src/h2s/defs/h2s_multi_station_training.py`**
- Add config field on the training asset:
  ```python
  config_schema={
      "enable_smote_clf_30ppb": dg.Field(bool, default_value=True,
          description="Apply BorderlineSMOTE to oversample 30ppb positives"),
  }
  ```
- Pass through to `train_and_select` for the clf_30ppb task ONLY (not other tasks)

**`projects/h2s/src/h2s/defs/h2s_backfill_pipeline.py`** — function
`_train_one_variant_local` (line 206)
- Same config field on the backfill training asset
- Pass `use_smote_on_minority=True` only when `task == "clf_30ppb"`

### 4c. Bookkeeping for replayability

**`archive_metadata.json`** carries an `algorithm_choices` field per
station/variant/task. Add a `smote_applied: bool` companion field so future
analysis can tell which models used SMOTE. See `docs/ALGORITHM_CHOICES.md`
for schema.

---

## 5. Evaluation Workflow

```bash
cd projects/h2s

# 1. Train new models on SAN_YSIDRO + IB_CIVIC_CTR (NESTOR unchanged baseline)
uv run dg launch --job station_model_training_job \
  --partition san_ysidro,ib_civic_ctr \
  --config-json '{"ops":{"h2s__per_station_trained_models":{"config":{"enable_smote_clf_30ppb":true}}}}'

# 2. Run backtest (monthly walk-forward) against new models
#    This populates s3://.../tijuana/forecast/backtest/{month}/backtest_results.json
uv run dg launch --job station_backfill_training_job  # or similar — check actual job name
uv run dg launch --job station_backtest_index_job

# 3. Pull backtest results and check gate
uv run python -c "
import urllib.request, json
url = 'https://oss.resilientservice.mooo.com/test/tijuana/forecast/backtest/2026-05/backtest_results.json'
data = json.loads(urllib.request.urlopen(url).read())
for st in ['SAN YSIDRO', 'IB CIVIC CTR']:
    ev = data['stations'][st]['variants']['evidence']['oos_metrics']
    print(f'{st}: prob_30 recall = {ev[\"prob\"][\"30\"][\"recall\"]:.3f}')
"

# 4. Re-run threshold sweep per-station
env $(grep -v '^#' ../../.env | xargs) uv run python scripts/sweep_prob_30_threshold.py

# 5. If gate met: deploy
uv run dg launch --job station_model_deployment_job \
  --partition san_ysidro,ib_civic_ctr
```

---

## 6. Decision Tree

```
Step 1: Experiment C (threshold sweep on existing models, per-station)
├── SAN_YSIDRO recall ≥ 0.50 at some threshold?
│   ├── YES → Document threshold, mark SAN_YSIDRO done, move to IB_CIVIC_CTR
│   └── NO  → Go to Step 2
└── (repeat for IB_CIVIC_CTR)

Step 2: SMOTE retraining (Experiment A)
├── New backtest meets gate?
│   ├── YES → Re-enable clf_30ppb for the station in backfill_validation.py,
│   │        deploy, document
│   └── NO  → Go to Step 3

Step 3: SMOTE + scale_pos_weight (Experiment B)
├── Gate met?
│   ├── YES → Deploy with elevated FAR caveat in docs
│   └── NO  → Drop clf_30ppb for that station permanently; document in
│            CLAUDE.md as known limitation. Keep 2-class fallback.
```

---

## 7. Reverting the "broken station" workaround on success

If retraining succeeds, remove the NESTOR-only guard in
`projects/h2s/scripts/backfill_validation.py` (around line 373):

```python
# Currently:
if station_key == "NESTOR__BES":
    prob_30 = clf30.predict_proba(X)[:, 1] if clf30 else np.zeros(len(X))
else:
    prob_30 = np.zeros(len(X))

# Change back to:
prob_30 = clf30.predict_proba(X)[:, 1] if clf30 else np.zeros(len(X))
```

---

## 8. Environment & Secrets

`.env` is at repo root (`/Users/valentin/development/dev_resilient/tj_h2s_prediction/.env`)
and is symlinked from `projects/h2s/.env`. Contains:
- `S3_BUCKET=test` (dev) — override with `S3_BUCKET=resilentpublic` for prod
- `PUBLIC_BUCKET=resilentpublic` — observation data always loaded from here
- `S3_ACCESS_KEY` / `S3_SECRET_KEY`
- `S3_ADDRESS=oss.resilientservice.mooo.com`

Run commands as:
```bash
cd projects/h2s
env $(grep -v '^#' ../../.env | xargs) uv run <command>
```

---

## 9. Risks / Things to Watch

1. **SMOTE leakage** — Standard SMOTE generates synthetic samples from
   k-nearest neighbors. With time-series data, neighbors are often adjacent
   in time. Verify `apply_smote` is called PER FOLD inside CV, not on the
   full train set, to avoid test-set leakage. Inspect
   `model_trainer.train_classifier` carefully.
2. **Precision collapse** — SMOTE inflates recall but precision often drops.
   The FAR ≤ 0.10 part of the gate exists to catch this. If FAR spikes,
   reduce the SMOTE ratio (e.g., 1.5x positives instead of 2x).
3. **Overfit on tiny minority** — With only ~60 positives, SMOTE could
   manufacture a smooth decision boundary that doesn't generalize. The OOS
   backtest is the only true test — don't rely on in-sample training metrics.
4. **NESTOR regression** — Don't retrain NESTOR. If you do, verify it still
   hits ≥ 80% backtest recall before merging.

---

## 10. References

- `CLAUDE.md` — repo overview, operational runbooks, S3 path conventions
- `docs/ALGORITHM_CHOICES.md` — schema for `algorithm_choices` field
- `projects/h2s/VALIDATION_AND_ACCURACY_REPORTING.md` — validation pipeline
- `projects/h2s/src/h2s/training/calibration_eval.py` — calibration harness
- Backtest report (current): `s3://test/tijuana/forecast/backtest/2026-05/backtest_results.json`
- Sweep script: `projects/h2s/scripts/sweep_prob_30_threshold.py`
- This brief: `projects/h2s/docs/RETRAIN_STATION_CLF30PPB_BRIEF.md`
