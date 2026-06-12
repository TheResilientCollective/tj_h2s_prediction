# Workplan — Nowcast / Nearcast / Forecast rework

Implements [rename.md](rename.md). Decisions locked 2026-06-12:

- **Cascading tiers** (all probabilities at NESTOR-BES, ppb not ppm):
  Tier 1 = P(H₂S>5) > 0.5 in **nowcast** → run nearcast.
  Tier 2 = P(>10) > 0.5 in **nearcast** → run forecast.
  Tier 3 = P(>30) > 0.5 in **forecast** → health team + all 3 sites.
- **`clf_30ppb`** becomes a fourth training task per station per variant
  (clean calibrated probability for Tier 3; the regression-threshold and
  reuse-clf_10ppb alternatives were rejected).
- **Both variants reported**: every product runs Evidence (33 feat) and
  Lean (19 feat) and reports both side by side. **Evidence drives the
  triggers** (production default); Lean is comparison context.
- **Evolve `tiered_alerts` in place** — the existing Tiers 1–3 defs are
  reworked into the new cascade; no parallel alert system.

Honest scope boundary, inherited from the tj_calibration arc: at the
forecast tier (6–24 h) magnitude skill is exogenous-bounded
(Spearman ceiling ≈ 0.33 on calm-night extremes). The forecast product is
a risk-ranker at that horizon, and the validation store (Phase 5) is
expected to show that. This is by design, not a defect.

---

## Phase 1 — Foundations: `clf_30ppb` + product constants *(small PR)*

- Add `exceed_30` target in `prepare_multi_station_features`; add
  `clf_30ppb` to the per-station task list (training, report, deployment).
  Per station per variant → 24 pickles per full cycle (up from 18).
  NESTOR has ~700 h >30 ppb; IB/SY will be weaker — report per-site
  honestly.
- Product constants in `constants.py`: nowcast (0–3 h), nearcast (3–6 h),
  forecast (6–24 h) windows; cascade trigger table
  (product, threshold_ppb, prob_cutoff).
- Verify: training cycle on `nestor_bes` produces
  `clf_30ppb_{evidence,lean}.pkl`; training report carries its AUC/recall;
  S3 smoke test.

## Phase 2 — Model versioning + human-in-the-loop promotion *(medium PR)*

- Deployment writes an immutable archive
  `models/archive/{version_tag}/stations/{key}/...`
  (version = UTC timestamp + git sha). The existing `models/stations/...`
  prefix stays as the production pointer — daily pipeline unchanged.
- `model_version` stamped into `deployment_metadata.json` and carried into
  every prediction-run output (the spec's "runs should store the model
  version").
- Monthly retrain posts a Slack report: new-vs-production on the
  calibration-aligned metrics (Spearman, recall@5/10/30/100), a
  recommendation, and the promote command.
- `promote_station_models_job`: copies `archive/{version}` → production.
  Running it IS the approval (PR #29 pattern).
- Verify: train → archive → promote → daily pipeline loads it → version
  visible in outputs; old versions re-loadable for replaying analyses.

## Phase 3 — Recursive inference engine + the three products *(core PR)*

- New module `h2s/forecasting/recursive.py`: hour-by-hour loop — predict
  hour t, append the prediction to the H₂S series, rebuild
  `h2s_lag_1h/3h/6h` + `h2s_rolling_6h/24h` from the blended
  actual+predicted series, predict t+1. Boundary construction (e.g.
  rolling_24h at lead 6 = 18 actual + 6 predicted hours) pinned by unit
  tests FIRST (tests-first; synthetic inputs only for mechanics tests,
  never for training/eval claims).
- Product assets: `nowcast` (actual lags only — successor of the current
  hourly behavior, plus P>30), `nearcast` (recursion seeded at last
  actual), `forecast` (full recursion). Each runs Evidence + Lean, tagged
  with model_version + variant.
- Every run appends hive-partitioned parquet rows:
  (run_ts, product, site, lead_hour, variant, model_version, h2s_pred,
  p5, p10, p30) — the substrate for Phase 5.

## Phase 4 — Cascade + alerting rework *(evolve `tiered_alerts`)*

- Hourly nowcast at NESTOR-BES. Cascade per the locked tier decisions;
  Tier 1 report = nowcast + nearcast + estimated hours of H₂S in next 6 h
  (ops Slack); Tier 2 adds Slack email (health-team email = future hook);
  Tier 3 = health team + all-3-sites report.
- Observed-exceedance state machine: open at observed >10 ppb (Slack),
  close after 2 h below → close-out report to Slack + S3, including hours
  above yellow/orange in the past 12 h.
- Existing observation tiers (watch 30 / critical 100) unchanged.

## Phase 5 — Validation store + accuracy reporting

- Daily join of stored product rows vs actuals → one consolidated
  accuracy parquet in S3, REBUILDABLE by a backfill job that recomputes
  everything from the stored runs.
- Per-hour report: forecasted probability vs measured outcome;
  per-product skill curves by lead hour (calibration harness:
  Spearman + recall@5/10/30/100 per lead-hour bucket, Evidence vs Lean).

## Phase 6 — Cadence + cutover

- Hourly nowcast schedule live (requires confirming the obs feed updates
  hourly — open item).
- Daily pipeline's decay-based forecast features
  (`_engineer_forecast_features`) retired in favor of the recursive
  engine; `forecast_prediction_job` naming reconciled with "nowcast".

---

## Open items (not blocking Phase 1)

1. **Hourly data freshness** — hourly nowcast only makes sense if the obs
   parquet / APCD feed updates hourly. Check before Phase 6.
2. **Health-team email** — Slack first; real email (SES/SMTP) later per
   the spec's "future option".
3. **0.5 cutoffs** — starting point; Phase 5 accuracy data tunes them
   (the prior `PROB_30_ALERT` was 0.35, so 0.5 at Tier 3 is a deliberate
   tightening).
4. **Trigger source** — Evidence-only triggers for now; revisit if Phase 5
   shows Lean catching events Evidence misses.

Sequencing is linear (1 → 2 → 3 → 4 → 5 → 6); versioning lands before the
products so every run is stamped from day one. Each phase is its own PR
off master.
