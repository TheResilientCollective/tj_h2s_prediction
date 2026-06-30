# Risk Classification — Two-Head Agreement, clf_30ppb Gating & Underfitting

> **New in the 2026-06 model revision** (`claude/h2s-model-underfitting`, PR #57).
> This document covers three coupled changes to how the per-station models turn
> raw outputs (`h2s_pred`, `p5`, `p10`, `p30`) into an operational hazard tier.

---

## 1. Primary Variant: Lean

Both variants (Evidence 33-feat, Lean 19-feat) are still trained and deployed in
parallel, but every operational surface now routes through **one** primary
variant, set by a single constant:

```python
# constants.py
PRIMARY_VARIANT = "lean"
```

Surfaces that follow `PRIMARY_VARIANT`:
- Daily forecast / dashboard (`h2s_daily_pipeline._VARIANT = PRIMARY_VARIANT`)
- Tier 1–3 cascade trigger (`cascade.TRIGGER_VARIANT = PRIMARY_VARIANT`)
- Heatmap board, digest, performance report, accuracy rollups

Evidence is still produced and shown **alongside** Lean (the cascade Slack report
renders both as `E … / L …`) as the published "not-overdetermined" cross-check.
Flipping `PRIMARY_VARIANT` back to `"evidence"` re-points every report and the
alert trigger in one edit — no path-string sprawl.

---

## 2. Per-Station clf_30ppb Gating

`clf_30ppb` is trained and deployed for **all** three stations (the artifact set
stays uniform), but the products/daily engines **emit `p30` only for stations we
trust**:

```python
# constants.py
CLF_30PPB_STATIONS = frozenset({"NESTOR - BES"})
```

For any station not in this set, `clf_30ppb` is passed as `None` to the recursive
engine, so `p30` comes out **NaN** — the exact same code path a genuinely missing
classifier already takes. NaN (not 0.0) is deliberate: it means the probability
head simply *cannot* reach ORANGE (`nan > cutoff` is `False`), rather than reading
as a confident "definitely not orange."

### Why gate?

Per-station ≥30 ppb recall is only dependable where there are enough orange
positives to learn from:

| Station | Training orange positives (≥30 ppb) | Base rate | Per-station recall | Status |
|---------|-------------------------------------|-----------|--------------------|--------|
| NESTOR-BES | ~344–550 | 4.6% | ~0.77 (→0.86–0.95 tuned) | **Emitted** |
| IB Civic Ctr | ~51–84 | 0.9% | ~0.70 | Suppressed (NaN) |
| San Ysidro | ~40–85 | 0.6% | 0.16–0.43 (collapses) | Suppressed (NaN) |

AUC stays high everywhere (0.96–0.98) — the *ranking* is fine — but at a fixed
operating point the positive-starved stations produce more noise than signal. So
IB Civic Center and San Ysidro fall back to the **≥10 ppb (yellow-high)** tier as
their top operational alert.

### Path to re-enabling

A **pooled cross-station ≥30 model** (train one classifier on all stations'
positives, optionally with a station feature) recovers recall on the sparse
stations. Experimental result (`experiment_pooled_clf30.py`):

| San Ysidro clf_30ppb | AUC | recall @ default | recall @ 0.25 cutoff |
|----------------------|-----|------------------|----------------------|
| Per-station (current) | 0.972 | 0.16 | 0.42 |
| Pooled (with station feat) | 0.982 | 0.64 | 0.84 |
| Pooled (no station feat) | 0.982 | 0.73 | 0.84 |

Once a pooled model lands, re-enabling a station is a one-line change: add its
`site_name` to `CLF_30PPB_STATIONS`.

---

## 3. Two-Head Agreement Classification

Each forecast row carries two **independent** hazard signals:

- **Magnitude head** — the regression `h2s_pred` (ppb), cut at 5/10/30
- **Probability head** — the classifiers `p5`/`p10`/`p30`, each compared to its
  alert cutoff

```python
# constants.py
def magnitude_risk_tier(h2s_pred) -> str:        # cuts 5/10/30 ppb
    ...
def probability_risk_tier(p5, p10, p30) -> str:  # cutoffs PROB_5/10/30_ALERT
    ...
```

Alert cutoffs for the probability head:

| Probability | Cutoff constant | Value | Tier it can raise |
|-------------|-----------------|-------|-------------------|
| p5  | `PROB_5_ALERT`  | 0.50 | YELLOW_LOW |
| p10 | `PROB_10_ALERT` | 0.50 | YELLOW_HIGH |
| p30 | `PROB_30_ALERT` | 0.25 | ORANGE |

### The agreement rubric

```python
def classify_risk_agreement(p5, p10, h2s_pred, p30=nan) -> (risk, risk_possible, risk_confidence):
    mag  = magnitude_risk_tier(h2s_pred)
    prob = probability_risk_tier(p5, p10, p30)
    lower  = min(mag, prob)   # by RISK_ORDER rank
    higher = max(mag, prob)
    if mag == prob:
        return lower, None,   "confirmed"
    return     lower, higher, "provisional"
```

- **Both heads at the same tier** → `risk` = that tier, `risk_possible = None`,
  `risk_confidence = "confirmed"`.
- **Heads disagree** → headline `risk` = the **lower** (confirmed) tier;
  `risk_possible` = the **higher** tier; `risk_confidence = "provisional"`.

The headline is never escalated on a single head alone — a hard alert needs
corroboration (fewer false ORANGE calls) — while a lone strong signal still
surfaces as `risk_possible` so it is never a silent miss.

### Composition with the gate

The two changes interlock: where `p30` is NaN (any non-`CLF_30PPB_STATIONS`
station), the probability head **cannot** reach ORANGE. So a magnitude-only
ORANGE there is always **provisional** (`risk = YELLOW_HIGH`,
`risk_possible = ORANGE`), never a confirmed hard alert.

### Row schema additions

Every daily-forecast row now carries:

| Field | Meaning |
|-------|---------|
| `risk` | Headline tier — the level both heads confirm |
| `risk_possible` | Higher single-head tier when heads disagree, else `None` |
| `risk_confidence` | `"confirmed"` or `"provisional"` |
| `risk_legacy` | Old single-call `classify_risk()` result, kept for continuity |
| `prob_30` | `p30 × 100`, or `None` for gated stations |

Station summaries add `hours_possible_orange` / `hours_possible_yellow_high`
counts, and the dashboard sparkline annotates provisional oranges as `O:3 (+2?)`.

---

## 4. The Underfitting Investigation (6/27/2026 Berry Elementary miss)

**Event:** Berry Elementary recorded 103–219 ppb H2S overnight; the model
completely missed it (predicted green). **Root cause:** extreme class imbalance +
a train/extreme distribution mismatch.

### Key data findings

- **Class imbalance is severe at 30 ppb.** NESTOR-BES is only 4.6% orange;
  San Ysidro 0.6%. At 5/10 ppb positives are 10–35× more common (9–22% base
  rate), so the baseline already works there — **the underfitting problem is
  almost entirely a 30 ppb problem.**
- **Orange is a nighttime phenomenon.** 90–94% of ≥30 ppb positives occur at
  night, and the nocturnal fraction rises with severity.
- **Heavy tail at NESTOR-BES.** Mean 7.5 ppb, P99 134 ppb, max 498 ppb — the
  bulk-fit regression underweights the extreme regime.

### Subset-training experiments (`experiment_underfitting.py`, Lean features)

Orange (≥30 ppb) recall vs baseline, per station:

| Station | Baseline | Best approach | Recall | Δ |
|---------|---------:|---------------|-------:|----|
| NESTOR-BES | 77.2% | Nighttime-only | **86.2%** | +9.0 pp |
| IB Civic Ctr | 69.7% | Dual H2S-nights | **80.0%** | +10.3 pp |
| San Ysidro | 15.6% | Non-zero H2S | **40.5%** | +24.9 pp |

(`Dual green` collapses to a single negative class and cannot train — needs an
anomaly-detection / regression-then-threshold approach instead.)

### Status

These subset-training and pooled-model results are **experiments, not yet
deployed** (`projects/h2s/experiments/underfitting_results/`). What *did* ship in
this revision is the defensive triad above: **Lean primary**, **clf_30ppb
gating**, and **two-head agreement** — which together prevent a sparse-station
orange model from firing false alarms while keeping a lone strong magnitude
signal visible as provisional. The pooled ≥30 model is the planned next step to
restore confirmed orange calls at IB Civic Center and San Ysidro.

---

## References

- `constants.py` — `PRIMARY_VARIANT`, `CLF_30PPB_STATIONS`,
  `magnitude_risk_tier`, `probability_risk_tier`, `classify_risk_agreement`
- `defs/h2s_daily_pipeline.py` — wiring of the gate + agreement into forecast rows
- `defs/cascade_alerts/cascade.py` — `TRIGGER_VARIANT = PRIMARY_VARIANT`
- `experiments/underfitting_results/FINDINGS.md` — 30 ppb underfitting analysis
- `experiments/underfitting_results/FINDINGS_5_10_PPB.md` — 5/10 ppb companion
- `experiments/underfitting_results/pooled_clf30_results.json` — pooled-model recall
- `tests/test_risk_agreement.py`, `tests/test_clf30_station_gating.py`
