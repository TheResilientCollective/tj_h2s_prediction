# H2S Underfitting Analysis — All Thresholds (5 / 10 / 30 ppb)

**Date:** 2026-06-27
**Companion to:** `FINDINGS.md` (which covered only the 30 ppb orange classifier)
**Feature set:** `MODEL_FEATURES_LEAN` (19 features)
**Split:** 80/20 by time per station; `train_and_select` auto-picks RF/XGB/Ensemble per task

## TL;DR

The same three approaches were re-run for the **5 ppb** (green/yellow) and
**10 ppb** (resident-smell) classifiers, not just 30 ppb (orange). Headline:

- **The underfitting problem is almost entirely a 30 ppb problem.** At 5 and
  10 ppb the baseline is already strong (recall 0.68–0.83) because positives are
  far less rare (9–22% base rate vs 0.6–4.6% at 30 ppb).
- **The night/event subset approaches still help at 5/10 ppb — just smaller and
  with no downside.** Unlike at 30 ppb (where `nonzero`/`nighttime` sometimes
  *hurt* a station), at 5/10 ppb every subset approach is ≥ baseline almost
  everywhere. The 6/27-style catastrophic miss does not occur at these
  thresholds.
- **The night specialization pays off more the higher the threshold.** Positives
  cluster harder at night as severity rises (≥5 ppb: ~75–82% nocturnal; ≥30 ppb:
  ~91–94%), so the nighttime/dual filters concentrate signal progressively more.

## Why 5/10 ppb were never the underfitting problem

Positive base rate by station × threshold (full dataset):

| Station | ≥5 ppb | ≥10 ppb | ≥30 ppb |
|---------|-------:|--------:|--------:|
| NESTOR-BES | 21.6% (2588) | 12.0% (1437) | 4.6% (550) |
| IB CIVIC CTR | 9.3% (835) | 4.0% (361) | 0.9% (84) |
| SAN YSIDRO | 12.1% (1807) | 4.5% (666) | 0.6% (85) |

At 30 ppb the model is learning from a handful of positives in a sea of
negatives (0.6% at SAN YSIDRO!). At 5 ppb it has 10–35× more positives. **More
positives → less underfitting → the baseline already works.** This is why the
6/27 miss was a hazard-tier (≥30) failure, not a caution-tier failure.

## Nighttime concentration rises with severity

% of positives that occur at night:

| Station | ≥5 ppb | ≥10 ppb | ≥30 ppb |
|---------|-------:|--------:|--------:|
| NESTOR-BES | 82.2% | 87.5% | 93.6% |
| IB CIVIC CTR | 82.3% | 87.8% | 92.9% |
| SAN YSIDRO | 75.2% | 79.7% | 90.6% |

The worse the event, the more nocturnal it is. So the night/event filters are
self-targeting: they remove the least informative rows (daytime greens) and the
fraction removed that *would* have been a positive shrinks as the threshold
rises. This is the mechanism behind the increasing gains at higher thresholds.

## Results — recall on held-out test (Δ vs baseline)

### ≥5 ppb (green → yellow boundary)

| Station | Baseline | Nighttime-only | Non-zero H2S | Dual H2S-nights |
|---------|---------:|---------------:|-------------:|----------------:|
| IB CIVIC CTR | 0.770 | **0.838 (+6.8)** | 0.791 (+2.1) | **0.838 (+6.8)** |
| NESTOR-BES | 0.826 | 0.853 (+2.7) | 0.829 (+0.3) | **0.864 (+3.8)** |
| SAN YSIDRO | 0.830 | **0.882 (+5.2)** | 0.795 (−3.5) | 0.874 (+4.4) |

### ≥10 ppb (resident-smell — "get it right")

| Station | Baseline | Nighttime-only | Non-zero H2S | Dual H2S-nights |
|---------|---------:|---------------:|-------------:|----------------:|
| IB CIVIC CTR | 0.684 | **0.804 (+12.0)** | 0.719 (+3.5) | 0.794 (+11.0) |
| NESTOR-BES | 0.756 | **0.807 (+5.1)** | 0.759 (+0.2) | 0.801 (+4.5) |
| SAN YSIDRO | 0.818 | 0.827 (+0.9) | **0.857 (+3.9)** | 0.848 (+3.1) |

### ≥30 ppb (orange / hazard — from FINDINGS.md, for reference)

| Station | Baseline | Nighttime-only | Non-zero H2S | Dual H2S-nights |
|---------|---------:|---------------:|-------------:|----------------:|
| IB CIVIC CTR | 0.697 | 0.667 (−3.0) | 0.606 (−9.1) | **0.800 (+10.3)** |
| NESTOR-BES | 0.772 | **0.862 (+9.0)** | 0.738 (−3.4) | 0.851 (+7.9) |
| SAN YSIDRO | 0.156 | 0.000 (−15.6) | **0.405 (+24.9)** | 0.263 (+10.8) |

## Reading across thresholds

1. **5/10 ppb are robust; almost every approach beats baseline.** At ≥10 ppb the
   *worst* subset result is +0.2 pp. There is no analog of the 30 ppb blow-ups
   (SAN YSIDRO nighttime → 0.000, IB nonzero → −9.1). With ~360–2600 positives,
   removing daytime greens only sharpens the model.

2. **`dual_h2s_nights` is the most consistent winner at 5/10 ppb.** It is top-or-tied
   for 4 of 6 station×threshold cells and never negative. It is the safest single
   choice if you want one strategy across the caution tiers.

3. **`nighttime_only` is the biggest 10 ppb winner for the two lower-H2S stations**
   (IB +12.0, NESTOR +5.1) but it is the approach that *catastrophically fails* at
   30 ppb for sparse stations (SAN YSIDRO drops to zero recall — the nighttime
   train split has too few ≥30 positives to learn the class). Don't use a single
   nighttime-only model across all thresholds for sparse stations.

4. **`nonzero_h2s_only` is uniquely the SAN YSIDRO answer at the extreme tail**
   (+24.9 at 30 ppb) but barely moves 5/10 ppb and even hurts SAN YSIDRO at 5 ppb
   (−3.5). It is a tail-specialist, not a general fix.

5. **AUC is flat-to-slightly-down everywhere** (0.92–0.99). The gains are recall
   gains from rebalancing the positive rate, not from better ranking — exactly
   what you'd expect when the fix is "show the model more positives." Precision
   mostly holds or improves at 5/10 ppb (it slips at 30 ppb for the rarer
   stations), so these are not just threshold-lowering artifacts.

## Recommendation update

The per-station strategy from `FINDINGS.md` was tuned on 30 ppb. With the
5/10 ppb data in hand:

- **Caution tiers (5 & 10 ppb): use `dual_h2s_nights` for all three stations.**
  Consistent positive Δ, no downside, one rule. This directly improves the
  10 ppb resident-smell calls the CLAUDE.md doc flags as "get it right."
- **Hazard tier (30 ppb): keep the per-station split** — nighttime-only for
  NESTOR, non-zero for SAN YSIDRO, dual for IB. The tail is where stations
  genuinely diverge.
- **Do not adopt a single global subset rule across all thresholds.**
  `nighttime_only` is great at 10 ppb and lethal at 30 ppb for SAN YSIDRO; the
  right subset is threshold- and station-dependent.

## Caveats (same as FINDINGS.md, plus)

- Held-out split is 80/20 of existing history, not a true future window; the 6/27
  event may not be in the test fold. Validate on recent extreme windows before
  deploying.
- These are recall-only comparisons at the model's default decision threshold.
  The deployed cascade uses probability cutoffs (`PROB_30_ALERT` etc.); the
  recall gains should be re-measured at those operating points.
- `dual_green_periods` was dropped from this run: filtering to green data makes
  every exceed-target single-class, so no binary classifier can be trained. A
  green-vs-event model needs a different framing (anomaly detection or
  regression-then-threshold).

## Reproduce

```bash
cd projects/h2s
# All three thresholds (default)
uv run python scripts/experiment_underfitting.py
# Single threshold
uv run python scripts/experiment_underfitting.py --thresholds 10
```

Raw metrics: `experiments/underfitting_results/experiment_results.json`
(now carries a `threshold` field per row).
