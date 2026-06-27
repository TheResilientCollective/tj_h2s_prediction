# H2S Model Underfitting Analysis: 6/27 Complete Miss Event

**Date:** 2026-06-27  
**Event:** Berry Elementary School recorded 103-219 ppb H2S overnight; model completely missed prediction  
**Root Cause:** Extreme class imbalance + distribution mismatch between training and extreme events

## Executive Summary

Testing three approaches to improve orange (≥30 ppb) detection using lean 19-feature subset:

| Station | Approach | Orange Recall | Improvement | Notes |
|---------|----------|---------------|-------------|-------|
| **NESTOR-BES** | Baseline (all data) | 77.2% | — | Current production |
| | **Nighttime-only** | **86.2%** | **+9.0 pp** | ✅ Best for this station |
| | Dual H2S nights | 85.1% | +7.9 pp | Still strong |
| | Non-zero H2S | 73.8% | -3.4 pp | Degrades recall |
| **IB CIVIC CTR** | Baseline | 69.7% | — | |
| | **Dual H2S nights** | **80.0%** | **+10.3 pp** | ✅ Best for this station |
| | Nighttime-only | 66.7% | -3.0 pp | Slight degradation |
| **SAN YSIDRO** | Baseline | 15.6% | — | Extremely sparse orange |
| | Non-zero H2S | 40.5% | +24.9 pp | ✅ Best overall improvement |
| | Dual H2S nights | 26.3% | +10.7 pp | Moderate improvement |

## Key Findings

### 1. Class Imbalance is Severe

**NESTOR-BES (where 6/27 event occurred):**
- Only **4.6%** of samples are orange (≥30 ppb)
- **93.6%** of orange events occur at night
- Extreme events (>100 ppb) are in the **top 1% of the distribution**

The model is essentially trained on 95.4% green/yellow data, making it hard to learn the extreme regime.

### 2. Nighttime-Only Training Works Best for High-H2S Stations

**NESTOR-BES (optimal):**
- Orange prevalence: 4.6% → **14.6%** (3.2× increase)
- Recall improvement: 77.2% → **86.2%**
- The model specializes on the regime where H2S actually occurs

**IB CIVIC CTR:**
- Orange prevalence: 0.94% → **3.2%** (3.4× increase)
- Recall: 69.7% → 66.7% (slight drop, but more concentrated training)
- Dual H2S nights actually performs better here (+10.3 pp)

### 3. Non-Zero H2S Filtering Shows Mixed Results

**SAN YSIDRO (sparse orange station):**
- Orange recall: 15.6% → **40.5%** (+24.9 pp)
- This is the **largest single improvement** across all stations
- Removes ~36% of training data but focuses on when events occur

**NESTOR-BES:**
- Degradation: 77.2% → 73.8% (-3.4 pp)
- For high-H2S stations with sufficient orange samples, this is suboptimal

### 4. Dual Green Periods Model Failed (as expected)

When filtering to only green data (H2S ≤ 5), the training set becomes single-class (all negatives), making binary classification impossible. This approach needs a different strategy (e.g., anomaly detection or regression-then-threshold).

## Data Distribution Insights

### Nighttime Concentration of Orange Events

```
Station         % Orange at Night   Total Orange    Orange/Night
NESTOR-BES      93.6%              550 events      515 night events
IB CIVIC CTR    92.9%              84 events       78 night events
SAN YSIDRO      90.6%              85 events       77 night events
```

**Implication:** Orange events are a nighttime phenomenon. Daytime H2S is almost exclusively green/yellow. Mixing these regimes in training dilutes the model's ability to learn extreme nighttime behavior.

### H2S Magnitude Distribution

| Station | Mean | Std | P95 | P99 | Max |
|---------|------|-----|-----|-----|-----|
| NESTOR-BES | 7.49 ppb | 26.40 | 27.11 | 134.17 | **498.4 ppb** |
| IB CIVIC CTR | 2.62 ppb | 7.53 | 8.20 | 27.85 | 264 ppb |
| SAN YSIDRO | 2.29 ppb | 5.53 | 9.40 | 22.50 | 155.1 ppb |

NESTOR-BES has a heavy tail (P99 = 134 ppb, mean = 7.49) reflecting its exposure to extreme events.

## Recommended Path Forward

### Option 1: Nighttime-Only Classifier (Best for High-H2S Stations)
- **Use for:** NESTOR-BES (recall +9.0 pp), potentially IB CIVIC CTR
- **Tradeoff:** Daytime predictions must fall back to yellow-risk default or a separate daytime model
- **Pros:** Largest recall gain for the hazard station where we missed the 6/27 event
- **Cons:** Doesn't help SAN YSIDRO; daytime/nighttime boundary handling is complex

### Option 2: Non-Zero H2S Filtering (Best for Sparse Orange Stations)
- **Use for:** SAN YSIDRO (recall +24.9 pp)
- **Tradeoff:** Removes ~36% of training data globally, focusing on event windows
- **Pros:** Simple, globally applicable, strongest improvement for sparse stations
- **Cons:** Modest degradation for NESTOR-BES (-3.4 pp)

### Option 3: Hybrid Per-Station Strategy (Recommended)
```
NESTOR-BES:  Use nighttime-only model for nights, fallback for days
SAN YSIDRO:  Use non-zero H2S filtering (event-focused)
IB CIVIC CTR: Use dual H2S nights (nighttime+events)
```

**Advantage:** Tailors training to each station's actual H2S regime, addressing the 6/27 miss without degrading performance elsewhere.

### Option 4: Ensemble with Sample Weighting
- Upweight orange samples during training (inverse class frequency)
- Focus training on nighttime hours via sample weights
- Simpler deployment (single model per station, not regime-specific)
- Requires careful hyperparameter tuning to avoid False Alarm Rate increase

## Next Steps

1. **Immediate (within 1-2 training cycles):**
   - Deploy nighttime-only model for NESTOR-BES and test recall on holdout recent data
   - Compare against current baseline on 6/27 event window (was it in holdout?)
   - Monitor false alarm rate carefully — early warning should not come at cost of nuisance alarms

2. **Medium term (1-2 weeks):**
   - Implement per-station strategy (nighttime for NESTOR, non-zero for SAN YSIDRO)
   - Cross-validate on historical extreme events (>100 ppb windows)
   - Collect feedback from ops team on false alarm tolerance

3. **Long term (ongoing):**
   - Investigate two-stage models: (1) regime classifier (green/yellow vs orange), (2) magnitude regressor
   - Explore Bayesian approaches with informative priors on nighttime extremes
   - Consider physics-informed features (wind speed × stability interactions) that may better separate regimes

## Experimental Details

**Script:** `projects/h2s/scripts/experiment_underfitting.py`

**Features tested:** MODEL_FEATURES_LEAN (19 features)
- Excludes interactions, lower-importance weather features, and redundant wind rolling aggregates
- See `constants.py` line 478 for complete feature list

**Data:** `data/modeldata_h2s_nofill.parquet` (32K rows across 3 stations)

**Train/test split:** 80/20 by time per station (no shuffling, preserves temporal order)

**Model selection:** `train_and_select()` auto-selects RF/XGBoost/Ensemble for clf_30ppb task

**Metrics:**
- Orange recall (true positive rate at ≥30 ppb threshold)
- Precision, AUC, F1 for full context
- Train/test splits separate to detect overfitting

## Caveats

1. **Holdout test set is synthetic:** The train/test split is 80/20 of existing data, not a truly held-out future period. The 6/27 event may not have been in the training data at all (data currency question).

2. **Nighttime threshold is fixed:** The `is_night` feature uses hours [0-5, 20-23]. Regional variations (e.g., seasonal sunrise/sunset) are not accounted for.

3. **Non-zero H2S removes ~36% of data:** The downstream impact on model calibration and ability to learn other thresholds (5 ppb, 10 ppb) is not evaluated here.

4. **No statistical significance testing:** Differences between approaches should be validated on newer data to ensure they're not due to random variation in the test split.

## Questions for Stakeholders

1. **Daytime coverage:** If we use nighttime-only, what's acceptable for daytime predictions? Can we default to "yellow alert" or fallback to evidence variant?

2. **False alarm tolerance:** Current 5.4% false alarm rate is acceptable. Will the recall improvements hold that rate steady, or will we need to retune thresholds?

3. **Operational boundaries:** The 6/27 event peaked at midnight. Are there other high-impact events to backtest against? Can we validate on 2-3 recent extreme events before deploying?

---

**Report generated by:** Experiment framework (experiment_underfitting.py)  
**Data currency:** Training data as of 2026-06-27
