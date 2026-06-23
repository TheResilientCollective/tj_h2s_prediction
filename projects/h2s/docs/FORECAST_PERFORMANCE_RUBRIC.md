# Forecast Performance Rubric

The forecast performance report scores every predicted hour using an **asymmetric, temporally-tolerant rubric** that reflects operational hazard-mitigation priorities: under-predicting a hazard is far worse than over-predicting it, the ≥10 ppb resident-smell level matters, and an over-prediction the actual reaches within ±2 hours is an acceptable early warning rather than a false alarm.

## Verdict Categories and Cost Weights

| Verdict | Cost | When it happens | Interpretation |
|---|---:|---|---|
| **match** | 0.0 | Predicted tier = actual tier | Perfect prediction |
| **yellow_band** | 0.25 | Predicted orange, actual yellow | Over-stated but hazard still flagged |
| **early_warning_ok** | 0.25 | Over-predicted, actual reaches that tier within ±2h | Valid early warning |
| **expected_low** | 0.5 | Predicted green, actual 5–10 ppb | Low yellow acceptable per health officials |
| **soft_miss** | 1.0 | Predicted yellow, actual orange | Hazard under-stated but still flagged |
| **false_alarm** | 1.5 | Over-predicted, actual never materialized | Persistent false positive |
| **smell_miss** | 2.5 | Predicted green, actual ≥10 ppb | Resident-smell level (10 ppb) missed |
| **dangerous_miss** | 5.0 | Predicted green, actual ≥30 ppb orange | Critical hazard missed — residents get no warning |

## Key Parameters

- **Tolerance window:** ±2 hours (`TOLERANCE_HOURS`)
  - An over-prediction is forgiven if the actual H2S reaches the predicted tier within 2 hours before or after the forecast hour.
  - Example: predict orange for 14:00, actual reaches 35 ppb at 14:30 → early_warning_ok, cost 0.25.

- **Smell threshold:** 10 ppb (yellow-high / resident-smell level)
  - Below 10 ppb is considered low yellow and may be acceptable per health officials.
  - Missing ≥10 ppb (actual smell level) is a distinctive error category (smell_miss, cost 2.5).

- **Hazard threshold:** 30 ppb (orange)
  - Missing an actual orange event is the worst possible outcome (dangerous_miss, cost 5.0).

## Rubric Guidance (for LLM Narrative)

When generating the plain-language narrative:

1. **Under-predicting a real ORANGE (≥30 ppb) event as GREEN is the worst outcome** — a dangerous miss (cost 5.0). Residents get no warning of a hazard.

2. **Under-predicting ORANGE as some YELLOW is acceptable** — a hazard was still flagged, just under-stated (soft_miss, cost 1.0).

3. **Predicting ORANGE when it is ORANGE is the goal** — the target outcome (match, cost 0.0).

4. **Missing the ≥10 ppb YELLOW-HIGH (resident-smell) level as GREEN matters** — residents smell it, so we want to get this level right (smell_miss, cost 2.5).

5. **Predicting GREEN when the measurement is a low YELLOW (5–10 ppb) is often acceptable** — confirm the low-end cut with health officials (expected_low, cost 0.5).

6. **Over-predicting GREEN as YELLOW/ORANGE is acceptable when the actual reaches that level within ±2 hours** — an early warning rather than a false alarm (early_warning_ok, cost 0.25). Only a persistent over-prediction is a false alarm (cost 1.5).

## Headline Metrics

The scorecard report computes:

- **Tolerant accuracy:** Fraction of hours where the verdict was match / yellow_band / early_warning_ok (cost ≤ 0.25)
- **Mean cost:** Average penalty across all forecast-hours. Range [0.0, 5.0]; lower is better.
- **Dangerous miss rate:** Fraction of hours classified as dangerous_miss (worst case).
- **Smell miss rate:** Fraction of hours classified as smell_miss (resident-smell level).
- **Verdict counts:** Breakdown of all verdicts by type.

## Design Rationale

The rubric is intentionally asymmetric because the operational consequences are asymmetric:

- A **missed hazard** (under-prediction) prevents residents from taking shelter and puts them at health risk → high cost.
- A **false alarm** (over-prediction) causes unnecessary concern but is not dangerous → moderate cost.
- **Correct predictions** and **acceptable early warnings** are equally valuable for situational awareness → zero cost.

The ±2-hour tolerance reflects the timescale of H2S events in the Tijuana River Valley — forecasts that are off by 1–2 hours but predict the right hazard tier still provide valid early warning for resident response.

## Tuning Notes

The cost weights and thresholds (especially the low-yellow cut at 10 ppb for the smell_miss vs expected_low distinction) are **starting points** and should be refined with Tijuana health officials, environmental authorities, and the community based on operational experience and risk tolerance.
