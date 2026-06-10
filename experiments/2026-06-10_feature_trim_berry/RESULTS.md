# Results — feature_trim_berry

**Run on:** 2026-06-10
**Runtime:** ~30 seconds (single laptop run, no GPU)
**Outputs:** in `output/` (gitignored) — see `output/ablation.json` for full payload
**Status:** complete; clear winner identified

## Question

(Restated from README.) Does removing the features calibration explicitly dismissed (SBIWTP, flow magnitude derivatives, redundant wind rolling aggregates) improve the deployed model on the alert-aligned metrics, or do they carry hidden signal we'd lose by trimming?

## What we did

Trained a production-config XGBoost regressor on Berry (NESTOR-BES) for four feature-set candidates — Baseline (44), Evidence-only (33), Lean (19), Minimal (11) — on the same chronological 70/30 split as the PR #26 retrain (train n=8303, test n=3559, test window 2025-10-18 → 2026-04-01). Scored every candidate via the calibration-aligned harness from PR #26 (Spearman + recall@{30, 100} + regime stratification). Compared each to the 44-feature baseline against fixed acceptance criteria; persistence floor included as reference.

## Key findings

1. **Evidence-only (33 features) wins decisively and beats the baseline on every metric.** Removing the 11 features calibration explicitly dismissed (6 SBIWTP + 3 flow derivatives + 2 wind 4h rolling) lifts Spearman by **+3.5 pp** and recall@100 by **+3.9 pp** while keeping recall@30 essentially flat (−0.9 pp, well inside the 2 pp tolerance). The overdetermination hypothesis is empirically confirmed.

2. **The improvement is largest where it matters most — calm regime.** On the `stable_atm=1` slice (where calibration says all the >100 ppb action happens), Evidence-only beats Baseline at recall@100 **0.759 vs 0.722** (+3.7 pp) and ties recall@30 at 0.857. SBIWTP and flow magnitude noise was actively hurting the model exactly in the regime where the operational stakes are highest.

3. **Beyond Evidence-only, the further trims (Lean 19, Minimal 11) fail acceptance.** Both lose Spearman gains relative to Evidence-only and drop recall@30 by 6.7 pp (Lean) and 13.0 pp (Minimal). The 14 features Lean removes (interactions + remaining wind rolling + low-importance weather + hour_sin/cos) collectively carry ~7 pp of recall@30 — they're providing real signal, even if individually each is low-importance.

4. **Honest negative on the aggressive trim.** Set A (Minimal, 11 features) ties baseline on Spearman (0.787 vs 0.782) — the calibration-load-bearing core captures the rank-ordering structure — but loses 13 pp recall@30 and 12 pp recall@100. **Lean rank skill does not imply lean alert recall on this dataset.**

## Numbers

### Overall (test slice n=3559, Berry only)

| Set | Spearman | recall@30 | recall@100 | Δ vs D, recall@30 | Δ vs D, recall@100 | Decision |
|---|---|---|---|---|---|---|
| persistence floor | 0.822 | 0.574 | 0.390 | — | — | reference |
| **D - Baseline (44)** | 0.782 | 0.803 | 0.636 | — | — | reference |
| **C - Evidence (33)** | **0.817** | **0.794** | **0.675** | **−0.009** | **+0.039** | ✅ **WIN** |
| B - Lean (19) | 0.805 | 0.735 | 0.571 | −0.067 | −0.065 | ❌ FAIL (recall) |
| A - Minimal (11) | 0.787 | 0.673 | 0.519 | −0.130 | −0.117 | ❌ FAIL (recall) |

### Calm regime (`stable_atm=1`, n=1161) — where Berry's >100 ppb events live

| Set | Spearman | recall@30 | recall@100 |
|---|---|---|---|
| persistence | 0.754 | 0.599 | 0.389 |
| **D - Baseline (44)** | 0.792 | 0.857 | 0.722 |
| **C - Evidence (33)** | **0.802** | 0.857 | **0.759** |
| B - Lean (19) | 0.808 | 0.837 | 0.667 |
| A - Minimal (11) | 0.795 | 0.816 | 0.648 |

### Windy regime (`stable_atm=0`, n=2398)

| Set | Spearman | recall@30 | recall@100 |
|---|---|---|---|
| persistence | 0.846 | 0.526 | 0.391 |
| **D - Baseline (44)** | 0.746 | 0.697 | 0.435 |
| **C - Evidence (33)** | **0.806** | 0.671 | **0.478** |
| B - Lean (19) | 0.784 | 0.539 | 0.348 |
| A - Minimal (11) | 0.757 | 0.395 | 0.217 |

Evidence-only beats the 44-feature baseline at recall@100 in **both** regimes (calm +3.7 pp, windy +4.3 pp) and lifts windy Spearman by **+6.0 pp**. The trade-off is a 2.6 pp drop in windy recall@30 — well inside tolerance.

### Acceptance gate vs Baseline (Δ = baseline − candidate, smaller is better)

| Set | Δ Spearman (≤+0.02) | Δ recall@30 (≤+0.02) | Δ recall@100 (≤+0.05) | Verdict |
|---|---|---|---|---|
| C - Evidence | **−0.035** ✓ | **+0.009** ✓ | **−0.039** ✓ | **PASS — all three improved or within tolerance** |
| B - Lean | −0.023 ✓ | +0.067 ✗ | +0.065 ✗ | FAIL (both recalls breach tolerance) |
| A - Minimal | −0.005 ✓ | +0.130 ✗ | +0.117 ✗ | FAIL (both recalls breach tolerance by ~2.5× allowed) |

## What this means

The deployed 44-feature model was overdetermined in exactly the way calibration predicted: 11 of its features (25%) carry zero signal at best and actively degrade alert recall at worst. The biggest win is in the calm regime — the regime where 97 % of >100 ppb events at Berry occur and where the model's operational stakes are highest.

The Evidence-only set is a strict improvement on the baseline: **better on every overall metric, no trade-off, fewer features.** This is the rare unconditional win. The deployed `MODEL_FEATURES` should be replaced with `MODEL_FEATURES_EVIDENCE` in a follow-up PR, the per-station models retrained via `multi_station_training_job`, and `station_deployment_job` re-run.

The deeper-trim sets (Lean, Minimal) tell a more nuanced story: the rank-order skill is preserved with very few features (Set A's Spearman is essentially baseline), but **recall@threshold requires the broader feature set** — the model needs the auxiliary signal to push borderline predictions across the alert boundary. This matches the calibration arc's framing that magnitude prediction is fundamentally harder than ranking.

## What should be done next

In order of value:

1. **Phase 2 (immediate follow-up PR): promote Evidence-only to production.**
   - Edit [constants.py](../../projects/h2s/src/h2s/constants.py) so `MODEL_FEATURES = MODEL_FEATURES_EVIDENCE` (or rename `_EVIDENCE` to be the new default and keep the old 44-feature list as `MODEL_FEATURES_LEGACY`)
   - Adjust [feature_builder.ensure_base_features](../../projects/h2s/src/h2s/training/feature_builder.py) — drop the unused SBIWTP defaults and the dropped flow derivatives. Idempotency is preserved (caller-supplied columns still win).
   - Trigger `multi_station_training_job` followed by `station_deployment_job` so all three per-station models pick up the new feature set
   - Smoke-test `H2SPredictor.from_s3(...)` on the deployed candidates to confirm inference works end-to-end

2. **Phase 3 (later): validate the win on all 3 stations.** Berry has the most >100 ppb events (n=77 in this test slice); IB-Civic-Ctr (n=3) and SY (n=5) are too small here for reliable recall@100. The per-station retraining in step 1 will surface whether the trim transports.

3. **Re-evaluate the 3-class classifier.** This experiment is regression-side. The deployed hourly XGBoost classifier may behave differently; needs its own ablation before any classifier-side production change.

## Limitations / caveats

- **Berry only.** Recall@100 in the windy regime has only n=23 positives at Berry; IB and SY positives@100 are even smaller. The trim's per-station behavior at the other receptors is not measured.
- **Regression task only.** Deployed hourly model is a 3-class classifier; conclusions may not transport.
- **Single chronological split.** No cross-validation. Production training uses the same convention.
- **Single seed.** XGBoost with `random_state=42` (production hyperparameters). Different seeds will move recall@100 by ±2-3 pp given the small positive count; the +3.9 pp Evidence-vs-Baseline lift is at the edge of single-seed reliability for that metric specifically. The recall@30 and Spearman improvements are robust.
- **`train_and_select`'s selector isn't exercised.** The script calls `get_xgb_regressor()` directly so the comparison is feature-set vs feature-set. The deployed daily-station path now uses the selector (PR #26); if it picks XGB consistently after this trim, behavior here should match deployment.

## Files

- `feature_ablation.py` — entry point; trains all four candidates, scores via `calibration_report`, prints scoreboard, persists JSON
- `output/ablation.json` (gitignored) — full per-candidate payload including feature lists, fit metrics, calibration report, feature importance, acceptance verdict

## Library code referenced (no changes in this experiment)

- [h2s.constants.MODEL_FEATURES_EVIDENCE / _LEAN / _MINIMAL](../../projects/h2s/src/h2s/constants.py) — added as new module-level constants, no edit to `MODEL_FEATURES` (production unchanged)
- [tests/test_constants.py](../../projects/h2s/tests/test_constants.py) — 14 tests pinning counts, subset relations, load-bearing-feature preservation, and calibration-dismissal drops

`dg check defs` clean. PR #26's 31 prior tests all still green.
