"""Pure cascade-trigger logic — no Dagster, no S3, no Slack.

Evaluates the compute-on-trigger cascade (docs/feature/rename.md, Phase 4) at
NESTOR-BES off the product probabilities the Phase-3 engine already produces:

  Tier 1  P(H2S > 5)  in the **nowcast**  window  > cutoff  (PLANT-SIGNAL)
  Tier 2  P(H2S > 10) in the **nearcast** window  > cutoff  (MULTI-SITE-RISK)
  Tier 3  P(H2S > 30) in the **forecast** window  > cutoff  (EXCEEDANCE-RISK)

Cutoffs and windows are the single source of truth in
``h2s.constants.CASCADE_TRIGGERS``.

Two deliberate departures from the retired gate/sigmoid system:

- **Triggers read the classifier directly.** The ~0.9-AUC per-threshold
  classifiers already encode the met regime; the cascade trusts P(exceed)
  rather than re-deriving a risk score from hand-weighted z-scores.
- **Tiers fire independently.** A high forecast-window P(>30) is surfaced even
  when the nowcast is calm — the old nesting invariant (Tier 3 ⇒ Tier 2 ⇒
  Tier 1) would have suppressed exactly the early-warning signal we want. The
  trigger probability per tier is the **peak** over that tier's product window.

The products are computed unconditionally upstream (Phase 3, for the validation
store), so this module gates *alerting*, not computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from h2s.constants import (
    CASCADE_TRIGGERS,
    H2S_THRESHOLD_HIGH,
    H2S_THRESHOLD_LOW,
    H2S_THRESHOLD_MED,
    PRIMARY_VARIANT,
)

# NESTOR-BES is the bellwether station that drives the cascade. The primary
# variant (lean, 19 feat — see PRIMARY_VARIANT in constants) fires triggers and
# drives the reports that import TRIGGER_VARIANT; Evidence is reported alongside
# but never gates.
NB_SITE = "NESTOR - BES"
TRIGGER_VARIANT = PRIMARY_VARIANT

# Map a tier's exceedance threshold (ppb) to its probability column.
THRESHOLD_PROB_COL: dict[int, str] = {
    H2S_THRESHOLD_LOW: "p5",    # 5  ppb
    H2S_THRESHOLD_MED: "p10",   # 10 ppb
    H2S_THRESHOLD_HIGH: "p30",  # 30 ppb
}

# Highest-first so ``highest_fired`` and audience routing are unambiguous.
TIER_ORDER = ("tier_1", "tier_2", "tier_3")
_TIER_RANK = {t: i + 1 for i, t in enumerate(TIER_ORDER)}


@dataclass(frozen=True)
class TierEvaluation:
    """One tier's verdict for one (station, variant)."""

    tier: str            # "tier_1" | "tier_2" | "tier_3"
    product: str         # nowcast | nearcast | forecast
    threshold_ppb: int
    prob_cutoff: float
    prob_col: str        # p5 | p10 | p30
    trigger_prob: float  # peak P(exceed) over the product window; NaN if unavailable
    fired: bool          # trigger_prob > cutoff (False when NaN — never fires on missing data)
    peak_lead_hour: int | None
    peak_time: pd.Timestamp | None


@dataclass(frozen=True)
class CascadeResult:
    station: str
    variant: str
    run_ts: str | None
    model_version: str | None
    evaluations: dict[str, TierEvaluation]   # keyed by tier
    est_hours_h2s_6h: int                     # predicted hours ≥ 5 ppb in leads 1–6

    @property
    def fired_tiers(self) -> list[str]:
        """Tiers that fired, ascending (tier_1 first)."""
        return [t for t in TIER_ORDER if self.evaluations[t].fired]

    @property
    def highest_fired(self) -> str | None:
        fired = self.fired_tiers
        return max(fired, key=lambda t: _TIER_RANK[t]) if fired else None

    @property
    def any_fired(self) -> bool:
        return bool(self.fired_tiers)


def _empty_evaluation(tier: str, cfg: dict) -> TierEvaluation:
    return TierEvaluation(
        tier=tier,
        product=cfg["product"],
        threshold_ppb=cfg["threshold_ppb"],
        prob_cutoff=cfg["prob_cutoff"],
        prob_col=THRESHOLD_PROB_COL[cfg["threshold_ppb"]],
        trigger_prob=float("nan"),
        fired=False,
        peak_lead_hour=None,
        peak_time=None,
    )


def _evaluate_tier(tier: str, cfg: dict, sdf: pd.DataFrame) -> TierEvaluation:
    """Evaluate one tier: peak P(exceed) over its product window vs the cutoff."""
    prob_col = THRESHOLD_PROB_COL[cfg["threshold_ppb"]]
    window = sdf[sdf["product"] == cfg["product"]]
    probs = window[prob_col].dropna() if prob_col in window.columns else pd.Series(dtype=float)

    if probs.empty:
        return _empty_evaluation(tier, cfg)

    peak_idx = probs.idxmax()
    trigger_prob = float(probs.loc[peak_idx])
    peak_row = window.loc[peak_idx]
    return TierEvaluation(
        tier=tier,
        product=cfg["product"],
        threshold_ppb=cfg["threshold_ppb"],
        prob_cutoff=cfg["prob_cutoff"],
        prob_col=prob_col,
        trigger_prob=trigger_prob,
        fired=trigger_prob > cfg["prob_cutoff"],
        peak_lead_hour=int(peak_row["lead_hour"]),
        peak_time=peak_row["time"],
    )


def _estimate_hours_h2s_6h(sdf: pd.DataFrame, threshold: float = H2S_THRESHOLD_LOW) -> int:
    """Predicted hours of detectable H2S (≥ threshold ppb) over leads 1–6."""
    next6 = sdf[(sdf["lead_hour"] >= 1) & (sdf["lead_hour"] <= 6)]
    return int((next6["h2s_pred"] >= threshold).sum())


def evaluate_cascade(
    products_df: pd.DataFrame,
    station: str = NB_SITE,
    variant: str = TRIGGER_VARIANT,
    triggers: dict | None = None,
) -> CascadeResult:
    """Evaluate Tiers 1–3 for one (station, variant) from a products frame.

    ``products_df`` is the output of ``h2s_products`` (columns: run_ts, product,
    station, lead_hour, time, variant, model_version, h2s_pred, p5, p10, p30).
    Returns a :class:`CascadeResult`; missing rows or missing probability
    columns degrade to non-firing tiers rather than raising.
    """
    triggers = triggers or CASCADE_TRIGGERS
    sdf = products_df[
        (products_df["station"] == station) & (products_df["variant"] == variant)
    ]

    if sdf.empty:
        return CascadeResult(
            station=station,
            variant=variant,
            run_ts=None,
            model_version=None,
            evaluations={t: _empty_evaluation(t, triggers[t]) for t in TIER_ORDER},
            est_hours_h2s_6h=0,
        )

    evaluations = {tier: _evaluate_tier(tier, triggers[tier], sdf) for tier in TIER_ORDER}
    return CascadeResult(
        station=station,
        variant=variant,
        run_ts=str(sdf["run_ts"].iloc[0]) if "run_ts" in sdf.columns else None,
        model_version=str(sdf["model_version"].iloc[0]) if "model_version" in sdf.columns else None,
        evaluations=evaluations,
        est_hours_h2s_6h=_estimate_hours_h2s_6h(sdf),
    )
