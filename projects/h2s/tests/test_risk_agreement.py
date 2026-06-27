"""Tests for the two-head agreement classification (constants.classify_risk_agreement).

Design (user decision 2026-06):
- Two independent site signals: regression MAGNITUDE (h2s_pred) and classifier
  PROBABILITY (p5/p10/p30).
- Headline risk = the level BOTH heads confirm (the LOWER of the two single-head
  tiers). The higher single-head tier surfaces as risk_possible / provisional —
  it never escalates the headline on one signal alone.
- Applies across all tiers (5 / 10 / 30 ppb).
- Composes with the per-station clf_30ppb gate: a NaN p30 cannot raise the
  probability head to ORANGE, so magnitude-only orange is always provisional.
"""

from __future__ import annotations

import math

from h2s.constants import (
    PROB_5_ALERT,
    PROB_10_ALERT,
    PROB_30_ALERT,
    RISK_GREEN,
    RISK_ORANGE,
    RISK_YELLOW_HIGH,
    RISK_YELLOW_LOW,
    classify_risk_agreement,
    magnitude_risk_tier,
    probability_risk_tier,
)

NAN = float("nan")
# Probabilities comfortably above / below each alert cutoff.
HI5, LO5 = PROB_5_ALERT + 0.2, PROB_5_ALERT - 0.2
HI10, LO10 = PROB_10_ALERT + 0.2, PROB_10_ALERT - 0.2
HI30, LO30 = PROB_30_ALERT + 0.2, PROB_30_ALERT - 0.1


# ---- single-head tier helpers -------------------------------------------------

def test_magnitude_tiers_at_cuts():
    assert magnitude_risk_tier(0) == RISK_GREEN
    assert magnitude_risk_tier(4.9) == RISK_GREEN
    assert magnitude_risk_tier(5) == RISK_YELLOW_LOW
    assert magnitude_risk_tier(10) == RISK_YELLOW_HIGH
    assert magnitude_risk_tier(30) == RISK_ORANGE
    assert magnitude_risk_tier(NAN) == RISK_GREEN


def test_probability_tiers():
    assert probability_risk_tier(LO5, LO10, LO30) == RISK_GREEN
    assert probability_risk_tier(HI5, LO10, LO30) == RISK_YELLOW_LOW
    assert probability_risk_tier(HI5, HI10, LO30) == RISK_YELLOW_HIGH
    assert probability_risk_tier(HI5, HI10, HI30) == RISK_ORANGE
    # NaN p30 cannot reach ORANGE — caps at the p10 tier.
    assert probability_risk_tier(HI5, HI10, NAN) == RISK_YELLOW_HIGH


# ---- agreement truth table ----------------------------------------------------

def test_both_heads_agree_is_confirmed():
    # Both say orange.
    risk, possible, conf = classify_risk_agreement(HI5, HI10, 40.0, HI30)
    assert risk == RISK_ORANGE
    assert possible is None
    assert conf == "confirmed"


def test_magnitude_only_is_provisional_lower_headline():
    # Magnitude=orange, probability=green → headline GREEN, possible ORANGE.
    risk, possible, conf = classify_risk_agreement(LO5, LO10, 40.0, LO30)
    assert risk == RISK_GREEN
    assert possible == RISK_ORANGE
    assert conf == "provisional"


def test_probability_only_is_provisional():
    # Probability=orange, magnitude=green → headline GREEN, possible ORANGE.
    risk, possible, conf = classify_risk_agreement(HI5, HI10, 1.0, HI30)
    assert risk == RISK_GREEN
    assert possible == RISK_ORANGE
    assert conf == "provisional"


def test_partial_agreement_headline_is_lower_of_two():
    # Magnitude=yellow_high (12 ppb), probability=orange (p30 high)
    # → confirmed up to YELLOW_HIGH, ORANGE only possible.
    risk, possible, conf = classify_risk_agreement(HI5, HI10, 12.0, HI30)
    assert risk == RISK_YELLOW_HIGH
    assert possible == RISK_ORANGE
    assert conf == "provisional"


def test_agreement_applies_to_caution_tiers():
    # Magnitude=yellow_low (6 ppb), probability=green → headline GREEN, possible YL.
    risk, possible, conf = classify_risk_agreement(LO5, LO10, 6.0, LO30)
    assert risk == RISK_GREEN
    assert possible == RISK_YELLOW_LOW
    assert conf == "provisional"

    # Both at yellow_low → confirmed YL.
    risk, possible, conf = classify_risk_agreement(HI5, LO10, 6.0, LO30)
    assert risk == RISK_YELLOW_LOW
    assert possible is None
    assert conf == "confirmed"


def test_all_green_is_confirmed_green():
    risk, possible, conf = classify_risk_agreement(LO5, LO10, 1.0, LO30)
    assert risk == RISK_GREEN
    assert possible is None
    assert conf == "confirmed"


# ---- composition with the clf_30ppb gate -------------------------------------

def test_gated_p30_makes_orange_only_provisional():
    """When p30 is NaN (gated station), a strong magnitude orange can only ever
    be provisional — the probability head caps at yellow_high."""
    # Magnitude orange (40 ppb), p10 high, p30 NaN.
    risk, possible, conf = classify_risk_agreement(HI5, HI10, 40.0, NAN)
    assert risk == RISK_YELLOW_HIGH   # both confirm YH (mag≥10, p10>cut)
    assert possible == RISK_ORANGE    # magnitude alone reaches orange
    assert conf == "provisional"


def test_gated_p30_can_still_confirm_yellow_high():
    # Magnitude yellow_high + p10 high, p30 NaN → confirmed YELLOW_HIGH, no orange.
    risk, possible, conf = classify_risk_agreement(HI5, HI10, 15.0, NAN)
    assert risk == RISK_YELLOW_HIGH
    assert possible is None
    assert conf == "confirmed"
