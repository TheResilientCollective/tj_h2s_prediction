"""Tests for the product-probability cascade logic (Phase 4).

Pins the trigger arithmetic BEFORE the cascade drives any Slack dispatch.
Synthetic product frames exercise mechanics only — never training or skill
claims.

What's load-bearing here:
- each tier reads its OWN probability column (p5 / p10 / p30) over its OWN
  product window, and fires on the PEAK of that window strictly above the cutoff;
- tiers fire INDEPENDENTLY (no nesting invariant) — the retired gate system
  would have suppressed a forecast-window P(>30) when the nowcast was calm;
- missing classifier columns degrade to a non-firing NaN tier, never a crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from h2s.constants import PRODUCT_FORECAST, PRODUCT_NEARCAST, PRODUCT_NOWCAST
from h2s.defs.cascade_alerts.cascade import (
    NB_SITE,
    TRIGGER_VARIANT,
    evaluate_cascade,
)


# ---------------------------------------------------------------------------
# Synthetic products frame
# ---------------------------------------------------------------------------

def _product_for_lead(lead: int) -> str:
    if lead <= 3:
        return PRODUCT_NOWCAST
    if lead <= 6:
        return PRODUCT_NEARCAST
    return PRODUCT_FORECAST


def _station_frame(
    prob_overrides: dict[int, tuple[float, float, float]] | None = None,
    h2s_overrides: dict[int, float] | None = None,
    station: str = NB_SITE,
    variant: str = TRIGGER_VARIANT,
    run_ts: str = "2026-06-14T1200Z",
    model_version: str = "v-test",
    drop_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """One station × variant: leads 1–24, low default probabilities (0.1)."""
    prob_overrides = prob_overrides or {}
    h2s_overrides = h2s_overrides or {}
    times = pd.date_range("2026-06-14 13:00", periods=24, freq="h", tz="UTC")
    rows = []
    for i, lead in enumerate(range(1, 25)):
        p5, p10, p30 = prob_overrides.get(lead, (0.1, 0.1, 0.1))
        rows.append({
            "run_ts": run_ts,
            "product": _product_for_lead(lead),
            "station": station,
            "lead_hour": lead,
            "time": times[i],
            "variant": variant,
            "model_version": model_version,
            "h2s_pred": h2s_overrides.get(lead, 0.0),
            "p5": p5,
            "p10": p10,
            "p30": p30,
        })
    df = pd.DataFrame(rows)
    if drop_cols:
        df = df.drop(columns=list(drop_cols))
    return df


# ---------------------------------------------------------------------------
# Single-tier firing
# ---------------------------------------------------------------------------

class TestTierFiring:
    def test_tier1_fires_when_nowcast_p5_above_cutoff(self):
        df = _station_frame(prob_overrides={2: (0.6, 0.1, 0.1)})
        result = evaluate_cascade(df)
        assert result.evaluations["tier_1"].fired
        assert not result.evaluations["tier_2"].fired
        assert not result.evaluations["tier_3"].fired
        assert result.fired_tiers == ["tier_1"]
        assert result.highest_fired == "tier_1"

    def test_not_fired_at_exact_cutoff(self):
        # strict > 0.5 — exactly 0.5 must NOT fire
        df = _station_frame(prob_overrides={1: (0.5, 0.1, 0.1)})
        result = evaluate_cascade(df)
        assert not result.evaluations["tier_1"].fired

    def test_trigger_prob_is_peak_over_window(self):
        df = _station_frame(prob_overrides={
            1: (0.2, 0.1, 0.1), 2: (0.7, 0.1, 0.1), 3: (0.3, 0.1, 0.1),
        })
        ev = evaluate_cascade(df).evaluations["tier_1"]
        assert ev.trigger_prob == pytest.approx(0.7)
        assert ev.peak_lead_hour == 2

    def test_each_tier_reads_its_own_probability_column(self):
        # Only p30 elevated (in the forecast window) — only Tier 3 may fire.
        df = _station_frame(prob_overrides={10: (0.1, 0.1, 0.9)})
        result = evaluate_cascade(df)
        assert result.evaluations["tier_1"].prob_col == "p5"
        assert result.evaluations["tier_2"].prob_col == "p10"
        assert result.evaluations["tier_3"].prob_col == "p30"
        assert result.fired_tiers == ["tier_3"]


# ---------------------------------------------------------------------------
# Independent firing (no nesting invariant)
# ---------------------------------------------------------------------------

class TestIndependentFiring:
    def test_tier3_fires_without_tier1(self):
        # Calm nowcast/nearcast, dangerous forecast — must still surface Tier 3.
        df = _station_frame(prob_overrides={12: (0.1, 0.1, 0.8)})
        result = evaluate_cascade(df)
        assert not result.evaluations["tier_1"].fired
        assert not result.evaluations["tier_2"].fired
        assert result.evaluations["tier_3"].fired
        assert result.highest_fired == "tier_3"

    def test_all_tiers_fire(self):
        df = _station_frame(prob_overrides={
            2: (0.9, 0.1, 0.1),   # nowcast p5
            5: (0.1, 0.7, 0.1),   # nearcast p10
            9: (0.1, 0.1, 0.6),   # forecast p30
        })
        result = evaluate_cascade(df)
        assert result.fired_tiers == ["tier_1", "tier_2", "tier_3"]
        assert result.highest_fired == "tier_3"
        assert result.any_fired

    def test_nearcast_p10_does_not_leak_into_nowcast_window(self):
        # p10 spike lives in the nearcast window; Tier 2 reads nearcast, so it
        # fires — but Tier 1 (nowcast p5) stays calm.
        df = _station_frame(prob_overrides={5: (0.1, 0.9, 0.1)})
        result = evaluate_cascade(df)
        assert result.fired_tiers == ["tier_2"]


# ---------------------------------------------------------------------------
# Degradation — missing classifiers / data
# ---------------------------------------------------------------------------

class TestDegradation:
    def test_missing_p30_column_yields_nan_not_crash(self):
        df = _station_frame(prob_overrides={10: (0.1, 0.1, 0.9)}, drop_cols=("p30",))
        ev = evaluate_cascade(df).evaluations["tier_3"]
        assert np.isnan(ev.trigger_prob)
        assert not ev.fired

    def test_all_nan_p30_yields_nan_not_crash(self):
        df = _station_frame()
        df["p30"] = np.nan
        ev = evaluate_cascade(df).evaluations["tier_3"]
        assert np.isnan(ev.trigger_prob)
        assert not ev.fired

    def test_empty_frame_no_fire(self):
        df = _station_frame().iloc[0:0]
        result = evaluate_cascade(df)
        assert not result.any_fired
        assert result.run_ts is None
        assert result.est_hours_h2s_6h == 0

    def test_other_station_only_no_fire(self):
        df = _station_frame(prob_overrides={2: (0.9, 0.1, 0.1)}, station="SAN YSIDRO")
        result = evaluate_cascade(df, station=NB_SITE)
        assert not result.any_fired


# ---------------------------------------------------------------------------
# Station / variant isolation
# ---------------------------------------------------------------------------

class TestStationVariantFilter:
    def test_only_nb_evidence_drives_triggers(self):
        # NB-evidence is calm; a screaming SY-evidence and NB-lean must not fire NB-evidence.
        nb_evidence = _station_frame()  # all 0.1
        sy_evidence = _station_frame(prob_overrides={2: (0.99, 0.99, 0.99)}, station="SAN YSIDRO")
        nb_lean = _station_frame(prob_overrides={2: (0.99, 0.99, 0.99)}, variant="lean")
        df = pd.concat([nb_evidence, sy_evidence, nb_lean], ignore_index=True)
        result = evaluate_cascade(df, station=NB_SITE, variant="evidence")
        assert not result.any_fired

    def test_picks_run_ts_and_model_version(self):
        df = _station_frame(run_ts="2026-06-14T1800Z", model_version="20260614-abc")
        result = evaluate_cascade(df)
        assert result.run_ts == "2026-06-14T1800Z"
        assert result.model_version == "20260614-abc"


# ---------------------------------------------------------------------------
# Estimated hours of H2S in the next 6h
# ---------------------------------------------------------------------------

class TestEstimatedHours:
    def test_counts_leads_1_through_6_above_5ppb(self):
        df = _station_frame(h2s_overrides={1: 8.0, 2: 3.0, 3: 6.0, 4: 0.0, 5: 12.0, 6: 5.0})
        # ≥5 ppb at leads 1,3,5,6 → 4 hours
        assert evaluate_cascade(df).est_hours_h2s_6h == 4

    def test_ignores_leads_beyond_6(self):
        df = _station_frame(h2s_overrides={7: 50.0, 12: 99.0})
        assert evaluate_cascade(df).est_hours_h2s_6h == 0
