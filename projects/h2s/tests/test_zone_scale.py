"""Tests for the per-zone emission scale applied to the Gaussian forward forecast."""

import numpy as np
import pandas as pd
import pytest

from h2s.constants import DISPERSION_ZONE_SCALE
from h2s.dispersion.gaussian import (
    SENSORS,
    SOURCES,
    apply_emission_scale,
    apply_zone_scale,
    run_forward_model,
)


def test_default_zone_scale_lowers_east_only():
    assert set(DISPERSION_ZONE_SCALE) == set(SOURCES) == {"east", "west", "south"}
    assert 0.0 < DISPERSION_ZONE_SCALE["east"] < 1.0
    assert DISPERSION_ZONE_SCALE["west"] == 1.0
    assert DISPERSION_ZONE_SCALE["south"] == 1.0


def test_zone_scale_on_zone_keyed_rates():
    rates = {"east": 40.0, "west": 10.0, "south": 100.0}
    out = apply_zone_scale(rates, {"east": 0.5})
    assert out == {"east": 20.0, "west": 10.0, "south": 100.0}
    assert rates["east"] == 40.0  # input untouched


def test_zone_scale_on_source_keyed_rates_via_zone_of():
    rates = {"stewarts_drain": 8.0, "oneonta_slough": 6.0, "mystery_source": 2.0}
    zone_of = {"stewarts_drain": "east", "oneonta_slough": "west"}
    out = apply_zone_scale(rates, {"east": 0.25, "west": 2.0}, zone_of=zone_of)
    assert out == {"stewarts_drain": 2.0, "oneonta_slough": 12.0, "mystery_source": 2.0}


def test_zone_scale_rejects_bad_factors():
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            apply_zone_scale({"east": 1.0}, {"east": bad})
    # zero is allowed: it switches a zone off
    assert apply_zone_scale({"east": 5.0}, {"east": 0.0}) == {"east": 0.0}


def _met(n_hours: int = 2, wind_direction: float = 215.0) -> pd.DataFrame:
    """Calm stable night; wind FROM the SW so the east source blows toward SAN YSIDRO."""
    times = pd.date_range("2026-04-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")
    rows = []
    for t in times:
        for sname in SENSORS:
            rows.append({
                "time": t, "site_name": sname,
                "wind_speed_10m": 2.0, "wind_direction_10m": wind_direction,
                "temperature_2m": 15.0, "is_night": 1,
            })
    return pd.DataFrame(rows)


def test_east_scale_composes_with_global_scale_and_only_moves_east_contribution():
    df = _met()
    start = df["time"].min()
    raw = {"east": 49.5, "west": 32.3, "south": 85.2}
    base = apply_emission_scale(raw, 0.15)
    east_half = apply_zone_scale(base, {"east": 0.5})
    east_off = apply_zone_scale(base, {"east": 0.0})

    full = run_forward_model(df, base, start, 2, cadence_minutes=60)
    half = run_forward_model(df, east_half, start, 2, cadence_minutes=60)
    off = run_forward_model(df, east_off, start, 2, cadence_minutes=60)

    for sensor in SENSORS:
        a = np.array(full.concentrations[sensor], dtype=float)
        b = np.array(half.concentrations[sensor], dtype=float)
        c = np.array(off.concentrations[sensor], dtype=float)
        ok = ~np.isnan(a)
        # east contribution = full - off; halving east removes exactly half of it
        east_contrib = a[ok] - c[ok]
        assert np.allclose(b[ok], a[ok] - 0.5 * east_contrib, rtol=1e-6, atol=2e-3)
        # non-east contribution is untouched
        assert np.all(c[ok] <= a[ok] + 1e-9)

    sy_full = np.nan_to_num(np.array(full.concentrations["SAN YSIDRO"], dtype=float))
    sy_off = np.nan_to_num(np.array(off.concentrations["SAN YSIDRO"], dtype=float))
    assert sy_full.max() > 0, "test met should put the east plume on SAN YSIDRO"
    assert sy_off.max() < sy_full.max(), "switching east off must lower SAN YSIDRO"
