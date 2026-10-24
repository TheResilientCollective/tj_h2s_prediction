"""Tests for the uniform emission scale applied to the Gaussian forward forecast."""

import numpy as np
import pandas as pd
import pytest

from h2s.constants import DISPERSION_DEFAULT_EMISSION_RATES_GS, DISPERSION_EMISSION_SCALE
from h2s.dispersion.gaussian import (
    SENSORS,
    apply_emission_scale,
    run_forward_model,
    run_forward_model_from_Q_field,
)


def test_default_scale_lowers_rates_but_stays_positive():
    assert 0.0 < DISPERSION_EMISSION_SCALE < 1.0


def test_scale_dict_returns_new_scaled_dict():
    rates = {"east": 20.0, "west": 10.0, "south": 137.0}
    out = apply_emission_scale(rates, 0.5)
    assert out == {"east": 10.0, "west": 5.0, "south": 68.5}
    assert rates == {"east": 20.0, "west": 10.0, "south": 137.0}  # input untouched
    assert apply_emission_scale(rates, 1.0) == rates


def test_scale_q_field_rows_scales_only_q():
    rows = [
        {"lat": 32.5, "lon": -117.0, "Q_g_s": 4.0, "segment_id": "a"},
        {"lat": 32.6, "lon": -117.1, "Q_g_s": 0.0, "segment_id": "b"},
    ]
    out = apply_emission_scale(rows, 0.25)
    assert [r["Q_g_s"] for r in out] == [1.0, 0.0]
    assert [r["segment_id"] for r in out] == ["a", "b"]
    assert rows[0]["Q_g_s"] == 4.0  # input untouched


def test_scale_none_passthrough_and_invalid_scale():
    assert apply_emission_scale(None, 0.5) is None
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            apply_emission_scale({"east": 1.0}, bad)


def _met(n_hours: int = 2) -> pd.DataFrame:
    """Calm, stable night with a southerly wind — the regime where the plume is strongest."""
    times = pd.date_range("2026-04-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")
    rows = []
    for t in times:
        for sname in SENSORS:
            rows.append({
                "time": t, "site_name": sname,
                "wind_speed_10m": 2.0, "wind_direction_10m": 180.0,
                "temperature_2m": 15.0, "is_night": 1,
            })
    return pd.DataFrame(rows)


def _assert_half(full, half):
    saw_signal = False
    for sensor, vals in full.concentrations.items():
        a = np.array(vals, dtype=float)
        b = np.array(half.concentrations[sensor], dtype=float)
        ok = ~np.isnan(a)
        assert np.allclose(b[ok], 0.5 * a[ok], rtol=1e-6, atol=1e-3)  # ppb are rounded to 3 dp
        saw_signal |= bool((a[ok] > 0).any())
    assert saw_signal, "test met should produce a non-zero plume at some sensor"


def test_forward_forecast_is_linear_in_emission_scale():
    """Halving the rates must halve every forecast ppb — the scale is a clean knob."""
    df = _met()
    start = df["time"].min()
    full = run_forward_model(df, dict(DISPERSION_DEFAULT_EMISSION_RATES_GS), start, 2, cadence_minutes=60)
    half = run_forward_model(
        df, apply_emission_scale(DISPERSION_DEFAULT_EMISSION_RATES_GS, 0.5), start, 2, cadence_minutes=60
    )
    _assert_half(full, half)


def test_q_field_forward_is_linear_in_emission_scale():
    df = _met()
    start = df["time"].min()
    q_field = [
        {"lat": 32.545, "lon": -117.07, "Q_g_s": 50.0},
        {"lat": 32.550, "lon": -117.08, "Q_g_s": 20.0},
    ]
    full = run_forward_model_from_Q_field(df, q_field, start, 2, cadence_minutes=60)
    half = run_forward_model_from_Q_field(df, apply_emission_scale(q_field, 0.5), start, 2, cadence_minutes=60)
    _assert_half(full, half)
