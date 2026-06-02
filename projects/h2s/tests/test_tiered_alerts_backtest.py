"""Regression tests for the tiered alert backtest.

Acceptance criteria per design §6.1 (Tier 3 per horizon):
  nowcast:   precision ≥ 0.65, recall ≥ 0.80
  near:      precision ≥ 0.60, recall ≥ 0.75
  mid:       precision ≥ 0.55, recall ≥ 0.70
  day_ahead: precision ≥ 0.50, recall ≥ 0.65

The full backtest against the public parquet is marked @pytest.mark.slow and
requires a network connection; the unit-level test verifies the backtest
machinery itself against a synthetic fixture.
"""

import pytest
import numpy as np
import pandas as pd

from h2s.defs.tiered_alerts.backtest import run_backtest, _site_max_h2s, _is_above_threshold
from h2s.defs.tiered_alerts.tiers import load_config


# ---------------------------------------------------------------------------
# Helpers for synthetic data
# ---------------------------------------------------------------------------

_SITES = ["NESTOR - BES", "IB CIVIC CTR", "SAN YSIDRO"]


def _make_df(hours: int = 48) -> pd.DataFrame:
    """Synthetic modeldata parquet with all required columns."""
    times = pd.date_range("2026-01-01", periods=hours, freq="1h", tz="UTC")
    n = len(times)
    rng = np.random.default_rng(42)

    rows = []
    for site in _SITES:
        rows.append(pd.DataFrame({
            "time":             times,
            "site_name":        site,
            "H2S":              rng.uniform(0, 10, n),
            "sbiwtp_flow_mgd":  rng.uniform(18, 28, n),
            "sbiwtp_anomaly":   rng.uniform(-3, 2, n),
            "sbiwtp_sli":       rng.uniform(-1, 1, n),
            "sbiwtp_deficit":   rng.uniform(-2, 5, n),
            "wind_speed_10m":   rng.uniform(1, 8, n),
            "wind_direction_10m": rng.uniform(0, 360, n),
            "temperature_2m":   rng.uniform(10, 22, n),
            "dewpoint_2m":      rng.uniform(8, 16, n),
            "stable_atm":       rng.choice([0, 1], n).astype(float),
            "precipitation":    rng.uniform(0, 0.5, n),
            "flow_log":         rng.uniform(0, 2, n),
            "is_night":         pd.Series(times).dt.hour.pipe(lambda h: ((h >= 20) | (h <= 7)).astype(int)).values,
        }))
    return pd.concat(rows, ignore_index=True)


def _inject_event(df: pd.DataFrame, start_idx: int, h2s: float = 35.0, n_rows: int = 3) -> pd.DataFrame:
    """Force H2S exceedance at a given row index for NB station."""
    df = df.copy()
    mask = df["site_name"] == "NESTOR - BES"
    nb_rows = df[mask].index
    for i in range(min(n_rows, len(nb_rows) - start_idx)):
        df.loc[nb_rows[start_idx + i], "H2S"] = h2s
    return df


# ---------------------------------------------------------------------------
# Unit tests of backtest machinery
# ---------------------------------------------------------------------------

def test_site_max_h2s_returns_max():
    df = _make_df(6)
    df.loc[df["site_name"] == "NESTOR - BES", "H2S"] = 0.0
    df.loc[
        (df["site_name"] == "NESTOR - BES") & (df["time"] == df["time"].iloc[0]),
        "H2S"
    ] = 42.0
    t_start = df["time"].min()
    t_end   = df["time"].max()
    v = _site_max_h2s(df, "NESTOR - BES", t_start, t_end)
    assert v == pytest.approx(42.0)


def test_site_max_h2s_empty_returns_zero():
    df = _make_df(4)
    t_start = pd.Timestamp("2099-01-01", tz="UTC")
    t_end   = pd.Timestamp("2099-01-02", tz="UTC")
    assert _site_max_h2s(df, "NESTOR - BES", t_start, t_end) == 0.0


def test_is_above_threshold_detects_exceedance():
    df = _inject_event(_make_df(12), start_idx=2, h2s=50.0)
    t_start = df["time"].iloc[0]
    t_end   = df["time"].iloc[-1] + pd.Timedelta(hours=1)
    assert _is_above_threshold(df, "NESTOR - BES", t_start, t_end, 30.0) is True


def test_is_above_threshold_no_exceedance():
    df = _make_df(12)
    df["H2S"] = 1.0
    t_start = df["time"].iloc[0]
    t_end   = df["time"].iloc[-1] + pd.Timedelta(hours=1)
    assert _is_above_threshold(df, "NESTOR - BES", t_start, t_end, 30.0) is False


def test_backtest_runs_on_synthetic_data():
    df = _make_df(48)
    config = load_config()
    records_df, _ = run_backtest(df, config)
    assert not records_df.empty
    # 3 tiers × 4 horizons × 1 eval step minimum
    assert set(records_df["tier"].unique()) == {"tier_1", "tier_2", "tier_3"}
    assert set(records_df["horizon"].unique()) == {"nowcast", "near", "mid", "day_ahead"}


def test_backtest_nesting_never_violated():
    """Nesting invariant must hold for all evaluation timestamps."""
    df = _make_df(48)
    # Force conditions for a Tier 1 fire but not Tier 3
    df["sbiwtp_flow_mgd"] = 20.0
    df["sbiwtp_anomaly"]  = -2.0
    df["wind_speed_10m"]  = 5.0  # above T2 threshold → T2/T3 won't fire
    config = load_config()
    # Should not raise TierNestingError
    records_df, _ = run_backtest(df, config)
    # No T3 fires when wind is 5 m/s (above T2's 4 m/s gate)
    t3_fires = records_df[(records_df["tier"] == "tier_3") & records_df["fired"]]
    assert len(t3_fires) == 0


@pytest.mark.slow
def test_backtest_acceptance_criteria():
    """Full backtest against public parquet — may take several minutes."""
    from h2s.defs.tiered_alerts.backtest import _load_data, _TARGETS, _ensure_sbiwtp_anomaly
    try:
        from h2s.training.feature_builder import ensure_base_features
        df = _load_data(None)
        df = ensure_base_features(df)
        df = _ensure_sbiwtp_anomaly(df)
    except Exception as e:
        pytest.skip(f"Could not load backtest data: {e}")

    config = load_config()
    _, stats = run_backtest(df, config)

    failed = []
    for horizon, s in stats.items():
        target_prec, target_rec = _TARGETS[horizon]
        if s["precision"] < target_prec or s["recall"] < target_rec:
            failed.append(
                f"{horizon}: prec={s['precision']:.3f}≥{target_prec}, "
                f"rec={s['recall']:.3f}≥{target_rec}"
            )

    if failed:
        pytest.fail(
            "Per-horizon Tier 3 acceptance targets missed:\n" + "\n".join(failed) +
            "\nTODO(weights): retrain logistic regression — see design §8.6"
        )
