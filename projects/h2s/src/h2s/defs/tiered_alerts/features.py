"""Horizon-windowed aggregation of forecast features for tier scoring."""

import os

import numpy as np
import pandas as pd

from h2s.constants import (
    ALERT_LOCAL_TZ,
    ALERT_SBIWTP_BASELINE_MGD,
    FORECAST_DATA_PATH,
    STATIONS,
)

from .tiers import Horizon, HORIZON_WINDOWS_H

# Canonical met-source station (bellwether per design §2.2)
_NB_SITE  = "NESTOR - BES"
_IB_SITE  = "IB CIVIC CTR"

# Night hours in local time: 20:00 – 07:59
_NIGHT_HOURS = set(range(20, 24)) | set(range(0, 8))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_forecast_df(s3) -> pd.DataFrame:
    """Load model_forecast.parquet from S3 and apply base feature engineering."""
    from h2s.training.feature_builder import ensure_base_features

    public_bucket = os.environ.get("PUBLIC_BUCKET", s3.S3_BUCKET)
    url = s3.publicUrl(path=FORECAST_DATA_PATH, bucket=public_bucket)
    df = pd.read_parquet(url)

    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = ensure_base_features(df)

    # Compute sbiwtp_anomaly if not present (flow - baseline)
    if "sbiwtp_flow_mgd" in df.columns and "sbiwtp_anomaly" not in df.columns:
        df["sbiwtp_anomaly"] = df["sbiwtp_flow_mgd"] - ALERT_SBIWTP_BASELINE_MGD

    return df


def _ensure_multistation(df: pd.DataFrame) -> pd.DataFrame:
    """If forecast has no site_name or only one station, replicate for all 3."""
    if "site_name" not in df.columns:
        frames = []
        for site_name in STATIONS:
            row = df.copy()
            row["site_name"] = site_name
            frames.append(row)
        return pd.concat(frames, ignore_index=True)

    present = set(df["site_name"].unique())
    if len(present) >= 2:
        return df

    # Single station present — replicate for the others
    base = df.copy()
    frames = [base]
    for site_name in STATIONS:
        if site_name not in present:
            extra = base.copy()
            extra["site_name"] = site_name
            frames.append(extra)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Horizon slicing and aggregation
# ---------------------------------------------------------------------------

def slice_horizon(df: pd.DataFrame, t: pd.Timestamp, horizon: str) -> pd.DataFrame:
    start_h, end_h = HORIZON_WINDOWS_H[horizon]
    start = t + pd.Timedelta(hours=start_h)
    end   = t + pd.Timedelta(hours=end_h)
    return df[(df["time"] >= start) & (df["time"] < end)].copy()


def _is_night_hour(ts: pd.Timestamp) -> bool:
    local = ts.tz_convert(ALERT_LOCAL_TZ)
    return local.hour in _NIGHT_HOURS


def _daytime_horizon(window_df: pd.DataFrame) -> bool:
    """True if <75% of rows in the window are night hours."""
    if window_df.empty:
        return False
    night_frac = window_df["time"].apply(_is_night_hour).mean()
    return night_frac < 0.75


def _vector_wind_dir(window_df: pd.DataFrame) -> float | None:
    if "wind_speed_10m" not in window_df.columns or "wind_direction_10m" not in window_df.columns:
        return None
    ws  = window_df["wind_speed_10m"].values
    wd  = window_df["wind_direction_10m"].values
    u = -ws * np.sin(np.deg2rad(wd))
    v = -ws * np.cos(np.deg2rad(wd))
    u_mean = np.nanmean(u)
    v_mean = np.nanmean(v)
    return float(np.rad2deg(np.arctan2(-u_mean, -v_mean)) % 360)


def _aggregate_station_window(window_df: pd.DataFrame) -> dict:
    """Mean-aggregate all numeric features over a horizon window."""
    if window_df.empty:
        return {}

    agg: dict = {}
    numeric_cols = window_df.select_dtypes(include="number").columns
    for col in numeric_cols:
        vals = window_df[col].dropna()
        if len(vals) == 0:
            continue
        agg[col] = float(vals.mean())

    # Overrides
    if "wind_direction_10m" in window_df.columns:
        wd = _vector_wind_dir(window_df)
        if wd is not None:
            agg["wind_direction_10m"] = wd

    if "wind_speed_10m" in window_df.columns:
        agg["wind_speed_min"] = float(window_df["wind_speed_10m"].min())

    if "temperature_2m" in window_df.columns:
        agg["temp_min"] = float(window_df["temperature_2m"].min())

    # stable_atm_fraction = mean of boolean column
    if "stable_atm" in window_df.columns:
        agg["stable_atm_fraction"] = float(window_df["stable_atm"].mean())

    return agg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_horizon_features(
    df: pd.DataFrame,
    t: pd.Timestamp,
) -> tuple[dict[tuple[str, str], dict], bool]:
    """Compute per-(horizon, station) feature aggregates.

    Returns:
        cell_features: dict keyed by (horizon_key, site_name) → feature dict
        degraded: True if NESTOR-BES data was missing and IB was used
    """
    df = _ensure_multistation(df)
    degraded = False

    # Resolve met source: use NB; fall back to IB per design §7.1
    nb_data = df[df["site_name"] == _NB_SITE]
    if nb_data.empty:
        degraded = True
        met_df = df[df["site_name"] == _IB_SITE]
    else:
        met_df = nb_data

    cell_features: dict[tuple[str, str], dict] = {}

    for horizon in HORIZON_WINDOWS_H:
        met_window = slice_horizon(met_df, t, horizon)
        is_daytime = _daytime_horizon(met_window)

        # Per-station: use met from bellwether, but evaluate gate per station
        for site_name in STATIONS:
            station_df = df[df["site_name"] == site_name]
            station_window = slice_horizon(station_df, t, horizon)

            if station_window.empty and met_window.empty:
                continue

            # Use per-station window if available, else met window (shared grid)
            source = station_window if not station_window.empty else met_window
            agg = _aggregate_station_window(source)
            agg["_daytime_horizon"] = is_daytime
            agg["_degraded"] = degraded
            cell_features[(horizon, site_name)] = agg

    return cell_features, degraded
