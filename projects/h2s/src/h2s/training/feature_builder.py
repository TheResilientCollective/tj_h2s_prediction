"""Idempotent feature engineering for H2S prediction.

Centralized feature calculation that only computes missing features.
Used across training (multi_station_trainer), forecast (H2SPredictor),
and multi-horizon pipelines.
"""

import numpy as np
import pandas as pd

from h2s.constants import FLOW_COL, WIND_COL


def ensure_base_features(df: pd.DataFrame, flow_col: str = FLOW_COL) -> pd.DataFrame:
    """Add base features to DataFrame, skipping any already present.

    Idempotent — safe to call multiple times. Only calculates features
    not already in the DataFrame.

    **Excluded:** H2S lags (h2s_lag_1h/3h/6h, h2s_rolling_6h/24h) are
    context-dependent — use shift() in training, seed from obs_state in
    forecast. Caller must handle separately.

    Args:
        df: DataFrame with 'time' column (datetime64) and raw weather/tidal data
        flow_col: Column name for flow rate (default: FLOW_COL constant)

    Returns:
        DataFrame with added base features (in-place modification + return)

    Features added (if missing):
        - Time cyclicals: hour_sin, hour_cos, month_sin, month_cos
        - Day/Night: is_night
        - Source regime: source_regime (dominant H2S source based on wind/flow)
        - Wind cyclicals: wind_direction_sin, wind_direction_cos
        - Wind rolling: wind_speed_10m_avg_2h/3h/4h, wind_gusts_10m_max_2h/3h/4h
        - Flow derivatives: flow_log, flow_low, flow_high, flow_lag_6h, flow_rolling_24h
        - Interactions: wind_temp_interaction, humidity_temp_interaction
        - Stability: stable_atm
        - Tidal encoding: tidal_state_encoded
        - SBIWTP defaults: sbiwtp_flow_mgd, sbiwtp_anomaly, sbiwtp_deficit,
                          sbiwtp_flow_x_temp, sbiwtp_hourly_mgd, sbiwtp_sli
    """
    # Ensure time is datetime
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    # ========================================================================
    # Group A: Time cyclicals (always cheap to recalculate)
    # ========================================================================
    if 'time' in df.columns:
        if 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
        if 'hour_cos' not in df.columns:
            df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
        if 'month_sin' not in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['time'].dt.month / 12)
        if 'month_cos' not in df.columns:
            df['month_cos'] = np.cos(2 * np.pi * df['time'].dt.month / 12)

    # ========================================================================
    # Group B: Day/Night indicator
    # ========================================================================
    if 'is_night' not in df.columns and 'time' in df.columns:
        df['is_night'] = df['time'].dt.hour.isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23]).astype(int)

    # ========================================================================
    # Group C: Source regime (dominant H2S source)
    # ========================================================================
    if 'source_regime' not in df.columns:
        if WIND_COL in df.columns:
            df['source_regime'] = df.apply(_infer_source_regime, axis=1)
        else:
            df['source_regime'] = 0

    # ========================================================================
    # Group D: Wind direction cyclicals
    # ========================================================================
    if WIND_COL in df.columns:
        if 'wind_direction_sin' not in df.columns:
            df['wind_direction_sin'] = np.sin(np.deg2rad(df[WIND_COL]))
        if 'wind_direction_cos' not in df.columns:
            df['wind_direction_cos'] = np.cos(np.deg2rad(df[WIND_COL]))

    # ========================================================================
    # Group E: Wind rolling averages
    # ========================================================================
    if 'wind_speed_10m' in df.columns:
        for window in [2, 3, 4]:
            col = f'wind_speed_10m_avg_{window}h'
            if col not in df.columns:
                df[col] = df['wind_speed_10m'].rolling(window, min_periods=1).mean()

    if 'wind_gusts_10m' in df.columns:
        for window in [2, 3, 4]:
            col = f'wind_gusts_10m_max_{window}h'
            if col not in df.columns:
                df[col] = df['wind_gusts_10m'].rolling(window, min_periods=1).max()

    # ========================================================================
    # Group F: Flow derivatives
    # ========================================================================
    if flow_col in df.columns:
        if 'flow_log' not in df.columns:
            df['flow_log'] = np.log1p(df[flow_col].clip(lower=0))
        if 'flow_low' not in df.columns:
            df['flow_low'] = (df[flow_col] < 1).astype(int)
        if 'flow_high' not in df.columns:
            df['flow_high'] = (df[flow_col] > 5).astype(int)
        if 'flow_lag_6h' not in df.columns:
            df['flow_lag_6h'] = df[flow_col].shift(6).fillna(df[flow_col].median() if len(df) > 0 else 2.0)
        if 'flow_rolling_24h' not in df.columns:
            df['flow_rolling_24h'] = df[flow_col].rolling(24, min_periods=1).mean()

    # ========================================================================
    # Group G: Interaction features
    # ========================================================================
    if 'wind_temp_interaction' not in df.columns:
        if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
            df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']
        else:
            df['wind_temp_interaction'] = 0.0

    if 'humidity_temp_interaction' not in df.columns:
        if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
            df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']
        else:
            df['humidity_temp_interaction'] = 0.0

    # ========================================================================
    # Group H: Atmospheric stability
    # ========================================================================
    if 'stable_atm' not in df.columns:
        if 'wind_speed_10m' in df.columns and 'is_night' in df.columns:
            df['stable_atm'] = ((df['wind_speed_10m'] < 5) & (df['is_night'] == 1)).astype(int)
        else:
            df['stable_atm'] = 0

    # ========================================================================
    # Group I: Tidal state encoding
    # ========================================================================
    if 'tidal_state_encoded' not in df.columns:
        if 'tidal_state' in df.columns:
            tidal_mapping = {'flood': 0, 'ebb': 1, 'slack high': 2, 'slack low': 3}
            df['tidal_state_encoded'] = df['tidal_state'].map(tidal_mapping).fillna(-1).astype(int)  # type: ignore[arg-type]
        else:
            df['tidal_state_encoded'] = -1

    # ========================================================================
    # Group J: SBIWTP defaults (when feed unavailable)
    # ========================================================================
    sbiwtp_defaults = {
        'sbiwtp_flow_mgd': 23.5,
        'sbiwtp_anomaly': 0.0,
        'sbiwtp_deficit': 0.0,
        'sbiwtp_flow_x_temp': 23.5 * 18.0,  # flow × typical temp
        'sbiwtp_hourly_mgd': 23.5 / 24,
        'sbiwtp_sli': 0.0,
    }
    for col, default in sbiwtp_defaults.items():
        if col not in df.columns:
            df[col] = default

    return df


def _infer_source_regime(row: pd.Series) -> int:
    """Infer source regime from wind direction during nighttime.

    Returns:
        0: Day or no wind data
        1: Night + wind 22.5-135° (NE-SE quadrant)
        2: Night + wind 247.5-22.5° (W-NW-N quadrant)
        3: Night + wind 135-247.5° (SE-SW quadrant)
    """
    if not row.get('is_night', 0):
        return 0

    wind_dir = row.get(WIND_COL, 0.0)
    if not isinstance(wind_dir, (int, float)) or pd.isna(wind_dir):
        return 0

    if 22.5 <= wind_dir < 135:
        return 1
    elif wind_dir >= 247.5 or wind_dir < 22.5:
        return 2
    elif 135 <= wind_dir < 247.5:
        return 3
    return 0