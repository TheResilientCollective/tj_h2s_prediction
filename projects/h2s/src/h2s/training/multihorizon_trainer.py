"""Multi-horizon H2S forecast training and inference utilities.

Provides horizon-specific feature construction on top of pre-featurized parquet data.
Each horizon uses features that honestly reflect what's known at that lead time:
  - 0-6h:   fresh H2S lags (1h, 3h, 6h)
  - 6-24h:  stale lags (6h, 12h, 24h) + daily stats + spill/crisis
  - 24-48h: old lags (24h, 48h) + yesterday + 2-days-ago stats
  - 48-72h: oldest lags (48h, 72h) + same daily stats

Ensemble classes must be importable from this module for pickle deserialization.
"""

import numpy as np
import pandas as pd

from h2s.training.feature_builder import ensure_base_features


# ============================================================
# HORIZON DEFINITIONS
# ============================================================

HORIZONS = {
    '0_6h': {
        'description': '0-6 hour ahead: fresh lag features',
        'lag_offsets': [1, 3, 6],
        'rolling_windows': [6, 24],
        'flow_lag': 6,
        'use_daily_stats': False,
        'use_spill_flag': False,
    },
    '6_24h': {
        'description': '6-24 hour ahead: stale lags, daily stats emerge',
        'lag_offsets': [6, 12, 24],
        'rolling_windows': [6, 24],
        'flow_lag': 24,
        'use_daily_stats': True,
        'use_spill_flag': True,
    },
    '24_48h': {
        'description': '24-48 hour ahead: yesterday stats only',
        'lag_offsets': [24, 48],
        'rolling_windows': [24],
        'flow_lag': 24,
        'use_daily_stats': True,
        'use_spill_flag': True,
    },
    '48_72h': {
        'description': '48-72 hour ahead: 2-day-old stats + climatology',
        'lag_offsets': [48, 72],
        'rolling_windows': [24],
        'flow_lag': 48,
        'use_daily_stats': True,
        'use_spill_flag': True,
    },
}

HORIZON_NAMES = list(HORIZONS.keys())

# Horizon boundaries for assigning forecast hours
HORIZON_BOUNDS = [
    ('0_6h',   0,  6),
    ('6_24h',  6,  24),
    ('24_48h', 24, 48),
    ('48_72h', 48, 999),
]

# BASE_FEATURES imported from h2s.constants

from h2s.constants import (  # noqa: F401 — re-exported for downstream imports
    BASE_FEATURES,
    FLOW_COL,
    SOURCES,
    STATION_PARTITION_MAP,
    STATIONS,
    classify_risk,
)

TASKS = ['regression', 'clf_5ppb', 'clf_10ppb']


# ============================================================
# ENSEMBLE CLASSES (must live here for pickle deserialization)
# ============================================================

class EnsembleRegressor:
    """Weighted average of two regressors."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict(self, X):
        return self.weight_a * self.model_a.predict(X) + self.weight_b * self.model_b.predict(X)

    @property
    def feature_importances_(self):
        a = getattr(self.model_a, 'feature_importances_', None)
        b = getattr(self.model_b, 'feature_importances_', None)
        if a is None and b is None:
            return None
        a = np.asarray(a) if a is not None else np.zeros_like(b)
        b = np.asarray(b) if b is not None else np.zeros_like(a)
        return self.weight_a * a + self.weight_b * b


class EnsembleClassifier:
    """Weighted probability average of two classifiers."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict_proba(self, X):
        return self.weight_a * self.model_a.predict_proba(X) + self.weight_b * self.model_b.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    @property
    def feature_importances_(self):
        a = getattr(self.model_a, 'feature_importances_', None)
        b = getattr(self.model_b, 'feature_importances_', None)
        if a is None and b is None:
            return None
        a = np.asarray(a) if a is not None else np.zeros_like(b)
        b = np.asarray(b) if b is not None else np.zeros_like(a)
        return self.weight_a * a + self.weight_b * b


# STATIONS, STATION_PARTITION_MAP, SOURCES imported from h2s.constants


# ============================================================
# TRAINING: BUILD HORIZON-SPECIFIC FEATURES
# ============================================================

def build_horizon_features(site_df, horizon_name, horizon_cfg):
    """Construct horizon-specific features on top of pre-featurized parquet data.

    Adds H2S/flow lags at the appropriate offset for the horizon,
    daily stats, exceedance climatology, and spill indicators.

    Args:
        site_df: DataFrame for one station from the training parquet.
                 Must contain BASE_FEATURES + 'H2S' + 'Flow (m^3/s)--Border' + 'time'.
        horizon_name: e.g. '0_6h', '6_24h', etc.
        horizon_cfg: dict from HORIZONS[horizon_name].

    Returns:
        (df, feature_cols) where df has new columns and feature_cols is the ordered list.
    """
    df = site_df.copy()
    h2s = df['H2S'].values
    flow = df[FLOW_COL].values

    feature_cols = list(BASE_FEATURES)

    # H2S lags at horizon-appropriate offsets
    for offset in horizon_cfg['lag_offsets']:
        col = f'h2s_lag_{offset}h'
        df[col] = pd.Series(h2s).shift(offset).values
        feature_cols.append(col)

    # Rolling means at horizon-appropriate staleness
    min_offset = min(horizon_cfg['lag_offsets'])
    for window in horizon_cfg['rolling_windows']:
        col = f'h2s_roll_{window}h_at_{min_offset}h'
        df[col] = pd.Series(h2s).shift(min_offset).rolling(window, min_periods=1).mean().values
        feature_cols.append(col)

    # Flow lag
    flow_lag = horizon_cfg['flow_lag']
    col = f'flow_lag_{flow_lag}h'
    df[col] = pd.Series(flow).shift(flow_lag).values
    feature_cols.append(col)

    # Flow rolling 24h ending at flow_lag
    col = f'flow_roll_24h_at_{flow_lag}h'
    df[col] = pd.Series(flow).shift(flow_lag).rolling(24, min_periods=1).mean().values
    feature_cols.append(col)

    # Daily statistics
    if horizon_cfg['use_daily_stats']:
        df['date'] = df['time'].dt.date
        daily = df.groupby('date')['H2S'].agg(['max', 'mean', 'std']).reset_index()
        daily.columns = ['date', 'daily_max', 'daily_mean', 'daily_std']
        daily['daily_std'] = daily['daily_std'].fillna(0)

        # Yesterday = the day before the target hour's date
        df['yesterday'] = (df['time'] - pd.Timedelta(days=1)).dt.date
        df = df.merge(daily.rename(columns={
            'date': 'yesterday', 'daily_max': 'yest_max',
            'daily_mean': 'yest_mean', 'daily_std': 'yest_std'
        }), on='yesterday', how='left')

        for c in ['yest_max', 'yest_mean', 'yest_std']:
            df[c] = df[c].fillna(0)
            feature_cols.append(c)

        # For 48-72h horizons, also add 2-days-ago stats
        if '48' in horizon_name:
            df['two_days_ago'] = (df['time'] - pd.Timedelta(days=2)).dt.date
            df = df.merge(daily.rename(columns={
                'date': 'two_days_ago', 'daily_max': 'twoday_max',
                'daily_mean': 'twoday_mean', 'daily_std': 'twoday_std'
            }), on='two_days_ago', how='left')
            for c in ['twoday_max', 'twoday_mean', 'twoday_std']:
                df[c] = df[c].fillna(0)
                feature_cols.append(c)

        # 7-day exceedance rate (climatological context)
        roll_7d = pd.Series(h2s).rolling(168, min_periods=24).apply(
            lambda x: (x > 5).mean(), raw=True).values
        df['exceed_rate_7d'] = pd.Series(roll_7d).shift(min_offset).values
        df['exceed_rate_7d'] = df['exceed_rate_7d'].fillna(0)
        feature_cols.append('exceed_rate_7d')

    # Spill state flag
    if horizon_cfg['use_spill_flag']:
        roll_72h_max = pd.Series(h2s).shift(min_offset).rolling(72, min_periods=1).max().values
        df['spill_active'] = (roll_72h_max > 100).astype(float)
        roll_24h_max = pd.Series(h2s).shift(min_offset).rolling(24, min_periods=1).max().values
        df['crisis_days'] = pd.Series(roll_24h_max > 30).rolling(168, min_periods=1).sum().values / 24
        df['crisis_days'] = df['crisis_days'].fillna(0)
        feature_cols.extend(['spill_active', 'crisis_days'])

    return df, feature_cols


# ============================================================
# FORECASTING: OBSERVATION STATE EXTRACTION
# ============================================================

def get_obs_state(obs_df, site_name):
    """Extract observation history for a station to seed forecast lag features.

    Args:
        obs_df: Full observation DataFrame (filtered to measured, H2S <= 500).
        site_name: Station name, e.g. 'SAN YSIDRO'.

    Returns:
        Dict with h2s_series, flow_series, times, daily_stats, last_h2s, last_flow,
        exceed_7d, spill_active, crisis_days.
    """
    ss = obs_df[obs_df['site_name'] == site_name].sort_values('time')
    if len(ss) == 0:
        return {
            'h2s_series': np.array([], dtype=float),
            'flow_series': np.array([], dtype=float),
            'times': pd.Series(dtype='datetime64[ns, UTC]'),
            'daily_stats': pd.DataFrame(),
            'last_h2s': 0, 'last_flow': 2.0,
            'exceed_7d': 0, 'spill_active': 0, 'crisis_days': 0,
        }

    # Daily stats for yesterday/two-days-ago lookups
    ss_copy = ss.copy()
    ss_copy['date'] = ss_copy['time'].dt.date
    daily = ss_copy.groupby('date')['H2S'].agg(['max', 'mean', 'std']).reset_index()
    daily.columns = ['date', 'daily_max', 'daily_mean', 'daily_std']
    daily['daily_std'] = daily['daily_std'].fillna(0)

    # 7-day exceedance rate
    h2s_arr = ss['H2S'].values
    exceed_7d = pd.Series(h2s_arr).rolling(168, min_periods=24).apply(
        lambda x: (x > 5).mean(), raw=True).iloc[-1] if len(h2s_arr) >= 24 else 0

    # Spill/crisis indicators
    last_72h = h2s_arr[-72:] if len(h2s_arr) >= 72 else h2s_arr
    spill_active = float(np.max(last_72h) > 100) if len(last_72h) > 0 else 0

    last_168h = h2s_arr[-168:] if len(h2s_arr) >= 168 else h2s_arr
    if len(last_168h) >= 24:
        daily_maxes = [np.max(last_168h[i:i+24]) for i in range(0, len(last_168h)-23, 24)]
        crisis_days = sum(1 for m in daily_maxes if m > 30)
    else:
        crisis_days = 0

    return {
        'h2s_series': ss['H2S'].values,
        'flow_series': ss[FLOW_COL].values,
        'times': ss['time'].values,
        'daily_stats': daily,
        'last_h2s': float(ss.iloc[-1]['H2S']),
        'last_flow': float(ss.iloc[-1][FLOW_COL]),
        'exceed_7d': exceed_7d,
        'spill_active': spill_active,
        'crisis_days': crisis_days,
    }


# ============================================================
# FORECASTING: BUILD HORIZON-SPECIFIC FEATURES
# ============================================================

def build_forecast_features(fc_site, obs_state, hz_name, hz_cfg):
    """Construct horizon-appropriate features for forecast hours.

    Uses observation history to seed lag/daily features honestly.
    Base features are already in the forecast parquet; this adds only
    horizon-specific H2S lags, flow lags, daily stats, and spill flags.

    Args:
        fc_site: Forecast DataFrame for one station+horizon slice.
                 Must contain BASE_FEATURES columns.
        obs_state: Dict from get_obs_state().
        hz_name: Horizon name, e.g. '0_6h'.
        hz_cfg: Horizon config from HORIZONS[hz_name].

    Returns:
        (df, feature_cols) with new horizon-specific columns added.
    """
    df = fc_site.copy()
    n = len(df)

    h2s_hist = obs_state['h2s_series']
    flow_hist = obs_state['flow_series']

    # Fill any missing base features (time cyclicals, wind features, flow derivatives, etc.)
    # The forecast parquet should already have these, but this ensures idempotence
    df = ensure_base_features(df)

    # Wind gust estimation if missing (data-specific, not in feature_builder)
    if 'wind_gusts_10m' not in df.columns or df['wind_gusts_10m'].isna().all():
        df['wind_gusts_10m'] = df['wind_speed_10m'] * 1.8

    # SBIWTP: forward-fill nulls (forecast-specific persistence assumption)
    for c in ['sbiwtp_flow_mgd', 'sbiwtp_anomaly', 'sbiwtp_deficit', 'sbiwtp_hourly_mgd', 'sbiwtp_sli']:
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    if 'sbiwtp_flow_x_temp' not in df.columns or df['sbiwtp_flow_x_temp'].isna().any():
        df['sbiwtp_flow_x_temp'] = df.get('sbiwtp_flow_mgd', pd.Series(20, index=df.index)) * df['temperature_2m']

    # Start building feature list with BASE_FEATURES (already added by ensure_base_features)
    feature_cols = list(BASE_FEATURES)

    # H2S lag features from observation history
    for offset in hz_cfg['lag_offsets']:
        col = f'h2s_lag_{offset}h'
        if len(h2s_hist) >= offset:
            vals = np.zeros(n)
            for i in range(n):
                actual_offset = offset + i
                if actual_offset <= len(h2s_hist):
                    vals[i] = h2s_hist[-(actual_offset)]
                else:
                    vals[i] = 0
            df[col] = vals
        else:
            df[col] = obs_state['last_h2s'] if len(h2s_hist) > 0 else 0
        feature_cols.append(col)

    # Rolling means with decay
    min_offset = min(hz_cfg['lag_offsets'])
    for window in hz_cfg['rolling_windows']:
        col = f'h2s_roll_{window}h_at_{min_offset}h'
        if len(h2s_hist) >= min_offset + window:
            base_val = (
                np.mean(h2s_hist[-(min_offset + window):-min_offset])
                if min_offset > 0
                else np.mean(h2s_hist[-window:])
            )
            decay = np.exp(-np.arange(n) / max(window, 1))
            df[col] = base_val * decay
        else:
            df[col] = (
                np.mean(h2s_hist[-window:])
                if len(h2s_hist) >= window
                else (obs_state['last_h2s'] if len(h2s_hist) > 0 else 0)
            )
        feature_cols.append(col)

    # Flow lag (flow changes slowly, persistence is fine)
    flow_lag = hz_cfg['flow_lag']
    col = f'flow_lag_{flow_lag}h'
    df[col] = obs_state['last_flow']
    feature_cols.append(col)

    col = f'flow_roll_24h_at_{flow_lag}h'
    df[col] = np.mean(flow_hist[-24:]) if len(flow_hist) >= 24 else obs_state['last_flow']
    feature_cols.append(col)

    # Daily stats
    if hz_cfg['use_daily_stats']:
        daily = obs_state['daily_stats']
        if len(daily) >= 1:
            yest = daily.iloc[-1]
            df['yest_max'] = yest['daily_max']
            df['yest_mean'] = yest['daily_mean']
            df['yest_std'] = yest['daily_std']
        else:
            df['yest_max'] = df['yest_mean'] = df['yest_std'] = 0
        feature_cols.extend(['yest_max', 'yest_mean', 'yest_std'])

        if '48' in hz_name and len(daily) >= 2:
            twoday = daily.iloc[-2]
            df['twoday_max'] = twoday['daily_max']
            df['twoday_mean'] = twoday['daily_mean']
            df['twoday_std'] = twoday['daily_std']
        elif '48' in hz_name:
            df['twoday_max'] = df['twoday_mean'] = df['twoday_std'] = 0

        if '48' in hz_name:
            feature_cols.extend(['twoday_max', 'twoday_mean', 'twoday_std'])

        df['exceed_rate_7d'] = obs_state.get('exceed_7d', 0)
        feature_cols.append('exceed_rate_7d')

    if hz_cfg['use_spill_flag']:
        df['spill_active'] = obs_state.get('spill_active', 0)
        df['crisis_days'] = obs_state.get('crisis_days', 0)
        feature_cols.extend(['spill_active', 'crisis_days'])

    return df, feature_cols


# ============================================================
# RISK CLASSIFICATION & SOURCE ATTRIBUTION
# ============================================================

# classify_risk imported from h2s.constants


def bearing_from(lat1, lon1, lat2, lon2):
    """Simple bearing calculation (adequate for short distances)."""
    return np.degrees(np.arctan2(lon2 - lon1, lat2 - lat1)) % 360


def find_aligned_source(site_name, wind_dir, is_night):
    """Attribute potential H2S source based on wind alignment at night."""
    if not is_night:
        return 'Daytime'
    stn = STATIONS[site_name]
    best_src, best_diff = 'Unaligned', 999
    for src_name, si in SOURCES.items():
        brg = bearing_from(stn['lat'], stn['lon'], si['lat'], si['lon'])
        diff = abs(((wind_dir - brg + 180) % 360) - 180)
        if diff < best_diff:
            best_diff = diff
            best_src = src_name if diff < 30 else 'Unaligned'
    return best_src
