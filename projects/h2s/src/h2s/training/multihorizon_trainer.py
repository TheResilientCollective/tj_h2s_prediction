"""Multi-horizon H2S forecast training and inference utilities.

Each horizon model is a true forecaster: it maps
``(origin features, forecast-time exogenous features, lead_hour) -> H2S(origin + lead_hour)``.

Origin-anchored features (lags, rolling means, daily stats, spill flags) are
computed at the forecast issue time ``t`` and are constant across every
forecast row in a horizon slice. Forecast-time features (weather, wind,
SBIWTP, time cyclicals, tide) come from the row at ``t + lead_hour`` and
vary per row. ``lead_hour`` itself is a feature so a single model per
horizon bucket can predict every hour in the bucket.

Each horizon honestly limits which lags are observable at issue time:
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
        'lead_range': (1, 6),
    },
    '6_24h': {
        'description': '6-24 hour ahead: stale lags, daily stats emerge',
        'lag_offsets': [6, 12, 24],
        'rolling_windows': [6, 24],
        'flow_lag': 24,
        'use_daily_stats': True,
        'use_spill_flag': True,
        'lead_range': (6, 23),
    },
    '24_48h': {
        'description': '24-48 hour ahead: yesterday stats only',
        'lag_offsets': [24, 48],
        'rolling_windows': [24],
        'flow_lag': 24,
        'use_daily_stats': True,
        'use_spill_flag': True,
        'lead_range': (24, 47),
    },
    '48_72h': {
        'description': '48-72 hour ahead: 2-day-old stats + climatology',
        'lag_offsets': [48, 72],
        'rolling_windows': [24],
        'flow_lag': 48,
        'use_daily_stats': True,
        'use_spill_flag': True,
        'lead_range': (48, 71),
    },
}

HORIZON_NAMES = list(HORIZONS.keys())

# Horizon boundaries for assigning forecast hours
HORIZON_BOUNDS = [
    ('0_6h',   0,  6),
    ('6_24h',  6,  24),
    ('24_48h', 24, 48),
    ('48_72h', 48, 72),
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

# Origin-anchored columns inside BASE_FEATURES that this module replaces with
# horizon-specific equivalents. They are excluded from the per-row exogenous
# feature set (which is sourced at forecast time = t + lead_hour) because their
# meaning is "history relative to the forecast issue time t."
_ORIGIN_ANCHORED_BASE_COLS = {
    'h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h',
    'h2s_rolling_6h', 'h2s_rolling_24h',
    'flow_lag_6h', 'flow_rolling_24h',
}

BASE_FEATURES_NON_LAG = [c for c in BASE_FEATURES if c not in _ORIGIN_ANCHORED_BASE_COLS]

TASKS = ['regression', 'clf_5ppb', 'clf_10ppb', 'clf_30ppb']


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
# FEATURE COLUMN ORDERING (shared by training + inference)
# ============================================================

def _horizon_feature_cols(horizon_name, horizon_cfg):
    """Canonical ordered feature list for a horizon.

    The same function is used at training and inference so the column order
    seen by the model is identical in both phases. Order:
      1. BASE_FEATURES_NON_LAG   (forecast-time exogenous features)
      2. h2s_lag_{offset}h       (origin-anchored, in lag_offsets order)
      3. h2s_roll_{w}h_at_{min_offset}h
      4. flow_lag_{flow_lag}h, flow_roll_24h_at_{flow_lag}h
      5. yest_*, twoday_* (if '48' in name), exceed_rate_7d
      6. spill_active, crisis_days
      7. lead_hour
    """
    cols = list(BASE_FEATURES_NON_LAG)

    for offset in horizon_cfg['lag_offsets']:
        cols.append(f'h2s_lag_{offset}h')

    min_offset = min(horizon_cfg['lag_offsets'])
    for window in horizon_cfg['rolling_windows']:
        cols.append(f'h2s_roll_{window}h_at_{min_offset}h')

    flow_lag = horizon_cfg['flow_lag']
    cols.append(f'flow_lag_{flow_lag}h')
    cols.append(f'flow_roll_24h_at_{flow_lag}h')

    if horizon_cfg['use_daily_stats']:
        cols.extend(['yest_max', 'yest_mean', 'yest_std'])
        if '48' in horizon_name:
            cols.extend(['twoday_max', 'twoday_mean', 'twoday_std'])
        cols.append('exceed_rate_7d')

    if horizon_cfg['use_spill_flag']:
        cols.extend(['spill_active', 'crisis_days'])

    cols.append('lead_hour')
    return cols


# ============================================================
# TRAINING: BUILD HORIZON-SPECIFIC FEATURES (long format)
# ============================================================

def build_horizon_features(site_df, horizon_name, horizon_cfg):
    """Construct training rows for one horizon in long ``(origin, lead_hour)`` format.

    For every origin time ``t`` in ``site_df`` and every integer ``h`` in
    ``horizon_cfg['lead_range']`` we emit one training row whose:

      - origin-anchored features (lags, rolling means, daily stats, spill
        flags) are computed at ``t``,
      - forecast-time features (BASE_FEATURES_NON_LAG: weather, wind,
        SBIWTP, time cyclicals, tide, ...) are taken from the row at
        ``t + h`` — this is the same vector that will arrive at inference
        from the forecast parquet,
      - ``lead_hour`` is ``h``,
      - target is ``H2S`` (and the precomputed ``exceed_5/10/30`` flags) at
        ``t + h``.

    Pairs are skipped when either the origin lookback or the target lookup
    falls outside the available data, or when any required value is NaN.

    Args:
        site_df: DataFrame for one station from the training parquet. Must
            contain BASE_FEATURES + ``H2S`` + ``Flow (m^3/s)--Border`` +
            ``time`` + ``exceed_5`` + ``exceed_10`` + ``exceed_30``. Rows
            must be hourly so the ``t + h`` lookup is well-defined.
        horizon_name: e.g. ``'0_6h'``, ``'6_24h'``.
        horizon_cfg: dict from ``HORIZONS[horizon_name]``.

    Returns:
        (long_df, feature_cols, targets) where:
          - ``long_df`` is the expanded DataFrame including ``origin_time``
            (used for splitting only) and the feature columns;
          - ``feature_cols`` is the ordered list returned by
            :func:`_horizon_feature_cols`;
          - ``targets`` is a dict with keys ``y_reg``, ``y_5``, ``y_10``,
            ``y_30`` aligned to the rows of ``long_df``.
    """
    df = site_df.copy().sort_values('time').reset_index(drop=True)
    h2s = df['H2S'].values
    flow = df[FLOW_COL].values if FLOW_COL in df.columns else np.zeros(len(df))

    # ------------------------------------------------------------------
    # Origin-anchored features: one value per origin row at time t.
    # ------------------------------------------------------------------
    origin = pd.DataFrame({'origin_time': df['time'].reset_index(drop=True)})

    for offset in horizon_cfg['lag_offsets']:
        origin[f'h2s_lag_{offset}h'] = pd.Series(h2s).shift(offset).values

    min_offset = min(horizon_cfg['lag_offsets'])
    for window in horizon_cfg['rolling_windows']:
        origin[f'h2s_roll_{window}h_at_{min_offset}h'] = (
            pd.Series(h2s).shift(min_offset).rolling(window, min_periods=1).mean().values
        )

    flow_lag = horizon_cfg['flow_lag']
    origin[f'flow_lag_{flow_lag}h'] = pd.Series(flow).shift(flow_lag).values
    origin[f'flow_roll_24h_at_{flow_lag}h'] = (
        pd.Series(flow).shift(flow_lag).rolling(24, min_periods=1).mean().values
    )

    if horizon_cfg['use_daily_stats']:
        date_series = pd.Series(df['time'].dt.date.values, name='date')
        daily = (
            pd.DataFrame({'date': date_series, 'H2S': h2s})
            .groupby('date')['H2S'].agg(['max', 'mean', 'std']).reset_index()
        )
        daily.columns = ['date', 'daily_max', 'daily_mean', 'daily_std']
        daily['daily_std'] = daily['daily_std'].fillna(0)

        yest_key = (df['time'] - pd.Timedelta(days=1)).dt.date.values
        ydf = pd.DataFrame({'_yesterday': yest_key}).merge(
            daily.rename(columns={
                'date': '_yesterday', 'daily_max': 'yest_max',
                'daily_mean': 'yest_mean', 'daily_std': 'yest_std',
            }),
            on='_yesterday', how='left',
        )
        for c in ['yest_max', 'yest_mean', 'yest_std']:
            origin[c] = ydf[c].fillna(0).values

        if '48' in horizon_name:
            two_key = (df['time'] - pd.Timedelta(days=2)).dt.date.values
            tdf = pd.DataFrame({'_two_days_ago': two_key}).merge(
                daily.rename(columns={
                    'date': '_two_days_ago', 'daily_max': 'twoday_max',
                    'daily_mean': 'twoday_mean', 'daily_std': 'twoday_std',
                }),
                on='_two_days_ago', how='left',
            )
            for c in ['twoday_max', 'twoday_mean', 'twoday_std']:
                origin[c] = tdf[c].fillna(0).values

        roll_7d = pd.Series(h2s).rolling(168, min_periods=24).apply(
            lambda x: (x > 5).mean(), raw=True
        ).values
        origin['exceed_rate_7d'] = pd.Series(roll_7d).shift(min_offset).fillna(0).values

    if horizon_cfg['use_spill_flag']:
        roll_72h_max = pd.Series(h2s).shift(min_offset).rolling(72, min_periods=1).max().values
        origin['spill_active'] = (roll_72h_max > 100).astype(float)
        roll_24h_max = pd.Series(h2s).shift(min_offset).rolling(24, min_periods=1).max().values
        origin['crisis_days'] = (
            pd.Series(roll_24h_max > 30).rolling(168, min_periods=1).sum().fillna(0).values / 24
        )

    # ------------------------------------------------------------------
    # Forecast-time rows: per lead hour, look up row at t + h and join.
    # ------------------------------------------------------------------
    fc_cols = [c for c in BASE_FEATURES_NON_LAG if c in df.columns]
    target_cols = ['H2S']
    for c in ('exceed_5', 'exceed_10', 'exceed_30'):
        if c in df.columns:
            target_cols.append(c)

    target_base = df[['time'] + fc_cols + target_cols].copy()

    h_low, h_high = horizon_cfg['lead_range']
    pieces = []
    for h in range(int(h_low), int(h_high) + 1):
        piece = target_base.copy()
        piece['origin_time'] = piece['time'] - pd.Timedelta(hours=h)
        piece['lead_hour'] = float(h)
        piece = piece.merge(origin, on='origin_time', how='inner')
        pieces.append(piece)

    if not pieces:
        empty = pd.DataFrame()
        return empty, _horizon_feature_cols(horizon_name, horizon_cfg), {
            'y_reg': np.array([]), 'y_5': np.array([]),
            'y_10': np.array([]), 'y_30': np.array([]),
        }

    long_df = pd.concat(pieces, ignore_index=True)
    long_df = long_df.sort_values(['origin_time', 'lead_hour']).reset_index(drop=True)

    feature_cols = _horizon_feature_cols(horizon_name, horizon_cfg)

    # Drop rows where any feature is NaN (e.g. lookback ran off the start of the
    # data, or target lookup landed on a row missing some BASE_FEATURE_NON_LAG
    # column). Targets must also be present.
    required_cols = feature_cols + ['H2S']
    long_df = long_df.dropna(subset=[c for c in required_cols if c in long_df.columns])
    long_df = long_df.reset_index(drop=True)

    # exceed_* are integer flags — backfill missing as int(H2S > thr).
    if 'exceed_5' not in long_df.columns:
        long_df['exceed_5'] = (long_df['H2S'] > 5).astype(int)
    if 'exceed_10' not in long_df.columns:
        long_df['exceed_10'] = (long_df['H2S'] > 10).astype(int)
    if 'exceed_30' not in long_df.columns:
        long_df['exceed_30'] = (long_df['H2S'] > 30).astype(int)

    targets = {
        'y_reg': long_df['H2S'].values,
        'y_5':   long_df['exceed_5'].values.astype(int),
        'y_10':  long_df['exceed_10'].values.astype(int),
        'y_30':  long_df['exceed_30'].values.astype(int),
    }

    return long_df, feature_cols, targets


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

def build_forecast_features(fc_site, obs_state, hz_name, hz_cfg, origin_time):
    """Build the per-row feature matrix for one horizon slice at inference.

    Mirrors the training contract: forecast-time exogenous features come
    from each row of ``fc_site`` (already at time ``origin + lead_hour``);
    origin-anchored features (lags, rolling means, daily stats, spill
    flags) are computed once from ``obs_state`` and broadcast to every row
    of the slice; ``lead_hour`` is computed per row from
    ``df['time'] - origin_time``.

    Args:
        fc_site: Forecast DataFrame for one station+horizon slice. Must
            contain ``time`` plus the BASE_FEATURES_NON_LAG columns at
            forecast resolution (15-min in the live pipeline).
        obs_state: Dict from :func:`get_obs_state` — supplies recent H2S
            and flow history, daily stats, spill flags, etc.
        hz_name: Horizon name, e.g. ``'0_6h'``.
        hz_cfg: Horizon config from ``HORIZONS[hz_name]``.
        origin_time: Forecast issue time. Anchors ``lead_hour`` and any
            staleness adjustments.

    Returns:
        (df, feature_cols) where ``feature_cols`` matches the canonical
        ordering returned by :func:`_horizon_feature_cols` and is exactly
        the column list seen at training time.
    """
    df = fc_site.copy().reset_index(drop=True)

    h2s_hist = np.asarray(obs_state['h2s_series'], dtype=float)
    flow_hist = np.asarray(obs_state['flow_series'], dtype=float)
    last_h2s = float(obs_state.get('last_h2s', 0.0) or 0.0)
    last_flow = float(obs_state.get('last_flow', 0.0) or 0.0)

    # Idempotently ensure exogenous base features exist on each forecast row.
    df = ensure_base_features(df)

    if 'wind_gusts_10m' not in df.columns or df['wind_gusts_10m'].isna().all():
        df['wind_gusts_10m'] = df['wind_speed_10m'] * 1.8

    for c in ('sbiwtp_flow_mgd', 'sbiwtp_anomaly', 'sbiwtp_deficit',
              'sbiwtp_hourly_mgd', 'sbiwtp_sli'):
        if c in df.columns:
            df[c] = df[c].ffill().fillna(0)
    if 'sbiwtp_flow_x_temp' not in df.columns or df['sbiwtp_flow_x_temp'].isna().any():
        df['sbiwtp_flow_x_temp'] = (
            df.get('sbiwtp_flow_mgd', pd.Series(20.0, index=df.index)) * df['temperature_2m']
        )

    # ------------------------------------------------------------------
    # lead_hour per row (forecast time minus origin time, in hours).
    # ------------------------------------------------------------------
    origin_ts = pd.Timestamp(origin_time)
    if pd.api.types.is_datetime64tz_dtype(df['time']):
        if origin_ts.tz is None:
            origin_ts = origin_ts.tz_localize(df['time'].dt.tz)
        else:
            origin_ts = origin_ts.tz_convert(df['time'].dt.tz)
    df['lead_hour'] = ((df['time'] - origin_ts).dt.total_seconds() / 3600.0).astype(float)

    # ------------------------------------------------------------------
    # Origin-anchored features: one constant value broadcast to every row.
    # ------------------------------------------------------------------
    for offset in hz_cfg['lag_offsets']:
        col = f'h2s_lag_{offset}h'
        if len(h2s_hist) >= offset:
            df[col] = float(h2s_hist[-offset])
        else:
            df[col] = last_h2s if len(h2s_hist) > 0 else 0.0

    min_offset = min(hz_cfg['lag_offsets'])
    for window in hz_cfg['rolling_windows']:
        col = f'h2s_roll_{window}h_at_{min_offset}h'
        if len(h2s_hist) >= min_offset + window:
            if min_offset > 0:
                base_val = float(np.mean(h2s_hist[-(min_offset + window):-min_offset]))
            else:
                base_val = float(np.mean(h2s_hist[-window:]))
        elif len(h2s_hist) >= window:
            base_val = float(np.mean(h2s_hist[-window:]))
        else:
            base_val = last_h2s if len(h2s_hist) > 0 else 0.0
        df[col] = base_val

    flow_lag = hz_cfg['flow_lag']
    flow_lag_col = f'flow_lag_{flow_lag}h'
    if len(flow_hist) >= flow_lag:
        df[flow_lag_col] = float(flow_hist[-flow_lag])
    else:
        df[flow_lag_col] = last_flow

    flow_roll_col = f'flow_roll_24h_at_{flow_lag}h'
    if len(flow_hist) >= 24:
        df[flow_roll_col] = float(np.mean(flow_hist[-24:]))
    else:
        df[flow_roll_col] = last_flow

    if hz_cfg['use_daily_stats']:
        daily = obs_state.get('daily_stats')
        if daily is not None and len(daily) >= 1:
            yest = daily.iloc[-1]
            df['yest_max'] = float(yest['daily_max'])
            df['yest_mean'] = float(yest['daily_mean'])
            df['yest_std'] = float(yest['daily_std'])
        else:
            df['yest_max'] = df['yest_mean'] = df['yest_std'] = 0.0

        if '48' in hz_name:
            if daily is not None and len(daily) >= 2:
                twoday = daily.iloc[-2]
                df['twoday_max'] = float(twoday['daily_max'])
                df['twoday_mean'] = float(twoday['daily_mean'])
                df['twoday_std'] = float(twoday['daily_std'])
            else:
                df['twoday_max'] = df['twoday_mean'] = df['twoday_std'] = 0.0

        df['exceed_rate_7d'] = float(obs_state.get('exceed_7d', 0) or 0)

    if hz_cfg['use_spill_flag']:
        df['spill_active'] = float(obs_state.get('spill_active', 0) or 0)
        df['crisis_days'] = float(obs_state.get('crisis_days', 0) or 0)

    feature_cols = _horizon_feature_cols(hz_name, hz_cfg)
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
