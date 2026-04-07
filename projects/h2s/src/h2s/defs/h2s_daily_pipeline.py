"""H2S Daily Analysis Pipeline — Multi-Station Source Attribution & Forecast.

Ported from src/h2s_daily_analysis.py as Dagster assets.
Runs daily at 14:00 UTC (6 AM PST).

Assets:
  1. multi_station_model_artifacts — Load all 9 per-station models from S3
  2. source_attribution            — Last 7 days attribution via wind bearing + Gaussian plume
  3. daily_station_forecasts       — 48h forward prediction per station
  4. daily_dashboard_viz           — 5-row PNG dashboard
  5. daily_summary_json            — JSON summary for web dashboards
"""

import io
import json
import os
import pickle
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import (
    ALIGNMENT_THRESHOLD_DEG,
    DAILY_SUMMARY_PATH,
    FLOW_COL,
    MODEL_FEATURES,
    MODEL_PATH,
    SOURCES,
    SPEED_COL,
    STATION_MODELS_S3_BASE,
    STATION_PARTITION_MAP,
    STATIONS,
    WIND_COL,
    classify_risk,
)

ENV_LABEL = os.environ.get("ENV_LABEL", "add_ENV_LABEL").upper()
_KEY = lambda name: dg.AssetKey(["h2s", name])


# ---- Utility functions ----

def _bearing_from(lat1, lon1, lat2, lon2):
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return float(np.degrees(np.arctan2(dlon, dlat)) % 360)


def _angular_diff(a, b):
    return float(((a - b + 180) % 360) - 180)


def _dist_km(lat1, lon1, lat2, lon2):
    dlat = lat1 - lat2
    dlon = (lon1 - lon2) * np.cos(np.radians((lat1 + lat2) / 2))
    return float(np.sqrt(dlat**2 + dlon**2) * 111.32)




def _estimate_emission_rate_gs(h2s_ppb, wind_speed_ms, dist_m):
    """Back-calculate emission rate (g/s) via simplified Gaussian plume, PG class D."""
    if wind_speed_ms < 0.5:
        wind_speed_ms = 0.5
    C = h2s_ppb * 1.42  # µg/m³
    sigma_y = 0.08 * dist_m / np.sqrt(1 + 0.0001 * dist_m)
    sigma_z = 0.06 * dist_m / np.sqrt(1 + 0.0015 * dist_m)
    Q_ug = C * np.pi * sigma_y * sigma_z * wind_speed_ms
    return Q_ug / 1e6  # g/s


def _find_aligned_source(station_name, wind_dir, is_night, threshold=ALIGNMENT_THRESHOLD_DEG):
    if not is_night:
        return 'Daytime (mixed)'
    stn = STATIONS[station_name]
    best_src, best_diff = None, 999.0
    for src_name, src_info in SOURCES.items():
        brg = _bearing_from(stn['lat'], stn['lon'], src_info['lat'], src_info['lon'])
        diff = abs(_angular_diff(wind_dir, brg))
        if diff < best_diff:
            best_diff = diff
            best_src = src_name if diff < threshold else None
    return best_src if best_src else 'Unaligned'


def _assign_source_regime(is_night, wind_dir):
    if not is_night:
        return 0
    if 22.5 <= wind_dir < 135:
        return 1
    elif wind_dir >= 247.5 or wind_dir < 22.5:
        return 2
    elif 135 <= wind_dir < 247.5:
        return 3
    return 0


def _engineer_forecast_features(fc_site: pd.DataFrame, last_state: dict) -> pd.DataFrame:
    """Add derived features to a per-station forecast DataFrame."""
    df = fc_site.copy()
    n = len(df)
    h = np.arange(n)

    # Gust estimates
    df['wind_gusts_10m'] = df[SPEED_COL] * 1.8
    df['wind_gusts_10m_max_2h'] = df['wind_gusts_10m'].rolling(2, min_periods=1).max()
    df['wind_gusts_10m_max_3h'] = df['wind_gusts_10m'].rolling(3, min_periods=1).max()
    df['wind_gusts_10m_max_4h'] = df['wind_gusts_10m'].rolling(4, min_periods=1).max()

    # Wind speed rolling averages
    df['wind_speed_10m_avg_2h'] = df[SPEED_COL].rolling(2, min_periods=1).mean()
    df['wind_speed_10m_avg_3h'] = df[SPEED_COL].rolling(3, min_periods=1).mean()
    df['wind_speed_10m_avg_4h'] = df[SPEED_COL].rolling(4, min_periods=1).mean()

    # Wind direction cyclicals
    df['wind_direction_sin'] = np.sin(np.radians(df[WIND_COL]))
    df['wind_direction_cos'] = np.cos(np.radians(df[WIND_COL]))

    # Time cyclicals
    utc_h = df['time'].dt.hour
    month = df['time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * utc_h / 24)
    df['hour_cos'] = np.cos(2 * np.pi * utc_h / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)

    # Night flag and source regime
    if 'day_night' in df.columns:
        df['is_night'] = (df['day_night'] == 'night').astype(int)
    else:
        df['is_night'] = ((utc_h < 6) | (utc_h >= 20)).astype(int)

    df['source_regime'] = [
        _assign_source_regime(bool(n_), float(w_))
        for n_, w_ in zip(df['is_night'], df[WIND_COL])
    ]

    # Flow features
    if FLOW_COL in df.columns:
        flow = df[FLOW_COL]
        df['flow_log'] = np.log1p(flow)
        df['flow_low'] = (flow < 1).astype(int)
        df['flow_high'] = (flow > 5).astype(int)
    else:
        df['flow_log'] = np.log1p(last_state['flow'])
        df['flow_low'] = int(last_state['flow'] < 1)
        df['flow_high'] = int(last_state['flow'] > 5)

    df['flow_lag_6h'] = last_state['flow']
    df['flow_rolling_24h'] = last_state['flow_24h']

    # Atmospheric stability
    df['stable_atm'] = ((df[SPEED_COL] < 5) & (df['is_night'] == 1)).astype(int)

    # Interaction features
    if 'wind_temp_interaction' not in df.columns:
        df['wind_temp_interaction'] = df[SPEED_COL] * df['temperature_2m']
    if 'humidity_temp_interaction' not in df.columns:
        df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

    # H2S persistence via exponential decay from last known state
    lh = last_state['h2s']
    lh6 = last_state['h2s_6h']
    lh24 = last_state['h2s_24h']
    decay_fast = np.exp(-h / 12)
    decay_slow = np.exp(-h / 36)

    df['h2s_lag_1h'] = np.concatenate([[lh], lh * decay_fast[:-1]])
    df['h2s_lag_3h'] = np.concatenate([[lh] * min(3, n), (lh * decay_fast)[:max(n - 3, 0)]])
    df['h2s_lag_6h'] = np.concatenate([[lh] * min(6, n), (lh * decay_fast)[:max(n - 6, 0)]])
    df['h2s_rolling_6h'] = lh6 * decay_fast
    df['h2s_rolling_24h'] = lh24 * decay_slow

    # SBIWTP defaults for forecast mode (persistence)
    for col in ['sbiwtp_flow_mgd', 'sbiwtp_anomaly', 'sbiwtp_deficit',
                'sbiwtp_flow_x_temp', 'sbiwtp_hourly_mgd', 'sbiwtp_sli']:
        if col not in df.columns:
            df[col] = last_state.get(f'sbiwtp_{col.split("_", 1)[1]}', 0.0)

    # Tidal state encoding
    tidal_map = {'flood': 0, 'ebb': 1, 'slack high': 2, 'slack low': 3}
    if 'tidal_state' in df.columns and 'tidal_state_encoded' not in df.columns:
        df['tidal_state_encoded'] = df['tidal_state'].map(tidal_map).fillna(-1).astype(int)
    elif 'tidal_state_encoded' not in df.columns:
        df['tidal_state_encoded'] = -1

    return df


# ==============================================================================
# Asset 1: Load all per-station models from S3
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_daily",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Load all 9 per-station models (3 stations × 3 tasks) from S3",
)
def multi_station_model_artifacts(context: dg.AssetExecutionContext) -> dict:
    """Load regression + >5ppb + >10ppb models for all stations from S3.

    Returns nested dict: {station_name: {task: model}}
    Falls back gracefully if models for a station are not yet deployed.
    """
    s3 = context.resources.s3
    artifacts = {}
    tasks = ['regression', 'clf_5ppb', 'clf_10ppb']

    for site_name, info in STATIONS.items():
        station_key = info['key']
        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        station_models = {}

        for task in tasks:
            s3_path = f"{base_path}/{task}.pkl"
            try:
                model_bytes = s3.getFile(path=s3_path, bucket=s3.S3_BUCKET)
                model = pickle.loads(model_bytes)
                station_models[task] = model
                context.log.info(f"  ✓ {site_name} / {task}")
            except Exception as e:
                context.log.warning(f"  ✗ {site_name} / {task}: {e}")

        # Load stored feature list (ensures inference matches training shape)
        feat_path = f"{base_path}/features.json"
        try:
            feat_bytes = s3.getFile(path=feat_path, bucket=s3.S3_BUCKET)
            station_models['_feature_cols'] = json.loads(feat_bytes.decode('utf-8'))
            context.log.info(f"  ✓ {site_name} / features.json ({len(station_models['_feature_cols'])} features)")
        except Exception:
            context.log.info(f"  ⚠ {site_name} / features.json not found, will use MODEL_FEATURES default")

        if station_models:
            artifacts[site_name] = station_models

    context.log.info(f"Loaded models for {len(artifacts)} stations")
    context.add_output_metadata({
        "stations_loaded": list(artifacts.keys()),
        "tasks_per_station": {s: list(m.keys()) for s, m in artifacts.items()},
    })
    return artifacts


# ==============================================================================
# Asset 2: Source attribution (last 7 days)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_daily",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Wind-bearing source attribution for last 7 days of H2S observations",
    config_schema={
        "lookback_days": dg.Field(int, default_value=7),
        "obs_bucket": dg.Field(str, default_value="resilentpublic"),
    },
)
def source_attribution(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Compute hourly source attribution for recent H2S observations.

    For each elevated hour, identifies the most wind-aligned source location
    and back-calculates emission rate via simplified Gaussian plume.
    """
    s3 = context.resources.s3
    lookback = context.op_config["lookback_days"]
    bucket = context.op_config["obs_bucket"]

    # Load observations from S3 via presigned URL
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
    parquet_url = s3.get_presigned_url(path=s3_path, bucket=bucket)
    obs_df = pd.read_parquet(parquet_url)
    context.log.info(f"✓ Loaded observations from S3: {len(obs_df)} rows")

    obs_df['time'] = pd.to_datetime(obs_df['time'], utc=True)
    obs_df = obs_df[(obs_df['h2s_measured'] == True) & (obs_df['H2S'] <= 500)].copy()
    obs_df['H2S'] = obs_df['H2S'].clip(lower=0)

    cutoff = obs_df['time'].max() - pd.Timedelta(days=lookback)
    recent = obs_df[obs_df['time'] >= cutoff].copy()

    results = []
    for _, row in recent.iterrows():
        site = row['site_name']
        if site not in STATIONS:
            continue
        h2s = float(row['H2S'])
        wd = float(row.get(WIND_COL, 0))
        ws = float(row.get(SPEED_COL, 0))
        is_night = bool(row.get('day_night', 'day') == 'night') if 'day_night' in row.index else bool(
            (row['time'].hour < 6) or (row['time'].hour >= 20)
        )

        aligned_src = _find_aligned_source(site, wd, is_night)

        Q_gs = None
        if h2s > 2 and aligned_src not in ('Unaligned', 'Daytime (mixed)'):
            src = SOURCES[aligned_src]
            d = _dist_km(STATIONS[site]['lat'], STATIONS[site]['lon'],
                         src['lat'], src['lon']) * 1000
            Q_gs = float(_estimate_emission_rate_gs(h2s, ws, d))

        results.append({
            'time': row['time'],
            'station': site,
            'H2S': h2s,
            'wind_dir': wd,
            'wind_speed': ws,
            'is_night': is_night,
            'aligned_source': aligned_src,
            'emission_rate_gs': Q_gs,
            'tide_height': float(row.get('tide_height', 0) or 0),
            'flow': float(row.get(FLOW_COL, 0) or 0),
            'temperature': float(row.get('temperature_2m', 0) or 0),
        })

    attr_df = pd.DataFrame(results)
    context.log.info(f"✓ Attribution complete: {len(attr_df)} rows (lookback {lookback} days)")

    if len(attr_df) > 0:
        elevated = attr_df[attr_df['H2S'] > 5]
        context.log.info(f"  Elevated hours (>5 ppb): {len(elevated)}")

    context.add_output_metadata({
        "rows": len(attr_df),
        "lookback_days": lookback,
        "elevated_hours": int((attr_df['H2S'] > 5).sum()) if len(attr_df) > 0 else 0,
    })
    return attr_df


# ==============================================================================
# Asset 3: 48-hour station forecasts
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_daily",
    required_resource_keys={"s3"},
    kinds={"python", "ml"},
    description="48-hour H2S forecast per station using per-station regression + classifier models",
    ins={
        "multi_station_model_artifacts": dg.AssetIn(key=_KEY("multi_station_model_artifacts")),
    },
    config_schema={
        "obs_bucket": dg.Field(str, default_value="resilentpublic"),
        "forecast_hours": dg.Field(int, default_value=48),
    },
)
def daily_station_forecasts(
    context: dg.AssetExecutionContext,
    multi_station_model_artifacts: dict,
) -> pd.DataFrame:
    """Run 48h per-station forecasts using loaded per-station models.

    Loads recent observations for lag state initialization, then applies
    the regression + classifier models to the environmental forecast data.
    """
    s3 = context.resources.s3
    bucket = context.op_config["obs_bucket"]
    forecast_hours = context.op_config["forecast_hours"]

    # Load recent observations from S3 via presigned URL
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
    parquet_url = s3.get_presigned_url(path=s3_path, bucket=bucket)
    obs_df = pd.read_parquet(parquet_url)
    context.log.info(f"✓ Loaded observations from S3: {len(obs_df)} rows")

    obs_df['time'] = pd.to_datetime(obs_df['time'], utc=True)
    obs_df = obs_df[(obs_df['h2s_measured'] == True) & (obs_df['H2S'] <= 500)].copy()
    obs_df['H2S'] = obs_df['H2S'].clip(lower=0)

    # Load model forecast from S3 (try parquet first, then CSV)
    try:
        fc_url = s3.get_presigned_url(path="latest/tijuana/forecast_data/model_forecast.parquet")
        fc_df = pd.read_parquet(fc_url)
        context.log.info(f"✓ Loaded model forecast (parquet) from S3: {len(fc_df)} rows")
    except Exception:
        fc_url = s3.get_presigned_url(path="latest/tijuana/forecast_data/model_forecast.csv")
        fc_df = pd.read_csv(fc_url)
        context.log.info(f"✓ Loaded model forecast (csv) from S3: {len(fc_df)} rows")
    if 'time' not in fc_df.columns and 'date' in fc_df.columns:
        fc_df = fc_df.rename(columns={'date': 'time'})
    fc_df['time'] = pd.to_datetime(fc_df['time'], utc=True)

    # Load tidal forecast
    try:
        tidal_url = s3.get_presigned_url(path="latest/tijuana/tidal_forecast/latest.csv")
        tidal_df = pd.read_csv(tidal_url)
        tidal_df['time'] = pd.to_datetime(tidal_df['time'], utc=True)
        tidal_df['_mtime'] = tidal_df['time'].dt.floor('h')
        fc_df['_mtime'] = pd.to_datetime(fc_df['time']).dt.floor('h')
        fc_df = fc_df.merge(
            tidal_df[['_mtime', 'tide_height', 'tidal_state']].drop_duplicates('_mtime'),
            on='_mtime', how='left'
        ).drop(columns=['_mtime'])
    except Exception as e:
        context.log.warning(f"Tidal forecast unavailable: {e}")
        fc_df['tide_height'] = 0.5
        fc_df['tidal_state'] = 'ebb'

    # Streamflow is now included in the forecast data feed
    # Default to 2.0 m³/s if not present
    if FLOW_COL not in fc_df.columns:
        fc_df[FLOW_COL] = 2.0
        context.log.info(f"⚠ {FLOW_COL} not in forecast data, using default 2.0 m³/s")
    else:
        fc_df[FLOW_COL] = fc_df[FLOW_COL].fillna(2.0)
        context.log.info(f"✓ {FLOW_COL} loaded from forecast data")

    results = []
    for site, info in STATIONS.items():
        if site not in multi_station_model_artifacts:
            raise ValueError(f"No models for {site}")

        station_models = multi_station_model_artifacts[site]
        if 'regression' not in station_models:
            context.log.warning(f"Missing regression model for {site}, skipping")
            continue

        # Use stored feature list if available, otherwise fall back to MODEL_FEATURES
        feature_cols = station_models.get('_feature_cols', MODEL_FEATURES)

        # Get last known state for lag initialization
        ss = obs_df[obs_df['site_name'] == site].sort_values('time')
        if len(ss) > 0:
            last_state = {
                'h2s': float(ss.iloc[-1]['H2S']),
                'h2s_6h': float(ss.tail(6)['H2S'].mean()),
                'h2s_24h': float(ss.tail(24)['H2S'].mean()),
                'flow': float(ss.iloc[-1].get(FLOW_COL, 2.0) or 2.0),
                'flow_24h': float(ss.tail(24).get(FLOW_COL, pd.Series([2.0])).mean()),
            }
        else:
            last_state = {'h2s': 0, 'h2s_6h': 0, 'h2s_24h': 0, 'flow': 2.0, 'flow_24h': 2.0}

        # Replicate forecast for this station (weather is same for all)
        sfc = fc_df.head(forecast_hours).copy().reset_index(drop=True)
        sfc['site_name'] = site

        sfc = _engineer_forecast_features(sfc, last_state)

        # Fill any missing model features with 0
        for col in feature_cols:
            if col not in sfc.columns:
                sfc[col] = 0.0

        X = sfc[feature_cols].fillna(0).values

        reg = station_models['regression']
        clf5 = station_models.get('clf_5ppb')
        clf10 = station_models.get('clf_10ppb')

        h2s_pred = np.clip(reg.predict(X), 0, None)
        prob_5 = clf5.predict_proba(X)[:, 1] * 100 if clf5 else np.zeros(len(X))
        prob_10 = clf10.predict_proba(X)[:, 1] * 100 if clf10 else np.zeros(len(X))

        for i in range(len(sfc)):
            is_night = bool(sfc['is_night'].iloc[i])
            wd = float(sfc[WIND_COL].iloc[i]) if WIND_COL in sfc.columns else 0.0
            aligned = _find_aligned_source(site, wd, is_night)

            results.append({
                'time': sfc['time'].iloc[i],
                'station': site,
                'h2s_pred': round(float(h2s_pred[i]), 1),
                'prob_5': round(float(prob_5[i]), 1),
                'prob_10': round(float(prob_10[i]), 1),
                'risk': classify_risk(prob_5[i] / 100, prob_10[i] / 100, h2s_pred[i]),
                'wind_speed': round(float(sfc.get(SPEED_COL, pd.Series([0])).iloc[i]), 1),
                'wind_dir': round(float(wd)),
                'temp': round(float(sfc['temperature_2m'].iloc[i]) if 'temperature_2m' in sfc.columns else 0, 1),
                'tide': round(float(sfc['tide_height'].iloc[i]) if 'tide_height' in sfc.columns else 0, 2),
                'flow': round(float(sfc[FLOW_COL].iloc[i]) if FLOW_COL in sfc.columns else 2.0, 2),
                'is_night': is_night,
                'aligned_source': aligned,
            })

    forecast_df = pd.DataFrame(results)
    context.log.info(f"✓ Generated forecasts: {len(forecast_df)} rows ({len(STATIONS)} stations × {forecast_hours}h)")

    context.add_output_metadata({
        "rows": len(forecast_df),
        "stations": list(forecast_df['station'].unique()) if len(forecast_df) > 0 else [],
        "hours": forecast_hours,
    })
    return forecast_df


def _generate_synthetic_forecast(obs_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Generate a persistence-based synthetic forecast when weather data is unavailable."""
    last = obs_df.sort_values('time').dropna(subset=['temperature_2m', WIND_COL]).tail(1)
    if len(last) == 0:
        last_row = {'temperature_2m': 18.0, WIND_COL: 270.0, 'wind_speed_10m': 3.0,
                    'relative_humidity_2m': 75.0, 'precipitation': 0.0, 'surface_pressure': 1013.0,
                    'cloud_cover': 50.0, 'dewpoint_2m': 12.0, FLOW_COL: 2.0}
    else:
        last_row = last.iloc[0].to_dict()

    now = pd.Timestamp.now('UTC').floor('h')
    rows = []
    for i in range(hours):
        row = dict(last_row)
        row['time'] = now + pd.Timedelta(hours=i)
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# Asset 4: Dashboard visualization
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_daily",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="5-row daily dashboard PNG (obs + source map + forecasts + exceedance probs)",
    ins={
        "source_attribution": dg.AssetIn(key=_KEY("source_attribution")),
        "daily_station_forecasts": dg.AssetIn(key=_KEY("daily_station_forecasts")),
    },
    config_schema={
        "obs_bucket": dg.Field(str, default_value="resilentpublic"),
    },
)
def daily_dashboard_viz(
    context: dg.AssetExecutionContext,
    source_attribution: pd.DataFrame,
    daily_station_forecasts: pd.DataFrame,
) -> None:
    """Generate and upload the 5-row daily dashboard PNG to S3."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    s3 = context.resources.s3
    bucket = context.op_config["obs_bucket"]

    # Load observations from S3 for historical rows
    parquet_url = s3.get_presigned_url(
        path="latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet",
        bucket=bucket
    )
    obs_df = pd.read_parquet(parquet_url)
    context.log.info(f"✓ Loaded observations from S3: {len(obs_df)} rows")

    obs_df['time'] = pd.to_datetime(obs_df['time'], utc=True)
    obs_df = obs_df[(obs_df['h2s_measured'] == True) & (obs_df['H2S'] <= 500)].copy()
    obs_df['H2S'] = obs_df['H2S'].clip(lower=0)

    # Source probability grid
    lat_grid, lon_grid, prob_grid = _compute_source_probability_grid(obs_df)

    station_list = ['SAN YSIDRO', 'NESTOR - BES', 'IB CIVIC CTR']
    attr_df = source_attribution
    fc_df = daily_station_forecasts

    fig = plt.figure(figsize=(24, 28))
    fig.set_facecolor('#0f0f1a')
    gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.25,
                          height_ratios=[0.3, 1.0, 1.0, 1.0, 1.0])

    run_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f'H₂S Daily Analysis & Forecast Dashboard\n{run_str}',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    fc48 = (fc_df[fc_df['time'] <= fc_df['time'].min() + pd.Timedelta(hours=48)]
            if len(fc_df) > 0 else pd.DataFrame())

    # Row 0: Summary cards
    for idx, site in enumerate(station_list):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#1a1a2e')
        ss = obs_df[obs_df['site_name'] == site].sort_values('time')
        lh = float(ss.iloc[-1]['H2S']) if len(ss) > 0 else 0
        sf = fc48[fc48['station'] == site] if len(fc48) > 0 else pd.DataFrame()
        max_fc = float(sf['h2s_pred'].max()) if len(sf) > 0 else 0
        max_p5 = float(sf['prob_5'].max()) if len(sf) > 0 else 0
        oranges = int((sf['risk'] == 'ORANGE').sum()) if len(sf) > 0 else 0
        yellow_highs = int((sf['risk'] == 'YELLOW_HIGH').sum()) if len(sf) > 0 else 0
        yellow_lows = int((sf['risk'] == 'YELLOW_LOW').sum()) if len(sf) > 0 else 0
        cc = '#e74c3c' if oranges > 0 else '#e67e22' if yellow_highs > 0 else '#f39c12' if yellow_lows > 0 else '#27ae60'
        ax.text(0.5, 0.75, site, ha='center', fontsize=13, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.45, f'Last: {lh:.0f} ppb | Fcst max: {max_fc:.0f} ppb', ha='center', fontsize=9, color='#ccc', transform=ax.transAxes)
        ax.text(0.5, 0.15, f'P(>5): {max_p5:.0f}% | O:{oranges} YH:{yellow_highs} YL:{yellow_lows}', ha='center', fontsize=9, color='#aaa', transform=ax.transAxes)
        for s_ in ax.spines.values():
            s_.set_color(cc)
            s_.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Row 1: Recent observations (7 days)
    recent = obs_df[obs_df['time'] >= obs_df['time'].max() - pd.Timedelta(days=7)]
    for idx, site in enumerate(station_list):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor('#1a1a2e')
        ss = recent[recent['site_name'] == site].sort_values('time')
        if len(ss) > 0:
            ax.plot(ss['time'], ss['H2S'], color=STATIONS[site]['color'], linewidth=1.2)
            ax.fill_between(ss['time'], 0, ss['H2S'], alpha=0.15, color=STATIONS[site]['color'])
        ax.axhline(5, color='#f39c12', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(30, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_ylabel('H₂S (ppb)', color='white', fontsize=9)
        ax.set_title(f'{site} — Last 7 Days', fontsize=10, fontweight='bold', color='white')
        ax.tick_params(colors='white', labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.grid(True, alpha=0.15, color='white')
        for s_ in ax.spines.values():
            s_.set_color('#333')

    # Row 2: Source probability map + attribution bar
    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_facecolor('#1a1a2e')
    colors_rgba = [(0.07, 0.07, 0.12, 0), (1, 1, 0.4, 0.2), (1, 0.75, 0, 0.4),
                   (1, 0.4, 0, 0.6), (0.85, 0.1, 0, 0.75), (0.5, 0, 0.08, 0.9)]
    cmap = mcolors.LinearSegmentedColormap.from_list('prob', colors_rgba, N=256)
    levels = np.linspace(0.15, 1.0, 25)
    ax.contourf(lon_grid, lat_grid, prob_grid, levels=levels, cmap=cmap)
    ax.contour(lon_grid, lat_grid, prob_grid, levels=[0.5, 0.7, 0.9],
               colors=['#ff9800', '#ff5722', '#d50000'], linewidths=[0.6, 1.0, 1.5], alpha=0.7)
    from matplotlib.lines import Line2D
    contour_legend = [
        Line2D([0], [0], color='#ff9800', linewidth=0.6, alpha=0.7, label='50% probability'),
        Line2D([0], [0], color='#ff5722', linewidth=1.0, alpha=0.7, label='70% probability'),
        Line2D([0], [0], color='#d50000', linewidth=1.5, alpha=0.7, label='90% probability'),
    ]
    ax.legend(handles=contour_legend, loc='lower left', fontsize=7, framealpha=0.75,
              facecolor='#1a1a2e', labelcolor='white', edgecolor='#555')
    for site, info in STATIONS.items():
        ax.plot(info['lon'], info['lat'], '^', markersize=10, color=info['color'],
                markeredgecolor='white', markeredgewidth=1.5, zorder=20)
    for src, si in SOURCES.items():
        ax.plot(si['lon'], si['lat'], 'o', markersize=6, color=si['color'],
                markeredgecolor='white', markeredgewidth=0.8, zorder=22)
        ax.annotate(src, (si['lon'], si['lat']), textcoords="offset points", xytext=(6, -8),
                    fontsize=6, color=si['color'],
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#1a1a2e', alpha=0.85), zorder=23)
    river_lons = [-117.13, -117.12, -117.11, -117.10, -117.09, -117.08, -117.07, -117.06, -117.05, -117.04, -117.03]
    river_lats = [32.557, 32.556, 32.555, 32.556, 32.556, 32.555, 32.553, 32.551, 32.549, 32.547, 32.543]
    ax.plot(river_lons, river_lats, color='#4fc3f7', linewidth=1.5, alpha=0.5, zorder=15)
    ax.set_xlim(lon_grid.min(), lon_grid.max())
    ax.set_ylim(lat_grid.min(), lat_grid.max())
    ax.set_aspect('equal')
    ax.set_title('Source Probability — Last 7 Days', fontsize=11, fontweight='bold', color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(True, alpha=0.1, color='white')
    for s_ in ax.spines.values():
        s_.set_color('#333')

    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor('#1a1a2e')
    if len(attr_df) > 0:
        elev = attr_df[(attr_df['H2S'] > 5) &
                       (~attr_df['aligned_source'].isin(['Unaligned', 'Daytime (mixed)']))]
        if len(elev) > 0:
            src_ct = elev['aligned_source'].value_counts().head(6)
            colors_bar = [SOURCES.get(s, {'color': '#888'})['color'] for s in src_ct.index]
            ax.barh(range(len(src_ct)), src_ct.values, color=colors_bar, alpha=0.8)
            ax.set_yticks(range(len(src_ct)))
            ax.set_yticklabels(src_ct.index, color='white', fontsize=8)
            ax.set_xlabel('Wind-aligned elevated hours', color='white', fontsize=9)
    ax.set_title('Source Attribution (7d)', fontsize=11, fontweight='bold', color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(True, alpha=0.15, color='white', axis='x')
    for s_ in ax.spines.values():
        s_.set_color('#333')

    # Row 3: Forecast H2S
    if len(fc_df) > 0:
        for idx, site in enumerate(station_list):
            ax = fig.add_subplot(gs[3, idx])
            ax.set_facecolor('#1a1a2e')
            obs_last3d = obs_df[
                (obs_df['site_name'] == site) &
                (obs_df['time'] >= obs_df['time'].max() - pd.Timedelta(days=3))
            ].sort_values('time')
            if len(obs_last3d) > 0:
                ax.plot(obs_last3d['time'], obs_last3d['H2S'], color=STATIONS[site]['color'],
                        linewidth=1, alpha=0.5, label='Observed')
            sf = fc_df[fc_df['station'] == site].sort_values('time')
            ax.plot(sf['time'], sf['h2s_pred'], color=STATIONS[site]['color'], linewidth=2, label='Forecast')
            ax.fill_between(sf['time'], 0, sf['h2s_pred'], alpha=0.2, color=STATIONS[site]['color'])
            if len(sf) > 0:
                ax.axvline(sf['time'].min(), color='yellow', linewidth=1.5, linestyle='-', alpha=0.5)
            ax.axhline(5, color='#f39c12', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.axhline(30, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.set_ylabel('H₂S (ppb)', color='white', fontsize=9)
            ax.set_title(f'{site} — Forecast', fontsize=10, fontweight='bold', color='white')
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(colors='white', labelsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
            ax.grid(True, alpha=0.15, color='white')
            for s_ in ax.spines.values():
                s_.set_color('#333')

    # Row 4: Exceedance probabilities
    if len(fc_df) > 0:
        for idx, site in enumerate(station_list):
            ax = fig.add_subplot(gs[4, idx])
            ax.set_facecolor('#1a1a2e')
            sf = fc_df[fc_df['station'] == site].sort_values('time')
            ax.fill_between(sf['time'], 0, sf['prob_5'], alpha=0.3, color='#f39c12', label='P(>5)')
            ax.plot(sf['time'], sf['prob_5'], color='#f39c12', linewidth=2)
            ax.fill_between(sf['time'], 0, sf['prob_10'], alpha=0.4, color='#e74c3c', label='P(>10)')
            ax.plot(sf['time'], sf['prob_10'], color='#e74c3c', linewidth=2)
            ax.axhline(50, color='white', linewidth=0.5, linestyle=':', alpha=0.3)
            ax.set_ylim(0, 100)
            ax.set_ylabel('Probability (%)', color='white', fontsize=9)
            ax.set_title(f'{site} — Exceedance Prob', fontsize=10, fontweight='bold', color='white')
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(colors='white', labelsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
            ax.grid(True, alpha=0.15, color='white')
            for s_ in ax.spines.values():
                s_.set_color('#333')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    buf.seek(0)
    img_bytes = buf.read()

    # Upload to S3 (latest + timestamped)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')
    latest_path = "latest/tijuana/forecast/visualizations/daily_dashboard.png"
    ts_path = f"{DAILY_SUMMARY_PATH}/{ts}/daily_dashboard.png"

    for path in [latest_path, ts_path]:
        try:
            s3.putFile(img_bytes, path, bucket=s3.S3_BUCKET, content_type='image/png')
            context.log.info(f"✓ Uploaded dashboard to S3: {path}")
        except Exception as e:
            context.log.warning(f"Upload failed ({path}): {e}")

    context.add_output_metadata({"image_size_kb": len(img_bytes) // 1024})


def _compute_source_probability_grid(obs_df: pd.DataFrame):
    """Compute geographic source probability surface from recent observations."""
    cutoff = obs_df['time'].max() - pd.Timedelta(days=7)
    recent = obs_df[obs_df['time'] >= cutoff]

    lat_min, lat_max = 32.525, 32.595
    lon_min, lon_max = -117.135, -117.025
    grid_res = 0.0006
    lats = np.arange(lat_min, lat_max, grid_res)
    lons = np.arange(lon_min, lon_max, grid_res)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    prob = np.zeros_like(lon_grid)

    for site, info in STATIONS.items():
        sdf = recent[(recent['site_name'] == site) & (recent['H2S'] > 1)]
        slat, slon = info['lat'], info['lon']
        dlat = lat_grid - slat
        dlon = lon_grid - slon
        bearing = np.degrees(np.arctan2(dlon, dlat)) % 360
        dist = np.sqrt(dlat**2 + (dlon * np.cos(np.radians(slat)))**2)

        for _, row in sdf.iterrows():
            wd = float(row.get(WIND_COL, 0))
            h2s = float(row['H2S'])
            ws = float(row.get(SPEED_COL, 1))
            ad = ((bearing - wd + 180) % 360) - 180
            dw = np.exp(-0.5 * (ad / 15) ** 2)
            cw = np.log1p(h2s)
            td = max((ws * 3600) / 111000, 0.005)
            distw = np.exp(-0.5 * ((dist - td * 0.3) / 0.015) ** 2)
            prob += dw * cw * distw

    if prob.max() > 0:
        prob /= prob.max()
    return lat_grid, lon_grid, prob


# ==============================================================================
# Asset 5: JSON summary for web dashboards
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_daily",
    required_resource_keys={"s3", "slack"},
    kinds={"json", "s3", "slack"},
    description="JSON summary for web dashboards (station stats, 48h rollup, active sources)",
    ins={
        "source_attribution": dg.AssetIn(key=_KEY("source_attribution")),
        "daily_station_forecasts": dg.AssetIn(key=_KEY("daily_station_forecasts")),
    },
    deps=[_KEY("daily_station_forecasts")],
    config_schema={
        "obs_bucket": dg.Field(str, default_value="resilentpublic"),
    },
)
def daily_summary_json(
    context: dg.AssetExecutionContext,
    source_attribution: pd.DataFrame,
    daily_station_forecasts: pd.DataFrame,
) -> dict:
    """Generate and upload daily summary JSON to S3."""
    s3 = context.resources.s3
    bucket = context.op_config["obs_bucket"]

    # Load observations from S3 for recent stats
    parquet_url = s3.get_presigned_url(
        path="latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet",
        bucket=bucket
    )
    obs_df = pd.read_parquet(parquet_url)
    context.log.info(f"✓ Loaded observations from S3: {len(obs_df)} rows")

    obs_df['time'] = pd.to_datetime(obs_df['time'], utc=True)
    obs_df = obs_df[(obs_df['h2s_measured'] == True) & (obs_df['H2S'] <= 500)].copy()
    obs_df['H2S'] = obs_df['H2S'].clip(lower=0)

    attr_df = source_attribution
    fc_df = daily_station_forecasts

    summary = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'stations': {},
        'forecast_48h': {},
        'active_sources': {},
    }

    # Per-station observation summary (last 24h)
    cutoff_24h = obs_df['time'].max() - pd.Timedelta(hours=24)
    for site, info in STATIONS.items():
        ss = obs_df[(obs_df['site_name'] == site) & (obs_df['time'] >= cutoff_24h)]
        if len(ss) == 0:
            continue
        summary['stations'][info['short']] = {
            'name': site,
            'last_h2s': round(float(ss.iloc[-1]['H2S']), 1),
            'last_time': ss.iloc[-1]['time'].isoformat(),
            'mean_24h': round(float(ss['H2S'].mean()), 1),
            'max_24h': round(float(ss['H2S'].max()), 1),
            'pct_exceed_5': round(float((ss['H2S'] > 5).mean() * 100), 1),
        }

    # Forecast 48h summary
    if len(fc_df) > 0:
        t48 = fc_df['time'].min() + pd.Timedelta(hours=48)
        fc48 = fc_df[fc_df['time'] <= t48]
        for site, info in STATIONS.items():
            sf = fc48[fc48['station'] == site]
            if len(sf) == 0:
                continue
            risk_counts = sf['risk'].value_counts().to_dict()
            summary['forecast_48h'][info['short']] = {
                'max_h2s': round(float(sf['h2s_pred'].max()), 1),
                'max_prob_5': round(float(sf['prob_5'].max()), 1),
                'max_prob_10': round(float(sf['prob_10'].max()), 1),
                'hours_orange': int(risk_counts.get('ORANGE', 0)),
                'hours_yellow_high': int(risk_counts.get('YELLOW_HIGH', 0)),
                'hours_yellow_low': int(risk_counts.get('YELLOW_LOW', 0)),
                'hours_green': int(risk_counts.get('GREEN', 0)),
            }

    # Active sources from attribution
    if len(attr_df) > 0:
        elev = attr_df[(attr_df['H2S'] > 5) &
                       (~attr_df['aligned_source'].isin(['Unaligned', 'Daytime (mixed)']))]
        if len(elev) > 0:
            src_stats = elev.groupby('aligned_source').agg(
                hours=('H2S', 'count'),
                mean_h2s=('H2S', 'mean'),
                max_h2s=('H2S', 'max'),
                median_Q=('emission_rate_gs', 'median'),
            ).to_dict('index')
            for src, stats in src_stats.items():
                summary['active_sources'][src] = {
                    'aligned_hours': int(stats['hours']),
                    'mean_h2s': round(float(stats['mean_h2s']), 1),
                    'max_h2s': round(float(stats['max_h2s']), 1),
                    'median_emission_gs': round(float(stats['median_Q']), 2) if pd.notna(stats.get('median_Q')) else None,
                }

    # Upload to S3
    summary_bytes = json.dumps(summary, indent=2, default=str).encode('utf-8')
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')
    for path in [
        "latest/tijuana/forecast_data/daily_summary.json",
        f"{DAILY_SUMMARY_PATH}/{ts}/daily_summary.json",
    ]:
        try:
            s3.putFile(summary_bytes, path, bucket=s3.S3_BUCKET, content_type='application/json')
            context.log.info(f"✓ Uploaded summary JSON to S3: {path}")
        except Exception as e:
            context.log.warning(f"Upload failed ({path}): {e}")

    context.add_output_metadata({
        "stations_in_summary": list(summary['stations'].keys()),
        "active_sources": list(summary['active_sources'].keys()),
    })

    # --- Slack summary ---
    station_fields = []
    for short, stats in summary.get('stations', {}).items():
        fc = summary.get('forecast_48h', {}).get(short, {})
        hours_orange = fc.get('hours_orange', 0)
        hours_yellow = fc.get('hours_yellow_high', 0) + fc.get('hours_yellow_low', 0)
        risk_icon = "🟠" if hours_orange > 0 else ("🟡" if hours_yellow > 0 else "🟢")
        station_fields.append({
            "type": "mrkdwn",
            "text": (
                f"*{risk_icon} {stats.get('name', short)}*\n"
                f"Now: {stats.get('last_h2s', 'N/A')} ppb | "
                f"24h max: {stats.get('max_24h', 'N/A')} ppb\n"
                f"48h forecast: {fc.get('max_h2s', 'N/A')} ppb max "
                f"({hours_orange}h orange, {hours_yellow}h yellow)"
            ),
        })

    active_sources = summary.get('active_sources', {})
    source_text = (
        ", ".join(
            f"{src} ({s['aligned_hours']}h, {s['max_h2s']} ppb max)"
            for src, s in active_sources.items()
        )
        if active_sources
        else "None"
    )

    # Forecast time range in Pacific time
    pacific = ZoneInfo("America/Los_Angeles")
    if len(fc_df) > 0:
        fc_start = fc_df['time'].min().astimezone(pacific).strftime("%-I %p %-m/%-d")
        fc_end = fc_df['time'].max().astimezone(pacific).strftime("%-I %p %-m/%-d")
        time_range = f"{fc_start} → {fc_end} PT"
    else:
        time_range = "N/A"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{ENV_LABEL} H2S Daily Summary — {ts}"},
        },
        {
            "type": "section",
            "fields": station_fields,
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"*Forecast window:* {time_range}"},
                {"type": "mrkdwn", "text": f"*Active sources:* {source_text}"},
                {"type": "mrkdwn", "text": f"<{context.resources.s3.publicUrl(f'{DAILY_SUMMARY_PATH}/{ts}/daily_dashboard.png', bucket=context.resources.s3.S3_BUCKET)}|View Dashboard>"},
            ],
        },
    ]

    try:
        slack = context.resources.slack
        client = slack.get_client()
        client.chat_postMessage(
            channel=slack.channel,
            text=f"H2S Daily Summary {ts}",
            blocks=blocks,
        )
        context.log.info("Slack daily summary sent")
    except Exception as e:
        context.log.warning(f"Slack notification failed: {e}")

    return summary


# ==============================================================================
# Job definition
# ==============================================================================

daily_analysis_job = dg.define_asset_job(
    name="daily_analysis_job",
    description="Run daily H2S source attribution + 48h forecast + dashboard + JSON summary",
    selection=dg.AssetSelection.assets(
        multi_station_model_artifacts,
        source_attribution,
        daily_station_forecasts,
        daily_dashboard_viz,
        daily_summary_json,
    ),
    tags={"environment": "production", "pipeline": "h2s_daily"},
)
