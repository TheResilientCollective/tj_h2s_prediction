#!/usr/bin/env python3
"""
H2S Daily Source Attribution & Forecast System
Tijuana River Valley — 3-Station Network

Usage:
    python h2s_daily_analysis.py --obs modeldata_h2s_nofill.parquet --forecast model_forecast.parquet --spills spills.csv --output ./output

Produces:
    - Daily source attribution for last 7 days (PNG + CSV)
    - 48-hour forward model forecast (PNG + CSV)
    - Combined dashboard image
    - JSON summary for web dashboard ingestion
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ENSEMBLE WRAPPERS (needed to deserialize ensemble model pickles)
# ============================================================

class EnsembleRegressor:
    """Simple averaging ensemble of two regressors."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict(self, X):
        return self.weight_a * self.model_a.predict(X) + self.weight_b * self.model_b.predict(X)


class EnsembleClassifier:
    """Simple probability-averaging ensemble of two classifiers."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict_proba(self, X):
        return self.weight_a * self.model_a.predict_proba(X) + self.weight_b * self.model_b.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# ============================================================
# CONFIGURATION
# ============================================================

STATIONS = {
    'SAN YSIDRO':  {'lat': 32.552794, 'lon': -117.047286, 'color': '#e74c3c', 'short': 'SY'},
    'NESTOR - BES': {'lat': 32.567097, 'lon': -117.090656, 'color': '#2ecc71', 'short': 'NB'},
    'IB CIVIC CTR': {'lat': 32.576139, 'lon': -117.115361, 'color': '#3498db', 'short': 'IB'},
}

SOURCES = {
    "Stewart's Drain":      {'lat': 32.54064, 'lon': -117.05801, 'color': '#ff4444'},
    "Smuggler's Gulch":     {'lat': 32.5377,  'lon': -117.08623, 'color': '#ffaa00'},
    "Hollister St PS":      {'lat': 32.5476,  'lon': -117.088374,'color': '#ff6600'},
    "Goat Canyon":          {'lat': 32.5369,  'lon': -117.09916, 'color': '#cc44cc'},
    "Goat Canyon PS":       {'lat': 32.543476,'lon': -117.108026,'color': '#aa44aa'},
    "Del Sol Canyon":       {'lat': 32.5393,  'lon': -117.06885, 'color': '#44aacc'},
    "Silva Drain":          {'lat': 32.539743,'lon': -117.064269,'color': '#88cc44'},
}

WIND_COL = 'wind_direction_10m'
SPEED_COL = 'wind_speed_10m'
ALIGNMENT_THRESHOLD_DEG = 30

MODEL_FEATURES = [
    'temperature_2m', 'wind_speed_10m', 'wind_direction_sin', 'wind_direction_cos',
    'wind_gusts_10m', 'precipitation', 'relative_humidity_2m', 'surface_pressure',
    'cloud_cover', 'dewpoint_2m',
    'wind_speed_10m_avg_2h', 'wind_speed_10m_avg_3h', 'wind_speed_10m_avg_4h',
    'wind_gusts_10m_max_2h', 'wind_gusts_10m_max_3h', 'wind_gusts_10m_max_4h',
    'tide_height', 'tidal_state_encoded',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_night', 'source_regime',
    'flow_log', 'flow_low', 'flow_high',
    'wind_temp_interaction', 'humidity_temp_interaction',
    'stable_atm',
    'h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h',
    'h2s_rolling_6h', 'h2s_rolling_24h',
    'flow_lag_6h', 'flow_rolling_24h',
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def bearing_from(lat1, lon1, lat2, lon2):
    """Compute bearing (degrees, 0=N clockwise) from point 1 to point 2."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return np.degrees(np.arctan2(dlon, dlat)) % 360


def angular_diff(a, b):
    """Signed angular difference, wrapped to [-180, 180]."""
    return ((a - b + 180) % 360) - 180


def dist_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km between two lat/lon points."""
    dlat = lat1 - lat2
    dlon = (lon1 - lon2) * np.cos(np.radians((lat1 + lat2) / 2))
    return np.sqrt(dlat**2 + dlon**2) * 111.32


def classify_risk(prob_5, prob_10, h2s_pred):
    """Assign risk tier from predictions."""
    if prob_10 > 0.5 or h2s_pred > 30:
        return 'RED'
    elif prob_5 > 0.5 or h2s_pred > 10:
        return 'ORANGE'
    elif prob_5 > 0.25 or h2s_pred > 5:
        return 'YELLOW'
    return 'GREEN'


def estimate_emission_rate_gs(h2s_ppb, wind_speed_ms, dist_m):
    """
    Back-calculate emission rate Q (g/s) using simplified Gaussian plume.
    Ground-level source, ground-level receptor, Pasquill-Gifford class D.
    1 ppb H2S ≈ 1.42 µg/m³ at 20°C, 1 atm.
    """
    if wind_speed_ms < 0.5:
        wind_speed_ms = 0.5
    C = h2s_ppb * 1.42  # µg/m³
    sigma_y = 0.08 * dist_m / np.sqrt(1 + 0.0001 * dist_m)
    sigma_z = 0.06 * dist_m / np.sqrt(1 + 0.0015 * dist_m)
    Q_ug = C * np.pi * sigma_y * sigma_z * wind_speed_ms
    return Q_ug / 1e6  # g/s


def assign_source_regime(is_night, wind_dir):
    """Classify wind regime by direction and time of day."""
    if not is_night:
        return 0  # daytime mixed
    if 22.5 <= wind_dir < 135:
        return 1  # East
    elif wind_dir >= 247.5 or wind_dir < 22.5:
        return 2  # West
    elif 135 <= wind_dir < 247.5:
        return 3  # South
    return 0


def find_aligned_source(station_name, wind_dir, is_night, threshold=ALIGNMENT_THRESHOLD_DEG):
    """Determine which source the wind direction points to from a given station."""
    if not is_night:
        return 'Daytime (mixed)'
    stn = STATIONS[station_name]
    best_src = None
    best_diff = 999
    for src_name, src_info in SOURCES.items():
        brg = bearing_from(stn['lat'], stn['lon'], src_info['lat'], src_info['lon'])
        diff = abs(angular_diff(wind_dir, brg))
        if diff < best_diff:
            best_diff = diff
            best_src = src_name if diff < threshold else None
    return best_src if best_src else 'Unaligned'


# ============================================================
# DATA LOADING
# ============================================================

def load_observations(path):
    """Load and clean the observation dataset."""
    df = pd.read_parquet(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df[(df['h2s_measured'] == True) & (df['H2S'] <= 500)].copy()
    df['H2S'] = df['H2S'].clip(lower=0)
    return df


def load_forecast(path):
    """Load forecast data."""
    df = pd.read_parquet(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df


def load_spills(path):
    """Load spill event data."""
    df = pd.read_csv(path)
    df['Start Date'] = pd.to_datetime(df['Start Date'], format='mixed')
    df['End Date'] = pd.to_datetime(df['End Date'], format='mixed')
    return df


# ============================================================
# SOURCE ATTRIBUTION (LAST 7 DAYS)
# ============================================================

def run_source_attribution(obs_df, lookback_days=7):
    """
    Compute hourly source attribution for the recent observation window.
    Returns a DataFrame with one row per station-hour, annotated with
    the most likely source, bearing alignment, and emission rate estimate.
    """
    cutoff = obs_df['time'].max() - pd.Timedelta(days=lookback_days)
    recent = obs_df[obs_df['time'] >= cutoff].copy()

    results = []
    for _, row in recent.iterrows():
        site = row['site_name']
        h2s = row['H2S']
        wd = row[WIND_COL]
        ws = row[SPEED_COL]
        is_night = row['day_night'] == 'night'

        # Find aligned source
        aligned_src = find_aligned_source(site, wd, is_night)

        # Compute emission rate if elevated and aligned
        Q_gs = None
        if h2s > 2 and aligned_src not in ('Unaligned', 'Daytime (mixed)'):
            src = SOURCES[aligned_src]
            d = dist_km(STATIONS[site]['lat'], STATIONS[site]['lon'],
                        src['lat'], src['lon']) * 1000
            Q_gs = estimate_emission_rate_gs(h2s, ws, d)

        results.append({
            'time': row['time'],
            'station': site,
            'H2S': h2s,
            'wind_dir': wd,
            'wind_speed': ws,
            'is_night': is_night,
            'aligned_source': aligned_src,
            'emission_rate_gs': Q_gs,
            'tide_height': row.get('tide_height', None),
            'flow': row.get('Flow (m^3/s)--Border', None),
            'temperature': row.get('temperature_2m', None),
        })

    return pd.DataFrame(results)


def compute_source_probability_grid(obs_df, lookback_days=7):
    """Compute the geographic source probability surface for the recent window."""
    cutoff = obs_df['time'].max() - pd.Timedelta(days=lookback_days)
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
            wd = row[WIND_COL]
            h2s = row['H2S']
            ws = row[SPEED_COL]
            ad = angular_diff(bearing, wd)
            dw = np.exp(-0.5 * (ad / 15)**2)
            cw = np.log1p(h2s)
            td = max((ws * 3600) / 111000, 0.005)
            distw = np.exp(-0.5 * ((dist - td * 0.3) / 0.015)**2)
            prob += dw * cw * distw

    if prob.max() > 0:
        prob /= prob.max()

    return lat_grid, lon_grid, prob


# ============================================================
# FORWARD MODEL (FORECAST)
# ============================================================

def engineer_forecast_features(fc_site, last_state):
    """Add derived features to forecast data for a single station."""
    df = fc_site.copy()
    n = len(df)
    h = np.arange(n)

    # Gusts (estimate)
    df['wind_gusts_10m'] = df['wind_speed_10m'] * 1.8
    df['wind_gusts_10m_max_2h'] = df['wind_gusts_10m'].rolling(2, min_periods=1).max()
    df['wind_gusts_10m_max_3h'] = df['wind_gusts_10m'].rolling(3, min_periods=1).max()
    df['wind_gusts_10m_max_4h'] = df['wind_gusts_10m'].rolling(4, min_periods=1).max()

    # Time
    utc_h = df['time'].dt.hour
    month = df['time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * utc_h / 24)
    df['hour_cos'] = np.cos(2 * np.pi * utc_h / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['is_night'] = (df['day_night'] == 'night').astype(int)

    # Source regime
    df['source_regime'] = [
        assign_source_regime(n_, w_)
        for n_, w_ in zip(df['is_night'], df[WIND_COL])
    ]

    # Flow
    flow = df['Flow (m^3/s)--Border']
    df['flow_log'] = np.log1p(flow)
    df['flow_low'] = (flow < 1).astype(int)
    df['flow_high'] = (flow > 5).astype(int)
    df['flow_lag_6h'] = last_state['flow']
    df['flow_rolling_24h'] = last_state['flow_24h']

    # Stability
    df['stable_atm'] = ((df['wind_speed_10m'] < 5) & (df['is_night'] == 1)).astype(int)

    # Interaction features (create if missing)
    if 'wind_temp_interaction' not in df.columns:
        df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']
    if 'humidity_temp_interaction' not in df.columns:
        df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

    # H2S persistence (decay from last known)
    lh = last_state['h2s']
    lh6 = last_state['h2s_6h']
    lh24 = last_state['h2s_24h']
    decay_fast = np.exp(-h / 12)
    decay_slow = np.exp(-h / 36)

    df['h2s_lag_1h'] = np.concatenate([[lh], lh * decay_fast[:-1]])
    df['h2s_lag_3h'] = np.concatenate([[lh]*min(3,n), (lh * decay_fast)[:max(n-3,0)]])
    df['h2s_lag_6h'] = np.concatenate([[lh]*min(6,n), (lh * decay_fast)[:max(n-6,0)]])
    df['h2s_rolling_6h'] = lh6 * decay_fast
    df['h2s_rolling_24h'] = lh24 * decay_slow

    return df


def run_forward_model(fc_df, obs_df, model_dir):
    """Run the RF forward model for all stations."""
    # Get last known state
    last_state = {}
    for site in STATIONS:
        ss = obs_df[obs_df['site_name'] == site].sort_values('time')
        if len(ss) == 0:
            last_state[site] = {'h2s': 0, 'h2s_6h': 0, 'h2s_24h': 0, 'flow': 2.0, 'flow_24h': 2.0}
            continue
        last_state[site] = {
            'h2s': ss.iloc[-1]['H2S'],
            'h2s_6h': ss.tail(6)['H2S'].mean(),
            'h2s_24h': ss.tail(24)['H2S'].mean(),
            'flow': ss.iloc[-1]['Flow (m^3/s)--Border'],
            'flow_24h': ss.tail(24)['Flow (m^3/s)--Border'].mean(),
        }

    results = []
    for site, info in STATIONS.items():
        sfc = fc_df[fc_df['site_name'] == site].copy().sort_values('time').reset_index(drop=True)
        if len(sfc) == 0:
            continue

        sfc = engineer_forecast_features(sfc, last_state[site])

        # Load models
        skey = site.replace(' ', '_').replace('-', '')
        try:
            reg = pickle.load(open(os.path.join(model_dir, f'best_reg_{skey}.pkl'), 'rb'))
            clf5 = pickle.load(open(os.path.join(model_dir, f'best_clf_5ppb_{skey}.pkl'), 'rb'))
            clf10 = pickle.load(open(os.path.join(model_dir, f'best_clf_10ppb_{skey}.pkl'), 'rb'))
        except FileNotFoundError:
            print(f"  Warning: models not found for {site}, skipping")
            continue

        X = sfc[MODEL_FEATURES].values
        h2s_pred = np.clip(reg.predict(X), 0, None)
        prob_5 = clf5.predict_proba(X)[:, 1]
        prob_10 = clf10.predict_proba(X)[:, 1]

        for i in range(len(sfc)):
            is_night = bool(sfc['is_night'].iloc[i])
            wd = sfc[WIND_COL].iloc[i]
            aligned = find_aligned_source(site, wd, is_night)

            results.append({
                'time': sfc['time'].iloc[i],
                'station': site,
                'h2s_pred': round(float(h2s_pred[i]), 1),
                'prob_5': round(float(prob_5[i]) * 100, 1),
                'prob_10': round(float(prob_10[i]) * 100, 1),
                'risk': classify_risk(prob_5[i], prob_10[i], h2s_pred[i]),
                'wind_speed': round(float(sfc[SPEED_COL].iloc[i]), 1),
                'wind_dir': round(float(wd)),
                'temp': round(float(sfc['temperature_2m'].iloc[i]), 1),
                'tide': round(float(sfc['tide_height'].iloc[i]), 2),
                'flow': round(float(sfc['Flow (m^3/s)--Border'].iloc[i]), 2),
                'is_night': is_night,
                'aligned_source': aligned,
            })

    return pd.DataFrame(results)


# ============================================================
# JSON SUMMARY (for web dashboard)
# ============================================================

def generate_json_summary(attribution_df, forecast_df, obs_df):
    """Generate a JSON summary for web dashboard consumption."""
    summary = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
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
    if len(forecast_df) > 0:
        t48 = forecast_df['time'].min() + pd.Timedelta(hours=48)
        fc48 = forecast_df[forecast_df['time'] <= t48]
        for site, info in STATIONS.items():
            sf = fc48[fc48['station'] == site]
            if len(sf) == 0:
                continue
            risk_counts = sf['risk'].value_counts().to_dict()
            summary['forecast_48h'][info['short']] = {
                'max_h2s': round(float(sf['h2s_pred'].max()), 1),
                'max_prob_5': round(float(sf['prob_5'].max()), 1),
                'max_prob_10': round(float(sf['prob_10'].max()), 1),
                'hours_red': int(risk_counts.get('RED', 0)),
                'hours_orange': int(risk_counts.get('ORANGE', 0)),
                'hours_yellow': int(risk_counts.get('YELLOW', 0)),
                'hours_green': int(risk_counts.get('GREEN', 0)),
            }

    # Active sources (last 7 days attribution)
    if len(attribution_df) > 0:
        elev = attribution_df[(attribution_df['H2S'] > 5) &
                               (~attribution_df['aligned_source'].isin(['Unaligned', 'Daytime (mixed)']))]
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
                    'median_emission_gs': round(float(stats['median_Q']), 2) if pd.notna(stats['median_Q']) else None,
                }

    return summary


# ============================================================
# VISUALIZATION
# ============================================================

def generate_dashboard(attribution_df, forecast_df, obs_df, lat_grid, lon_grid, prob_grid, output_dir):
    """Generate the combined dashboard PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors

    fig = plt.figure(figsize=(24, 28))
    fig.set_facecolor('#0f0f1a')
    gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.25,
                          height_ratios=[0.3, 1.0, 1.0, 1.0, 1.0])

    run_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    fig.suptitle(f'H₂S Daily Analysis & Forecast Dashboard\n{run_str}',
                 fontsize=16, fontweight='bold', color='white', y=0.99)

    station_list = ['SAN YSIDRO', 'NESTOR - BES', 'IB CIVIC CTR']

    # --- ROW 0: Summary cards ---
    fc48 = forecast_df[forecast_df['time'] <= forecast_df['time'].min() + pd.Timedelta(hours=48)] if len(forecast_df) > 0 else pd.DataFrame()

    for idx, site in enumerate(station_list):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#1a1a2e')
        info = STATIONS[site]

        # Last observation
        ss = obs_df[obs_df['site_name'] == site].sort_values('time')
        lh = ss.iloc[-1]['H2S'] if len(ss) > 0 else 0

        sf = fc48[fc48['station'] == site] if len(fc48) > 0 else pd.DataFrame()
        max_fc = sf['h2s_pred'].max() if len(sf) > 0 else 0
        max_p5 = sf['prob_5'].max() if len(sf) > 0 else 0
        reds = (sf['risk'] == 'RED').sum() if len(sf) > 0 else 0
        oranges = (sf['risk'] == 'ORANGE').sum() if len(sf) > 0 else 0
        yellows = (sf['risk'] == 'YELLOW').sum() if len(sf) > 0 else 0

        if reds > 0: cc = '#e74c3c'
        elif oranges > 0: cc = '#e67e22'
        elif yellows > 0: cc = '#f39c12'
        else: cc = '#27ae60'

        ax.text(0.5, 0.75, site, ha='center', fontsize=13, fontweight='bold', color='white', transform=ax.transAxes)
        ax.text(0.5, 0.45, f'Last: {lh:.0f} ppb | Fcst max: {max_fc:.0f} ppb', ha='center', fontsize=9, color='#ccc', transform=ax.transAxes)
        ax.text(0.5, 0.15, f'P(>5): {max_p5:.0f}% | R:{reds} O:{oranges} Y:{yellows}', ha='center', fontsize=9, color='#aaa', transform=ax.transAxes)
        for s in ax.spines.values(): s.set_color(cc); s.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    # --- ROW 1: Recent observations (7 days) ---
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
        for s in ax.spines.values(): s.set_color('#333')

    # --- ROW 2: Source probability map + source attribution bar ---
    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_facecolor('#1a1a2e')
    colors_rgba = [(0.07,0.07,0.12,0),(1,1,0.4,0.2),(1,0.75,0,0.4),(1,0.4,0,0.6),(0.85,0.1,0,0.75),(0.5,0,0.08,0.9)]
    cmap = mcolors.LinearSegmentedColormap.from_list('prob', colors_rgba, N=256)
    levels = np.linspace(0.15, 1.0, 25)
    cf = ax.contourf(lon_grid, lat_grid, prob_grid, levels=levels, cmap=cmap)
    ax.contour(lon_grid, lat_grid, prob_grid, levels=[0.5,0.7,0.9],
               colors=['#ff9800','#ff5722','#d50000'], linewidths=[0.6,1.0,1.5], alpha=0.7)
    for site, info in STATIONS.items():
        ax.plot(info['lon'], info['lat'], '^', markersize=10, color=info['color'],
                markeredgecolor='white', markeredgewidth=1.5, zorder=20)
    for src, si in SOURCES.items():
        ax.plot(si['lon'], si['lat'], 'o', markersize=6, color=si['color'],
                markeredgecolor='white', markeredgewidth=0.8, zorder=22)
        ax.annotate(src, (si['lon'], si['lat']), textcoords="offset points", xytext=(6,-8),
                    fontsize=6, color=si['color'], bbox=dict(boxstyle='round,pad=0.15',
                    facecolor='#1a1a2e', alpha=0.85), zorder=23)
    river_lons = [-117.13,-117.12,-117.11,-117.10,-117.09,-117.08,-117.07,-117.06,-117.05,-117.04,-117.03]
    river_lats = [32.557,32.556,32.555,32.556,32.556,32.555,32.553,32.551,32.549,32.547,32.543]
    ax.plot(river_lons, river_lats, color='#4fc3f7', linewidth=1.5, alpha=0.5, zorder=15)
    ax.axhline(y=32.5355, color='#aaa', linestyle='--', linewidth=1, alpha=0.4, zorder=14)
    ax.set_xlim(lon_grid.min(), lon_grid.max()); ax.set_ylim(lat_grid.min(), lat_grid.max())
    ax.set_aspect('equal')
    ax.set_title('Source Probability — Last 7 Days', fontsize=11, fontweight='bold', color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(True, alpha=0.1, color='white')
    for s in ax.spines.values(): s.set_color('#333')

    # Source contribution bar chart
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor('#1a1a2e')
    if len(attribution_df) > 0:
        elev = attribution_df[(attribution_df['H2S'] > 5) &
                               (~attribution_df['aligned_source'].isin(['Unaligned', 'Daytime (mixed)']))]
        if len(elev) > 0:
            src_ct = elev['aligned_source'].value_counts().head(6)
            colors = [SOURCES.get(s, {'color': '#888'})['color'] for s in src_ct.index]
            ax.barh(range(len(src_ct)), src_ct.values, color=colors, alpha=0.8)
            ax.set_yticks(range(len(src_ct)))
            ax.set_yticklabels(src_ct.index, color='white', fontsize=8)
            ax.set_xlabel('Wind-aligned elevated hours', color='white', fontsize=9)
    ax.set_title('Source Attribution (7d)', fontsize=11, fontweight='bold', color='white')
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(True, alpha=0.15, color='white', axis='x')
    for s in ax.spines.values(): s.set_color('#333')

    # --- ROW 3: Forecast H2S ---
    if len(forecast_df) > 0:
        for idx, site in enumerate(station_list):
            ax = fig.add_subplot(gs[3, idx])
            ax.set_facecolor('#1a1a2e')

            # Recent obs
            obs_last3d = obs_df[(obs_df['site_name'] == site) &
                                (obs_df['time'] >= obs_df['time'].max() - pd.Timedelta(days=3))].sort_values('time')
            if len(obs_last3d) > 0:
                ax.plot(obs_last3d['time'], obs_last3d['H2S'], color=STATIONS[site]['color'],
                        linewidth=1, alpha=0.5, label='Observed')

            sf = forecast_df[forecast_df['station'] == site].sort_values('time')
            ax.plot(sf['time'], sf['h2s_pred'], color=STATIONS[site]['color'], linewidth=2, label='Forecast')
            ax.fill_between(sf['time'], 0, sf['h2s_pred'], alpha=0.2, color=STATIONS[site]['color'])
            ax.axvline(sf['time'].min(), color='yellow', linewidth=1.5, linestyle='-', alpha=0.5, label='Fcst start')
            ax.axhline(5, color='#f39c12', linewidth=0.8, linestyle='--', alpha=0.5)
            ax.axhline(30, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.5)

            ax.set_ylabel('H₂S (ppb)', color='white', fontsize=9)
            ax.set_title(f'{site} — Forecast', fontsize=10, fontweight='bold', color='white')
            ax.legend(fontsize=7, loc='upper right')
            ax.tick_params(colors='white', labelsize=7)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:00'))
            ax.grid(True, alpha=0.15, color='white')
            for s in ax.spines.values(): s.set_color('#333')

    # --- ROW 4: Exceedance probability ---
    if len(forecast_df) > 0:
        for idx, site in enumerate(station_list):
            ax = fig.add_subplot(gs[4, idx])
            ax.set_facecolor('#1a1a2e')
            sf = forecast_df[forecast_df['station'] == site].sort_values('time')
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
            for s in ax.spines.values(): s.set_color('#333')

    plt.savefig(os.path.join(output_dir, 'dashboard.png'), dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='H2S Daily Source Attribution & Forecast')
    parser.add_argument('--obs', required=True, help='Path to modeldata_h2s_nofill.parquet')
    parser.add_argument('--forecast', required=True, help='Path to model_forecast.parquet')
    parser.add_argument('--spills', default=None, help='Path to spills.csv')
    parser.add_argument('--models', default='.', help='Directory containing RF model .pkl files')
    parser.add_argument('--output', default='./output', help='Output directory')
    parser.add_argument('--lookback', type=int, default=7, help='Days of observation lookback')
    parser.add_argument('--no-plot', action='store_true', help='Skip PNG generation')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("H2S DAILY SOURCE ATTRIBUTION & FORECAST")
    print(f"Run: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    obs = load_observations(args.obs)
    print(f"  Observations: {len(obs):,} records, {obs['time'].min().date()} to {obs['time'].max().date()}")

    fc = load_forecast(args.forecast)
    print(f"  Forecast: {len(fc):,} records, {fc['time'].min()} to {fc['time'].max()}")

    spills = load_spills(args.spills) if args.spills else None
    if spills is not None:
        print(f"  Spills: {len(spills)} events")

    # 1. Source attribution (last N days)
    print(f"\nRunning source attribution (last {args.lookback} days)...")
    attr = run_source_attribution(obs, lookback_days=args.lookback)
    attr.to_csv(os.path.join(args.output, 'attribution.csv'), index=False)
    print(f"  {len(attr):,} hourly records analyzed")

    elevated = attr[attr['H2S'] > 5]
    print(f"  Elevated hours (>5 ppb): {len(elevated):,}")
    if len(elevated) > 0:
        src_summary = elevated['aligned_source'].value_counts()
        print(f"  Source alignment:")
        for src, ct in src_summary.items():
            print(f"    {src}: {ct} hours")

    # 2. Source probability grid
    print("\nComputing source probability grid...")
    lat_grid, lon_grid, prob_grid = compute_source_probability_grid(obs, lookback_days=args.lookback)

    # 3. Forward model
    print("\nRunning forward model...")
    forecast_results = run_forward_model(fc, obs, args.models)
    forecast_results.to_csv(os.path.join(args.output, 'forecast.csv'), index=False)

    if len(forecast_results) > 0:
        fc48 = forecast_results[forecast_results['time'] <= forecast_results['time'].min() + pd.Timedelta(hours=48)]
        print("\n  48-HOUR FORECAST SUMMARY:")
        for site in STATIONS:
            sf = fc48[fc48['station'] == site]
            if len(sf) == 0: continue
            rc = sf['risk'].value_counts().to_dict()
            print(f"  {site}:")
            print(f"    Max H2S: {sf['h2s_pred'].max():.1f} ppb")
            print(f"    P(>5): {sf['prob_5'].max():.0f}% | P(>10): {sf['prob_10'].max():.0f}%")
            print(f"    GREEN:{rc.get('GREEN',0)} YELLOW:{rc.get('YELLOW',0)} ORANGE:{rc.get('ORANGE',0)} RED:{rc.get('RED',0)}")

    # 4. JSON summary
    summary = generate_json_summary(attr, forecast_results, obs)
    with open(os.path.join(args.output, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # 5. Dashboard
    if not args.no_plot:
        print("\nGenerating dashboard...")
        generate_dashboard(attr, forecast_results, obs, lat_grid, lon_grid, prob_grid, args.output)
        print(f"  Saved to {args.output}/dashboard.png")

    print(f"\nAll outputs saved to {args.output}/")
    print("Done.")


if __name__ == '__main__':
    main()
