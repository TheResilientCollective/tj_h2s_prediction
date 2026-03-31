#!/usr/bin/env python3
"""
H2S Prediction System
=====================

Generates H2S predictions from CSV data using trained models produced by
train_models_auto.py. Uses regression + binary classifiers to assign risk tiers
(GREEN/YELLOW_LOW/YELLOW_HIGH/ORANGE), consistent with SD County guidance.

Runs all stations by default; use --site to filter to one.

Usage:
    python predict_h2s.py --input data.parquet --models ./models
    python predict_h2s.py --input data.csv --models ./models --site "IB CIVIC CTR"
    python predict_h2s.py --input data.csv --models ./models --filter-alerts

Requirements:
    - Model .pkl files from train_models_auto.py (best_reg_*, best_clf_5ppb_*, best_clf_10ppb_*)
    - pandas, numpy, scikit-learn
"""

import os
import pickle
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# Ensemble classes must be importable for pickle to load models saved by train_models_auto.py
class EnsembleRegressor:
    """Simple averaging ensemble of two regressors."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a, self.model_b = model_a, model_b
        self.weight_a, self.weight_b = weight_a, 1.0 - weight_a
    def predict(self, X):
        return self.weight_a * self.model_a.predict(X) + self.weight_b * self.model_b.predict(X)


class EnsembleClassifier:
    """Simple probability-averaging ensemble of two classifiers."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a, self.model_b = model_a, model_b
        self.weight_a, self.weight_b = weight_a, 1.0 - weight_a
    def predict_proba(self, X):
        return self.weight_a * self.model_a.predict_proba(X) + self.weight_b * self.model_b.predict_proba(X)
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# Same feature list as train_models_auto.py and h2s_daily_analysis.py
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

# Station name → model file key (same mapping as train_models_auto.py)
STATION_KEYS = {
    'SAN YSIDRO':   'SAN_YSIDRO',
    'NESTOR - BES': 'NESTOR__BES',
    'IB CIVIC CTR': 'IB_CIVIC_CTR',
}

# Tidal state encoding (matching h2s_daily_analysis.py)
TIDAL_ENCODING = {'ebb': 0, 'flood': 1, 'slack': 2, 'slack high': 3, 'slack low': 4}


def classify_risk(prob_5, prob_10, h2s_pred):
    """Assign risk tier from predictions (SD County guidance).

    GREEN:       H2S < 5 ppb
    YELLOW_LOW:  5 ≤ H2S < 10 ppb  (prob >5 classifier)
    YELLOW_HIGH: 10 ≤ H2S < 30 ppb (prob >10 classifier)
    ORANGE:      H2S ≥ 30 ppb
    """
    if prob_10 > 0.5 or h2s_pred > 30:
        return 'ORANGE'
    elif prob_5 > 0.5 or h2s_pred > 10:
        return 'YELLOW_HIGH'
    elif prob_5 > 0.25 or h2s_pred > 5:
        return 'YELLOW_LOW'
    return 'GREEN'


class _ModelUnpickler(pickle.Unpickler):
    """Resolve EnsembleRegressor/EnsembleClassifier regardless of which module pickled them."""
    _CLASSES = {
        'EnsembleRegressor': EnsembleRegressor,
        'EnsembleClassifier': EnsembleClassifier,
    }
    def find_class(self, module, name):
        if name in self._CLASSES:
            return self._CLASSES[name]
        return super().find_class(module, name)


def load_models(models_dir, station_key):
    """Load regression + classifier models for a station."""
    def _load(path):
        with open(path, 'rb') as f:
            return _ModelUnpickler(f).load()
    reg = _load(os.path.join(models_dir, f'best_reg_{station_key}.pkl'))
    clf5 = _load(os.path.join(models_dir, f'best_clf_5ppb_{station_key}.pkl'))
    clf10 = _load(os.path.join(models_dir, f'best_clf_10ppb_{station_key}.pkl'))
    return reg, clf5, clf10


def engineer_features(df):
    """Engineer features to match train_models_auto.py format."""
    df = df.copy()

    # Time features
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        hour = df['time'].dt.hour
        month = df['time'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['is_night'] = ((hour < 6) | (hour >= 20)).astype(int)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        hour = df['date'].dt.hour
        month = df['date'].dt.month
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)
        df['is_night'] = ((hour < 6) | (hour >= 20)).astype(int)

    # Wind direction cyclicals
    if 'wind_direction_10m' in df.columns:
        rad = np.deg2rad(df['wind_direction_10m'].fillna(0))
        df['wind_direction_sin'] = np.sin(rad)
        df['wind_direction_cos'] = np.cos(rad)

    # Wind rolling averages and gust maxes
    if 'wind_gusts_10m' not in df.columns and 'wind_speed_10m' in df.columns:
        df['wind_gusts_10m'] = df['wind_speed_10m'] * 1.8
    for h in (2, 3, 4):
        if f'wind_speed_10m_avg_{h}h' not in df.columns and 'wind_speed_10m' in df.columns:
            df[f'wind_speed_10m_avg_{h}h'] = df['wind_speed_10m'].rolling(h, min_periods=1).mean()
        if f'wind_gusts_10m_max_{h}h' not in df.columns and 'wind_gusts_10m' in df.columns:
            df[f'wind_gusts_10m_max_{h}h'] = df['wind_gusts_10m'].rolling(h, min_periods=1).max()

    # Interaction features
    if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
        df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']
    if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
        df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

    # Source regime
    if 'is_night' in df.columns and 'wind_direction_10m' in df.columns:
        def _source_regime(row):
            if not row['is_night']:
                return 0
            wd = row.get('wind_direction_10m', 0)
            if 22.5 <= wd < 135:
                return 1
            elif wd >= 247.5 or wd < 22.5:
                return 2
            elif 135 <= wd < 247.5:
                return 3
            return 0
        df['source_regime'] = df.apply(_source_regime, axis=1)

    # Atmospheric stability
    if 'wind_speed_10m' in df.columns and 'is_night' in df.columns:
        df['stable_atm'] = ((df['wind_speed_10m'] < 5) & (df['is_night'] == 1)).astype(int)

    # Tidal state encoding
    if 'tidal_state' in df.columns and 'tidal_state_encoded' not in df.columns:
        df['tidal_state_encoded'] = df['tidal_state'].map(TIDAL_ENCODING).fillna(-1).astype(int)

    # Tide height alias
    if 'tide_height_m' in df.columns and 'tide_height' not in df.columns:
        df['tide_height'] = df['tide_height_m']

    # Flow features
    flow_col = None
    for c in ['Flow (m^3/s)--Border', 'flow_rate_cms']:
        if c in df.columns:
            flow_col = c
            break
    if flow_col:
        df['flow_log'] = np.log1p(df[flow_col])
        df['flow_low'] = (df[flow_col] < 1).astype(int)
        df['flow_high'] = (df[flow_col] > 5).astype(int)
        df['flow_lag_6h'] = df[flow_col].shift(6).fillna(df[flow_col].median())
        df['flow_rolling_24h'] = df[flow_col].rolling(24, min_periods=1).mean()
    else:
        for c in ['flow_log', 'flow_low', 'flow_high', 'flow_lag_6h', 'flow_rolling_24h']:
            if c not in df.columns:
                df[c] = 0.0

    # H2S lags — forecast mode, no measurements available
    for col in ('h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h', 'h2s_rolling_6h', 'h2s_rolling_24h'):
        if col not in df.columns:
            df[col] = 0.0

    return df


def predict(df, reg, clf5, clf10):
    """Run prediction using regression + classifiers, return results DataFrame."""
    avail = [f for f in MODEL_FEATURES if f in df.columns]
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} missing features (zero-filled): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        for f in missing:
            df[f] = 0.0

    X = df[MODEL_FEATURES].fillna(0.0).values
    h2s_pred = np.clip(reg.predict(X), 0, None)
    prob_5 = clf5.predict_proba(X)[:, 1]
    prob_10 = clf10.predict_proba(X)[:, 1]

    result = df.copy()
    result['h2s_predicted'] = np.round(h2s_pred, 1)
    result['prob_exceed_5ppb'] = np.round(prob_5 * 100, 1)
    result['prob_exceed_10ppb'] = np.round(prob_10 * 100, 1)
    result['risk'] = [classify_risk(prob_5[i], prob_10[i], h2s_pred[i]) for i in range(len(df))]
    result['alert'] = result['risk'].isin(['ORANGE', 'YELLOW_HIGH'])

    return result


def _predict_station(df, site, skey, models_dir, filter_alerts):
    """Run prediction for a single station. Returns results DataFrame or None."""
    # Filter to site
    if 'site_name' in df.columns:
        sdf = df[df['site_name'] == site].copy()
        if len(sdf) == 0:
            print(f"  No data for {site}, skipping")
            return None
    else:
        sdf = df.copy()

    # Load models
    try:
        reg, clf5, clf10 = load_models(models_dir, skey)
    except FileNotFoundError:
        print(f"  Models not found for {skey}, skipping")
        return None

    # Engineer features + predict
    sdf = engineer_features(sdf)
    results = predict(sdf, reg, clf5, clf10)
    results['station'] = site

    if filter_alerts:
        results = results[results['alert']].copy()

    # Per-station summary
    risk_counts = results['risk'].value_counts()
    parts = [f"{tier}:{risk_counts.get(tier, 0)}" for tier in ['GREEN', 'YELLOW_LOW', 'YELLOW_HIGH', 'ORANGE']]
    print(f"  {len(results)} rows | {' | '.join(parts)}")

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate H2S predictions using trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # All stations (default)
  python predict_h2s.py --input data.parquet --models ./models

  # Single station
  python predict_h2s.py --input data.csv --models ./models --site "IB CIVIC CTR"

  # Alerts only (ORANGE + YELLOW_HIGH)
  python predict_h2s.py --input data.csv --models ./models --filter-alerts

  # Output to a specific directory
  python predict_h2s.py --input data.csv --models ./models --output ./results
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV or parquet with environmental data')
    parser.add_argument('--output', '-o', default='.',
                        help='Output directory (default: current directory)')
    parser.add_argument('--models', required=True,
                        help='Directory containing model .pkl files from train_models_auto.py')
    parser.add_argument('--filter-alerts', action='store_true',
                        help='Only output ORANGE and YELLOW_HIGH predictions')
    parser.add_argument('--site', default=None,
                        choices=list(STATION_KEYS.keys()),
                        help='Station to predict for (default: all stations)')

    args = parser.parse_args()

    sites = {args.site: STATION_KEYS[args.site]} if args.site else STATION_KEYS

    # Header
    print("=" * 70)
    print("H2S PREDICTION SYSTEM")
    print(f"  Stations:  {', '.join(sites.keys())}")
    print(f"  Input:     {args.input}")
    os.makedirs(args.output, exist_ok=True)
    print(f"  Output:    {args.output}")
    print(f"  Models:    {args.models}")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.input}...")
    try:
        if args.input.endswith('.parquet'):
            df = pd.read_parquet(args.input)
        else:
            df = pd.read_csv(args.input)
        print(f"  {len(df)} records loaded")
    except Exception as e:
        print(f"  Error loading data: {e}")
        return 1

    # Run predictions per station
    all_results = []
    for site, skey in sites.items():
        print(f"\n--- {site} ({skey}) ---")
        results = _predict_station(df, site, skey, args.models, args.filter_alerts)
        if results is not None and len(results) > 0:
            # Save per-station CSV
            out_path = os.path.join(args.output, f'predictions_{skey}.csv')
            results.to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")
            all_results.append(results)

    if not all_results:
        print("\nNo predictions generated.")
        return 1

    # Combined output
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = os.path.join(args.output, 'predictions_all.csv')
    combined.to_csv(combined_path, index=False)

    # Overall summary
    risk_counts = combined['risk'].value_counts()
    alert_count = int(combined['alert'].sum())
    print(f"\nPrediction Summary ({len(combined)} total across {len(all_results)} stations):")
    for tier in ['GREEN', 'YELLOW_LOW', 'YELLOW_HIGH', 'ORANGE']:
        count = risk_counts.get(tier, 0)
        pct = count / len(combined) * 100 if len(combined) > 0 else 0
        print(f"  {tier:12s}: {count:4d} ({pct:.0f}%)")
    if alert_count:
        print(f"  {'Alerts':12s}: {alert_count}")
    print(f"\nCombined: {combined_path}")

    print("\n" + "=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    exit(main())
