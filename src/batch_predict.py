#!/usr/bin/env python3
"""
Batch H2S Prediction — All Stations
=====================================

Runs predictions for all stations using trained models from train_models_auto.py.
Outputs one CSV per station plus a combined summary.

Usage:
    python batch_predict.py --obs data/model_forecast.csv --models ./models --output ./output
    python batch_predict.py --obs data/model_forecast.csv --models ./models --output ./output --filter-alerts
"""

import os
import pandas as pd
import argparse

from predict_h2s import (
    load_models, engineer_features, predict, STATION_KEYS,
)


def main():
    parser = argparse.ArgumentParser(
        description='Batch H2S predictions for all stations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--obs', required=True,
                        help='Input CSV or parquet with environmental data')
    parser.add_argument('--models', required=True,
                        help='Directory containing best_*.pkl files from train_models_auto.py')
    parser.add_argument('--output', '-o', default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--filter-alerts', action='store_true',
                        help='Only output ORANGE and RED predictions')

    args = parser.parse_args()

    print("=" * 70)
    print("BATCH H2S PREDICTION — ALL STATIONS")
    print(f"  Input:   {args.obs}")
    print(f"  Models:  {args.models}")
    print(f"  Output:  {args.output}")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {args.obs}...")
    if args.obs.endswith('.parquet'):
        df_all = pd.read_parquet(args.obs)
    else:
        df_all = pd.read_csv(args.obs)
    print(f"  {len(df_all)} records loaded")

    os.makedirs(args.output, exist_ok=True)
    all_results = []

    for site, skey in STATION_KEYS.items():
        print(f"\n--- {site} ({skey}) ---")

        # Filter to site
        if 'site_name' in df_all.columns:
            df = df_all[df_all['site_name'] == site].copy()
            if len(df) == 0:
                print(f"  No data for {site}, skipping")
                continue
        else:
            df = df_all.copy()

        # Load models
        try:
            reg, clf5, clf10 = load_models(args.models, skey)
        except FileNotFoundError:
            print(f"  Models not found for {skey}, skipping")
            continue

        # Engineer features + predict
        df = engineer_features(df)
        results = predict(df, reg, clf5, clf10)
        results['station'] = site

        if args.filter_alerts:
            results = results[results['alert']].copy()

        # Save per-station CSV
        out_path = os.path.join(args.output, f'predictions_{skey}.csv')
        results.to_csv(out_path, index=False)

        # Summary
        risk_counts = results['risk'].value_counts()
        parts = [f"{tier}:{risk_counts.get(tier, 0)}" for tier in ['GREEN', 'YELLOW', 'ORANGE', 'RED']]
        print(f"  {len(results)} rows -> {out_path}")
        print(f"  {' | '.join(parts)}")

        all_results.append(results)

    # Combined output
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.output, 'predictions_all_stations.csv')
        combined.to_csv(combined_path, index=False)
        print(f"\nCombined: {len(combined)} rows -> {combined_path}")

    print(f"\n{'=' * 70}")
    print("BATCH PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
