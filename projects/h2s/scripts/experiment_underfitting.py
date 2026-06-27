#!/usr/bin/env python3
"""Experiment: H2S Model Underfitting Hypotheses

Tests three approaches to improve hazard (orange ≥30 ppb) detection:
1. Nighttime-only training (focus on when H2S actually occurs)
2. Non-zero H2S only (event-focused training)
3. Dual models (one for H2S nights, one for green periods)

All experiments use the lean 19-feature subset.

Usage:
  cd projects/h2s
  uv run python scripts/experiment_underfitting.py --help
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from h2s.constants import (
    H2S_THRESHOLD_HIGH,
    MODEL_FEATURES_LEAN,
    STATION_PARTITION_MAP,
    STATIONS,
)
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)


class ExperimentResult(NamedTuple):
    """Results from one experiment configuration."""
    name: str
    station: str
    variant: str
    n_train: int
    n_test: int
    n_orange_train: int
    n_orange_test: int
    orange_recall_train: float
    orange_recall_test: float
    orange_precision_train: float
    orange_precision_test: float
    orange_auc_train: float
    orange_auc_test: float
    orange_f1_train: float
    orange_f1_test: float
    model_obj: dict  # Model pickle-able dict


def load_training_data(data_path: str = "../../data/modeldata_h2s_nofill.parquet") -> pd.DataFrame:
    """Load and prepare training data."""
    df = pd.read_parquet(data_path)
    df = prepare_multi_station_features(df)
    return df


def split_and_train(
    df: pd.DataFrame,
    features: list[str],
    station: str,
    variant_name: str,
    experiment_filter=None,
    random_state: int = 42,
) -> ExperimentResult:
    """Train a model on the given data subset."""
    if experiment_filter is not None:
        df_train = df[experiment_filter(df)].copy()
    else:
        df_train = df.copy()

    # Ensure we have the station data
    df_train = df_train[df_train['site_name'] == station].copy()

    if len(df_train) == 0:
        raise ValueError(f"No data for station {station} after filtering")

    # Train/test split (preserving time order within station)
    n_split = int(len(df_train) * TRAIN_FRACTION)
    df_train_split = df_train.iloc[:n_split].copy()
    df_test = df_train.iloc[n_split:].copy()

    X_train = df_train_split[features].copy()
    X_test = df_test[features].copy()
    y_orange_train = df_train_split['exceed_30'].values
    y_orange_test = df_test['exceed_30'].values

    # Train the orange classifier
    model, choice, metrics = train_and_select(X_train, X_test, y_orange_train, y_orange_test, task='clf_30ppb')

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Metrics
    results = ExperimentResult(
        name=variant_name,
        station=station,
        variant=variant_name,
        n_train=len(df_train_split),
        n_test=len(df_test),
        n_orange_train=(y_orange_train == 1).sum(),
        n_orange_test=(y_orange_test == 1).sum(),
        orange_recall_train=recall_score(y_orange_train, y_pred_train, zero_division=0),
        orange_recall_test=recall_score(y_orange_test, y_pred_test, zero_division=0),
        orange_precision_train=precision_score(y_orange_train, y_pred_train, zero_division=0),
        orange_precision_test=precision_score(y_orange_test, y_pred_test, zero_division=0),
        orange_auc_train=roc_auc_score(y_orange_train, y_prob_train) if len(np.unique(y_orange_train)) > 1 else 0.5,
        orange_auc_test=roc_auc_score(y_orange_test, y_prob_test) if len(np.unique(y_orange_test)) > 1 else 0.5,
        orange_f1_train=f1_score(y_orange_train, y_pred_train, zero_division=0),
        orange_f1_test=f1_score(y_orange_test, y_pred_test, zero_division=0),
        model_obj={"model": model, "choice": choice, "metrics": metrics, "features": features},
    )
    return results


def analyze_data_distribution(df: pd.DataFrame) -> dict:
    """Analyze key characteristics of training data."""
    stats = {}
    for station in df['site_name'].unique():
        s = df[df['site_name'] == station]
        stats[station] = {
            'n_total': len(s),
            'n_night': (s['is_night'] == 1).sum(),
            'pct_night': 100 * (s['is_night'] == 1).sum() / len(s),
            'n_nonzero_h2s': (s['H2S'] > 0).sum(),
            'pct_nonzero_h2s': 100 * (s['H2S'] > 0).sum() / len(s),
            'n_orange': (s['exceed_30'] == 1).sum(),
            'pct_orange': 100 * (s['exceed_30'] == 1).sum() / len(s),
            'n_night_with_orange': ((s['is_night'] == 1) & (s['exceed_30'] == 1)).sum(),
            'pct_night_with_orange_of_all_orange': 100 * ((s['is_night'] == 1) & (s['exceed_30'] == 1)).sum() / max(1, (s['exceed_30'] == 1).sum()),
            'h2s_mean': s['H2S'].mean(),
            'h2s_std': s['H2S'].std(),
            'h2s_max': s['H2S'].max(),
            'h2s_min': s['H2S'].min(),
            'h2s_p95': s['H2S'].quantile(0.95),
            'h2s_p99': s['H2S'].quantile(0.99),
        }
    return stats


def run_experiments(df: pd.DataFrame, output_dir: Path = None):
    """Run all experiment variants."""
    if output_dir is None:
        output_dir = Path('./experiments/underfitting_results')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Map partition keys to display names
    partition_to_display = {v: v for v in STATION_PARTITION_MAP.values()}

    # 1. BASELINE: All data (lean features)
    print("\n=== BASELINE: All Data (Lean Features) ===")
    for partition_key, station_name in STATION_PARTITION_MAP.items():
        try:
            r = split_and_train(df, MODEL_FEATURES_LEAN, station_name, "baseline_all_data")
            results.append(r)
            print(f"{station_name}: n_train={r.n_train}, orange_recall_test={r.orange_recall_test:.3f}, orange_auc_test={r.orange_auc_test:.3f}")
        except Exception as e:
            print(f"{station_name}: FAILED - {e}")

    # 2. NIGHTTIME-ONLY: Focus on when H2S actually occurs
    print("\n=== NIGHTTIME-ONLY: Lean Features ===")
    for partition_key, station_name in STATION_PARTITION_MAP.items():
        try:
            r = split_and_train(
                df,
                MODEL_FEATURES_LEAN,
                station_name,
                "nighttime_only",
                experiment_filter=lambda d: d['is_night'] == 1
            )
            results.append(r)
            print(f"{station_name}: n_train={r.n_train}, orange_recall_test={r.orange_recall_test:.3f}, orange_auc_test={r.orange_auc_test:.3f}")
        except Exception as e:
            print(f"{station_name}: FAILED - {e}")

    # 3. NON-ZERO H2S: Event-focused training
    print("\n=== NON-ZERO H2S: Lean Features ===")
    for partition_key, station_name in STATION_PARTITION_MAP.items():
        try:
            r = split_and_train(
                df,
                MODEL_FEATURES_LEAN,
                station_name,
                "nonzero_h2s_only",
                experiment_filter=lambda d: d['H2S'] > 0
            )
            results.append(r)
            print(f"{station_name}: n_train={r.n_train}, orange_recall_test={r.orange_recall_test:.3f}, orange_auc_test={r.orange_auc_test:.3f}")
        except Exception as e:
            print(f"{station_name}: FAILED - {e}")

    # 4. DUAL MODEL - Part A: H2S Nights (high concentration training)
    print("\n=== DUAL MODEL - H2S Nights: Lean Features ===")
    for partition_key, station_name in STATION_PARTITION_MAP.items():
        try:
            r = split_and_train(
                df,
                MODEL_FEATURES_LEAN,
                station_name,
                "dual_h2s_nights",
                experiment_filter=lambda d: (d['is_night'] == 1) & (d['H2S'] > 0)
            )
            results.append(r)
            print(f"{station_name}: n_train={r.n_train}, orange_recall_test={r.orange_recall_test:.3f}, orange_auc_test={r.orange_auc_test:.3f}")
        except Exception as e:
            print(f"{station_name}: FAILED - {e}")

    # 5. DUAL MODEL - Part B: Green Periods (safe-state learning)
    print("\n=== DUAL MODEL - Green Periods: Lean Features ===")
    for partition_key, station_name in STATION_PARTITION_MAP.items():
        try:
            r = split_and_train(
                df,
                MODEL_FEATURES_LEAN,
                station_name,
                "dual_green_periods",
                experiment_filter=lambda d: (d['H2S'] <= 5) | (d['H2S'].isna())
            )
            results.append(r)
            print(f"{station_name}: n_train={r.n_train}, orange_recall_test={r.orange_recall_test:.3f}, orange_auc_test={r.orange_auc_test:.3f}")
        except Exception as e:
            print(f"{station_name}: FAILED - {e}")

    return results


def print_summary(results: list[ExperimentResult]):
    """Print summary table of results."""
    print("\n" + "=" * 120)
    print("EXPERIMENT SUMMARY: Orange (≥30 ppb) Detection Metrics")
    print("=" * 120)

    # Group by station
    for station in sorted(set(r.station for r in results)):
        print(f"\n{station}:")
        print("-" * 110)
        print(f"{'Experiment':<25} {'Train N':<10} {'Test N':<10} {'Orange%':<10} {'Recall':<10} {'Precision':<10} {'AUC':<10} {'F1':<10}")
        print("-" * 110)

        for r in sorted([x for x in results if x.station == station], key=lambda x: x.name):
            orange_pct_test = 100 * r.n_orange_test / max(1, r.n_test)
            print(
                f"{r.name:<25} {r.n_train:<10} {r.n_test:<10} "
                f"{orange_pct_test:<10.1f} {r.orange_recall_test:<10.3f} "
                f"{r.orange_precision_test:<10.3f} {r.orange_auc_test:<10.3f} "
                f"{r.orange_f1_test:<10.3f}"
            )


def save_results(results: list[ExperimentResult], output_dir: Path):
    """Save detailed results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tabular results (remove model objects)
    results_dict = [
        {
            'name': r.name,
            'station': r.station,
            'n_train': int(r.n_train),
            'n_test': int(r.n_test),
            'n_orange_train': int(r.n_orange_train),
            'n_orange_test': int(r.n_orange_test),
            'orange_recall_train': float(r.orange_recall_train),
            'orange_recall_test': float(r.orange_recall_test),
            'orange_precision_train': float(r.orange_precision_train),
            'orange_precision_test': float(r.orange_precision_test),
            'orange_auc_train': float(r.orange_auc_train),
            'orange_auc_test': float(r.orange_auc_test),
            'orange_f1_train': float(r.orange_f1_train),
            'orange_f1_test': float(r.orange_f1_test),
        }
        for r in results
    ]

    results_path = output_dir / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description='Test H2S model underfitting hypotheses')
    parser.add_argument(
        '--data-path',
        default='../../data/modeldata_h2s_nofill.parquet',
        help='Path to training data parquet'
    )
    parser.add_argument(
        '--output-dir',
        default='./experiments/underfitting_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze data distribution, do not train'
    )
    args = parser.parse_args()

    print("Loading training data...")
    df = load_training_data(args.data_path)

    print("\n=== DATA DISTRIBUTION ANALYSIS ===")
    stats = analyze_data_distribution(df)
    for station, station_stats in sorted(stats.items()):
        print(f"\n{station}:")
        for key, val in sorted(station_stats.items()):
            if isinstance(val, float):
                print(f"  {key:<30} {val:>10.2f}")
            else:
                print(f"  {key:<30} {val:>10}")

    if args.analyze_only:
        return

    print("\n\nStarting experiments (this may take a few minutes)...")
    results = run_experiments(df, Path(args.output_dir))

    print_summary(results)
    save_results(results, Path(args.output_dir))

    print("\n✓ Experiments complete!")


if __name__ == '__main__':
    main()
