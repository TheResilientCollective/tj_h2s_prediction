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
    """Results from one experiment configuration.

    Metric fields use the neutral name ``pos`` (positive class) so the same
    record works for any threshold: ``pos`` = H2S ≥ threshold ppb.
    """
    name: str
    station: str
    variant: str
    threshold: int          # 5, 10, or 30 ppb
    n_train: int
    n_test: int
    n_pos_train: int
    n_pos_test: int
    pos_recall_train: float
    pos_recall_test: float
    pos_precision_train: float
    pos_precision_test: float
    pos_auc_train: float
    pos_auc_test: float
    pos_f1_train: float
    pos_f1_test: float
    model_obj: dict  # Model pickle-able dict


# (threshold ppb -> target column, train_and_select task name)
_THRESHOLD_SPEC = {
    5:  ('exceed_5', 'clf_5ppb'),
    10: ('exceed_10', 'clf_10ppb'),
    30: ('exceed_30', 'clf_30ppb'),
}


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
    threshold: int = 30,
    experiment_filter=None,
    random_state: int = 42,
) -> ExperimentResult:
    """Train a binary classifier for one threshold on the given data subset."""
    target_col, task = _THRESHOLD_SPEC[threshold]

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
    y_train = df_train_split[target_col].values
    y_test = df_test[target_col].values

    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Train split single-class for {target_col} (all {y_train[0]})")

    # Train the threshold classifier
    model, choice, metrics = train_and_select(X_train, X_test, y_train, y_test, task=task)

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
        threshold=threshold,
        n_train=len(df_train_split),
        n_test=len(df_test),
        n_pos_train=(y_train == 1).sum(),
        n_pos_test=(y_test == 1).sum(),
        pos_recall_train=recall_score(y_train, y_pred_train, zero_division=0),
        pos_recall_test=recall_score(y_test, y_pred_test, zero_division=0),
        pos_precision_train=precision_score(y_train, y_pred_train, zero_division=0),
        pos_precision_test=precision_score(y_test, y_pred_test, zero_division=0),
        pos_auc_train=roc_auc_score(y_train, y_prob_train) if len(np.unique(y_train)) > 1 else 0.5,
        pos_auc_test=roc_auc_score(y_test, y_prob_test) if len(np.unique(y_test)) > 1 else 0.5,
        pos_f1_train=f1_score(y_train, y_pred_train, zero_division=0),
        pos_f1_test=f1_score(y_test, y_pred_test, zero_division=0),
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
            # Per-threshold prevalence (positive base rate)
            'n_exceed_5': (s['exceed_5'] == 1).sum(),
            'pct_exceed_5': 100 * (s['exceed_5'] == 1).sum() / len(s),
            'n_exceed_10': (s['exceed_10'] == 1).sum(),
            'pct_exceed_10': 100 * (s['exceed_10'] == 1).sum() / len(s),
            'n_exceed_30': (s['exceed_30'] == 1).sum(),
            'pct_exceed_30': 100 * (s['exceed_30'] == 1).sum() / len(s),
            # Nighttime concentration of positives per threshold
            'pct_night_of_exceed_5': 100 * ((s['is_night'] == 1) & (s['exceed_5'] == 1)).sum() / max(1, (s['exceed_5'] == 1).sum()),
            'pct_night_of_exceed_10': 100 * ((s['is_night'] == 1) & (s['exceed_10'] == 1)).sum() / max(1, (s['exceed_10'] == 1).sum()),
            'pct_night_of_exceed_30': 100 * ((s['is_night'] == 1) & (s['exceed_30'] == 1)).sum() / max(1, (s['exceed_30'] == 1).sum()),
            'h2s_mean': s['H2S'].mean(),
            'h2s_std': s['H2S'].std(),
            'h2s_max': s['H2S'].max(),
            'h2s_min': s['H2S'].min(),
            'h2s_p95': s['H2S'].quantile(0.95),
            'h2s_p99': s['H2S'].quantile(0.99),
        }
    return stats


# Training-subset approaches. Each maps a name to a row filter (or None for
# all data). The "dual_green_periods" approach is omitted: filtering to green
# data makes every threshold target single-class (no positives), so it cannot
# train a binary classifier.
_APPROACHES = [
    ("baseline_all_data", None),
    ("nighttime_only", lambda d: d['is_night'] == 1),
    ("nonzero_h2s_only", lambda d: d['H2S'] > 0),
    ("dual_h2s_nights", lambda d: (d['is_night'] == 1) & (d['H2S'] > 0)),
]


def run_experiments(df: pd.DataFrame, output_dir: Path = None, thresholds=(5, 10, 30)):
    """Run all approaches × thresholds × stations."""
    if output_dir is None:
        output_dir = Path('./experiments/underfitting_results')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for threshold in thresholds:
        print(f"\n{'#' * 70}")
        print(f"# THRESHOLD: H2S ≥ {threshold} ppb")
        print(f"{'#' * 70}")
        for approach_name, filt in _APPROACHES:
            print(f"\n=== {approach_name} (≥{threshold} ppb, Lean Features) ===")
            for partition_key, station_name in STATION_PARTITION_MAP.items():
                try:
                    r = split_and_train(
                        df, MODEL_FEATURES_LEAN, station_name, approach_name,
                        threshold=threshold, experiment_filter=filt,
                    )
                    results.append(r)
                    print(
                        f"{station_name}: n_train={r.n_train}, pos%={100 * r.n_pos_test / max(1, r.n_test):.1f}, "
                        f"recall_test={r.pos_recall_test:.3f}, auc_test={r.pos_auc_test:.3f}"
                    )
                except Exception as e:
                    print(f"{station_name}: FAILED - {e}")

    return results


def print_summary(results: list[ExperimentResult]):
    """Print summary tables, one block per threshold × station."""
    thresholds = sorted(set(r.threshold for r in results))
    for threshold in thresholds:
        print("\n" + "=" * 120)
        print(f"EXPERIMENT SUMMARY: H2S ≥ {threshold} ppb Detection Metrics")
        print("=" * 120)

        for station in sorted(set(r.station for r in results)):
            rows = [x for x in results if x.station == station and x.threshold == threshold]
            if not rows:
                continue
            print(f"\n{station}  (≥{threshold} ppb):")
            print("-" * 110)
            print(f"{'Experiment':<25} {'Train N':<10} {'Test N':<10} {'Pos%':<10} {'Recall':<10} {'Precision':<10} {'AUC':<10} {'F1':<10}")
            print("-" * 110)

            baseline = next((x for x in rows if x.name == 'baseline_all_data'), None)
            for r in sorted(rows, key=lambda x: x.name):
                pos_pct_test = 100 * r.n_pos_test / max(1, r.n_test)
                delta = ""
                if baseline is not None and r.name != 'baseline_all_data':
                    d = r.pos_recall_test - baseline.pos_recall_test
                    delta = f"  ({'+' if d >= 0 else ''}{d * 100:.1f} pp)"
                print(
                    f"{r.name:<25} {r.n_train:<10} {r.n_test:<10} "
                    f"{pos_pct_test:<10.1f} {r.pos_recall_test:<10.3f} "
                    f"{r.pos_precision_test:<10.3f} {r.pos_auc_test:<10.3f} "
                    f"{r.pos_f1_test:<10.3f}{delta}"
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
            'threshold': int(r.threshold),
            'n_train': int(r.n_train),
            'n_test': int(r.n_test),
            'n_pos_train': int(r.n_pos_train),
            'n_pos_test': int(r.n_pos_test),
            'pos_recall_train': float(r.pos_recall_train),
            'pos_recall_test': float(r.pos_recall_test),
            'pos_precision_train': float(r.pos_precision_train),
            'pos_precision_test': float(r.pos_precision_test),
            'pos_auc_train': float(r.pos_auc_train),
            'pos_auc_test': float(r.pos_auc_test),
            'pos_f1_train': float(r.pos_f1_train),
            'pos_f1_test': float(r.pos_f1_test),
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
    parser.add_argument(
        '--thresholds',
        default='5,10,30',
        help='Comma-separated ppb thresholds to evaluate (default: 5,10,30)'
    )
    args = parser.parse_args()
    thresholds = tuple(int(t) for t in args.thresholds.split(','))

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
    results = run_experiments(df, Path(args.output_dir), thresholds=thresholds)

    print_summary(results)
    save_results(results, Path(args.output_dir))

    print("\n✓ Experiments complete!")


if __name__ == '__main__':
    main()
