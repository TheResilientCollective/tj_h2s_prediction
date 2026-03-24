#!/usr/bin/env python3
"""
H2S Model Visualization Generator
===================================

Generates feature importance and evaluation plots using trained models
from train_models_auto.py.

Usage:
    python generate_visualizations.py --models ./models
    python generate_visualizations.py --models ./models --predictions pred.csv --actuals act.csv --output ./reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

from predict_h2s import load_models, STATION_KEYS, MODEL_FEATURES


def plot_feature_importance(models_dir, station_key, output_path='feature_importance.png', top_n=15):
    """Generate feature importance plot from trained .pkl models."""
    print("Generating feature importance plot...")

    reg, clf5, clf10 = load_models(models_dir, station_key)

    # Collect importances from all 3 models
    models_info = [
        ('Regression', reg),
        ('Clf >5ppb', clf5),
        ('Clf >10ppb', clf10),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    for ax, (name, model) in zip(axes, models_info):
        # Handle ensemble models (average importances from sub-models)
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'model_a') and hasattr(model.model_a, 'feature_importances_'):
            imp_a = model.model_a.feature_importances_
            imp_b = model.model_b.feature_importances_
            imp = model.weight_a * imp_a + model.weight_b * imp_b
        else:
            ax.set_title(f'{name}\n(no importances)')
            continue

        features = MODEL_FEATURES[:len(imp)]
        ranked = sorted(zip(features, imp), key=lambda x: x[1], reverse=True)[:top_n]
        names, values = zip(*ranked)

        ax.barh(range(len(names)), values, color='steelblue')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')

    site_name = [k for k, v in STATION_KEYS.items() if v == station_key][0]
    fig.suptitle(f'Feature Importance — {site_name} (top {top_n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_confusion_matrix(predictions_df, actuals_df, output_path='confusion_matrix.png'):
    """Generate confusion matrix comparing predictions vs actuals."""
    print("\nGenerating confusion matrix...")

    merged = predictions_df.merge(actuals_df, on='time', how='inner')
    if len(merged) == 0:
        print("  No matching timestamps between predictions and actuals")
        return None

    print(f"  Matched {len(merged)} records")

    # Categorize actuals into risk tiers
    def categorize(h2s):
        if h2s > 30:
            return 'RED'
        elif h2s > 10:
            return 'ORANGE'
        elif h2s > 5:
            return 'YELLOW'
        return 'GREEN'

    merged['actual_risk'] = merged['H2S'].apply(categorize)

    # Use 'risk' column if present, fall back to 'predicted_category'
    pred_col = 'risk' if 'risk' in merged.columns else 'predicted_category'
    tiers = ['GREEN', 'YELLOW', 'ORANGE', 'RED']
    cm = confusion_matrix(merged['actual_risk'], merged[pred_col], labels=tiers)

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    for i in range(len(tiers)):
        for j in range(len(tiers)):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.0%}\n(n={cm[i,j]})',
                    ha='center', va='center', color=color, fontsize=10)

    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers)
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(tiers)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix — H2S Risk Tiers', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Proportion')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    bal_acc = balanced_accuracy_score(merged['actual_risk'], merged[pred_col])
    print(f"  Balanced Accuracy: {bal_acc:.1%}")
    for i, tier in enumerate(tiers):
        total = cm[i, :].sum()
        recall = cm[i, i] / total if total > 0 else 0
        print(f"  {tier:8s} Recall: {recall:.0%} ({cm[i,i]}/{total})")

    print(f"  Saved to {output_path}")
    return cm


def generate_all(models_dir, station_key, predictions_path, actuals_path, output_dir):
    """Generate all visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    site_name = [k for k, v in STATION_KEYS.items() if v == station_key][0]
    print("=" * 70)
    print(f"H2S VISUALIZATION GENERATOR — {site_name}")
    print("=" * 70)

    plot_feature_importance(
        models_dir, station_key,
        os.path.join(output_dir, f'feature_importance_{station_key}.png'),
    )

    if predictions_path and actuals_path:
        predictions_df = pd.read_csv(predictions_path)
        actuals_df = pd.read_csv(actuals_path)
        print(f"\n  Predictions: {len(predictions_df)} records")
        print(f"  Actuals: {len(actuals_df)} records")

        plot_confusion_matrix(
            predictions_df, actuals_df,
            os.path.join(output_dir, f'confusion_matrix_{station_key}.png'),
        )
    else:
        print("\n  Skipping confusion matrix (requires --predictions and --actuals)")

    print(f"\n{'=' * 70}")
    print(f"Outputs saved to {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate H2S model visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feature importance for all stations
  python generate_visualizations.py --models ./models

  # With confusion matrix (requires predictions + actuals)
  python generate_visualizations.py --models ./models --predictions pred.csv --actuals act.csv

  # Specific station
  python generate_visualizations.py --models ./models --site "SAN YSIDRO" --output ./reports
        """
    )

    parser.add_argument('--models', required=True,
                        help='Directory containing best_*.pkl files from train_models_auto.py')
    parser.add_argument('--predictions', '-p',
                        help='CSV with predictions (must have "time" and "risk" columns)')
    parser.add_argument('--actuals', '-a',
                        help='CSV with actual H2S values (must have "time" and "H2S" columns)')
    parser.add_argument('--output', '-o', default='.',
                        help='Output directory for plots (default: current directory)')
    parser.add_argument('--site', default=None,
                        choices=list(STATION_KEYS.keys()),
                        help='Station (default: all stations)')

    args = parser.parse_args()

    if args.site:
        stations = {args.site: STATION_KEYS[args.site]}
    else:
        stations = STATION_KEYS

    for site, skey in stations.items():
        try:
            generate_all(args.models, skey, args.predictions, args.actuals, args.output)
        except FileNotFoundError:
            print(f"  Models not found for {site} ({skey}), skipping\n")


if __name__ == '__main__':
    main()
