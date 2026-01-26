#!/usr/bin/env python3
"""
H2S Model Visualization Generator
===================================

Generates three key visualizations:
1. Feature Importance - Which features matter most
2. Model Comparison - Performance across different models
3. Confusion Matrices - Detailed error analysis

Usage:
    python generate_visualizations.py --predictions predictions.csv --actuals actuals.csv
    python generate_visualizations.py --predictions predictions.csv --actuals actuals.csv --output-dir ./reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pickle
import argparse
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')


def plot_feature_importance(model_path, preprocessing_path, output_path='feature_importance.png'):
    """
    Generate feature importance plot.
    
    Args:
        model_path: Path to XGBoost model
        preprocessing_path: Path to preprocessing info
        output_path: Where to save the plot
    """
    print("Generating feature importance plot...")
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Load preprocessing info for feature names
    with open(preprocessing_path, 'rb') as f:
        prep_info = pickle.load(f)
    
    feature_names = prep_info['feature_cols']
    
    # Get feature importance
    importance_dict = model.get_booster().get_score(importance_type='gain')
    
    # Map to feature names
    importance_data = []
    for i, fname in enumerate(feature_names):
        key = f'f{i}'
        if key in importance_dict:
            importance_data.append({
                'feature': fname,
                'importance': importance_dict[key]
            })
    
    # Sort and get top 15
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    plt.title('Top 15 Most Important Features for H2S Prediction', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance saved to {output_path}")
    
    return importance_df


def plot_confusion_matrix(predictions_df, actuals_df, output_path='confusion_matrix.png'):
    """
    Generate confusion matrix plot.
    
    Args:
        predictions_df: DataFrame with predictions and 'time' column
        actuals_df: DataFrame with actual values and 'time' column
        output_path: Where to save the plot
    """
    print("\nGenerating confusion matrix...")
    
    # Merge predictions with actuals
    merged = predictions_df.merge(actuals_df, on='time', how='inner')
    
    if len(merged) == 0:
        print("✗ No matching timestamps between predictions and actuals")
        return None
    
    print(f"  Matched {len(merged)} records")
    
    # Create actual categories
    def categorize(value):
        if value < 5:
            return 'green'
        elif value < 15:
            return 'yellow'
        else:
            return 'orange'
    
    merged['actual_category'] = merged['H2S'].apply(categorize)
    
    # Get confusion matrix
    class_names = ['green', 'orange', 'yellow']
    cm = confusion_matrix(merged['actual_category'], merged['predicted_category'], 
                         labels=class_names)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'}, ax=ax,
                vmin=0, vmax=1)
    
    ax.set_title('Confusion Matrix - H2S Predictions vs Actuals', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Actual Category', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Category', fontsize=12, fontweight='bold')
    
    # Add counts as text
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]}',
                          ha="center", va="center", color="darkred", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to {output_path}")
    
    # Calculate metrics
    bal_acc = balanced_accuracy_score(merged['actual_category'], merged['predicted_category'])
    
    print(f"\n  Balanced Accuracy: {bal_acc:.1%}")
    for i, name in enumerate(class_names):
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"  {name.capitalize()} Recall: {recall:.1%} ({cm[i, i]}/{cm[i, :].sum()})")
    
    return cm


def plot_model_comparison(predictions_df, actuals_df, model_name="XGBoost", 
                          output_path='model_comparison.png'):
    """
    Generate model comparison plot showing performance metrics.
    
    Args:
        predictions_df: DataFrame with predictions
        actuals_df: DataFrame with actual values
        model_name: Name of the model
        output_path: Where to save the plot
    """
    print("\nGenerating model comparison plot...")
    
    # Merge
    merged = predictions_df.merge(actuals_df, on='time', how='inner')
    
    if len(merged) == 0:
        print("✗ No matching timestamps")
        return None
    
    # Create actual categories
    def categorize(value):
        if value < 5:
            return 'green'
        elif value < 15:
            return 'yellow'
        else:
            return 'orange'
    
    merged['actual_category'] = merged['H2S'].apply(categorize)
    
    # Calculate metrics
    class_names = ['green', 'orange', 'yellow']
    cm = confusion_matrix(merged['actual_category'], merged['predicted_category'], 
                         labels=class_names)
    
    bal_acc = balanced_accuracy_score(merged['actual_category'], merged['predicted_category'])
    
    recalls = []
    precisions = []
    for i in range(len(class_names)):
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        recalls.append(recall)
        precisions.append(precision)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Model Performance Analysis', fontsize=14, fontweight='bold')
    
    # 1. Balanced Accuracy
    ax = axes[0, 0]
    ax.bar([model_name], [bal_acc], color='steelblue', width=0.5)
    ax.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Overall Performance', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.text(0, bal_acc + 0.03, f'{bal_acc:.1%}', ha='center', fontsize=12, fontweight='bold')
    ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target: 60%')
    ax.legend()
    
    # 2. Per-Class Recall
    ax = axes[0, 1]
    x = np.arange(len(class_names))
    colors = ['#2ecc71', '#e74c3c', '#f39c12']
    bars = ax.bar(x, recalls, color=colors)
    ax.set_ylabel('Recall', fontsize=11, fontweight='bold')
    ax.set_title('Detection Rate by Category', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in class_names])
    ax.set_ylim([0, 1])
    for i, (bar, val) in enumerate(zip(bars, recalls)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.03, 
               f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
    
    # 3. Precision vs Recall
    ax = axes[1, 0]
    x = np.arange(len(class_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, precisions, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x + width/2, recalls, width, label='Recall', color='lightcoral')
    ax.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in class_names])
    ax.legend()
    ax.set_ylim([0, 1])
    
    # 4. Confusion Matrix (normalized)
    ax = axes[1, 1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=[c.capitalize() for c in class_names], 
                yticklabels=[c.capitalize() for c in class_names],
                ax=ax, cbar_kws={'label': 'Percentage'})
    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model comparison saved to {output_path}")
    
    return {
        'balanced_accuracy': bal_acc,
        'recalls': dict(zip(class_names, recalls)),
        'precisions': dict(zip(class_names, precisions))
    }


def generate_all_visualizations(predictions_path=None, actuals_path=None, 
                               model_path='nestor_xgboost_weighted_model.json',
                               preprocessing_path='nestor_preprocessing_info.pkl',
                               output_dir='.'):
    """
    Generate all three visualizations.
    
    Args:
        predictions_path: Path to predictions CSV (optional)
        actuals_path: Path to actuals CSV (optional)
        model_path: Path to model file
        preprocessing_path: Path to preprocessing file
        output_dir: Directory to save outputs
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("H2S MODEL VISUALIZATION GENERATOR")
    print("="*80)
    print()
    
    results = {}
    
    # 1. Feature Importance (always possible)
    try:
        importance_df = plot_feature_importance(
            model_path, 
            preprocessing_path,
            os.path.join(output_dir, 'feature_importance.png')
        )
        results['feature_importance'] = importance_df
    except Exception as e:
        print(f"✗ Feature importance failed: {e}")
    
    # 2 & 3. Confusion Matrix and Model Comparison (require predictions + actuals)
    if predictions_path and actuals_path:
        try:
            print("\nLoading prediction and actual data...")
            predictions_df = pd.read_csv(predictions_path)
            actuals_df = pd.read_csv(actuals_path)
            print(f"  Predictions: {len(predictions_df)} records")
            print(f"  Actuals: {len(actuals_df)} records")
            
            # Confusion Matrix
            cm = plot_confusion_matrix(
                predictions_df,
                actuals_df,
                os.path.join(output_dir, 'confusion_matrix.png')
            )
            results['confusion_matrix'] = cm
            
            # Model Comparison
            metrics = plot_model_comparison(
                predictions_df,
                actuals_df,
                output_path=os.path.join(output_dir, 'model_comparison.png')
            )
            results['metrics'] = metrics
            
        except Exception as e:
            print(f"✗ Confusion matrix/comparison failed: {e}")
            print("  Note: Requires predictions and actuals CSV files with:")
            print("    - predictions.csv: must have 'time' and 'predicted_category' columns")
            print("    - actuals.csv: must have 'time' and 'H2S' columns")
    else:
        print("\n⚠ Skipping confusion matrix and model comparison")
        print("  (Requires both --predictions and --actuals arguments)")
        print("  Feature importance plot generated from model alone.")
    
    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - feature_importance.png")
    if predictions_path and actuals_path:
        print("  - confusion_matrix.png")
        print("  - model_comparison.png")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate H2S model visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feature importance only (no actuals needed)
  python generate_visualizations.py
  
  # All three plots (requires predictions and actuals)
  python generate_visualizations.py --predictions predictions.csv --actuals actuals.csv
  
  # Save to specific directory
  python generate_visualizations.py --predictions pred.csv --actuals act.csv --output-dir ./reports
        """
    )
    
    parser.add_argument('--predictions', '-p',
                        help='CSV file with predictions (must have "time" and "predicted_category" columns)')
    parser.add_argument('--actuals', '-a',
                        help='CSV file with actual H2S values (must have "time" and "H2S" columns)')
    parser.add_argument('--model', default='nestor_xgboost_weighted_model.json',
                        help='Path to model file (default: nestor_xgboost_weighted_model.json)')
    parser.add_argument('--preprocessing', default='nestor_preprocessing_info.pkl',
                        help='Path to preprocessing file (default: nestor_preprocessing_info.pkl)')
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Output directory for plots (default: current directory)')
    
    args = parser.parse_args()
    
    # Generate visualizations
    generate_all_visualizations(
        predictions_path=args.predictions,
        actuals_path=args.actuals,
        model_path=args.model,
        preprocessing_path=args.preprocessing,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
