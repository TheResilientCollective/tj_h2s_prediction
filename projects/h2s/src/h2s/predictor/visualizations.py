"""H2S Model Visualization Generator for S3 storage.

Generates visualizations that can be stored directly to S3 without local files.
Returns plots as BytesIO objects for direct S3 upload.
"""

from io import BytesIO
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def generate_feature_importance(model, prep_info: Dict, top_n: int = 15) -> BytesIO:
    """Generate feature importance plot as BytesIO.

    Args:
        model: Trained XGBoost model
        prep_info: Preprocessing info dict with 'feature_cols'
        top_n: Number of top features to display

    Returns:
        BytesIO object with PNG image data
    """
    feature_names = prep_info['feature_cols']

    # Get feature importance from model
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

    # Sort and get top N
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features for H2S Prediction',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def generate_confusion_matrix(predictions: pd.DataFrame, actuals: pd.DataFrame) -> BytesIO:
    """Generate confusion matrix plot as BytesIO.

    Args:
        predictions: DataFrame with 'predicted_category' and 'time' columns
        actuals: DataFrame with 'H2S' values and 'time' column

    Returns:
        BytesIO object with PNG image data
    """
    # Merge predictions with actuals on time
    merged = predictions.merge(actuals, on='time', how='inner')

    if len(merged) == 0:
        # Return empty plot if no matching data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No matching timestamps', ha='center', va='center')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        return buf

    # Convert actual H2S values to categories
    def categorize_h2s(value):
        if value < 5:
            return 'green'
        elif value < 15:
            return 'yellow'
        else:
            return 'orange'

    merged['actual_category'] = merged['H2S'].apply(categorize_h2s)

    # Compute confusion matrix
    categories = ['green', 'yellow', 'orange']
    cm = confusion_matrix(merged['actual_category'], merged['predicted_category'],
                          labels=categories)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=categories, yticklabels=categories, ax=ax)
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_title('H2S Prediction Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def generate_confusion_matrix_with_metrics(predictions: pd.DataFrame, actuals: pd.DataFrame,
                                           time_col: str = 'time') -> BytesIO:
    """Generate enhanced confusion matrix plot with metrics as BytesIO.

    Args:
        predictions: DataFrame with 'predicted_category' and time column
        actuals: DataFrame with 'H2S' values and time column
        time_col: Name of the time column for merging (default: 'time')

    Returns:
        BytesIO object with PNG image data
    """
    # Merge predictions with actuals on time - use suffixes to handle column conflicts
    merged = predictions.merge(actuals, on=time_col, how='inner', suffixes=('_pred', '_actual'))

    if len(merged) == 0:
        # Return empty plot if no matching data
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No matching timestamps between predictions and actuals',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    # Determine which H2S column to use (handle conflicts from merge)
    h2s_col = 'H2S_actual' if 'H2S_actual' in merged.columns else 'H2S'

    # Filter to only rows with non-null H2S measurements
    merged = merged[merged[h2s_col].notna()].copy()

    if len(merged) == 0:
        # Return message if no actual measurements after filtering
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No H2S measurements available for comparison',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    # Convert actual H2S values to categories
    def categorize_h2s(value):
        if value < 5:
            return 'green'
        elif value < 15:
            return 'yellow'
        else:
            return 'orange'

    merged['actual_category'] = merged[h2s_col].apply(categorize_h2s)

    # Compute confusion matrix
    class_names = ['green', 'orange', 'yellow']
    cm = confusion_matrix(merged['actual_category'], merged['predicted_category'],
                          labels=class_names)

    # Normalize for percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create plot
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
            ax.text(j + 0.5, i + 0.7, f'n={cm[i, j]}',
                   ha="center", va="center", color="darkred", fontsize=9)

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def generate_model_comparison(predictions: pd.DataFrame, actuals: pd.DataFrame,
                              model_name: str = "XGBoost", time_col: str = 'time') -> BytesIO:
    """Generate model comparison plot showing performance metrics as BytesIO.

    Creates a 2x2 grid showing:
    1. Overall balanced accuracy
    2. Per-class recall (detection rate)
    3. Precision vs recall comparison
    4. Confusion matrix

    Args:
        predictions: DataFrame with 'predicted_category' and time column
        actuals: DataFrame with 'H2S' values and time column
        model_name: Name of the model (default: "XGBoost")
        time_col: Name of the time column for merging (default: 'time')

    Returns:
        BytesIO object with PNG image data
    """
    # Merge predictions with actuals
    merged = predictions.merge(actuals, on=time_col, how='inner')

    if len(merged) == 0:
        # Return empty plot if no matching data
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.text(0.5, 0.5, 'No matching timestamps between predictions and actuals',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    # Convert actual H2S values to categories
    def categorize_h2s(value):
        if value < 5:
            return 'green'
        elif value < 15:
            return 'yellow'
        else:
            return 'orange'

    merged['actual_category'] = merged['H2S'].apply(categorize_h2s)

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

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def generate_prediction_timeline(predictions: pd.DataFrame, raw_environmental_data: Optional[pd.DataFrame] = None) -> BytesIO:
    """Generate timeline plot of predictions with environmental variables.

    Args:
        predictions: DataFrame with 'time', 'predicted_category' columns and environmental variables
        raw_environmental_data: Optional DataFrame with raw data (may include 'H2S' measurements)

    Returns:
        BytesIO object with PNG image data
    """
    # Create multi-panel figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Prediction Timeline
    ax = axes[0]
    category_colors = {'green': 'green', 'yellow': 'gold', 'orange': 'orangered'}

    for category, color in category_colors.items():
        mask = predictions['predicted_category'] == category
        if mask.any():
            ax.scatter(predictions.loc[mask, 'time'],
                      [category] * mask.sum(),
                      c=color, s=50, alpha=0.6, label=f'Predicted {category}')

    # Plot actual H2S measurements if available in raw data
    if raw_environmental_data is not None:
        # Check if H2S column exists
        h2s_cols = [col for col in raw_environmental_data.columns if col.upper() == 'H2S' or 'h2s' in col.lower()]

        if h2s_cols:
            # Prepare raw data with time column
            raw_df = raw_environmental_data.copy()
            if 'time' not in raw_df.columns:
                if 'date' in raw_df.columns:
                    raw_df['time'] = pd.to_datetime(raw_df['date'])

            if 'time' in raw_df.columns:
                # Use the H2S column - filter to only non-null values
                h2s_col = h2s_cols[0]
                h2s_subset = raw_df[['time', h2s_col]].copy()
                # Keep only rows with actual H2S measurements
                h2s_subset = h2s_subset[h2s_subset[h2s_col].notna()]

                if len(h2s_subset) > 0:
                    # Merge with predictions - use suffixes to avoid column conflicts
                    merged = predictions.merge(h2s_subset, on='time', how='inner', suffixes=('', '_actual'))

                    # Use the actual H2S column (with _actual suffix if there was a conflict)
                    h2s_plot_col = f'{h2s_col}_actual' if f'{h2s_col}_actual' in merged.columns else h2s_col

                    if len(merged) > 0 and h2s_plot_col in merged.columns:
                        ax.plot(merged['time'], merged[h2s_plot_col], 'k-', alpha=0.5, label='Actual H2S', linewidth=2)
                        ax2 = ax.twinx()
                        ax2.set_ylabel('H2S (ppb)', fontsize=10)
                        ax2.plot(merged['time'], merged[h2s_plot_col], 'k-', alpha=0.5, linewidth=2)

    ax.set_ylabel('Predicted Category', fontsize=10, fontweight='bold')
    ax.set_title('H2S Predictions Timeline with Environmental Variables', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Flow
    ax = axes[1]
    flow_col = 'Flow (m^3/s)--Border' if 'Flow (m^3/s)--Border' in predictions.columns else None
    if flow_col:
        ax.plot(predictions['time'], predictions[flow_col], color='steelblue', linewidth=1.5)
        ax.fill_between(predictions['time'], predictions[flow_col], alpha=0.3, color='steelblue')
        ax.set_ylabel('Flow (m³/s)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Flow data not available', ha='center', va='center', transform=ax.transAxes)

    # Panel 3: Dewpoint
    ax = axes[2]
    if 'dewpoint_2m' in predictions.columns:
        ax.plot(predictions['time'], predictions['dewpoint_2m'], color='teal', linewidth=1.5)
        ax.fill_between(predictions['time'], predictions['dewpoint_2m'], alpha=0.3, color='teal')
        ax.set_ylabel('Dewpoint (°C)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Dewpoint data not available', ha='center', va='center', transform=ax.transAxes)

    # Panel 4: Humidity & Temperature (dual axis)
    ax = axes[3]
    if 'relative_humidity_2m' in predictions.columns and 'temperature_2m' in predictions.columns:
        # Humidity on left axis
        color = 'tab:blue'
        ax.plot(predictions['time'], predictions['relative_humidity_2m'],
                color=color, linewidth=1.5, label='Humidity')
        ax.fill_between(predictions['time'], predictions['relative_humidity_2m'],
                        alpha=0.2, color=color)
        ax.set_ylabel('Humidity (%)', fontsize=10, fontweight='bold', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.3)

        # Temperature on right axis
        ax2 = ax.twinx()
        color = 'tab:red'
        ax2.plot(predictions['time'], predictions['temperature_2m'],
                 color=color, linewidth=1.5, label='Temperature')
        ax2.set_ylabel('Temperature (°C)', fontsize=10, fontweight='bold', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Humidity/Temperature data not available', ha='center', va='center', transform=ax.transAxes)

    # Panel 5: Tide Height
    ax = axes[4]
    if 'tide_height' in predictions.columns:
        ax.plot(predictions['time'], predictions['tide_height'], color='darkblue', linewidth=1.5)
        ax.fill_between(predictions['time'], predictions['tide_height'], alpha=0.3, color='darkblue')
        ax.set_ylabel('Tide Height (m)', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Tide data not available', ha='center', va='center', transform=ax.transAxes)

    # Set x-axis label only on bottom plot
    ax.set_xlabel('Time', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf
