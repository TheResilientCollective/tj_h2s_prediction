"""H2S Model Visualization Generator for S3 storage.

Generates visualizations that can be stored directly to S3 without local files.
Returns plots as BytesIO objects for direct S3 upload.
"""

from io import BytesIO
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


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


def generate_prediction_timeline(predictions: pd.DataFrame, actuals: pd.DataFrame = None) -> BytesIO:
    """Generate timeline plot of predictions vs actuals.

    Args:
        predictions: DataFrame with 'time', 'predicted_category' columns
        actuals: Optional DataFrame with 'time' and 'H2S' columns

    Returns:
        BytesIO object with PNG image data
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot predictions
    category_colors = {'green': 'green', 'yellow': 'gold', 'orange': 'orangered'}

    for category, color in category_colors.items():
        mask = predictions['predicted_category'] == category
        if mask.any():
            ax.scatter(predictions.loc[mask, 'time'],
                      [category] * mask.sum(),
                      c=color, s=50, alpha=0.6, label=f'Predicted {category}')

    # Plot actuals if provided
    if actuals is not None:
        merged = predictions.merge(actuals, on='time', how='inner')
        if len(merged) > 0:
            ax.plot(merged['time'], merged['H2S'], 'k-', alpha=0.3, label='Actual H2S')
            ax2 = ax.twinx()
            ax2.set_ylabel('H2S (ppb)', fontsize=11)
            ax2.plot(merged['time'], merged['H2S'], 'k-', alpha=0.3)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Category', fontsize=12, fontweight='bold')
    ax.set_title('H2S Predictions Timeline', fontsize=14, fontweight='bold')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to BytesIO
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf
