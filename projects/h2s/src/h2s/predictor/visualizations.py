"""H2S Model Visualization Generator for S3 storage.

Generates visualizations that can be stored directly to S3 without local files.
Returns plots as BytesIO objects for direct S3 upload.
"""

from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def generate_feature_importance(model, prep_info: Dict, top_n: int = 15, model_name: str = "") -> BytesIO:
    """Generate feature importance plot as BytesIO.

    Args:
        model: Trained model (XGBoost or sklearn)
        prep_info: Preprocessing info dict with 'feature_cols'
        top_n: Number of top features to display
        model_name: Display name for the model (shown in plot title)

    Returns:
        BytesIO object with PNG image data
    """
    feature_names = prep_info['feature_cols']

    # Get feature importance — handle both XGBoost and sklearn estimators
    if hasattr(model, 'feature_importances_'):
        # sklearn (e.g. RandomForest): direct array aligned to feature_names
        importance_data = [
            {'feature': fname, 'importance': float(score)}
            for fname, score in zip(feature_names, model.feature_importances_)
            if score > 0
        ]
    else:
        # XGBoost: booster uses f0, f1, ... keys
        importance_dict = model.get_booster().get_score(importance_type='gain')
        importance_data = [
            {'feature': fname, 'importance': importance_dict[f'f{i}']}
            for i, fname in enumerate(feature_names)
            if f'f{i}' in importance_dict
        ]

    # Sort and get top N
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)

    # Create plot
    name_suffix = f" — {model_name}" if model_name else ""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.set_xlabel('Importance (Gain)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features for H2S Prediction{name_suffix}',
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
        elif value < 30:
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
                                           time_col: str = 'time', model_name: str = "") -> BytesIO:
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
        elif value < 30:
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

    name_suffix = f" — {model_name}" if model_name else ""
    ax.set_title(f'Confusion Matrix - H2S Predictions vs Actuals{name_suffix}',
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
    # Merge predictions with actuals - use suffixes to handle column conflicts
    merged = predictions.merge(actuals, on=time_col, how='inner', suffixes=('_pred', '_actual'))

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

    # Determine which H2S column to use (handle conflicts from merge)
    h2s_col = 'H2S_actual' if 'H2S_actual' in merged.columns else 'H2S'

    # Filter to only rows with non-null H2S measurements
    merged = merged[merged[h2s_col].notna()].copy()

    if len(merged) == 0:
        # Return message if no actual measurements after filtering
        fig, ax = plt.subplots(figsize=(14, 10))
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
        elif value < 30:
            return 'yellow'
        else:
            return 'orange'

    merged['actual_category'] = merged[h2s_col].apply(categorize_h2s)

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


def generate_cross_correlation_viz(
    df: pd.DataFrame,
    h2s_col: str = "H2S",
    max_lag_hours: int = 24,
    top_n: int = 6,
) -> BytesIO:
    """Compute and plot time-lagged cross-correlation between H2S and environmental features.

    For each feature, computes corr(H2S(t), feature(t - lag)) for lags in
    [-max_lag_hours, +max_lag_hours].  A positive lag means the feature *precedes* H2S
    (feature at time t-lag predicts H2S at time t).

    Args:
        df: DataFrame with a datetime index (or 'time'/'date' column) and H2S + feature columns.
        h2s_col: Column name for actual H2S measurements.
        max_lag_hours: Maximum lag in hours to compute (both positive and negative).
        top_n: Number of features to highlight in the line-plot panel.

    Returns:
        BytesIO PNG image.
    """
    CANDIDATE_FEATURES = [
        "tide_height",
        "Flow (m^3/s)--Border",
        "wind_speed_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "surface_pressure",
        "cloud_cover",
        "wind_direction_10m",
    ]

    # --- Prepare a clean, time-sorted series ---
    data = df.copy()
    if "time" in data.columns:
        data = data.set_index(pd.to_datetime(data["time"])).sort_index()
    elif "date" in data.columns:
        data = data.set_index(pd.to_datetime(data["date"])).sort_index()

    if h2s_col not in data.columns:
        # Return empty plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, f"Column '{h2s_col}' not found — cross-correlation unavailable",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    h2s = data[h2s_col].dropna()
    features = [f for f in CANDIDATE_FEATURES if f in data.columns]

    if not features:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No environmental feature columns found for cross-correlation",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf

    lags = list(range(-max_lag_hours, max_lag_hours + 1))
    corr_matrix = {}

    for feat in features:
        series = data[feat].reindex(h2s.index)
        corrs = []
        for lag in lags:
            # shift(lag) → feature(t - lag), so positive lag = feature leads H2S
            shifted = series.shift(lag)
            valid = pd.concat([h2s, shifted], axis=1).dropna()
            if len(valid) >= 10:
                corrs.append(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
            else:
                corrs.append(float("nan"))
        corr_matrix[feat] = corrs

    corr_df = pd.DataFrame(corr_matrix, index=lags)

    # Peak |correlation| for each feature — used to rank top_n and label the heatmap
    peak_corr = corr_df.abs().max()
    top_features = peak_corr.nlargest(top_n).index.tolist()

    # --- Layout: heatmap (top) + line chart (bottom) ---
    fig, (ax_heat, ax_line) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [1.4, 1]}
    )
    fig.suptitle(
        "Cross-Correlation: H2S vs Environmental Drivers\n"
        f"(positive lag = feature precedes H2S by N hours)",
        fontsize=13, fontweight="bold",
    )

    # Panel 1 — heatmap
    short_names = {
        "Flow (m^3/s)--Border": "Flow (m³/s)",
        "wind_speed_10m": "Wind speed",
        "temperature_2m": "Temperature",
        "relative_humidity_2m": "Humidity",
        "precipitation": "Precipitation",
        "surface_pressure": "Pressure",
        "cloud_cover": "Cloud cover",
        "wind_direction_10m": "Wind dir.",
        "tide_height": "Tide height",
    }
    display_cols = [short_names.get(f, f) for f in corr_df.columns]
    heat_data = corr_df.T.copy()
    heat_data.index = display_cols

    # Subsample lag axis to keep heatmap readable
    step = max(1, max_lag_hours // 12)
    heat_lags = [l for l in lags if l % step == 0]
    sns.heatmap(
        heat_data[heat_lags],
        ax=ax_heat,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=len(features) <= 6,
        fmt=".2f",
        linewidths=0.3,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax_heat.set_xlabel("Lag (hours)", fontsize=10, fontweight="bold")
    ax_heat.set_ylabel("")
    ax_heat.set_title("Correlation Heatmap (all features)", fontsize=11)
    ax_heat.tick_params(axis="x", rotation=0)

    # Panel 2 — top-N line chart
    palette = plt.get_cmap("tab10")(np.linspace(0, 1, len(top_features)))
    for feat, color in zip(top_features, palette):
        display_name = short_names.get(feat, feat)
        ax_line.plot(lags, corr_df[feat], label=display_name, color=color, linewidth=2)
        # Mark the lag of maximum absolute correlation
        peak_lag = corr_df[feat].abs().idxmax()
        peak_val = corr_df.loc[peak_lag, feat]
        ax_line.axvline(peak_lag, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax_line.scatter([peak_lag], [peak_val], color=color, s=60, zorder=5)

    ax_line.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax_line.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, label="Lag = 0")
    ax_line.set_xlabel("Lag (hours) — positive = feature leads H2S", fontsize=10, fontweight="bold")
    ax_line.set_ylabel("Pearson r", fontsize=10, fontweight="bold")
    ax_line.set_title(f"Top {top_n} Features by Peak |r|", fontsize=11)
    ax_line.legend(fontsize=8, loc="upper left", ncol=2)
    ax_line.grid(True, alpha=0.3)
    ax_line.set_xlim(lags[0], lags[-1])

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
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


# ==============================================================================
# Cell Comparison & Line Chart Visualizations
# ==============================================================================

CELL_COLORS = {
    'green': '#2ecc71',
    'yellow': '#f1c40f',
    'orange': '#e74c3c',
    None: '#d5d8dc',
}


def _categorize_h2s(value: float) -> Optional[str]:
    """Convert H2S ppb value to category string."""
    if pd.isna(value):
        return None
    if value < 5:
        return 'green'
    if value < 30:
        return 'yellow'
    return 'orange'


def _prepare_cell_data(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    time_col: str = 'time',
) -> pd.DataFrame:
    """Merge predictions and actuals, returning hourly cell data.

    Returns DataFrame with columns: date, hour, actual_category, predicted_category
    """
    pred = predictions.copy()
    act = actuals.copy()

    pred[time_col] = pd.to_datetime(pred[time_col])
    act[time_col] = pd.to_datetime(act[time_col])

    # Strip timezone for consistent merging
    if pred[time_col].dt.tz is not None:
        pred[time_col] = pred[time_col].dt.tz_localize(None)
    if act[time_col].dt.tz is not None:
        act[time_col] = act[time_col].dt.tz_localize(None)

    pred[time_col] = pred[time_col].dt.round('h')
    act[time_col] = act[time_col].dt.round('h')

    # Determine H2S column
    h2s_col = 'H2S'
    if h2s_col not in act.columns:
        h2s_cols = [c for c in act.columns if c.upper() == 'H2S' or 'h2s' in c.lower()]
        if h2s_cols:
            h2s_col = h2s_cols[0]
        else:
            h2s_col = None

    # Build actuals with categories
    if h2s_col and h2s_col in act.columns:
        act_slim = act[[time_col, h2s_col]].dropna(subset=[h2s_col]).copy()
        act_slim['actual_category'] = act_slim[h2s_col].apply(_categorize_h2s)
        act_slim = act_slim[[time_col, 'actual_category']].drop_duplicates(subset=[time_col], keep='last')
    else:
        act_slim = pd.DataFrame(columns=[time_col, 'actual_category'])

    # Build predictions slim
    pred_slim = pred[[time_col, 'predicted_category']].drop_duplicates(subset=[time_col], keep='last')

    # Full outer merge
    merged = pred_slim.merge(act_slim, on=time_col, how='outer')
    merged['date'] = merged[time_col].dt.strftime('%Y-%m-%d')
    merged['hour'] = merged[time_col].dt.hour

    return merged.sort_values(time_col)


def generate_cell_comparison_png(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    stations: Optional[List[tuple]] = None,
    time_col: str = 'time',
) -> BytesIO:
    """Generate cell comparison PNG showing actual vs predicted H2S per hour.

    Args:
        predictions: DataFrame with 'predicted_category' and time column.
        actuals: DataFrame with H2S values and time column.
            If it has a 'site_name' column, data is filtered per station.
        stations: List of (display_name, site_key) tuples.
            Defaults to [("NESTOR - BES", "NESTOR__BES")] if None.
        time_col: Name of time column.

    Returns:
        BytesIO PNG image.
    """
    from matplotlib.patches import Patch

    if stations is None:
        stations = [("NESTOR - BES", "NESTOR__BES")]

    has_site_col = 'site_name' in actuals.columns

    # Collect data per station
    station_data = []
    for display_name, site_key in stations:
        if has_site_col:
            act_station = actuals[actuals['site_name'].isin([site_key, display_name])].copy()
        else:
            act_station = actuals.copy()

        cell_df = _prepare_cell_data(predictions, act_station, time_col=time_col)
        dates = sorted(cell_df['date'].unique())
        station_data.append((display_name, cell_df, dates))

    # Collect all unique dates across stations
    all_dates = sorted(set(d for _, _, dates in station_data for d in dates))
    if not all_dates:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, 'No data available for cell comparison',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    hours = list(range(24))
    n_dates = len(all_dates)
    n_stations = len(stations)
    n_rows = n_dates * n_stations * 2

    fig_height = max(3, n_rows * 0.4 + 1.5)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    row_labels = []
    row_idx = 0

    for date_str in all_dates:
        for display_name, cell_df, _ in station_data:
            day_data = cell_df[cell_df['date'] == date_str]
            hour_actual = dict(zip(day_data['hour'], day_data['actual_category']))
            hour_pred = dict(zip(day_data['hour'], day_data['predicted_category']))

            # Actual row
            for h in hours:
                cat = hour_actual.get(h)
                color = CELL_COLORS.get(cat, CELL_COLORS[None])
                rect = plt.Rectangle((h, n_rows - row_idx - 1), 1, 1,
                                     facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            row_labels.append(f"{date_str}  {display_name} - Actual")
            row_idx += 1

            # Predicted row
            for h in hours:
                cat = hour_pred.get(h)
                color = CELL_COLORS.get(cat, CELL_COLORS[None])
                rect = plt.Rectangle((h, n_rows - row_idx - 1), 1, 1,
                                     facecolor=color, edgecolor='white', linewidth=0.5)
                ax.add_patch(rect)
            row_labels.append(f"{'':>10}  {display_name} - Predicted")
            row_idx += 1

    ax.set_xlim(0, 24)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([h + 0.5 for h in hours])
    ax.set_xticklabels([f'{h:02d}' for h in hours], fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(list(reversed(row_labels)), fontsize=7, family='monospace')
    ax.set_xlabel('Hour (UTC)', fontsize=10, fontweight='bold')
    ax.set_title('H2S Predictions vs Actuals — Cell Comparison', fontsize=13, fontweight='bold')

    legend_elements = [
        Patch(facecolor=CELL_COLORS['green'], label='Green (<5 ppb)'),
        Patch(facecolor=CELL_COLORS['yellow'], label='Yellow (5-30 ppb)'),
        Patch(facecolor=CELL_COLORS['orange'], label='Orange (>=30 ppb)'),
        Patch(facecolor=CELL_COLORS[None], label='No data'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, ncol=4,
              bbox_to_anchor=(1.0, -0.05))

    ax.tick_params(left=False, bottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_cell_comparison_html(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    stations: Optional[List[tuple]] = None,
    time_col: str = 'time',
) -> BytesIO:
    """Generate scrollable HTML cell comparison of actual vs predicted H2S per hour.

    Args:
        predictions: DataFrame with 'predicted_category' and time column.
        actuals: DataFrame with H2S values and time column.
        stations: List of (display_name, site_key) tuples.
        time_col: Name of time column.

    Returns:
        BytesIO with UTF-8 encoded HTML.
    """
    if stations is None:
        stations = [("NESTOR - BES", "NESTOR__BES")]

    has_site_col = 'site_name' in actuals.columns
    hours = list(range(24))

    # Gather per-station cell data
    station_cells = {}
    all_dates = set()
    total_matches = 0
    total_mismatches = 0

    for display_name, site_key in stations:
        if has_site_col:
            act_station = actuals[actuals['site_name'].isin([site_key, display_name])].copy()
        else:
            act_station = actuals.copy()

        cell_df = _prepare_cell_data(predictions, act_station, time_col=time_col)
        station_cells[display_name] = cell_df
        all_dates.update(cell_df['date'].unique())

        # Count matches/mismatches
        matched = cell_df.dropna(subset=['actual_category', 'predicted_category'])
        total_matches += int((matched['actual_category'] == matched['predicted_category']).sum())
        total_mismatches += int((matched['actual_category'] != matched['predicted_category']).sum())

    all_dates = sorted(all_dates)
    total_compared = total_matches + total_mismatches
    match_rate = (total_matches / total_compared * 100) if total_compared > 0 else 0

    # Build HTML
    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>H2S Cell Comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #eee; margin: 0; padding: 16px; }}
  h1 {{ font-size: 18px; margin: 0 0 8px; }}
  .stats {{ font-size: 13px; color: #aab; margin-bottom: 12px; }}
  .scroll-container {{ overflow-x: auto; max-width: 100%; }}
  table {{ border-collapse: collapse; font-size: 12px; white-space: nowrap; }}
  th {{ background: #16213e; padding: 4px 6px; font-weight: 600;
       position: sticky; top: 0; z-index: 2; border: 1px solid #2a2a4a; }}
  td {{ padding: 0; width: 28px; height: 24px; text-align: center;
       border: 1px solid #2a2a4a; }}
  td.label {{ padding: 2px 6px; text-align: left; font-weight: 500;
             background: #16213e; position: sticky; left: 0; z-index: 1; }}
  td.station-label {{ left: 80px; min-width: 108px; }}
  td.type-label {{ left: 190px; min-width: 72px; }}
  .cell-green {{ background: #2ecc71; }}
  .cell-yellow {{ background: #f1c40f; }}
  .cell-orange {{ background: #e74c3c; }}
  .cell-none {{ background: #34344a; }}
  .cell-mismatch {{ box-shadow: inset 0 0 0 2px #fff; }}
  tr.date-separator td {{ border-top: 3px solid #4a4a6a; }}
  .legend {{ display: flex; gap: 16px; margin-top: 12px; font-size: 12px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 4px; }}
  .legend-swatch {{ width: 16px; height: 16px; border-radius: 3px; display: inline-block; }}
</style>
</head>
<body>
<h1>H2S Predictions vs Actuals &mdash; Cell Comparison</h1>
<div class="stats">
  Dates: {all_dates[0] if all_dates else 'N/A'} to {all_dates[-1] if all_dates else 'N/A'} |
  Stations: {len(stations)} |
  Match rate: {match_rate:.1f}% ({total_matches}/{total_compared}) |
  Mismatches: {total_mismatches}
</div>
<div class="scroll-container">
<table>
<thead>
<tr>
  <th>Date</th>
  <th>Station</th>
  <th>Type</th>
  {"".join(f'<th>{h:02d}</th>' for h in hours)}
</tr>
</thead>
<tbody>
"""]

    for date_idx, date_str in enumerate(all_dates):
        for st_idx, (display_name, _site_key) in enumerate(stations):
            cell_df = station_cells[display_name]
            day_data = cell_df[cell_df['date'] == date_str]
            hour_actual = dict(zip(day_data['hour'], day_data['actual_category']))
            hour_pred = dict(zip(day_data['hour'], day_data['predicted_category']))

            separator_class = ' class="date-separator"' if (date_idx > 0 and st_idx == 0) else ''

            # Actual row
            date_label = date_str if st_idx == 0 else ''
            html_parts.append(f'<tr{separator_class}>')
            html_parts.append(f'<td class="label">{date_label}</td>')
            html_parts.append(f'<td class="label">{display_name}</td>')
            html_parts.append(f'<td class="label">Actual</td>')
            for h in hours:
                cat = hour_actual.get(h)
                cls = f'cell-{cat}' if cat else 'cell-none'
                html_parts.append(f'<td class="{cls}"></td>')
            html_parts.append('</tr>\n')

            # Predicted row
            html_parts.append('<tr>')
            html_parts.append('<td class="label"></td>')
            html_parts.append(f'<td class="label"></td>')
            html_parts.append(f'<td class="label">Predicted</td>')
            for h in hours:
                cat = hour_pred.get(h)
                actual_cat = hour_actual.get(h)
                cls = f'cell-{cat}' if cat else 'cell-none'
                if cat and actual_cat and cat != actual_cat:
                    cls += ' cell-mismatch'
                html_parts.append(f'<td class="{cls}"></td>')
            html_parts.append('</tr>\n')

    html_parts.append("""</tbody>
</table>
</div>
<div class="legend">
  <div class="legend-item"><span class="legend-swatch" style="background:#2ecc71"></span> Green (&lt;5 ppb)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#f1c40f"></span> Yellow (5-30 ppb)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#e74c3c"></span> Orange (&ge;30 ppb)</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#34344a"></span> No data</div>
  <div class="legend-item"><span class="legend-swatch" style="background:#e74c3c; box-shadow: inset 0 0 0 2px #fff"></span> Mismatch</div>
</div>
</body>
</html>""")

    html_content = ''.join(html_parts)
    buf = BytesIO()
    buf.write(html_content.encode('utf-8'))
    buf.seek(0)
    return buf


def generate_h2s_line_chart(
    actuals: pd.DataFrame,
    stations: Optional[List[tuple]] = None,
    time_col: str = 'time',
    h2s_col: str = 'H2S',
) -> BytesIO:
    """Generate line chart of actual H2S values with threshold zones.

    Args:
        actuals: DataFrame with H2S measurements, time column, and optionally site_name.
        stations: List of (display_name, site_key) tuples.
        time_col: Name of time column.
        h2s_col: Name of H2S column.

    Returns:
        BytesIO PNG image.
    """
    if stations is None:
        stations = [("NESTOR - BES", "NESTOR__BES")]

    # Find H2S column
    if h2s_col not in actuals.columns:
        h2s_candidates = [c for c in actuals.columns if c.upper() == 'H2S' or 'h2s' in c.lower()]
        if h2s_candidates:
            h2s_col = h2s_candidates[0]
        else:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.text(0.5, 0.5, 'No H2S column found', ha='center', va='center', fontsize=14)
            ax.axis('off')
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf

    has_site_col = 'site_name' in actuals.columns
    n_stations = len(stations)
    fig, axes = plt.subplots(n_stations, 1, figsize=(14, 3.5 * n_stations), sharex=True, squeeze=False)

    station_colors = {'NESTOR__BES': '#2ecc71', 'IB_CIVIC_CTR': '#3498db', 'SAN_YSIDRO': '#e74c3c'}

    for i, (display_name, site_key) in enumerate(stations):
        ax = axes[i, 0]

        if has_site_col:
            df = actuals[actuals['site_name'].isin([site_key, display_name])].copy()
        else:
            df = actuals.copy()

        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col)

        if len(df) == 0 or h2s_col not in df.columns:
            ax.text(0.5, 0.5, f'{display_name}: No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_ylabel(display_name, fontsize=10, fontweight='bold')
            continue

        times = df[time_col]
        h2s_values = df[h2s_col]

        # Background threshold zones
        y_max = max(h2s_values.max() * 1.1, 35)
        ax.axhspan(0, 5, color='#2ecc71', alpha=0.1)
        ax.axhspan(5, 30, color='#f1c40f', alpha=0.1)
        ax.axhspan(30, y_max, color='#e74c3c', alpha=0.1)

        # Threshold lines
        ax.axhline(5, color='#f1c40f', linestyle='--', linewidth=1, alpha=0.7, label='5 ppb')
        ax.axhline(30, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7, label='30 ppb')

        # H2S line
        color = station_colors.get(site_key, '#3498db')
        ax.plot(times, h2s_values, color=color, linewidth=1.5, label=f'{display_name} H2S')
        ax.fill_between(times, h2s_values, alpha=0.15, color=color)

        ax.set_ylabel('H2S (ppb)', fontsize=10, fontweight='bold')
        ax.set_title(display_name, fontsize=11, fontweight='bold')
        ax.set_ylim(0, y_max)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time', fontsize=10, fontweight='bold')
    plt.xticks(rotation=45)

    fig.suptitle('Actual H2S Measurements', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_daily_slack_chart(daily_station_forecasts: pd.DataFrame, env_label: str = "") -> BytesIO:
    """Generate a 3-panel 48h forecast chart for the daily pipeline Slack message.

    Args:
        daily_station_forecasts: Output of the daily_station_forecasts asset with columns:
            time, station, h2s_pred, risk, temp, wind_speed.
        env_label: Optional environment label (e.g. "DEV") for the title.

    Returns:
        BytesIO PNG image.
    """
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("America/Los_Angeles")
    station_colors = {"IB_CIVIC_CTR": "#1565c0", "NESTOR__BES": "#6a1b9a", "SAN_YSIDRO": "#2e7d32"}

    df = daily_station_forecasts.copy()
    if "time" in df.columns:
        df["time_pt"] = df["time"].apply(
            lambda t: t.astimezone(pacific) if hasattr(t, "astimezone") else t
        )
    else:
        df["time_pt"] = range(len(df))

    stations = df["station"].unique() if "station" in df.columns else []

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1.5, 1.5]}, sharex=True)
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    label_suffix = f" [{env_label}]" if env_label else ""
    fig.suptitle(f"H2S 48h Station Forecast{label_suffix}", fontsize=12, fontweight="bold", y=1.01)

    # Panel 1: h2s_pred per station + threshold lines
    ax = axes[0]
    for station in stations:
        sdf = df[df["station"] == station].sort_values("time_pt")
        color = station_colors.get(station, "#555555")
        ax.plot(sdf["time_pt"], sdf["h2s_pred"], color=color, linewidth=1.8, label=station.replace("__", " / "))
    ax.axhline(5, color="#ffc107", linewidth=1.0, linestyle="--", alpha=0.8, label="5 ppb")
    ax.axhline(30, color="#f44336", linewidth=1.0, linestyle="--", alpha=0.8, label="30 ppb")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("H2S (ppb)", fontsize=9, fontweight="bold")
    ax.set_title("Predicted H2S", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)

    # Panel 2: Temperature (use first station — same weather for all)
    ax = axes[1]
    if "temp" in df.columns:
        ref = df[df["station"] == stations[0]].sort_values("time_pt") if len(stations) else df.sort_values("time_pt")
        ax.plot(ref["time_pt"], ref["temp"], color="#e53935", linewidth=1.8)
        ax.fill_between(ref["time_pt"], ref["temp"], alpha=0.15, color="#e53935")
        ax.set_ylabel("°C", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Temperature unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Temperature", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.25)

    # Panel 3: Wind speed
    ax = axes[2]
    if "wind_speed" in df.columns:
        ref = df[df["station"] == stations[0]].sort_values("time_pt") if len(stations) else df.sort_values("time_pt")
        times = ref["time_pt"].tolist()
        bar_width_days = ((times[1] - times[0]).total_seconds() / 3600 * 0.85 / 24) if len(times) > 1 and hasattr(times[0], "toordinal") else 0.03
        ax.bar(ref["time_pt"], ref["wind_speed"], width=bar_width_days, color="#1565c0", alpha=0.7)
        ax.set_ylabel("m/s", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Wind data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Wind Speed", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.25, axis="y")

    if "time_pt" in df.columns and len(df) > 1 and hasattr(df["time_pt"].iloc[0], "toordinal"):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%-I %p\n%-m/%-d", tz=pacific))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), fontsize=7)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_mh_slack_chart(mh_forecasts: pd.DataFrame, env_label: str = "") -> BytesIO:
    """Generate a forecast chart for the multi-horizon pipeline Slack message.

    Shows predicted H2S per station across time, with risk-colored background bands.

    Args:
        mh_forecasts: Output of the mh_forecasts asset with columns:
            time, station, horizon, h2s_pred, risk.
        env_label: Optional environment label (e.g. "DEV") for the title.

    Returns:
        BytesIO PNG image.
    """
    import matplotlib.dates as mdates
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("America/Los_Angeles")
    station_colors = {"IB_CIVIC_CTR": "#1565c0", "NESTOR__BES": "#6a1b9a", "SAN_YSIDRO": "#2e7d32"}

    df = mh_forecasts.copy()
    if "time" in df.columns:
        df["time_pt"] = df["time"].apply(
            lambda t: t.astimezone(pacific) if hasattr(t, "astimezone") else t
        )
    else:
        df["time_pt"] = range(len(df))

    stations = list(df["station"].unique()) if "station" in df.columns else []
    n = len(stations)
    if n == 0:
        n = 1
        stations = ["unknown"]

    fig, axes = plt.subplots(n, 1, figsize=(12, max(4, n * 2.5)), sharex=True, squeeze=False)
    fig.patch.set_facecolor("#f8f9fa")
    for row in axes:
        for ax in row:
            ax.set_facecolor("#ffffff")

    label_suffix = f" [{env_label}]" if env_label else ""
    fig.suptitle(f"Multi-Horizon H2S Forecast{label_suffix}", fontsize=12, fontweight="bold", y=1.01)

    for i, station in enumerate(stations):
        ax = axes[i][0]
        sdf = df[df["station"] == station].sort_values("time_pt") if "station" in df.columns else df.sort_values("time_pt")
        color = station_colors.get(station, "#555555")

        # Plot each horizon as a separate line (dashed for longer horizons)
        linestyles = {"0_6h": "-", "6_24h": "--", "24_48h": "-.", "48_72h": ":"}
        if "horizon" in sdf.columns:
            for hz, ls in linestyles.items():
                hdf = sdf[sdf["horizon"] == hz]
                if hdf.empty:
                    continue
                ax.plot(hdf["time_pt"], hdf["h2s_pred"], color=color, linewidth=1.6, linestyle=ls,
                        label=hz.replace("_", "-") + "h", alpha=0.9)
        elif "h2s_pred" in sdf.columns:
            ax.plot(sdf["time_pt"], sdf["h2s_pred"], color=color, linewidth=1.8)

        # Threshold bands
        ax.axhline(5, color="#ffc107", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.axhline(30, color="#f44336", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("ppb", fontsize=8, fontweight="bold")
        ax.set_title(station.replace("__", " / "), fontsize=10, fontweight="bold", loc="left")
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.grid(True, alpha=0.25)

    if "time_pt" in df.columns and len(df) > 1 and hasattr(df["time_pt"].iloc[0], "toordinal"):
        axes[-1][0].xaxis.set_major_formatter(mdates.DateFormatter("%-I %p\n%-m/%-d", tz=pacific))
        axes[-1][0].xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(axes[-1][0].xaxis.get_majorticklabels(), fontsize=7)

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_forecast_slack_chart(predictions: pd.DataFrame, env_label: str = "") -> BytesIO:
    """Generate a 3-panel 48h forecast chart for Slack alerts.

    Args:
        predictions: Full h2s_predictions DataFrame with time, predicted_category,
            probability_orange, h2s_risk, temperature_2m, and wind_gusts_10m columns.
        env_label: Optional environment label (e.g. "DEV") appended to the chart title.

    Returns:
        BytesIO PNG image.
    """
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from zoneinfo import ZoneInfo

    pacific = ZoneInfo("America/Los_Angeles")
    category_colors = {"green": "#4caf50", "yellow": "#ffc107", "orange": "#f44336"}

    df = predictions.copy()

    if "time" in df.columns:
        df["time_pt"] = df["time"].apply(
            lambda t: t.astimezone(pacific) if hasattr(t, "astimezone") else t
        )
    else:
        df["time_pt"] = range(len(df))

    times = df["time_pt"].tolist()
    has_time = len(times) > 1

    # Bar width in matplotlib date units (days)
    if has_time and hasattr(times[0], "toordinal"):
        dt_hours = (times[1] - times[0]).total_seconds() / 3600
        bar_width_days = dt_hours * 0.85 / 24
    else:
        bar_width_days = 0.03

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
        sharex=True,
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    label_suffix = f" [{env_label}]" if env_label else ""
    fig.suptitle(
        f"H2S 48h Forecast — NESTOR / Berry Elementary School{label_suffix}",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )

    # --- Panel 1: Category bars + probability_orange line ---
    ax = axes[0]
    if has_time and "predicted_category" in df.columns:
        for _, row in df.iterrows():
            color = category_colors.get(row.get("predicted_category", "green"), "#4caf50")
            ax.bar(row["time_pt"], 1.0, width=bar_width_days, color=color, alpha=0.55, align="center")

    if "probability_orange" in df.columns and has_time:
        ax.plot(df["time_pt"], df["probability_orange"], color="#b71c1c", linewidth=1.8, label="P(orange)", zorder=5)
        ax.axhline(0.33, color="#888", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(0.66, color="#555", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_ylabel("P(orange)", fontsize=9, fontweight="bold")
    else:
        ax.set_ylabel("Category", fontsize=9, fontweight="bold")

    ax.set_ylim(0, 1.05)
    legend_patches = [Patch(facecolor=c, alpha=0.6, label=k.capitalize()) for k, c in category_colors.items()]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")
    ax.set_title("H2S Risk Category", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.25, axis="y")

    # --- Panel 2: Temperature ---
    ax = axes[1]
    if "temperature_2m" in df.columns and has_time:
        ax.plot(df["time_pt"], df["temperature_2m"], color="#e53935", linewidth=1.8)
        ax.fill_between(df["time_pt"], df["temperature_2m"], alpha=0.15, color="#e53935")
        ax.set_ylabel("°C", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Temperature unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Temperature", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.25)

    # --- Panel 3: Wind gusts ---
    ax = axes[2]
    if "wind_gusts_10m" in df.columns and has_time:
        ax.bar(df["time_pt"], df["wind_gusts_10m"], width=bar_width_days, color="#1565c0", alpha=0.7)
        ax.set_ylabel("m/s", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Wind gust data unavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Wind Gusts", fontsize=10, fontweight="bold", loc="left")
    ax.grid(True, alpha=0.25, axis="y")

    # X-axis time formatting
    if has_time and hasattr(times[0], "toordinal"):
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%-I %p\n%-m/%-d", tz=pacific))
        axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(axes[-1].xaxis.get_majorticklabels(), fontsize=7)

    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf
