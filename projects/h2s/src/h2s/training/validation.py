"""Model Validation and Comparison Utilities.

Functions for calculating classification metrics and comparing model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix as sklearn_confusion_matrix
)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     class_names: list = None) -> Dict:
    """Calculate comprehensive classification metrics.

    Args:
        y_true: True labels (encoded as 0, 1, 2)
        y_pred: Predicted labels (encoded as 0, 1, 2)
        class_names: Optional list of class names (default: ['green', 'orange', 'yellow'])

    Returns:
        Dictionary with metrics:
        - balanced_accuracy: Overall balanced accuracy
        - precision_{class}: Precision per class
        - recall_{class}: Recall per class
        - f1_{class}: F1 score per class
        - confusion_matrix: Confusion matrix as 2D list

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1])
        >>> y_pred = np.array([0, 1, 2, 1, 1])
        >>> metrics = calculate_metrics(y_true, y_pred)
        >>> metrics['balanced_accuracy']
        0.8333...
    """
    if class_names is None:
        class_names = ['green', 'orange', 'yellow']

    # Ensure y_true and y_pred are 1D integer arrays
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    # Determine which labels are actually present in the data
    unique_labels = sorted(np.union1d(y_true, y_pred).tolist())

    # Calculate precision, recall, f1 per class (only for present classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred,
        average=None,
        labels=unique_labels,
        zero_division=0
    )

    # Build metrics dictionary
    metrics = {
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'confusion_matrix': sklearn_confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
    }

    # Add per-class metrics (only for classes actually present)
    for idx, label_idx in enumerate(unique_labels):
        # Map label index to class name
        if label_idx < len(class_names):
            class_name = class_names[label_idx]
        else:
            class_name = f'class_{label_idx}'

        metrics[f'precision_{class_name}'] = float(precision[idx])
        metrics[f'recall_{class_name}'] = float(recall[idx])
        metrics[f'f1_{class_name}'] = float(f1[idx])

    return metrics


def calculate_false_alarm_rate(y_true: np.ndarray, y_pred: np.ndarray,
                               positive_class: int = 1) -> float:
    """Calculate false alarm rate (false positive rate).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        positive_class: Class to consider as positive (default: 1 for orange)

    Returns:
        False alarm rate as float (0-1)

    Notes:
        False alarm rate = FP / (FP + TN)
        For H2S: measures how often we predict orange when it's actually green
    """
    # Create binary labels (positive class vs all others)
    y_true_binary = (y_true == positive_class).astype(int)
    y_pred_binary = (y_pred == positive_class).astype(int)

    # Calculate confusion matrix for binary classification
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))

    if (fp + tn) == 0:
        return 0.0

    return float(fp / (fp + tn))


def compare_models(new_metrics: Dict, current_metrics: Dict,
                  min_balanced_acc_delta: float = -0.05,
                  min_orange_recall_delta: float = -0.05,
                  min_orange_precision_delta: float = -0.10) -> Tuple[bool, Dict]:
    """Compare new model vs current model and recommend approval.

    Compares only common metrics between models. If field sets differ,
    comparison is limited to accuracy, recall, and precision.

    Args:
        new_metrics: Metrics dict from new model
        current_metrics: Metrics dict from current production model
        min_balanced_acc_delta: Minimum acceptable balanced accuracy difference (negative = worse)
        min_orange_recall_delta: Minimum acceptable orange recall difference
        min_orange_precision_delta: Minimum acceptable orange precision difference

    Returns:
        Tuple of (approval_recommended: bool, comparison_details: dict)

    Example:
        >>> new_metrics = {'balanced_accuracy': 0.65, 'recall_orange': 0.60}
        >>> current_metrics = {'balanced_accuracy': 0.63, 'recall_orange': 0.61}
        >>> approved, details = compare_models(new_metrics, current_metrics)
        >>> approved
        True
    """
    # Identify available metrics
    new_keys = set(new_metrics.keys())
    current_keys = set(current_metrics.keys())
    common_keys = new_keys & current_keys
    missing_in_new = current_keys - new_keys
    missing_in_current = new_keys - current_keys

    field_mismatch = bool(missing_in_new or missing_in_current)

    # Calculate differences for common metrics
    metric_differences = {}
    quality_gates = {}

    # Balanced accuracy (primary metric)
    balanced_acc_diff = 0.0
    balanced_acc_ok = True
    if 'balanced_accuracy' in common_keys:
        balanced_acc_diff = new_metrics['balanced_accuracy'] - current_metrics['balanced_accuracy']
        balanced_acc_ok = balanced_acc_diff >= min_balanced_acc_delta
        metric_differences['balanced_accuracy'] = float(balanced_acc_diff)
        quality_gates['balanced_accuracy_gate'] = {
            'passed': balanced_acc_ok,
            'threshold': min_balanced_acc_delta,
            'actual': float(balanced_acc_diff)
        }
    elif 'accuracy' in common_keys:
        # Fallback to regular accuracy
        balanced_acc_diff = new_metrics['accuracy'] - current_metrics['accuracy']
        balanced_acc_ok = balanced_acc_diff >= min_balanced_acc_delta
        metric_differences['accuracy'] = float(balanced_acc_diff)
        quality_gates['accuracy_gate'] = {
            'passed': balanced_acc_ok,
            'threshold': min_balanced_acc_delta,
            'actual': float(balanced_acc_diff)
        }

    # Orange recall
    orange_recall_diff = 0.0
    orange_recall_ok = True
    if 'recall_orange' in common_keys:
        orange_recall_diff = new_metrics['recall_orange'] - current_metrics['recall_orange']
        orange_recall_ok = orange_recall_diff >= min_orange_recall_delta
        metric_differences['recall_orange'] = float(orange_recall_diff)
        quality_gates['orange_recall_gate'] = {
            'passed': orange_recall_ok,
            'threshold': min_orange_recall_delta,
            'actual': float(orange_recall_diff)
        }

    # Orange precision
    orange_precision_diff = 0.0
    orange_precision_ok = True
    if 'precision_orange' in common_keys:
        orange_precision_diff = new_metrics['precision_orange'] - current_metrics['precision_orange']
        orange_precision_ok = orange_precision_diff >= min_orange_precision_delta
        metric_differences['precision_orange'] = float(orange_precision_diff)
        quality_gates['orange_precision_gate'] = {
            'passed': orange_precision_ok,
            'threshold': min_orange_precision_delta,
            'actual': float(orange_precision_diff)
        }

    approval_recommended = balanced_acc_ok and orange_recall_ok and orange_precision_ok

    comparison_details = {
        'metric_differences': metric_differences,
        'quality_gates': quality_gates,
        'approval_recommended': approval_recommended,
        'field_mismatch': field_mismatch,
        'missing_in_new': list(missing_in_new) if missing_in_new else [],
        'missing_in_current': list(missing_in_current) if missing_in_current else [],
        'new_model_metrics': {k: float(v) for k, v in new_metrics.items() if isinstance(v, (int, float))},
        'current_model_metrics': {k: float(v) for k, v in current_metrics.items() if isinstance(v, (int, float))},
    }

    return approval_recommended, comparison_details


def format_metrics_report(metrics: Dict, model_name: str = "Model") -> str:
    """Format metrics dictionary as human-readable report.

    Args:
        metrics: Metrics dictionary from calculate_metrics()
        model_name: Name to display in report (default: "Model")

    Returns:
        Formatted string report

    Example:
        >>> metrics = {'balanced_accuracy': 0.63, 'recall_orange': 0.61, ...}
        >>> print(format_metrics_report(metrics, "New Model"))
        New Model Performance:
        ====================
        ...
    """
    report_lines = [
        f"{model_name} Performance:",
        "=" * (len(model_name) + 13),
        f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}",
        "",
        "Per-Class Metrics:",
    ]

    for class_name in ['green', 'yellow', 'orange']:
        precision = metrics.get(f'precision_{class_name}', 0)
        recall = metrics.get(f'recall_{class_name}', 0)
        f1 = metrics.get(f'f1_{class_name}', 0)

        report_lines.append(f"  {class_name.capitalize()}:")
        report_lines.append(f"    Precision: {precision:.3f}")
        report_lines.append(f"    Recall:    {recall:.3f}")
        report_lines.append(f"    F1 Score:  {f1:.3f}")

    return "\n".join(report_lines)
