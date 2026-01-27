"""H2S Model Training Infrastructure.

This package contains modules for automated monthly model retraining:
- relabeling: H2S threshold categorization logic
- model_trainer: XGBoost training with cross-validation
- validation: Model comparison and metrics calculation
"""

from h2s.training.relabeling import categorize_h2s
from h2s.training.model_trainer import train_model_with_cv, calculate_class_weights
from h2s.training.validation import calculate_metrics, compare_models

__all__ = [
    'categorize_h2s',
    'train_model_with_cv',
    'calculate_class_weights',
    'calculate_metrics',
    'compare_models',
]
