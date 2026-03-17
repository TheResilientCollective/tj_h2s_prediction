"""XGBoost Model Training with Cross-Validation.

Implements time-series aware cross-validation for H2S prediction models.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from h2s.training.validation import calculate_metrics


HAZARD_CLASSES = {'orange', 'yellow'}


def calculate_class_weights(
    y: pd.Series, label_map: Dict[str, int], hazard_multiplier: float = 3.0
) -> Dict[int, float]:
    """Calculate class weights to handle imbalanced data.

    Args:
        y: Series of class labels (as strings: 'green', 'yellow', 'orange')
        label_map: Mapping from class names to integers
        hazard_multiplier: Extra weight multiplier applied to orange and yellow classes

    Returns:
        Dictionary mapping class indices to weights

    Example:
        >>> y = pd.Series(['green', 'green', 'yellow', 'orange'])
        >>> label_map = {'green': 0, 'yellow': 2, 'orange': 1}
        >>> weights = calculate_class_weights(y, label_map)
        >>> weights[0]  # green appears 2/4 times
        0.666...
    """
    class_counts = y.value_counts()
    total = len(y)
    n_classes = len(label_map)

    weights = {}
    for class_name, class_idx in label_map.items():
        count = class_counts.get(class_name, 1)  # Avoid division by zero
        w = total / (n_classes * count)
        if class_name in HAZARD_CLASSES:
            w *= hazard_multiplier
        weights[class_idx] = w

    return weights


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42, logger=None) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to oversample minority classes.

    Only oversamples yellow/orange (hazard) classes to improve recall on alerts.
    Uses BorderlineSMOTE to focus on decision boundary samples.

    Args:
        X: Feature DataFrame
        y: Encoded integer labels
        random_state: Random seed
        logger: Optional logger

    Returns:
        Resampled (X, y) with minority classes oversampled
    """
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE

    class_counts = y.value_counts()
    min_samples = int(class_counts.min())
    if logger:
        logger.info(f"  Before SMOTE: {dict(class_counts)}")

    # BorderlineSMOTE needs k_neighbors + 1 samples in the minority class.
    # Fall back to plain SMOTE when the minority class is very small.
    k = min(5, min_samples - 1)
    if k < 1:
        if logger:
            logger.warning(f"  SMOTE k<1 (minority={min_samples}): falling back to RandomOverSampler")
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=random_state)
        X_res, y_res = ros.fit_resample(X, y)
        if logger:
            logger.info(f"  After RandomOverSampler: {dict(pd.Series(y_res).value_counts())}")
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)

    try:
        smote = BorderlineSMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)
    except ValueError:
        smote = SMOTE(random_state=random_state, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)

    if logger:
        resampled_counts = pd.Series(y_res).value_counts()
        logger.info(f"  After SMOTE: {dict(resampled_counts)}")

    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)


def train_model_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label_map: Dict[str, int],
    n_folds: int = 5,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    use_class_weights: bool = True,
    use_smote: bool = False,
    random_state: int = 42,
    logger = None,  # Optional logger for debugging
    hazard_multiplier: float = 3.0,
) -> Tuple[xgb.XGBClassifier, List[Dict]]:
    """Train XGBoost model with time-series cross-validation.

    Args:
        X_train: Training features (preprocessed)
        y_train: Training labels (as strings: 'green', 'yellow', 'orange')
        label_map: Mapping from class names to integers
        n_folds: Number of CV folds (default: 5)
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum tree depth (default: 6)
        learning_rate: Learning rate (default: 0.1)
        use_class_weights: Whether to balance class weights (default: True)
        use_smote: Whether to apply SMOTE oversampling on each fold's training data (default: False)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (trained_model, cv_metrics_list)
        - trained_model: Final model trained on full training set
        - cv_metrics_list: List of dicts with metrics for each fold

    Example:
        >>> model, cv_metrics = train_model_with_cv(X, y, {'green': 0, 'orange': 1, 'yellow': 2})
        >>> model.predict(X_test)
        array([0, 1, 2, ...])
    """
    # Encode labels
    y_encoded = y_train.map(label_map)

    # Calculate class weights if enabled
    class_weights = None
    sample_weights = None
    if use_class_weights:
        class_weights = calculate_class_weights(y_train, label_map, hazard_multiplier)
        sample_weights = y_encoded.map(class_weights).values

    # Initialize cross-validation
    tscv = TimeSeriesSplit(n_splits=n_folds)
    cv_metrics = []

    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_train_fold = X_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_train_fold = y_encoded.iloc[train_idx]
        y_val_fold = y_encoded.iloc[val_idx]

        # Apply SMOTE to fold training data only (not validation)
        if use_smote:
            if logger:
                logger.info(f"Fold {fold+1}: Applying SMOTE...")
            X_train_fold, y_train_fold = apply_smote(X_train_fold, y_train_fold, random_state=random_state, logger=logger)

        # Train fold model
        fold_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=len(label_map),  # Explicitly set number of classes
            random_state=random_state,
            n_jobs=-1
        )

        # Fit with or without class weights
        if use_class_weights:
            sample_weights_fold = y_train_fold.map(class_weights).values
            fold_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weights_fold)
        else:
            fold_model.fit(X_train_fold, y_train_fold)

        # Evaluate on validation fold
        y_pred_fold = fold_model.predict(X_val_fold)

        # Handle case where predict returns probabilities instead of classes
        if y_pred_fold.ndim > 1:
            # Convert probability matrix to class predictions
            y_pred_fold = np.argmax(y_pred_fold, axis=1)

        # Debug: Check shapes before passing to calculate_metrics
        y_val_array = y_val_fold.values
        if logger:
            logger.info(f"Fold {fold+1}: X_val_fold shape={X_val_fold.shape}, y_val_fold shape={y_val_array.shape}, y_pred_fold shape={y_pred_fold.shape}")
            logger.info(f"Fold {fold+1}: val_idx has {len(val_idx)} indices, y_pred has {len(y_pred_fold)} predictions")

        try:
            fold_metrics = calculate_metrics(y_val_array, y_pred_fold)
        except ValueError as e:
            error_msg = (
                f"Error in fold {fold+1} calculating metrics:\n"
                f"  y_val_array shape: {y_val_array.shape}, dtype: {y_val_array.dtype}\n"
                f"  y_pred_fold shape: {y_pred_fold.shape}, dtype: {y_pred_fold.dtype}\n"
                f"  val_idx length: {len(val_idx)}\n"
                f"  y_val_array[:5]: {y_val_array[:5]}\n"
                f"  y_pred_fold[:10]: {y_pred_fold[:10]}\n"
                f"  Original error: {str(e)}"
            )
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg) from e
        fold_metrics['fold'] = fold + 1
        fold_metrics['train_size'] = len(train_idx)
        fold_metrics['val_size'] = len(val_idx)

        cv_metrics.append(fold_metrics)

    # Apply SMOTE to full training set for final model
    X_train_final = X_train
    y_encoded_final = y_encoded
    sample_weights_final = sample_weights
    if use_smote:
        if logger:
            logger.info("Applying SMOTE to full training set for final model...")
        X_train_final, y_encoded_final = apply_smote(X_train, y_encoded, random_state=random_state, logger=logger)
        if use_class_weights:
            class_weights_final = calculate_class_weights(
                y_encoded_final.map({v: k for k, v in label_map.items()}),
                label_map,
                hazard_multiplier,
            )
            sample_weights_final = y_encoded_final.map(class_weights_final).values

    # Train final model on full training set
    final_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='multi:softprob',
        eval_metric='mlogloss',
        num_class=len(label_map),  # Explicitly set number of classes
        random_state=random_state,
        n_jobs=-1
    )

    if use_class_weights:
        final_model.fit(X_train_final, y_encoded_final, sample_weight=sample_weights_final)
    else:
        final_model.fit(X_train_final, y_encoded_final)

    return final_model, cv_metrics


def train_random_forest_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label_map: Dict[str, int],
    n_folds: int = 5,
    n_estimators: int = 300,
    max_depth: int = None,
    use_class_weights: bool = True,
    use_smote: bool = False,
    random_state: int = 42,
    logger=None,
    hazard_multiplier: float = 3.0,
) -> Tuple["sklearn.ensemble.RandomForestClassifier", List[Dict]]:  # noqa: F821
    """Train a Random Forest classifier with time-series cross-validation.

    Same interface as train_model_with_cv so callers can swap models transparently.
    """
    from sklearn.ensemble import RandomForestClassifier

    y_encoded = y_train.map(label_map)

    # Build explicit weight dict with hazard multiplier instead of "balanced"
    if use_class_weights:
        class_weight_arg = calculate_class_weights(y_train, label_map, hazard_multiplier)
    else:
        class_weight_arg = None

    tscv = TimeSeriesSplit(n_splits=n_folds)
    cv_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold = X_train.iloc[train_idx]
        X_val  = X_train.iloc[val_idx]
        y_fold = y_encoded.iloc[train_idx]
        y_val  = y_encoded.iloc[val_idx]

        if use_smote:
            X_fold, y_fold = apply_smote(X_fold, y_fold, random_state=random_state, logger=logger)

        fold_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight_arg,
            random_state=random_state,
            n_jobs=-1,
        )
        fold_model.fit(X_fold, y_fold)

        y_pred = fold_model.predict(X_val)
        try:
            fold_metrics = calculate_metrics(y_val.values, y_pred)
        except ValueError as e:
            if logger:
                logger.error(f"Fold {fold+1} metrics error: {e}")
            raise
        fold_metrics['fold'] = fold + 1
        fold_metrics['train_size'] = len(train_idx)
        fold_metrics['val_size']   = len(val_idx)
        cv_metrics.append(fold_metrics)

    # Final model on full training set
    X_final, y_final = X_train, y_encoded
    if use_smote:
        X_final, y_final = apply_smote(X_train, y_encoded, random_state=random_state, logger=logger)

    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight_arg,
        random_state=random_state,
        n_jobs=-1,
    )
    final_model.fit(X_final, y_final)

    return final_model, cv_metrics


def get_feature_importance(model: xgb.XGBClassifier,
                           feature_names: List[str],
                           importance_type: str = 'gain') -> Dict[str, float]:
    """Extract feature importance from trained XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (must match model features)
        importance_type: Type of importance ('gain', 'weight', 'cover')

    Returns:
        Dictionary mapping feature names to importance scores

    Example:
        >>> importance = get_feature_importance(model, ['temp', 'wind', 'tide'])
        >>> importance['temp']
        0.342
    """
    # RandomForest (and other sklearn estimators) expose feature_importances_ directly
    if hasattr(model, 'feature_importances_'):
        return {
            name: float(score)
            for name, score in zip(feature_names, model.feature_importances_)
        }

    # XGBoost: get importance dict from booster (uses f0, f1, f2... as keys)
    importance_dict = model.get_booster().get_score(importance_type=importance_type)
    return {
        name: float(importance_dict.get(f'f{i}', 0.0))
        for i, name in enumerate(feature_names)
    }


def calculate_cv_summary(cv_metrics: List[Dict]) -> Dict:
    """Calculate summary statistics across CV folds.

    Args:
        cv_metrics: List of metrics dicts from cross-validation

    Returns:
        Dictionary with mean and std for each metric across folds

    Example:
        >>> cv_metrics = [
        ...     {'balanced_accuracy': 0.63, 'recall_orange': 0.60},
        ...     {'balanced_accuracy': 0.65, 'recall_orange': 0.62}
        ... ]
        >>> summary = calculate_cv_summary(cv_metrics)
        >>> summary['balanced_accuracy_mean']
        0.64
    """
    summary = {}

    # Collect all unique metric names across all folds
    all_metric_names = set()
    for fold in cv_metrics:
        all_metric_names.update(fold.keys())

    # Filter out non-numeric metrics
    metric_names = [k for k in all_metric_names
                   if k not in ['fold', 'confusion_matrix', 'train_size', 'val_size']]

    for metric_name in metric_names:
        # Only include folds that have this metric
        values = [fold[metric_name] for fold in cv_metrics if metric_name in fold]
        if values:  # Only calculate stats if we have values
            summary[f'{metric_name}_mean'] = float(np.mean(values))
            summary[f'{metric_name}_std'] = float(np.std(values))
            summary[f'{metric_name}_min'] = float(np.min(values))
            summary[f'{metric_name}_max'] = float(np.max(values))

    return summary
