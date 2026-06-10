"""Multi-station H2S model training helper functions.

Shared logic for feature engineering, model training, and selection.
Ported from src/train_models_auto.py for use in Dagster training pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score,
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from h2s.constants import (  # noqa: F401 — re-exported for downstream imports
    FLOW_COL,
    H2S_THRESHOLD_EXTREME,
    H2S_THRESHOLD_HIGH,
    MODEL_FEATURES,
    STATION_PARTITION_MAP,
    STATIONS,
)
from h2s.training.calibration_eval import recall_at_threshold
from h2s.training.feature_builder import ensure_base_features

ENSEMBLE_AUC_MARGIN = 0.01
ENSEMBLE_R2_MARGIN = 0.02
ENSEMBLE_RECALL_MARGIN = 0.02  # 2 pp on recall — tight enough that obvious wins dominate
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

# Default ensemble margin per selection metric — used when caller doesn't override.
_DEFAULT_MARGINS = {
    "recall_30": ENSEMBLE_RECALL_MARGIN,
    "recall_100": ENSEMBLE_RECALL_MARGIN,
    "r2": ENSEMBLE_R2_MARGIN,
    "auc": ENSEMBLE_AUC_MARGIN,
}


class EnsembleRegressor:
    """Weighted average of two regressors."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict(self, X):
        return self.weight_a * self.model_a.predict(X) + self.weight_b * self.model_b.predict(X)

    @property
    def feature_importances_(self):
        a = getattr(self.model_a, 'feature_importances_', np.zeros(len(MODEL_FEATURES)))
        b = getattr(self.model_b, 'feature_importances_', np.zeros(len(MODEL_FEATURES)))
        return self.weight_a * np.asarray(a) + self.weight_b * np.asarray(b)


class EnsembleClassifier:
    """Weighted probability average of two classifiers."""
    def __init__(self, model_a, model_b, weight_a=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = 1.0 - weight_a

    def predict_proba(self, X):
        return self.weight_a * self.model_a.predict_proba(X) + self.weight_b * self.model_b.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    @property
    def feature_importances_(self):
        a = getattr(self.model_a, 'feature_importances_', np.zeros(len(MODEL_FEATURES)))
        b = getattr(self.model_b, 'feature_importances_', np.zeros(len(MODEL_FEATURES)))
        return self.weight_a * np.asarray(a) + self.weight_b * np.asarray(b)


def prepare_multi_station_features(df: pd.DataFrame, station: str = None) -> pd.DataFrame:
    """Load, clean, and engineer features from raw training parquet.

    Args:
        df: Raw DataFrame from modeldata_h2s_nofill.parquet
        station: If provided, filter to this station name only

    Returns:
        Feature-engineered DataFrame with MODEL_FEATURES columns + targets
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df[(df['h2s_measured'] == True) & (df['H2S'] <= 500)].copy()
    df = df.sort_values(['site_name', 'time']).reset_index(drop=True)
    df['H2S'] = df['H2S'].clip(lower=0)

    # Add base features (time cyclicals, source_regime, wind cyclicals, interactions, etc.)
    df = ensure_base_features(df, flow_col=FLOW_COL)

    # Gust estimation if missing (not in feature_builder since it's data-specific)
    if 'wind_gusts_10m' not in df.columns:
        df['wind_gusts_10m'] = df.get('wind_speed_10m', pd.Series(0, index=df.index)) * 1.8

    # Rolling window features per site (must be computed per-site to avoid cross-contamination)
    # NOTE: ensure_base_features() already creates simple rolling windows, but we overwrite
    # them here with per-site calculations to prevent cross-contamination between stations
    for site in df['site_name'].unique():
        m = df['site_name'] == site
        for h in (2, 3, 4):
            df.loc[m, f'wind_speed_10m_avg_{h}h'] = df.loc[m, 'wind_speed_10m'].rolling(h, min_periods=1).mean()
            df.loc[m, f'wind_gusts_10m_max_{h}h'] = df.loc[m, 'wind_gusts_10m'].rolling(h, min_periods=1).max()

    # H2S lag/rolling features per site
    for site in df['site_name'].unique():
        m = df['site_name'] == site
        s = df.loc[m].copy()
        s['h2s_lag_1h'] = s['H2S'].shift(1)
        s['h2s_lag_3h'] = s['H2S'].shift(3)
        s['h2s_lag_6h'] = s['H2S'].shift(6)
        s['h2s_rolling_6h'] = s['H2S'].rolling(6, min_periods=1).mean()
        s['h2s_rolling_24h'] = s['H2S'].rolling(24, min_periods=1).mean()
        if FLOW_COL in df.columns:
            s['flow_lag_6h'] = s[FLOW_COL].shift(6)
            s['flow_rolling_24h'] = s[FLOW_COL].rolling(24, min_periods=1).mean()
        else:
            s['flow_lag_6h'] = 0.0
            s['flow_rolling_24h'] = 0.0
        for col in ['h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h', 'h2s_rolling_6h',
                    'h2s_rolling_24h', 'flow_lag_6h', 'flow_rolling_24h']:
            df.loc[m, col] = s[col].values

    # Target variables
    df['exceed_5'] = (df['H2S'] > 5).astype(int)
    df['exceed_10'] = (df['H2S'] > 10).astype(int)

    # Drop rows missing required features
    df = df.dropna(subset=MODEL_FEATURES).reset_index(drop=True)

    if station is not None:
        df = df[df['site_name'] == station].copy().reset_index(drop=True)

    return df


# ---- Model factory functions ----

def get_rf_regressor():
    return RandomForestRegressor(
        n_estimators=500, max_depth=20, min_samples_leaf=5,
        max_features='sqrt', n_jobs=-1, random_state=RANDOM_STATE
    )


def get_rf_classifier():
    return RandomForestClassifier(
        n_estimators=500, max_depth=20, min_samples_leaf=5,
        max_features='sqrt', class_weight='balanced',
        n_jobs=-1, random_state=RANDOM_STATE
    )


def get_xgb_regressor():
    return XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1
    )


def get_xgb_classifier(scale_pos=1.0):
    return XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, scale_pos_weight=scale_pos,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss'
    )


# ---- Evaluation helpers ----

def eval_regressor(model, X_test, y_test):
    """Evaluate a regressor on test set.

    Returns absolute-fit metrics (MAE/RMSE/R²) **and** alert-aligned recall
    at the operational thresholds (30 ppb watch, 100 ppb critical). Recall
    is computed by cutting both the prediction and the truth at the
    threshold — matches the `calibration_eval.recall_at_threshold`
    contract used by the calibration-aligned report.
    """
    y_pred = np.clip(model.predict(X_test), 0, None)
    y_test_arr = np.asarray(y_test, dtype=float)
    r30 = recall_at_threshold(y_test_arr, y_pred, H2S_THRESHOLD_HIGH)
    r100 = recall_at_threshold(y_test_arr, y_pred, H2S_THRESHOLD_EXTREME)
    return {
        'MAE': float(mean_absolute_error(y_test_arr, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test_arr, y_pred))),
        'R2': float(r2_score(y_test_arr, y_pred)),
        'recall_30': float(r30['recall']),
        'precision_30': float(r30['precision']),
        'n_positives_30': int(r30['n_positives']),
        'recall_100': float(r100['recall']),
        'precision_100': float(r100['precision']),
        'n_positives_100': int(r100['n_positives']),
    }


def eval_classifier(model, X_test, y_test, threshold=0.3):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
    auc = roc_auc_score(y_test, y_prob) if 0 < int(y_test.sum()) < len(y_test) else 0.5
    return {
        'AUC': float(auc),
        'Brier': float(brier_score_loss(y_test, y_prob)),
        'F1': float(f1_score(y_test, y_pred, zero_division=0)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
    }


def get_feature_importance(model, top_n=10):
    imp = getattr(model, 'feature_importances_', None)
    if imp is None:
        return {}
    imp = np.asarray(imp)
    idx = np.argsort(imp)[::-1][:top_n]
    return {MODEL_FEATURES[i]: round(float(imp[i]), 4) for i in idx if i < len(MODEL_FEATURES)}


def train_and_select(X_train, X_test, y_train, y_test, task: str,
                     ensemble_margin: float | None = None,
                     selection_metric: str | None = None):
    """Train RF + XGBoost for one task, auto-select or ensemble.

    Args:
        task: 'regression', 'clf_5ppb', or 'clf_10ppb'.
        ensemble_margin: Override default margin for ensembling. Interpreted
            in the units of the chosen ``selection_metric``.
        selection_metric: How to pick between RF and XGB on test set.
            Regression default: ``'recall_30'`` — recall at 30 ppb (operational
                watch threshold). Aligned with the calibration finding that
                R² rewards bulk fit on heavy-tailed series but hides large
                gaps in extreme recall.
            Classifier default: ``'auc'`` — backward-compat rank metric.
            Other regression options: ``'recall_100'``, ``'r2'``.

    Returns:
        (best_model, choice_str, metrics_dict).
        ``metrics_dict`` always includes both RF and XGB eval dicts plus
        ``selection_metric``, ``selection_value_rf``, ``selection_value_xgb``,
        and ``feature_importance``.
    """
    if task == 'regression':
        metric = selection_metric or 'recall_30'
        if metric not in {'recall_30', 'recall_100', 'r2'}:
            raise ValueError(
                f"selection_metric={metric!r} unsupported for regression; "
                "expected 'recall_30', 'recall_100', or 'r2'"
            )
        margin = ensemble_margin if ensemble_margin is not None else _DEFAULT_MARGINS[metric]
        metric_key = {'recall_30': 'recall_30', 'recall_100': 'recall_100', 'r2': 'R2'}[metric]

        rf = get_rf_regressor()
        rf.fit(X_train, y_train)
        m_rf = eval_regressor(rf, X_test, y_test)

        if HAS_XGB:
            xgb = get_xgb_regressor()
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            m_xgb = eval_regressor(xgb, X_test, y_test)

            s_rf, s_xgb = m_rf[metric_key], m_xgb[metric_key]
            diff = abs(s_rf - s_xgb)
            if diff < margin:
                # Ensemble within margin — weight by metric value, floor at 0.
                # For R² this preserves the legacy weighting; for recall it
                # yields a sensible 0/1-bounded blend.
                total = max(s_rf + s_xgb, 0.01)
                w_rf = max(s_rf, 0.0) / total
                model = EnsembleRegressor(rf, xgb, weight_a=w_rf)
                choice = 'Ensemble'
            elif s_rf > s_xgb:
                model, choice = rf, 'RandomForest'
            else:
                model, choice = xgb, 'XGBoost'
        else:
            model, choice, m_xgb = rf, 'RandomForest', None
            s_rf, s_xgb = m_rf[metric_key], None

        return model, choice, {
            'RF': m_rf, 'XGB': m_xgb, 'selected': choice,
            'selection_metric': metric,
            'selection_value_rf': float(s_rf),
            'selection_value_xgb': float(s_xgb) if s_xgb is not None else None,
            'feature_importance': get_feature_importance(model),
        }

    else:  # classifier
        # NOTE: classification selection stays on AUC by default. The
        # clf_5ppb / clf_10ppb thresholds aren't tied to the 30/100 ppb
        # alert boundaries, so a recall@30 selector here wouldn't be
        # well-defined. Aligning classifier selection with operational
        # alert thresholds is a separate piece of work.
        metric = selection_metric or 'auc'
        if metric != 'auc':
            raise ValueError(
                f"selection_metric={metric!r} unsupported for {task!r}; "
                "expected 'auc' (only AUC is supported for binary classifiers today)"
            )
        margin = ensemble_margin if ensemble_margin is not None else _DEFAULT_MARGINS['auc']

        pos_rate = float(y_train.mean())
        scale_pos = (1 - pos_rate) / max(pos_rate, 0.01)

        rf = get_rf_classifier()
        rf.fit(X_train, y_train)
        m_rf = eval_classifier(rf, X_test, y_test)

        if HAS_XGB:
            xgb = get_xgb_classifier(scale_pos)
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            m_xgb = eval_classifier(xgb, X_test, y_test)

            auc_rf, auc_xgb = m_rf['AUC'], m_xgb['AUC']
            diff = abs(auc_rf - auc_xgb)
            if diff < margin:
                total = auc_rf + auc_xgb
                w_rf = auc_rf / total
                model = EnsembleClassifier(rf, xgb, weight_a=w_rf)
                choice = 'Ensemble'
            elif auc_rf > auc_xgb:
                model, choice = rf, 'RandomForest'
            else:
                model, choice = xgb, 'XGBoost'
        else:
            model, choice, m_xgb = rf, 'RandomForest', None
            auc_rf, auc_xgb = m_rf['AUC'], None

        return model, choice, {
            'RF': m_rf, 'XGB': m_xgb, 'selected': choice,
            'selection_metric': metric,
            'selection_value_rf': float(auc_rf),
            'selection_value_xgb': float(auc_xgb) if auc_xgb is not None else None,
            'feature_importance': get_feature_importance(model),
        }
