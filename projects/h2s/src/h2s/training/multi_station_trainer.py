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

STATIONS = {
    'SAN YSIDRO':   {'key': 'SAN_YSIDRO',    'partition': 'san_ysidro'},
    'NESTOR - BES': {'key': 'NESTOR__BES',    'partition': 'nestor_bes'},
    'IB CIVIC CTR': {'key': 'IB_CIVIC_CTR',   'partition': 'ib_civic_ctr'},
}

STATION_PARTITION_MAP = {v['partition']: k for k, v in STATIONS.items()}

MODEL_FEATURES = [
    'temperature_2m', 'wind_speed_10m', 'wind_direction_sin', 'wind_direction_cos',
    'wind_gusts_10m', 'precipitation', 'relative_humidity_2m', 'surface_pressure',
    'cloud_cover', 'dewpoint_2m',
    'wind_speed_10m_avg_2h', 'wind_speed_10m_avg_3h', 'wind_speed_10m_avg_4h',
    'wind_gusts_10m_max_2h', 'wind_gusts_10m_max_3h', 'wind_gusts_10m_max_4h',
    'tide_height', 'tidal_state_encoded',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_night', 'source_regime',
    'flow_log', 'flow_low', 'flow_high',
    'wind_temp_interaction', 'humidity_temp_interaction',
    'stable_atm',
    'h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h',
    'h2s_rolling_6h', 'h2s_rolling_24h',
    'flow_lag_6h', 'flow_rolling_24h',
    # SBIWTP effluent features (available when USIBWC feed is connected)
    'sbiwtp_flow_mgd', 'sbiwtp_anomaly', 'sbiwtp_deficit',
    'sbiwtp_flow_x_temp', 'sbiwtp_hourly_mgd', 'sbiwtp_sli',
]

ENSEMBLE_AUC_MARGIN = 0.01
ENSEMBLE_R2_MARGIN = 0.02
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

FLOW_COL = 'Flow (m^3/s)--Border'


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

    # Time cyclicals
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Day/night flag
    if 'day_night' in df.columns:
        df['is_night'] = (df['day_night'] == 'night').astype(int)
    else:
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] >= 20)).astype(int)

    # Source regime (wind direction × night)
    def _source_regime(row):
        if not row['is_night']:
            return 0
        wd = row.get('wind_direction_10m', 0)
        if 22.5 <= wd < 135:
            return 1
        elif wd >= 247.5 or wd < 22.5:
            return 2
        elif 135 <= wd < 247.5:
            return 3
        return 0

    df['source_regime'] = df.apply(_source_regime, axis=1)

    # Wind direction cyclical encoding (if not already present)
    if 'wind_direction_sin' not in df.columns and 'wind_direction_10m' in df.columns:
        df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction_10m']))
        df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction_10m']))

    # Flow features
    if FLOW_COL in df.columns:
        df['flow_log'] = np.log1p(df[FLOW_COL])
        df['flow_low'] = (df[FLOW_COL] < 1).astype(int)
        df['flow_high'] = (df[FLOW_COL] > 5).astype(int)
    else:
        df['flow_log'] = 0.0
        df['flow_low'] = 0
        df['flow_high'] = 0

    # Atmospheric stability
    if 'wind_speed_10m' in df.columns:
        df['stable_atm'] = ((df['wind_speed_10m'] < 5) & (df['is_night'] == 1)).astype(int)
    else:
        df['stable_atm'] = 0

    # Interaction features
    if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
        df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']
    if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
        df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

    # Gust estimation if missing
    if 'wind_gusts_10m' not in df.columns:
        df['wind_gusts_10m'] = df.get('wind_speed_10m', pd.Series(0, index=df.index)) * 1.8

    # Rolling window features per site (must be computed per-site to avoid cross-contamination)
    for site in df['site_name'].unique():
        m = df['site_name'] == site
        for h in (2, 3, 4):
            df.loc[m, f'wind_speed_10m_avg_{h}h'] = df.loc[m, 'wind_speed_10m'].rolling(h, min_periods=1).mean()
            df.loc[m, f'wind_gusts_10m_max_{h}h'] = df.loc[m, 'wind_gusts_10m'].rolling(h, min_periods=1).max()

    # Tidal state encoding
    tidal_map = {'flood': 0, 'ebb': 1, 'slack high': 2, 'slack low': 3}
    if 'tidal_state' in df.columns and 'tidal_state_encoded' not in df.columns:
        df['tidal_state_encoded'] = df['tidal_state'].map(tidal_map).fillna(-1).astype(int)
    elif 'tidal_state_encoded' not in df.columns:
        df['tidal_state_encoded'] = -1

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

    # SBIWTP defaults — fill NaN when feed is not yet connected
    sbiwtp_defaults = {
        'sbiwtp_flow_mgd': 23.5,
        'sbiwtp_hourly_mgd': 23.5 / 24,
        'sbiwtp_anomaly': 0.0,
        'sbiwtp_deficit': 0.0,
        'sbiwtp_flow_x_temp': 0.0,
        'sbiwtp_sli': 0.0,
    }
    for col, val in sbiwtp_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
        else:
            df[col] = val

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
    y_pred = np.clip(model.predict(X_test), 0, None)
    return {
        'MAE': float(mean_absolute_error(y_test, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'R2': float(r2_score(y_test, y_pred)),
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
                     ensemble_margin: float = None):
    """Train RF + XGBoost for one task, auto-select or ensemble.

    Args:
        task: 'regression', 'clf_5ppb', or 'clf_10ppb'
        ensemble_margin: Override default AUC/R² margins for ensembling

    Returns:
        (best_model, choice_str, metrics_dict)
    """
    auc_margin = ensemble_margin or ENSEMBLE_AUC_MARGIN
    r2_margin = (ensemble_margin * 2) if ensemble_margin else ENSEMBLE_R2_MARGIN

    if task == 'regression':
        rf = get_rf_regressor()
        rf.fit(X_train, y_train)
        m_rf = eval_regressor(rf, X_test, y_test)

        if HAS_XGB:
            xgb = get_xgb_regressor()
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            m_xgb = eval_regressor(xgb, X_test, y_test)

            r2_rf, r2_xgb = m_rf['R2'], m_xgb['R2']
            diff = abs(r2_rf - r2_xgb)
            if diff < r2_margin:
                total = max(r2_rf + r2_xgb, 0.01)
                w_rf = max(r2_rf, 0) / total
                model = EnsembleRegressor(rf, xgb, weight_a=w_rf)
                choice = 'Ensemble'
            elif r2_rf > r2_xgb:
                model, choice = rf, 'RandomForest'
            else:
                model, choice = xgb, 'XGBoost'
        else:
            model, choice, m_xgb = rf, 'RandomForest', None

        return model, choice, {
            'RF': m_rf, 'XGB': m_xgb, 'selected': choice,
            'feature_importance': get_feature_importance(model),
        }

    else:  # classifier
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
            if diff < auc_margin:
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

        return model, choice, {
            'RF': m_rf, 'XGB': m_xgb, 'selected': choice,
            'feature_importance': get_feature_importance(model),
        }
