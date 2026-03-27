"""H2S Prediction System for NESTOR - BES with S3 support.

This module provides H2S forecasting with production classification model,
optimized for S3 storage and Dagster integration.

Predicts H2S levels per SD County guidance:
- Green: H2S < 5 ppb (safe)
- Yellow: 5 ≤ H2S < 30 ppb (caution)
- Orange: H2S ≥ 30 ppb (alert)

Note: The standalone scripts (src/) further split yellow into YELLOW_LOW (5-10 ppb)
and YELLOW_HIGH (10-30 ppb) using >5 and >10 binary classifiers for operational granularity.
"""

import json
import tempfile
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# Default Hill/log-logistic parameters
# c = EC50 (H2S concentration at 50% risk) = 5 ppb (green/yellow boundary)
# b = Hill coefficient (slope) = 1.23
_HILL_C = 5.0
_HILL_B = 1.23

# Representative H2S concentrations for each class (used to compute expected H2S)
_CLASS_H2S_PPB = {"green": 2.5, "yellow": 17.5, "orange": 50.0}


def hill_forward(x, c: float = _HILL_C, b: float = _HILL_B) -> np.ndarray:
    """Map H2S concentration (ppb) to risk score [0, 1] via the Hill/log-logistic function.

    f(x) = x^b / (c^b + x^b)

    Args:
        x: H2S concentration in ppb (≥ 0)
        c: EC50 — concentration at which risk = 0.5 (default 5 ppb)
        b: Hill coefficient / slope (default 1.23)

    Returns:
        Risk score in [0, 1]
    """
    x = np.asarray(x, dtype=float)
    x_b = np.power(np.clip(x, 0, None), b)
    c_b = np.power(c, b)
    return x_b / (c_b + x_b)


def hill_backward(risk, c: float = _HILL_C, b: float = _HILL_B) -> np.ndarray:
    """Map risk score [0, 1) to equivalent H2S concentration (ppb) — inverse Hill function.

    x = c * (f / (1 - f))^(1/b)

    Args:
        risk: Risk score in [0, 1) — values of 1.0 return inf
        c: EC50 (default 5 ppb)
        b: Hill coefficient (default 1.23)

    Returns:
        Equivalent H2S concentration in ppb
    """
    risk = np.asarray(risk, dtype=float)
    odds = risk / (1.0 - risk)
    return c * np.power(odds, 1.0 / b)


class H2SPredictor:
    """H2S Forecasting Model with S3 support and JSON-based metadata."""

    def __init__(self, model, prep_info_dict, model_name: str = ""):
        """Initialize predictor with loaded model and preprocessing info.

        Args:
            model: Loaded XGBoost classifier
            prep_info_dict: Dictionary with preprocessing metadata (from JSON)
            model_name: Human-readable name for the model (e.g. variant name)
        """
        self.model = model
        self.model_name = model_name
        self.prep_info = prep_info_dict
        self.feature_cols = prep_info_dict['feature_cols']
        self.class_names = prep_info_dict['class_names']
        self.site_name = prep_info_dict.get('site_name', 'NESTOR - BES')

        # Convert label encoder mappings to dicts for lookup
        self.wind_cat_mapping = prep_info_dict.get('wind_cat_mapping', {})
        self.tidal_mapping = prep_info_dict.get('tidal_mapping', {})

    @classmethod
    def from_local(cls, model_path: str, preprocessing_json_path: str):
        """Load model and preprocessing info from local filesystem.

        Supports XGBoost (.json) and scikit-learn/joblib (.joblib) models.
        """
        if model_path.endswith('.joblib'):
            import joblib
            model = joblib.load(model_path)
        else:
            model = xgb.XGBClassifier()
            model.load_model(model_path)

        with open(preprocessing_json_path, 'r') as f:
            prep_info = json.load(f)

        return cls(model, prep_info)

    @classmethod
    def from_s3(cls, s3_resource, model_path: str, preprocessing_json_path: str, model_name: str = ""):
        """Load model and preprocessing info from S3.

        Args:
            s3_resource: S3Resource instance from resilient_workflows_public
            model_path: S3 path to model file (e.g., 'tijuana/forecast/models/model.json')
            preprocessing_json_path: S3 path to preprocessing JSON

        Returns:
            H2SPredictor instance
        """
        import os

        # Download model from S3 (returns bytes)
        model_bytes = s3_resource.getFile(path=model_path, bucket=s3_resource.S3_BUCKET)

        # Detect format by content (magic byte 0x80 = pickle/joblib),
        # not by extension — so a random_forest deployed to a .json path still loads correctly.
        is_joblib = model_bytes[:1] == b'\x80'

        if is_joblib or model_path.endswith('.joblib'):
            import joblib, io
            model = joblib.load(io.BytesIO(model_bytes))
        else:
            # XGBoost requires a file path, so use tempfile
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
                tmp.write(model_bytes)
                tmp_path = tmp.name
            model = xgb.XGBClassifier()
            model.load_model(tmp_path)
            os.unlink(tmp_path)

        # Download and parse preprocessing JSON from S3 (returns bytes)
        prep_bytes = s3_resource.getFile(path=preprocessing_json_path, bucket=s3_resource.S3_BUCKET)
        prep_info = json.loads(prep_bytes.decode('utf-8'))

        return cls(model, prep_info, model_name=model_name)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data to match model training format.

        Computes all features used by both the original XGBoost classifier
        and the new per-station multi-task models (regression + classifiers).

        Args:
            df: DataFrame with raw sensor data

        Returns:
            DataFrame with engineered features ready for prediction
        """
        df = df.copy()

        # Sort by time before computing rolling features
        time_col = next((c for c in ['date', 'time'] if c in df.columns), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(time_col).reset_index(drop=True)

        # ---- Time cyclicals ----
        ts = df.get(time_col)
        if ts is not None:
            ts = pd.to_datetime(ts)
            if 'hour_sin' not in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)
            if 'month_sin' not in df.columns:
                df['month_sin'] = np.sin(2 * np.pi * ts.dt.month / 12)
                df['month_cos'] = np.cos(2 * np.pi * ts.dt.month / 12)
            # Night flag
            if 'is_night' not in df.columns:
                hour = ts.dt.hour
                df['is_night'] = ((hour < 6) | (hour >= 20)).astype(int)

        # ---- Source regime ----
        if 'source_regime' not in df.columns:
            def _src_regime(row):
                if not row.get('is_night', 0):
                    return 0
                wd = row.get('wind_direction_10m', 0)
                if 22.5 <= wd < 135:
                    return 1
                elif wd >= 247.5 or wd < 22.5:
                    return 2
                elif 135 <= wd < 247.5:
                    return 3
                return 0
            df['source_regime'] = df.apply(_src_regime, axis=1)

        # ---- Wind direction cyclical encoding ----
        if 'wind_direction_10m' in df.columns:
            if 'wind_direction_sin' not in df.columns:
                df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction_10m']))
                df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction_10m']))

        # ---- Rolling wind features ----
        if 'wind_speed_10m' in df.columns:
            for h in (2, 3, 4):
                col = f'wind_speed_10m_avg_{h}h'
                if col not in df.columns:
                    df[col] = df['wind_speed_10m'].rolling(h, min_periods=1).mean()
        if 'wind_gusts_10m' in df.columns:
            for h in (2, 3, 4):
                col = f'wind_gusts_10m_max_{h}h'
                if col not in df.columns:
                    df[col] = df['wind_gusts_10m'].rolling(h, min_periods=1).max()

        # ---- Interaction features ----
        if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
            if 'wind_temp_interaction' not in df.columns:
                df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']
        if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
            if 'humidity_temp_interaction' not in df.columns:
                df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

        # ---- Atmospheric stability ----
        if 'stable_atm' not in df.columns and 'wind_speed_10m' in df.columns:
            df['stable_atm'] = ((df['wind_speed_10m'] < 5) & (df.get('is_night', 0) == 1)).astype(int)

        # ---- Flow derivative features ----
        flow_col = 'Flow (m^3/s)--Border'
        if flow_col in df.columns:
            if 'flow_log' not in df.columns:
                df['flow_log'] = np.log1p(df[flow_col])
            if 'flow_low' not in df.columns:
                df['flow_low'] = (df[flow_col] < 1).astype(int)
            if 'flow_high' not in df.columns:
                df['flow_high'] = (df[flow_col] > 5).astype(int)
            if 'flow_lag_6h' not in df.columns:
                df['flow_lag_6h'] = df[flow_col].shift(6).fillna(df[flow_col].median() if len(df) > 0 else 2.0)
            if 'flow_rolling_24h' not in df.columns:
                df['flow_rolling_24h'] = df[flow_col].rolling(24, min_periods=1).mean()

        # ---- H2S lag features (fill with 0 in forecast mode) ----
        h2s_col = next((c for c in ['H2S', 'h2s'] if c in df.columns), None)
        if h2s_col and h2s_col in df.columns and df[h2s_col].notna().sum() > 0:
            series = df[h2s_col].fillna(0)
            if 'h2s_lag_1h' not in df.columns:
                df['h2s_lag_1h'] = series.shift(1).fillna(0)
            if 'h2s_lag_3h' not in df.columns:
                df['h2s_lag_3h'] = series.shift(3).fillna(0)
            if 'h2s_lag_6h' not in df.columns:
                df['h2s_lag_6h'] = series.shift(6).fillna(0)
            if 'h2s_rolling_6h' not in df.columns:
                df['h2s_rolling_6h'] = series.rolling(6, min_periods=1).mean()
            if 'h2s_rolling_24h' not in df.columns:
                df['h2s_rolling_24h'] = series.rolling(24, min_periods=1).mean()
        else:
            # Forecast mode: no H2S measurements available
            for col in ['h2s_lag_1h', 'h2s_lag_3h', 'h2s_lag_6h', 'h2s_rolling_6h', 'h2s_rolling_24h']:
                if col not in df.columns:
                    df[col] = 0.0

        # ---- Encode tidal state ----
        if 'tidal_state' in df.columns and self.tidal_mapping:
            df['tidal_state_encoded'] = df['tidal_state'].map(self.tidal_mapping).fillna(-1).astype(int)
        elif 'tidal_state_encoded' not in df.columns:
            tidal_numeric = {'flood': 0, 'ebb': 1, 'slack high': 2, 'slack low': 3}
            if 'tidal_state' in df.columns:
                df['tidal_state_encoded'] = df['tidal_state'].map(tidal_numeric).fillna(-1).astype(int)
            else:
                df['tidal_state_encoded'] = -1

        # ---- SBIWTP pass-through (if present, else defaults) ----
        sbiwtp_defaults = {
            'sbiwtp_flow_mgd': 23.5,
            'sbiwtp_anomaly': 0.0,
            'sbiwtp_deficit': 0.0,
            'sbiwtp_flow_x_temp': 23.5 * 18.0,  # flow × typical temp
            'sbiwtp_hourly_mgd': 23.5 / 24,
            'sbiwtp_sli': 0.0,
        }
        for col, default in sbiwtp_defaults.items():
            if col not in df.columns:
                df[col] = default

        # ---- Add missing columns with default values ----
        missing_cols = set(self.feature_cols) - set(df.columns)
        for col in missing_cols:
            df[col] = 0.0

        # ---- Return only model features + essential metadata ----
        # Keep time/date column for tracking, site_name if present
        keep_cols = self.feature_cols.copy()
        time_col = next((c for c in ['date', 'time'] if c in df.columns), None)
        if time_col and time_col not in keep_cols:
            keep_cols.append(time_col)
        if 'site_name' in df.columns and 'site_name' not in keep_cols:
            keep_cols.append('site_name')

        # Only return columns that exist in the dataframe
        return df[[col for col in keep_cols if col in df.columns]]

    def predict(self, df: pd.DataFrame, orange_threshold: Optional[float] = None,
                yellow_threshold: Optional[float] = None) -> pd.DataFrame:
        """Generate predictions for new data.

        Args:
            df: DataFrame with preprocessed features
            orange_threshold: Custom threshold for orange prediction (default: 0.33)
            yellow_threshold: Custom threshold for yellow prediction (default: 0.33)

        Returns:
            DataFrame with original data plus predictions
        """
        # Extract features in correct order
        X = df[self.feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Get predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Apply custom thresholds if provided
        if orange_threshold is not None or yellow_threshold is not None:
            ot = orange_threshold if orange_threshold is not None else 0.33
            yt = yellow_threshold if yellow_threshold is not None else 0.33

            predictions = self._apply_custom_thresholds(probabilities, ot, yt)

        # Add predictions to dataframe
        result = df.copy()
        result['predicted_category'] = [self.class_names[p] for p in predictions]
        result['probability_green'] = probabilities[:, 0]
        result['probability_orange'] = probabilities[:, 1]
        result['probability_yellow'] = probabilities[:, 2]
        result['confidence'] = probabilities.max(axis=1)

        # Add alert flag
        result['alert'] = result['predicted_category'].isin(['orange', 'yellow'])

        # h2s_risk: Hill-function risk score derived from expected H2S concentration.
        # Expected H2S = weighted sum of class-representative concentrations.
        expected_h2s = (
            result['probability_green'] * _CLASS_H2S_PPB['green']
            + result['probability_yellow'] * _CLASS_H2S_PPB['yellow']
            + result['probability_orange'] * _CLASS_H2S_PPB['orange']
        )
        result['h2s_risk'] = hill_forward(expected_h2s.values)

        return result

    def _apply_custom_thresholds(self, probabilities: np.ndarray,
                                  orange_threshold: float,
                                  yellow_threshold: float) -> np.ndarray:
        """Apply custom decision thresholds."""
        predictions = []

        for prob in probabilities:
            if prob[1] >= orange_threshold:  # orange probability
                predictions.append(1)  # orange
            elif prob[2] >= yellow_threshold:  # yellow probability
                predictions.append(2)  # yellow
            else:
                predictions.append(0)  # green

        return np.array(predictions)

    def predict_with_alerts(self, df: pd.DataFrame,
                            orange_threshold: Optional[float] = None,
                            yellow_threshold: Optional[float] = None) -> pd.DataFrame:
        """Generate predictions and return only alerts (orange/yellow).

        Args:
            df: DataFrame with preprocessed features
            orange_threshold: Custom threshold for orange prediction
            yellow_threshold: Custom threshold for yellow prediction

        Returns:
            DataFrame with only orange and yellow predictions
        """
        results = self.predict(df, orange_threshold, yellow_threshold)
        alerts = results[results['alert'] == True].copy()

        return alerts
