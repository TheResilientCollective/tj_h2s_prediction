"""H2S Prediction System for NESTOR - BES with S3 support.

This module provides H2S forecasting with production classification model,
optimized for S3 storage and Dagster integration.

Predicts H2S levels in three categories:
- Green: H2S < 5 ppb (safe)
- Yellow: 5 ≤ H2S < 15 ppb (caution)
- Orange: H2S ≥ 15 ppb (alert)
"""

import json
import tempfile
from io import BytesIO
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb


class H2SPredictor:
    """H2S Forecasting Model with S3 support and JSON-based metadata."""

    def __init__(self, model, prep_info_dict):
        """Initialize predictor with loaded model and preprocessing info.

        Args:
            model: Loaded XGBoost classifier
            prep_info_dict: Dictionary with preprocessing metadata (from JSON)
        """
        self.model = model
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

        Args:
            model_path: Path to XGBoost model file (.json)
            preprocessing_json_path: Path to preprocessing info file (.json)

        Returns:
            H2SPredictor instance
        """
        # Load XGBoost model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Load preprocessing info from JSON
        with open(preprocessing_json_path, 'r') as f:
            prep_info = json.load(f)

        return cls(model, prep_info)

    @classmethod
    def from_s3(cls, s3_resource, model_path: str, preprocessing_json_path: str):
        """Load model and preprocessing info from S3.

        Args:
            s3_resource: S3Resource instance from resilient_workflows_public
            model_path: S3 path to model file (e.g., 'tijuana/forecast/models/model.json')
            preprocessing_json_path: S3 path to preprocessing JSON

        Returns:
            H2SPredictor instance
        """
        # Download model from S3 (returns bytes)
        model_bytes = s3_resource.getFile(path=model_path, bucket=s3_resource.S3_BUCKET)

        # XGBoost requires a file path, so use tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_model:
            tmp_model.write(model_bytes)
            tmp_model_path = tmp_model.name

        # Load model from temporary file
        model = xgb.XGBClassifier()
        model.load_model(tmp_model_path)

        # Clean up temp file
        import os
        os.unlink(tmp_model_path)

        # Download and parse preprocessing JSON from S3 (returns bytes)
        prep_bytes = s3_resource.getFile(path=preprocessing_json_path, bucket=s3_resource.S3_BUCKET)
        prep_info = json.loads(prep_bytes.decode('utf-8'))

        return cls(model, prep_info)

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data to match model training format.

        Args:
            df: DataFrame with raw sensor data

        Returns:
            DataFrame with engineered features ready for prediction
        """
        df = df.copy()

        # Convert time to datetime if present
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

            # Extract temporal features
            df['hour'] = df['time'].dt.hour
            df['day_of_week'] = df['time'].dt.dayofweek
            df['month'] = df['time'].dt.month

            # Cyclical encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Wind direction cyclical encoding
        if 'wind_direction_10m' in df.columns:
            df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction_10m']))
            df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction_10m']))

        # Interaction features
        if 'wind_speed_10m' in df.columns and 'temperature_2m' in df.columns:
            df['wind_temp_interaction'] = df['wind_speed_10m'] * df['temperature_2m']

        if 'relative_humidity_2m' in df.columns and 'temperature_2m' in df.columns:
            df['humidity_temp_interaction'] = df['relative_humidity_2m'] * df['temperature_2m']

        # Encode categorical variables using dict lookups (instead of LabelEncoder)
        if 'wind_direction_categorical' in df.columns and self.wind_cat_mapping:
            df['wind_direction_cat_encoded'] = df['wind_direction_categorical'].map(self.wind_cat_mapping).fillna(-1).astype(int)

        if 'tidal_state' in df.columns and self.tidal_mapping:
            df['tidal_state_encoded'] = df['tidal_state'].map(self.tidal_mapping).fillna(-1).astype(int)

        return df

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
