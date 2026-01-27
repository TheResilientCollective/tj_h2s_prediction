#!/usr/bin/env python3
"""
H2S Prediction System for NESTOR - BES
=======================================

This script generates H2S predictions from new data using the trained XGBoost model.

Usage:
    python predict_h2s.py --input new_data.csv --output predictions.csv
    python predict_h2s.py --input new_data.csv --output predictions.csv --threshold 0.25
    python predict_h2s.py --input new_data.csv --output predictions.csv --filter-alerts

Requirements:
    - nestor_xgboost_weighted_model.json
    - nestor_preprocessing_info.pkl
    - pandas, numpy, xgboost, scikit-learn
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class H2SPredictor:
    """
    H2S Forecasting Model for NESTOR - BES
    
    Predicts H2S levels in three categories:
    - Green: H2S < 5 ppb (safe)
    - Yellow: 5 ≤ H2S < 30 ppb (caution)
    - Orange: H2S ≥ 30 ppb (alert)
    """
    
    def __init__(self, model_path, preprocessing_path):
        """
        Load the trained model and preprocessing information.
        
        Args:
            model_path: Path to XGBoost model file (.json)
            preprocessing_path: Path to preprocessing info file (.pkl)
        """
        print("Loading H2S prediction model...")
        
        # Load XGBoost model
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Load preprocessing info
        with open(preprocessing_path, 'rb') as f:
            self.prep_info = pickle.load(f)
        
        self.feature_cols = self.prep_info['feature_cols']
        self.class_names = self.prep_info['class_names']
        self.le_wind_cat = self.prep_info.get('le_wind_cat')
        self.le_tidal = self.prep_info.get('le_tidal')
        
        print(f"✓ Preprocessing info loaded")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Classes: {self.class_names}")
        print()
    
    def preprocess_data(self, df):
        """
        Preprocess raw data to match model training format.
        
        Args:
            df: DataFrame with raw sensor data
            
        Returns:
            DataFrame with engineered features ready for prediction
        """
        print("Preprocessing data...")
        df = df.copy()
        
        # Convert time to datetime
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
        
        # Encode categorical variables
        if 'wind_direction_categorical' in df.columns and self.le_wind_cat is not None:
            df['wind_direction_cat_encoded'] = self.le_wind_cat.transform(df['wind_direction_categorical'])
        
        if 'tidal_state' in df.columns and self.le_tidal is not None:
            df['tidal_state_encoded'] = self.le_tidal.transform(df['tidal_state'])
        
        print(f"✓ Preprocessed {len(df)} samples")
        return df
    
    def predict(self, df, orange_threshold=None, yellow_threshold=None):
        """
        Generate predictions for new data.
        
        Args:
            df: DataFrame with preprocessed features
            orange_threshold: Custom threshold for orange prediction (default: 0.33)
            yellow_threshold: Custom threshold for yellow prediction (default: 0.33)
            
        Returns:
            DataFrame with original data plus predictions
        """
        print("\nGenerating predictions...")
        
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
        
        print(f"✓ Generated {len(result)} predictions")
        print("\nPrediction Summary:")
        print(result['predicted_category'].value_counts().to_string())
        print()
        
        return result
    
    def _apply_custom_thresholds(self, probabilities, orange_threshold, yellow_threshold):
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
    
    def predict_with_alerts(self, df, orange_threshold=None, yellow_threshold=None):
        """
        Generate predictions and return only alerts (orange/yellow).
        
        Args:
            df: DataFrame with preprocessed features
            orange_threshold: Custom threshold for orange prediction
            yellow_threshold: Custom threshold for yellow prediction
            
        Returns:
            DataFrame with only orange and yellow predictions
        """
        results = self.predict(df, orange_threshold, yellow_threshold)
        alerts = results[results['alert'] == True].copy()
        
        print(f"Found {len(alerts)} alerts ({(alerts['predicted_category']=='orange').sum()} orange, {(alerts['predicted_category']=='yellow').sum()} yellow)")
        
        return alerts


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate H2S predictions for NESTOR - BES',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python predict_h2s.py --input new_data.csv --output predictions.csv
  
  # With custom thresholds for more sensitive detection
  python predict_h2s.py --input new_data.csv --output predictions.csv --orange-threshold 0.25
  
  # Filter to show only alerts
  python predict_h2s.py --input new_data.csv --output alerts.csv --filter-alerts
  
  # Combine options
  python predict_h2s.py --input new_data.csv --output alerts.csv --orange-threshold 0.25 --filter-alerts
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file with new data')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for predictions')
    parser.add_argument('--model', default='nestor_xgboost_weighted_model.json',
                        help='Path to model file (default: nestor_xgboost_weighted_model.json)')
    parser.add_argument('--preprocessing', default='nestor_preprocessing_info.pkl',
                        help='Path to preprocessing file (default: nestor_preprocessing_info.pkl)')
    parser.add_argument('--orange-threshold', type=float,
                        help='Custom threshold for orange prediction (default: 0.33, lower=more sensitive)')
    parser.add_argument('--yellow-threshold', type=float,
                        help='Custom threshold for yellow prediction (default: 0.33)')
    parser.add_argument('--filter-alerts', action='store_true',
                        help='Only output orange and yellow predictions (exclude green)')
    parser.add_argument('--site', default='NESTOR - BES',
                        help='Filter to specific site (default: NESTOR - BES)')
    
    args = parser.parse_args()
    
    # Header
    print("="*80)
    print("H2S PREDICTION SYSTEM - NESTOR - BES")
    print("="*80)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} records")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return 1
    
    # Filter to site if multiple sites present
    if 'site_name' in df.columns:
        if args.site in df['site_name'].values:
            df = df[df['site_name'] == args.site].copy()
            print(f"✓ Filtered to {args.site}: {len(df)} records")
        else:
            print(f"⚠ Warning: Site '{args.site}' not found in data")
    
    # Initialize predictor
    try:
        predictor = H2SPredictor(args.model, args.preprocessing)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return 1
    
    # Preprocess
    try:
        df_processed = predictor.preprocess_data(df)
    except Exception as e:
        print(f"✗ Error preprocessing data: {e}")
        print(f"   Required columns: {predictor.feature_cols}")
        return 1
    
    # Generate predictions
    try:
        if args.filter_alerts:
            results = predictor.predict_with_alerts(
                df_processed,
                orange_threshold=args.orange_threshold,
                yellow_threshold=args.yellow_threshold
            )
        else:
            results = predictor.predict(
                df_processed,
                orange_threshold=args.orange_threshold,
                yellow_threshold=args.yellow_threshold
            )
    except Exception as e:
        print(f"✗ Error generating predictions: {e}")
        return 1
    
    # Save results
    try:
        results.to_csv(args.output, index=False)
        print(f"✓ Predictions saved to {args.output}")
        print(f"  Total records: {len(results)}")
        if 'alert' in results.columns:
            print(f"  Alerts: {results['alert'].sum()}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")
        return 1
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())
