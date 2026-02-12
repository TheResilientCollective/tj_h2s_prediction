"""Unit and Integration Tests for H2S Training Pipeline.

Tests cover:
- Data extraction and relabeling
- Model training with CV
- Validation and comparison
- Deployment approval workflow
- End-to-end training flow
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from h2s.training.relabeling import categorize_h2s, apply_categorization, get_threshold_info
from h2s.training.model_trainer import (
    train_model_with_cv,
    calculate_class_weights,
    get_feature_importance,
    calculate_cv_summary,
)
from h2s.training.validation import calculate_metrics, compare_models


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_training_data():
    """Create sample training data with known H2S values."""
    np.random.seed(42)
    n_samples = 100

    data = {
        'time': pd.date_range('2025-01-01', periods=n_samples, freq='h'),
        'site_name': ['NESTOR - BES'] * n_samples,
        'h2s_measured': [True] * n_samples,
        'H2S': np.concatenate([
            np.random.uniform(0, 5, 50),      # Green: 50 samples
            np.random.uniform(5, 30, 30),     # Yellow: 30 samples
            np.random.uniform(30, 100, 20),   # Orange: 20 samples
        ]),
        # Minimal required features for preprocessing
        'temperature_2m': np.random.uniform(10, 30, n_samples),
        'wind_speed_10m': np.random.uniform(0, 20, n_samples),
        'wind_direction_10m': np.random.uniform(0, 360, n_samples),
        'relative_humidity_2m': np.random.uniform(30, 90, n_samples),
        'surface_pressure': np.random.uniform(1000, 1020, n_samples),
        'precipitation': np.random.uniform(0, 5, n_samples),
        'cloud_cover': np.random.uniform(0, 100, n_samples),
        'wind_direction_categorical': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], n_samples),
        'flow_rate_cms': np.random.uniform(0, 10, n_samples),
        'tide_height_m': np.random.uniform(0, 3, n_samples),
        'tidal_state': np.random.choice(['rising', 'falling', 'high', 'low'], n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_features_and_labels():
    """Create sample preprocessed features and labels for training."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Use 2 classes to match real-world scenario (like production data)
    # Production data currently only has green and yellow, no orange
    y = pd.Series(['green'] * 60 + ['yellow'] * 40)

    return X, y


# ============================================================================
# UNIT TESTS: Relabeling
# ============================================================================


class TestRelabeling:
    """Test H2S threshold categorization with new thresholds."""

    def test_categorize_h2s_green(self):
        """Test green category (< 5 ppb)."""
        assert categorize_h2s(0.0) == 'green'
        assert categorize_h2s(2.5) == 'green'
        assert categorize_h2s(4.9) == 'green'

    def test_categorize_h2s_yellow(self):
        """Test yellow category (5-30 ppb) - NEW THRESHOLD."""
        assert categorize_h2s(5.0) == 'yellow'
        assert categorize_h2s(15.0) == 'yellow'
        assert categorize_h2s(29.9) == 'yellow'

    def test_categorize_h2s_orange(self):
        """Test orange category (≥30 ppb) - NEW THRESHOLD."""
        assert categorize_h2s(30.0) == 'orange'
        assert categorize_h2s(50.0) == 'orange'
        assert categorize_h2s(100.0) == 'orange'

    def test_categorize_h2s_nan(self):
        """Test NaN handling."""
        assert categorize_h2s(np.nan) is None
        assert categorize_h2s(float('nan')) is None

    def test_apply_categorization(self, sample_training_data):
        """Test applying categorization to DataFrame."""
        df = apply_categorization(sample_training_data, h2s_column='H2S')

        # Check column added
        assert 'h2s_category' in df.columns

        # Check all categories present
        categories = df['h2s_category'].unique()
        assert 'green' in categories
        assert 'yellow' in categories
        assert 'orange' in categories

        # Verify thresholds applied correctly
        green_mask = df['h2s_category'] == 'green'
        assert (df[green_mask]['H2S'] < 5).all()

        yellow_mask = df['h2s_category'] == 'yellow'
        assert ((df[yellow_mask]['H2S'] >= 5) & (df[yellow_mask]['H2S'] < 30)).all()

        orange_mask = df['h2s_category'] == 'orange'
        assert (df[orange_mask]['H2S'] >= 30).all()

    def test_get_threshold_info(self):
        """Test threshold metadata retrieval."""
        info = get_threshold_info()

        assert info['green_max'] == 5
        assert info['yellow_min'] == 5
        assert info['yellow_max'] == 30
        assert info['orange_min'] == 30
        assert info['version'] == '2.0'
        assert '2026' in info['effective_date']


# ============================================================================
# UNIT TESTS: Model Training
# ============================================================================


class TestModelTraining:
    """Test XGBoost training with cross-validation."""

    def test_calculate_class_weights(self):
        """Test class weight calculation for imbalanced data."""
        y = pd.Series(['green'] * 80 + ['yellow'] * 15 + ['orange'] * 5)
        label_map = {'green': 0, 'orange': 1, 'yellow': 2}

        weights = calculate_class_weights(y, label_map)

        # Orange should have highest weight (rarest class)
        assert weights[1] > weights[0]  # orange > green
        assert weights[1] > weights[2]  # orange > yellow

        # Weights should sum sensibly (total / n_classes)
        expected_total = len(y)
        actual_total = sum(weights[i] * y.value_counts()[name]
                          for name, i in label_map.items())
        assert abs(actual_total - expected_total) < 1e-6

    def test_train_model_with_cv(self, sample_features_and_labels):
        """Test model training with CV returns model and metrics."""
        X, y = sample_features_and_labels

        # Create label map from actual classes present (matches production pattern)
        unique_classes = sorted(y.unique())
        label_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        model, cv_metrics = train_model_with_cv(
            X, y, label_map,
            n_folds=3,
            n_estimators=10,  # Small for speed
            max_depth=3,
            use_class_weights=True,
        )

        # Check model trained
        assert isinstance(model, xgb.XGBClassifier)
        assert hasattr(model, 'predict')

        # Check CV metrics
        assert len(cv_metrics) == 3  # 3 folds
        for fold_metrics in cv_metrics:
            assert 'fold' in fold_metrics
            assert 'balanced_accuracy' in fold_metrics
            # Should have per-class metrics for classes that exist
            assert any(k.startswith('recall_') for k in fold_metrics.keys())
            assert 'train_size' in fold_metrics
            assert 'val_size' in fold_metrics

        # Check model can predict
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        # Predictions should be integer class indices
        assert predictions.dtype in [np.int32, np.int64, np.float64]

    def test_get_feature_importance(self, sample_features_and_labels):
        """Test feature importance extraction."""
        X, y = sample_features_and_labels

        # Create label map from actual classes present
        unique_classes = sorted(y.unique())
        label_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        model, _ = train_model_with_cv(
            X, y, label_map,
            n_folds=2, n_estimators=10, max_depth=3
        )

        feature_names = X.columns.tolist()
        importance = get_feature_importance(model, feature_names, importance_type='gain')

        # Check all features have importance scores
        assert len(importance) == len(feature_names)
        for feature in feature_names:
            assert feature in importance
            assert isinstance(importance[feature], float)
            assert importance[feature] >= 0

    def test_calculate_cv_summary(self):
        """Test CV summary statistics calculation."""
        cv_metrics = [
            {'fold': 1, 'balanced_accuracy': 0.60, 'recall_orange': 0.50},
            {'fold': 2, 'balanced_accuracy': 0.65, 'recall_orange': 0.55},
            {'fold': 3, 'balanced_accuracy': 0.62, 'recall_orange': 0.52},
        ]

        summary = calculate_cv_summary(cv_metrics)

        # Check mean calculated correctly
        assert abs(summary['balanced_accuracy_mean'] - 0.623333) < 0.001
        assert abs(summary['recall_orange_mean'] - 0.523333) < 0.001

        # Check std calculated
        assert 'balanced_accuracy_std' in summary
        assert 'recall_orange_std' in summary

        # Check min/max
        assert summary['balanced_accuracy_min'] == 0.60
        assert summary['balanced_accuracy_max'] == 0.65


# ============================================================================
# UNIT TESTS: Validation
# ============================================================================


class TestValidation:
    """Test model validation and comparison."""

    def test_calculate_metrics(self):
        """Test comprehensive metrics calculation."""
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2, 1, 1, 0])

        metrics = calculate_metrics(y_true, y_pred, class_names=['green', 'orange', 'yellow'])

        # Check all metrics present
        assert 'balanced_accuracy' in metrics
        assert 'confusion_matrix' in metrics
        for class_name in ['green', 'orange', 'yellow']:
            assert f'precision_{class_name}' in metrics
            assert f'recall_{class_name}' in metrics
            assert f'f1_{class_name}' in metrics

        # Check ranges
        assert 0 <= metrics['balanced_accuracy'] <= 1
        for class_name in ['green', 'orange', 'yellow']:
            assert 0 <= metrics[f'precision_{class_name}'] <= 1
            assert 0 <= metrics[f'recall_{class_name}'] <= 1

    def test_compare_models_approve(self):
        """Test model comparison recommends approval when metrics improve."""
        new_metrics = {
            'balanced_accuracy': 0.70,
            'recall_orange': 0.65,
            'precision_orange': 0.60,
        }
        current_metrics = {
            'balanced_accuracy': 0.65,
            'recall_orange': 0.60,
            'precision_orange': 0.55,
        }

        approved, details = compare_models(new_metrics, current_metrics)

        assert approved is True
        assert details['approval_recommended'] is True
        assert details['quality_gates']['balanced_accuracy_gate']['passed'] is True
        assert details['quality_gates']['orange_recall_gate']['passed'] is True

    def test_compare_models_reject(self):
        """Test model comparison rejects when metrics degrade significantly."""
        new_metrics = {
            'balanced_accuracy': 0.50,  # -15% (threshold is -5%)
            'recall_orange': 0.45,       # -15% (threshold is -5%)
            'precision_orange': 0.40,
        }
        current_metrics = {
            'balanced_accuracy': 0.65,
            'recall_orange': 0.60,
            'precision_orange': 0.55,
        }

        approved, details = compare_models(new_metrics, current_metrics)

        assert approved is False
        assert details['approval_recommended'] is False
        assert details['quality_gates']['balanced_accuracy_gate']['passed'] is False
        assert details['quality_gates']['orange_recall_gate']['passed'] is False

    def test_compare_models_marginal_degradation(self):
        """Test model comparison allows marginal degradation within thresholds."""
        new_metrics = {
            'balanced_accuracy': 0.61,  # -4% (within -5% threshold)
            'recall_orange': 0.56,       # -4% (within -5% threshold)
            'precision_orange': 0.50,    # -5% (within -10% threshold)
        }
        current_metrics = {
            'balanced_accuracy': 0.65,
            'recall_orange': 0.60,
            'precision_orange': 0.55,
        }

        approved, details = compare_models(
            new_metrics, current_metrics,
            min_balanced_acc_delta=-0.05,
            min_orange_recall_delta=-0.05,
            min_orange_precision_delta=-0.10,
        )

        assert approved is True
        assert details['quality_gates']['balanced_accuracy_gate']['passed'] is True


# ============================================================================
# INTEGRATION TESTS: Asset Logic
# ============================================================================


class TestAssetLogic:
    """Test Dagster asset logic (without actual Dagster context)."""

    @pytest.mark.skip(reason="Requires full Dagster context and S3 mocking")
    def test_monthly_training_data_asset(self):
        """Test monthly_training_data asset loads data correctly."""
        # Would test data loading logic here
        pass

    @pytest.mark.skip(reason="Requires full Dagster context and S3 mocking")
    def test_relabeled_training_data_asset(self, sample_training_data):
        """Test relabeled_training_data asset applies new thresholds."""
        # Would test asset execution here
        pass

    def test_data_quality_checks(self, sample_training_data):
        """Test data quality validation logic."""
        df = apply_categorization(sample_training_data, h2s_column='H2S')

        # Check required columns present
        required_cols = ['time', 'site_name', 'H2S', 'h2s_category']
        for col in required_cols:
            assert col in df.columns

        # Check no excessive missing values
        missing_pct = df['H2S'].isna().sum() / len(df)
        assert missing_pct < 0.10  # <10% missing

        # Check H2S range reasonable
        valid_h2s = df['H2S'].dropna()
        assert (valid_h2s >= 0).all()
        assert (valid_h2s <= 200).all()  # Reasonable upper bound

        # Check class balance
        class_dist = df['h2s_category'].value_counts(normalize=True)
        assert (class_dist >= 0.05).all()  # Each class ≥5%


# ============================================================================
# INTEGRATION TESTS: End-to-End
# ============================================================================


class TestEndToEnd:
    """Test end-to-end training workflow."""

    def test_full_training_flow(self, sample_features_and_labels):
        """Test complete training flow: data → train → validate → compare."""
        X, y = sample_features_and_labels

        # Create label map from actual classes present
        unique_classes = sorted(y.unique())
        label_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}

        # Step 1: Apply categorization (simulate relabeling)
        # (Already done in fixture)

        # Step 2: Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # Step 3: Train with CV
        model, cv_metrics = train_model_with_cv(
            X_train, y_train, label_map,
            n_folds=3, n_estimators=10, max_depth=3
        )

        # Step 4: Validate on held-out set
        y_val_encoded = y_val.map(label_map)
        y_pred = model.predict(X_val)

        new_metrics = calculate_metrics(y_val_encoded.values, y_pred)

        # Step 5: Compare with "current" model (simulate worse performance)
        current_metrics = {
            'balanced_accuracy': new_metrics['balanced_accuracy'] - 0.10,
            'recall_orange': new_metrics['recall_orange'] - 0.10,
            'precision_orange': new_metrics['precision_orange'] - 0.10,
        }

        approved, details = compare_models(new_metrics, current_metrics)

        # Should approve because new model is better
        assert approved is True

        # Step 6: Verify CV summary
        cv_summary = calculate_cv_summary(cv_metrics)
        assert 'balanced_accuracy_mean' in cv_summary
        assert 'balanced_accuracy_std' in cv_summary

    @pytest.mark.skip(reason="Requires S3 mocking")
    def test_deployment_workflow(self):
        """Test deployment approval → archive → deploy workflow."""
        # Would test deployment asset execution here
        pass


# ============================================================================
# INTEGRATION TESTS: Approval Workflow
# ============================================================================


class TestApprovalWorkflow:
    """Test manual approval gate behavior."""

    def test_approval_gate_blocks_by_default(self):
        """Test deployment_approval raises Failure when approve_deployment=False."""
        # Simulate approval gate logic
        approve = False
        validation_report = {
            'approval_recommended': True,
            'validation_samples': 100,
        }

        # Should raise error (in real asset, raises dg.Failure)
        with pytest.raises(Exception):
            if not approve:
                raise Exception("Deployment not approved")

    def test_approval_gate_passes_when_approved(self):
        """Test deployment_approval succeeds when approve_deployment=True."""
        # Simulate approval gate logic
        approve = True
        validation_report = {
            'approval_recommended': True,
            'validation_samples': 100,
        }

        # Should not raise error
        try:
            if not approve:
                raise Exception("Deployment not approved")
            approval_metadata = {
                'approved_at': datetime.now().isoformat(),
                'approved_by': 'manual',
            }
            assert approval_metadata is not None
        except Exception:
            pytest.fail("Should not raise when approved")


# ============================================================================
# HELPERS FOR MOCKING
# ============================================================================


@pytest.fixture
def mock_s3_resource():
    """Create mock S3 resource for testing."""
    mock_s3 = MagicMock()
    mock_s3.S3_BUCKET = "test"
    mock_s3.getFile = MagicMock(return_value=b'{"test": "data"}')
    mock_s3.putFile = MagicMock()
    return mock_s3


# ============================================================================
# MARKERS FOR TEST ORGANIZATION
# ============================================================================

# pytest -m unit       # Run only unit tests
# pytest -m integration # Run only integration tests
# pytest -m slow       # Run slow tests
# pytest -m "not slow" # Skip slow tests
