"""Pytest configuration and shared fixtures for H2S tests.

This file is automatically loaded by pytest and provides:
- Shared fixtures across all test files
- Test configuration
- Pytest markers
"""

import os
import sys
import pytest

# Add src directory to path for imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "s3: marks tests as requiring S3 connection (deselect with '-m \"not s3\"')"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )


@pytest.fixture(scope="session")
def s3_credentials_available():
    """Check if S3 credentials are available in environment."""
    required_vars = ['S3_BUCKET', 'S3_ADDRESS', 'S3_ACCESS_KEY', 'S3_SECRET_KEY']
    return all(os.getenv(var) for var in required_vars)


@pytest.fixture
def sample_env_data():
    """Provide sample environmental data for testing."""
    import pandas as pd
    import numpy as np

    return pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=10, freq='h'),
        'temperature_2m': np.random.uniform(15, 25, 10),
        'wind_speed_10m': np.random.uniform(0, 10, 10),
        'wind_direction_10m': np.random.uniform(0, 360, 10),
        'relative_humidity_2m': np.random.uniform(60, 90, 10),
        'surface_pressure': np.random.uniform(1010, 1020, 10),
        'precipitation': np.random.uniform(0, 5, 10),
        'cloud_cover': np.random.uniform(0, 100, 10),
        'wind_direction_categorical': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE'],
        'flow_rate_cms': np.random.uniform(1, 10, 10),
        'tide_height_m': np.random.uniform(0, 2, 10),
        'tidal_state': ['rising', 'falling'] * 5,
    })
