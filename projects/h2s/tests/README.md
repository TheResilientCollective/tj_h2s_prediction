# H2S Pipeline Tests

This directory contains comprehensive tests for the H2S prediction Dagster pipeline.

## Test Files

### `test_h2s_pipeline.py`
Tests the Dagster asset logic without requiring S3 connection.

**What it tests:**
- Preprocessing feature creation (temporal, cyclical, interaction features)
- Categorical encoding
- Prediction output format and columns
- Probability validation (sum to 1, values in range 0-1)
- Alert filtering logic
- Metadata generation

**Run:**
```bash
pytest tests/test_h2s_pipeline.py -v
```

### `test_predictor.py`
Tests the H2SPredictor class functionality in isolation.

**What it tests:**
- Predictor initialization
- Data preprocessing logic
- Cyclical encoding correctness
- Interaction feature calculations
- Prediction generation
- Alert filtering
- Missing value handling

**Run:**
```bash
pytest tests/test_predictor.py -v
```

### `test_s3_integration.py`
Tests S3 integration and resource connections (requires S3 credentials).

**What it tests:**
- S3Resource connection
- File upload/download operations
- Model loading from S3
- Visualization upload
- Dataframe export to S3
- Metadata creation

**Run:**
```bash
# Requires .env with S3 credentials
pytest tests/test_s3_integration.py -v

# Skip if credentials not available
pytest tests/test_s3_integration.py -v --skip-s3
```

### `test_asset_materialization.py`
Tests actual Dagster asset materialization using mocked resources.

**What it tests:**
- Raw environmental data loading from S3 and local fallback
- Model artifact loading from S3
- Preprocessed features materialization with dependencies
- H2S predictions pipeline materialization
- H2S alerts filtering materialization
- End-to-end pipeline execution
- Failure scenarios and error handling

**Run:**
```bash
pytest tests/test_asset_materialization.py -v
```

## Running Tests

### All Tests
```bash
cd projects/h2s
uv sync  # Install dev dependencies including pytest
uv run pytest
```

### Specific Test File
```bash
uv run pytest tests/test_h2s_pipeline.py -v
```

### Specific Test Class
```bash
uv run pytest tests/test_h2s_pipeline.py::TestPreprocessedFeatures -v
```

### Specific Test Function
```bash
uv run pytest tests/test_h2s_pipeline.py::TestPreprocessedFeatures::test_creates_temporal_features -v
```

### Skip S3 Tests
```bash
uv run pytest -m "not s3"
```

### Run Only S3 Tests
```bash
uv run pytest -m s3
```

### With Coverage Report
```bash
uv run pytest --cov=h2s --cov-report=html
# Open htmlcov/index.html to view coverage
```

### Verbose Output
```bash
uv run pytest -vv
```

### Stop on First Failure
```bash
uv run pytest -x
```

### Run in Parallel (faster)
```bash
uv pip install pytest-xdist
uv run pytest -n auto
```

## Test Markers

Tests are marked with custom markers for selective execution:

- `@pytest.mark.s3` - Requires S3 connection
- `@pytest.mark.slow` - Takes more than 1 second
- `@pytest.mark.integration` - Tests multiple components together

**Usage:**
```bash
# Run only fast tests
uv run pytest -m "not slow"

# Run only integration tests
uv run pytest -m integration

# Skip S3 tests
uv run pytest -m "not s3"
```

## Fixtures

Shared test fixtures are defined in `conftest.py`:

- `s3_credentials_available` - Check if S3 credentials exist
- `sample_env_data` - Sample environmental data for testing
- `s3_resource` - Configured S3Resource instance
- `mock_environmental_data` - Mock data for pipeline tests
- `mock_predictor` - Mock H2SPredictor for unit tests

## Writing New Tests

### Unit Test Example
```python
def test_feature_creation():
    """Test that specific feature is created correctly."""
    # Arrange
    input_data = pd.DataFrame({...})

    # Act
    result = preprocess(input_data)

    # Assert
    assert 'new_feature' in result.columns
    assert result['new_feature'].min() >= 0
```

### Integration Test Example
```python
@pytest.mark.integration
def test_end_to_end_pipeline(s3_resource):
    """Test complete pipeline from data to export."""
    # Load model
    predictor = H2SPredictor.from_s3(...)

    # Process data
    predictions = predictor.predict(...)

    # Export results
    export_to_s3(predictions, s3_resource)

    # Verify
    assert predictions_exist_on_s3()
```

### S3 Test Example
```python
@pytest.mark.s3
def test_s3_upload(s3_resource):
    """Test uploading file to S3."""
    s3_resource.putFile_text(
        data="test",
        path="test/file.txt"
    )

    downloaded = s3_resource.getFile(path="test/file.txt")
    assert downloaded.decode() == "test"
```

## Coverage Goals

Target coverage: **80%+**

Check coverage:
```bash
uv run pytest --cov=h2s --cov-report=term-missing
```

Areas requiring coverage:
- ✅ Preprocessing logic
- ✅ Prediction generation
- ✅ Alert filtering
- ✅ S3 operations
- ⚠️ Error handling (add more tests)
- ⚠️ Edge cases (add more tests)

## CI/CD Integration

For continuous integration, add to your CI pipeline:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    cd projects/h2s
    uv sync
    uv run pytest --cov=h2s --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'h2s'"**
- Run: `uv sync` from `projects/h2s/`
- Check: `sys.path` includes `src/`

**"S3 tests skipped"**
- Ensure `.env` file exists with S3 credentials
- Or skip with: `pytest -m "not s3"`

**"ImportError: cannot import name"**
- Ensure all dependencies installed: `uv sync`
- Check Python version: `python --version` (should be 3.10-3.14)

**Tests hang**
- Use timeout: `pytest --timeout=30`
- Check for infinite loops in code

**Coverage not working**
- Install: `uv pip install pytest-cov`
- Check config in `pytest.ini`
