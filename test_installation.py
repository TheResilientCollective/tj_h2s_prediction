#!/usr/bin/env python3
"""
Test H2S Prediction System Installation
=========================================

Verifies that all components are working correctly.

Usage:
    python test_installation.py
"""

import sys
import os

def test_imports():
    """Test that all required packages are installed."""
    print("Testing package imports...")

    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'xgboost': 'xgboost',
        'sklearn': 'scikit-learn',
        'pickle': 'pickle (built-in)'
    }

    failed = []

    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            failed.append(package)

    if failed:
        print(f"\n✗ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("✓ All packages installed")
    return True


def test_files():
    """Test that required files exist."""
    print("\nTesting required files...")

    required_files = {
        'nestor_xgboost_weighted_model.json': 'XGBoost model',
        'nestor_preprocessing_info.json': 'Preprocessing info',
        'src/predict_h2s.py': 'Main prediction script',
        'src/batch_predict.py': 'Batch processing script'
    }

    failed = []

    for filename, description in required_files.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✓ {description} ({size:,} bytes)")
        else:
            print(f"  ✗ {description} - FILE NOT FOUND")
            failed.append(filename)

    if failed:
        print(f"\n✗ Missing files: {', '.join(failed)}")
        return False

    print("✓ All files present")
    return True


def test_prediction():
    """Test running a prediction."""
    print("\nTesting prediction system...")

    try:
        from src.predict_h2s import H2SPredictor
        import pandas as pd

        # Load model
        print("  Loading model...", end=" ")
        predictor = H2SPredictor(
            'nestor_xgboost_weighted_model.json',
            'nestor_preprocessing_info.json'
        )
        print("✓")

        # Check if example file exists
        if os.path.exists('example_input.csv'):
            print("  Loading example data...", end=" ")
            df = pd.read_csv('example_input.csv')
            print(f"✓ ({len(df)} records)")

            print("  Preprocessing...", end=" ")
            df_processed = predictor.preprocess_data(df)
            print("✓")

            print("  Generating predictions...", end=" ")
            predictions = predictor.predict(df_processed)
            print("✓")

            print("\n  Example predictions:")
            for i, row in predictions.head(3).iterrows():
                print(f"    {row['time']}: {row['predicted_category']} (confidence: {row['confidence']:.2f})")

            print("\n✓ Prediction test successful")
            return True
        else:
            print("  ⚠ example_input.csv not found - skipping prediction test")
            print("  (Model loads correctly)")
            return True

    except Exception as e:
        print(f"\n✗ Prediction test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*80)
    print("H2S PREDICTION SYSTEM - INSTALLATION TEST")
    print("="*80)
    print()

    tests = [
        ("Package imports", test_imports),
        ("Required files", test_files),
        ("Prediction system", test_prediction)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append(False)
        print()

    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)

    for (name, _), passed in zip(tests, results):
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print()

    if all(results):
        print("🎉 ALL TESTS PASSED - System ready to use!")
        print("\nNext steps:")
        print("  1. Review DEPLOYMENT_GUIDE.md for usage")
        print("  2. Test with your data:")
        print("     python predict_h2s.py --input your_data.csv --output predictions.csv")
        return 0
    else:
        print("⚠ SOME TESTS FAILED - Fix issues before deploying")
        print("\nCommon fixes:")
        print("  - Missing packages: pip install -r requirements.txt")
        print("  - Missing files: Ensure all files from the package are present")
        return 1


if __name__ == '__main__':
    sys.exit(main())
