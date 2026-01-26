#!/usr/bin/env python3
"""
Batch H2S Prediction Update Script
====================================

Automatically processes new data files and generates predictions.
Useful for regular automated updates when new data becomes available.

Usage:
    python batch_predict.py --input-dir ./new_data --output-dir ./predictions
    python batch_predict.py --input-dir ./new_data --output-dir ./predictions --archive
"""

import os
import glob
import pandas as pd
import argparse
from datetime import datetime
import shutil
from src.predict_h2s import H2SPredictor


def process_batch(input_dir, output_dir, archive_dir=None, model_path='nestor_xgboost_weighted_model.json',
                 preprocessing_path='nestor_preprocessing_info.pkl', **kwargs):
    """
    Process all CSV files in input directory.

    Args:
        input_dir: Directory containing input CSV files
        output_dir: Directory for output prediction files
        archive_dir: Optional directory to move processed files
        model_path: Path to model file
        preprocessing_path: Path to preprocessing file
        **kwargs: Additional arguments passed to predictor
    """

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)

    # Find all CSV files
    input_files = glob.glob(os.path.join(input_dir, '*.csv'))

    if not input_files:
        print(f"No CSV files found in {input_dir}")
        return

    print("="*80)
    print("BATCH H2S PREDICTION")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(input_files)}")
    print("="*80)
    print()

    # Load model once
    print("Loading model...")
    predictor = H2SPredictor(model_path, preprocessing_path)
    print()

    # Process each file
    results_summary = []

    for i, input_file in enumerate(input_files, 1):
        filename = os.path.basename(input_file)
        print(f"[{i}/{len(input_files)}] Processing: {filename}")

        try:
            # Load data
            df = pd.read_csv(input_file)
            original_count = len(df)

            # Filter to NESTOR - BES if needed
            if 'site_name' in df.columns:
                if 'NESTOR - BES' in df['site_name'].values:
                    df = df[df['site_name'] == 'NESTOR - BES'].copy()

            # Preprocess
            df_processed = predictor.preprocess_data(df)

            # Predict
            predictions = predictor.predict(
                df_processed,
                orange_threshold=kwargs.get('orange_threshold'),
                yellow_threshold=kwargs.get('yellow_threshold')
            )

            # Generate output filename
            output_filename = filename.replace('.csv', '_predictions.csv')
            output_path = os.path.join(output_dir, output_filename)

            # Save
            predictions.to_csv(output_path, index=False)

            # Summary
            orange_count = (predictions['predicted_category'] == 'orange').sum()
            yellow_count = (predictions['predicted_category'] == 'yellow').sum()
            green_count = (predictions['predicted_category'] == 'green').sum()

            results_summary.append({
                'input_file': filename,
                'output_file': output_filename,
                'total_records': len(predictions),
                'orange': orange_count,
                'yellow': yellow_count,
                'green': green_count,
                'status': 'SUCCESS'
            })

            print(f"  ✓ Saved to: {output_filename}")
            print(f"    Records: {len(predictions)}, Orange: {orange_count}, Yellow: {yellow_count}, Green: {green_count}")

            # Archive original file if requested
            if archive_dir:
                archive_path = os.path.join(archive_dir, filename)
                shutil.move(input_file, archive_path)
                print(f"    Archived to: {archive_dir}")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results_summary.append({
                'input_file': filename,
                'output_file': None,
                'total_records': 0,
                'orange': 0,
                'yellow': 0,
                'green': 0,
                'status': f'ERROR: {str(e)}'
            })

        print()

    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(output_dir, f'batch_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    summary_df.to_csv(summary_path, index=False)

    # Print final summary
    print("="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Total files processed: {len(input_files)}")
    print(f"Successful: {(summary_df['status'] == 'SUCCESS').sum()}")
    print(f"Failed: {(summary_df['status'] != 'SUCCESS').sum()}")
    print(f"\nSummary saved to: {summary_path}")
    print("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Batch process H2S predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input-dir', '-i', required=True,
                        help='Directory containing input CSV files')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Directory for output prediction files')
    parser.add_argument('--archive', action='store_true',
                        help='Move processed files to archive directory')
    parser.add_argument('--archive-dir', default='./archive',
                        help='Archive directory (default: ./archive)')
    parser.add_argument('--model', default='nestor_xgboost_weighted_model.json',
                        help='Path to model file')
    parser.add_argument('--preprocessing', default='nestor_preprocessing_info.pkl',
                        help='Path to preprocessing file')
    parser.add_argument('--orange-threshold', type=float,
                        help='Custom orange threshold')
    parser.add_argument('--yellow-threshold', type=float,
                        help='Custom yellow threshold')

    args = parser.parse_args()

    # Process batch
    process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        archive_dir=args.archive_dir if args.archive else None,
        model_path=args.model,
        preprocessing_path=args.preprocessing,
        orange_threshold=args.orange_threshold,
        yellow_threshold=args.yellow_threshold
    )


if __name__ == '__main__':
    main()
