#!/usr/bin/env python3
"""Convert preprocessing info from pickle to JSON format."""

import pickle
import json
import sys
import numpy as np

# Load pickle file
import os
base_dir = '/Users/valentin/development/dev_resilient/tj_h2s_prediction'
pkl_path = os.path.join(base_dir, 'nestor_preprocessing_info.pkl')
json_path = os.path.join(base_dir, 'nestor_preprocessing_info.json')

try:
    with open(pkl_path, 'rb') as f:
        prep_info = pickle.load(f)

    print(f"✓ Loaded preprocessing info from {pkl_path}")
    print(f"  Keys: {list(prep_info.keys())}")

    # Convert sklearn LabelEncoders to simple dict mappings
    json_prep = {
        "feature_cols": prep_info['feature_cols'],
        "class_names": prep_info['class_names'],
        "site_name": prep_info.get('site_name', 'NESTOR - BES'),
    }

    # Convert class_weights if it exists (numpy array -> dict)
    if 'class_weights' in prep_info and prep_info['class_weights'] is not None:
        class_weights = prep_info['class_weights']
        if isinstance(class_weights, np.ndarray):
            json_prep["class_weights"] = {prep_info['class_names'][i]: float(class_weights[i]) for i in range(len(class_weights))}
        else:
            json_prep["class_weights"] = class_weights

    # Add label encoder mappings if they exist
    if prep_info.get('le_wind_cat'):
        json_prep["wind_cat_mapping"] = {str(cat): int(idx) for idx, cat in enumerate(prep_info['le_wind_cat'].classes_)}
        print(f"  Wind categories: {json_prep['wind_cat_mapping']}")

    if prep_info.get('le_tidal'):
        json_prep["tidal_mapping"] = {str(state): int(idx) for idx, state in enumerate(prep_info['le_tidal'].classes_)}
        print(f"  Tidal states: {json_prep['tidal_mapping']}")

    # Write to JSON with custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, 'w') as f:
        json.dump(json_prep, f, indent=2, cls=NumpyEncoder)

    print(f"\n✓ Converted to {json_path}")
    print(f"  Features: {len(json_prep['feature_cols'])}")
    print(f"  Classes: {json_prep['class_names']}")

except FileNotFoundError:
    print(f"Error: {pkl_path} not found")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
