# H2S Prediction System - Deployment Guide

## Overview

This package provides a complete production-ready system for generating H2S predictions at NESTOR - BES using the trained XGBoost model.

**Performance:**
- Orange Detection: 61.3% (84/137 events caught)
- Yellow Detection: 46.1% (113/245 events caught)
- Balanced Accuracy: 63.1%
- False Alarm Rate: 5.4%

---

## Quick Start

### 1. Installation

```bash
# Install required packages
pip install pandas numpy xgboost scikit-learn

# Verify installation
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
```

### 2. Required Files

Place these files in your working directory:

```
your_project/
├── predict_h2s.py                          # Main prediction script
├── batch_predict.py                        # Batch processing script
├── nestor_xgboost_weighted_model.json      # Trained model
├── nestor_preprocessing_info.pkl           # Preprocessing info
└── your_data.csv                           # Your input data
```

### 3. Generate Predictions

```bash
# Basic usage
python predict_h2s.py --input your_data.csv --output predictions.csv

# View results
head predictions.csv
```

**Output columns added:**
- `predicted_category`: green, yellow, or orange
- `probability_green`: Confidence in green prediction (0-1)
- `probability_orange`: Confidence in orange prediction (0-1)
- `probability_yellow`: Confidence in yellow prediction (0-1)
- `confidence`: Highest probability (model confidence)
- `alert`: True if yellow or orange

---

## Detailed Usage

### Single File Prediction

```bash
# Standard prediction
python predict_h2s.py --input new_data.csv --output predictions.csv

# More sensitive detection (catch more events, more false alarms)
python predict_h2s.py --input new_data.csv --output predictions.csv --orange-threshold 0.25

# Only show alerts (filter out green)
python predict_h2s.py --input new_data.csv --output alerts.csv --filter-alerts

# Combine options
python predict_h2s.py \
    --input new_data.csv \
    --output alerts.csv \
    --orange-threshold 0.25 \
    --yellow-threshold 0.30 \
    --filter-alerts
```

### Batch Processing (Multiple Files)

```bash
# Process all CSV files in a directory
python batch_predict.py --input-dir ./new_data --output-dir ./predictions

# With archiving (moves processed files)
python batch_predict.py \
    --input-dir ./new_data \
    --output-dir ./predictions \
    --archive \
    --archive-dir ./processed

# With custom thresholds
python batch_predict.py \
    --input-dir ./new_data \
    --output-dir ./predictions \
    --orange-threshold 0.25
```

---

## Input Data Requirements

### Required Columns

Your CSV file must contain these columns:

**Weather Data:**
- `temperature_2m`: Temperature at 2m (°C)
- `wind_speed_10m`: Wind speed at 10m (m/s)
- `wind_direction_10m`: Wind direction (degrees)
- `wind_gusts_10m`: Wind gusts (m/s)
- `precipitation`: Precipitation (mm)
- `relative_humidity_2m`: Relative humidity (%)
- `surface_pressure`: Surface pressure (hPa)
- `cloud_cover`: Cloud cover (%)
- `dewpoint_2m`: Dewpoint temperature (°C)

**Tidal/Flow Data:**
- `Flow (m^3/s)--Border`: Water flow rate (m³/s)
- `tide_height`: Tide height (m)
- `tidal_state`: Tidal state (flood, ebb, slack, slack low, slack high)

**Categorical:**
- `wind_direction_categorical`: N, NE, E, SE, S, SW, W, NW

**Temporal:**
- `time`: Timestamp (ISO format: 2024-01-15T12:00:00Z)

**Optional:**
- `site_name`: Site name (will filter to NESTOR - BES if present)
- `H2S`: Actual H2S value (for validation, not used in prediction)

### Example Input Format

```csv
time,site_name,temperature_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,relative_humidity_2m,surface_pressure,cloud_cover,dewpoint_2m,wind_direction_categorical,Flow (m^3/s)--Border,tide_height,tidal_state
2024-01-15T12:00:00Z,NESTOR - BES,15.2,3.5,180,5.2,0.0,65,1013.2,25,8.1,S,125.5,1.2,flood
2024-01-15T13:00:00Z,NESTOR - BES,16.1,4.2,190,6.1,0.0,62,1013.0,30,8.5,S,130.2,1.3,flood
```

---

## Output Format

### Prediction File

Output CSV contains all input columns plus:

```csv
time,site_name,...,predicted_category,probability_green,probability_orange,probability_yellow,confidence,alert
2024-01-15T12:00:00Z,NESTOR - BES,...,green,0.85,0.08,0.07,0.85,False
2024-01-15T13:00:00Z,NESTOR - BES,...,orange,0.15,0.62,0.23,0.62,True
2024-01-15T14:00:00Z,NESTOR - BES,...,yellow,0.25,0.18,0.57,0.57,True
```

### Understanding Predictions

**predicted_category:**
- `green`: H2S predicted < 5 ppb (safe)
- `yellow`: H2S predicted 5-30 ppb (caution, monitor)
- `orange`: H2S predicted ≥ 30 ppb (alert, take action)

**Probabilities:**
- Range: 0.0 to 1.0
- Sum to 1.0 across three categories
- Higher = more confident

**confidence:**
- Maximum probability across all categories
- > 0.7: High confidence
- 0.5-0.7: Moderate confidence
- < 0.5: Low confidence (uncertain)

**alert:**
- `True`: Yellow or orange prediction
- `False`: Green prediction

---

## Automated Updates

### Daily Cron Job

Add to crontab for daily automatic predictions:

```bash
# Edit crontab
crontab -e

# Add this line for daily processing at 2 AM
0 2 * * * /usr/bin/python3 /path/to/batch_predict.py --input-dir /data/incoming --output-dir /data/predictions --archive --archive-dir /data/processed >> /var/log/h2s_predictions.log 2>&1
```

### Hourly Updates

For real-time monitoring:

```bash
# Every hour
0 * * * * /usr/bin/python3 /path/to/predict_h2s.py --input /data/latest.csv --output /data/current_predictions.csv --filter-alerts
```

### Watch Directory Script

Create `watch_and_predict.sh`:

```bash
#!/bin/bash
# Watch directory for new files and process automatically

WATCH_DIR="/data/incoming"
OUTPUT_DIR="/data/predictions"
SCRIPT_DIR="/path/to/scripts"

inotifywait -m -e create -e moved_to --format '%f' $WATCH_DIR |
while read FILE; do
    if [[ $FILE == *.csv ]]; then
        echo "Processing $FILE..."
        python3 $SCRIPT_DIR/predict_h2s.py \
            --input "$WATCH_DIR/$FILE" \
            --output "$OUTPUT_DIR/${FILE%.csv}_predictions.csv"
        
        # Move to archive
        mv "$WATCH_DIR/$FILE" "$WATCH_DIR/processed/"
        echo "Done: $FILE"
    fi
done
```

---

## Threshold Tuning

### Default Thresholds

- Orange: 0.33 (probability ≥ 0.33 predicts orange)
- Yellow: 0.33 (probability ≥ 0.33 predicts yellow)

### Adjusting Sensitivity

**More Sensitive (catch more events, more false alarms):**

```bash
# Lower thresholds = more alerts
python predict_h2s.py \
    --input data.csv \
    --output predictions.csv \
    --orange-threshold 0.25 \
    --yellow-threshold 0.30
```

**Expected impact:**
- Orange recall: 61.3% → ~70%
- False positive rate: 5.4% → ~10%

**Less Sensitive (fewer false alarms, miss more events):**

```bash
# Higher thresholds = fewer alerts
python predict_h2s.py \
    --input data.csv \
    --output predictions.csv \
    --orange-threshold 0.45 \
    --yellow-threshold 0.40
```

**Expected impact:**
- Orange recall: 61.3% → ~50%
- False positive rate: 5.4% → ~3%

### Threshold Selection Guide

| Priority | Orange Threshold | Yellow Threshold | Orange Recall | False Positives |
|----------|------------------|------------------|---------------|-----------------|
| **Balanced** (default) | 0.33 | 0.33 | 61% | 5.4% |
| **Sensitive** | 0.25 | 0.30 | 70% | 10% |
| **Very Sensitive** | 0.20 | 0.25 | 75% | 15% |
| **Conservative** | 0.40 | 0.40 | 55% | 3% |
| **Very Conservative** | 0.50 | 0.50 | 45% | 1.5% |

**Recommendation:** Start with default (0.33), monitor for 2-4 weeks, then adjust based on feedback.

---

## Integration Examples

### Python Integration

```python
from src.predict_h2s import H2SPredictor
import pandas as pd

# Load model
predictor = H2SPredictor(
    '../nestor_xgboost_weighted_model.json',
    'nestor_preprocessing_info.pkl'
)

# Load your data
df = pd.read_csv('new_data.csv')

# Preprocess
df_processed = predictor.preprocess_data(df)

# Generate predictions
predictions = predictor.predict(df_processed)

# Get only alerts
alerts = predictions[predictions['alert'] == True]

# Send email if orange detected
if (alerts['predicted_category'] == 'orange').any():
    send_email_alert(alerts)
```

### R Integration

```r
# Call Python script from R
system("python3 predict_h2s.py --input data.csv --output predictions.csv")

# Load results
predictions <- read.csv("predictions.csv")

# Filter alerts
alerts <- predictions[predictions$alert == TRUE, ]

# Plot
library(ggplot2)
ggplot(predictions, aes(x=time, y=probability_orange)) +
  geom_line() +
  theme_minimal()
```

### API Endpoint (Flask)

```python
from flask import Flask, request, jsonify
from src.predict_h2s import H2SPredictor
import pandas as pd

app = Flask(__name__)
predictor = H2SPredictor('nestor_xgboost_weighted_model.json',
                         'nestor_preprocessing_info.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_processed = predictor.preprocess_data(df)
    result = predictor.predict(df_processed)

    return jsonify({
        'predicted_category': result['predicted_category'].iloc[0],
        'probability_orange': float(result['probability_orange'].iloc[0]),
        'probability_yellow': float(result['probability_yellow'].iloc[0]),
        'confidence': float(result['confidence'].iloc[0]),
        'alert': bool(result['alert'].iloc[0])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Monitoring & Maintenance

### Performance Monitoring

Track these metrics weekly:

```python
import pandas as pd

# Load predictions and actual values
predictions = pd.read_csv('predictions.csv')
actual = pd.read_csv('actual_h2s.csv')

# Merge on time
df = predictions.merge(actual, on='time')

# Calculate actual categories
df['actual_category'] = df['H2S'].apply(
    lambda x: 'green' if x < 5 else ('yellow' if x < 30 else 'orange')
)

# Metrics
from sklearn.metrics import classification_report
print(classification_report(df['actual_category'], df['predicted_category']))
```

**Alert if:**
- Orange recall drops below 55%
- False positive rate exceeds 10%
- Balanced accuracy drops below 60%

### Retraining Schedule

**When to retrain:**
- Quarterly (every 3 months) - routine update
- When orange recall drops below 55%
- After sensor calibration or maintenance
- When new data patterns emerge

**How to retrain:**
1. Collect new data (append to original dataset)
2. Run training script (contact data science team)
3. Validate on held-out test set
4. Deploy if performance improves
5. Archive old model

### Log Monitoring

Check logs regularly:

```bash
# View recent predictions
tail -f /var/log/h2s_predictions.log

# Count alerts per day
grep "alert.*True" predictions_*.csv | wc -l

# Check for errors
grep "Error" /var/log/h2s_predictions.log
```

---

## Troubleshooting

### Common Issues

**1. "FileNotFoundError: model file not found"**
```bash
# Verify files exist
ls -lh nestor_xgboost_weighted_model.json
ls -lh nestor_preprocessing_info.pkl

# Check current directory
pwd

# Provide full path
python predict_h2s.py \
    --input data.csv \
    --output predictions.csv \
    --model /full/path/to/nestor_xgboost_weighted_model.json \
    --preprocessing /full/path/to/nestor_preprocessing_info.pkl
```

**2. "KeyError: 'temperature_2m' not found"**
```bash
# Check column names in your data
head -1 your_data.csv

# Compare with required columns (see Input Data Requirements section)
# Fix column names to match exactly
```

**3. "ValueError: unknown categorical value"**
```
# This occurs when categorical variables have new values not seen in training

# For wind_direction_categorical, use only: N, NE, E, SE, S, SW, W, NW
# For tidal_state, use only: flood, ebb, slack, slack low, slack high

# Check your data:
cat your_data.csv | cut -d',' -f13 | sort | uniq  # wind_direction_categorical
cat your_data.csv | cut -d',' -f24 | sort | uniq  # tidal_state
```

**4. "Model predictions all green"**
```python
# This can happen if thresholds are too high
# Lower thresholds:
python predict_h2s.py \
    --input data.csv \
    --output predictions.csv \
    --orange-threshold 0.20 \
    --yellow-threshold 0.25

# Or check if input data is realistic
# Flow rate, wind speed, H2S patterns should match training range
```

**5. "Too many false alarms"**
```python
# Raise thresholds:
python predict_h2s.py \
    --input data.csv \
    --output predictions.csv \
    --orange-threshold 0.40 \
    --yellow-threshold 0.40

# Or model may need retraining if patterns changed
```

---

## Performance Expectations

### What the Model Can Do

✅ **Strengths:**
- Detect 61.3% of critical orange events (H2S ≥ 30 ppb)
- Provide 1-3 hour advance warning
- Low false alarm rate (5.4%)
- Process predictions in seconds
- Works 24/7 without manual intervention

### What the Model Cannot Do

❌ **Limitations:**
- Still misses ~39% of orange events
- Cannot detect sudden spikes (< 1 hour)
- Requires sensor data to be current and accurate
- Not 100% accurate (no ML model is)
- Should supplement, not replace, direct H2S monitoring

### Use Cases

**Good for:**
- Early warning system
- Planning maintenance during predicted high H2S
- Trend analysis and reporting
- Reducing operator workload
- Prioritizing direct monitoring efforts

**Not suitable for:**
- Emergency response (too slow, not 100% reliable)
- Replacing H2S sensors
- Regulatory compliance as sole measure
- Life-safety critical decisions without verification

---

## Support

### Getting Help

**For technical issues:**
1. Check this documentation
2. Review error messages carefully
3. Verify input data format
4. Check log files

**For model questions:**
- Review `Complete_Model_Testing_Summary.md`
- See `NESTOR_BES_H2S_Forecasting_Report.md`

**For updates:**
- Model retraining: Contact data science team
- Feature requests: Submit via project management system

---

## Files Included

```
h2s_prediction_system/
├── predict_h2s.py                          # Main prediction script
├── batch_predict.py                        # Batch processing
├── nestor_xgboost_weighted_model.json      # Trained XGBoost model
├── nestor_preprocessing_info.pkl           # Feature encoders & config
├── DEPLOYMENT_GUIDE.md                     # This file
├── NESTOR_BES_Quick_Start.md               # Quick reference
├── NESTOR_BES_H2S_Forecasting_Report.md    # Full technical report
└── Complete_Model_Testing_Summary.md       # All algorithms tested
```

---

## Version History

**v1.0** (December 2025)
- Initial production release
- XGBoost model with 61.3% orange recall
- Trained on 9,631 NESTOR - BES samples
- 20 engineered features
- Balanced class weighting

---

## License & Citation

**Model:** Proprietary
**Data:** NESTOR - BES site, November 2023 - January 2025
**Contact:** [Your organization]

When referencing this model, cite:
```
H2S Forecasting Model for NESTOR - BES, v1.0
Trained: December 2025
Performance: 61.3% orange recall, 63.1% balanced accuracy
Algorithm: XGBoost with class weighting
```

---

*Last updated: December 2025*
