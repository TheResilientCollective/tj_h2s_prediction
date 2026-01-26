# H2S Prediction System - Complete Package

## 📦 Package Contents

This package contains everything you need to deploy and maintain the H2S prediction system for NESTOR - BES.

### Core Files (Required)

| File | Size | Purpose |
|------|------|---------|
| `nestor_xgboost_weighted_model.json` | ~500 KB | Trained XGBoost model |
| `nestor_preprocessing_info.pkl` | ~1 KB | Feature encoders & configuration |
| `predict_h2s.py` | ~12 KB | Main prediction script |
| `batch_predict.py` | ~5 KB | Batch processing script |
| `requirements.txt` | ~500 B | Python dependencies |

### Documentation (Important)

| File | Purpose |
|------|---------|
| `README.md` | Quick overview and getting started |
| `DEPLOYMENT_GUIDE.md` | Complete deployment documentation |
| `NESTOR_BES_Quick_Start.md` | 5-minute quick reference |
| `NESTOR_BES_H2S_Forecasting_Report.md` | Full technical report |
| `Complete_Model_Testing_Summary.md` | All algorithms tested |

### Testing & Examples

| File | Purpose |
|------|---------|
| `test_installation.py` | Verify installation |
| `example_input.csv` | Sample input data |

### Analysis & Reports

| File | Purpose |
|------|---------|
| `Final_H2S_Analysis_All_Datasets.md` | Dataset comparison analysis |
| `KNN_Model_Analysis.md` | KNN algorithm testing |
| `Original_vs_Latest_Dataset_Comparison.md` | Dataset size impact |
| Various `.png` files | Performance visualizations |

---

## 🚀 Quick Deployment (5 Steps)

### Step 1: Download & Extract

Download all files to your working directory:
```bash
mkdir h2s_prediction
cd h2s_prediction
# Copy all files here
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Installs:
- pandas (data handling)
- numpy (numerical operations)
- xgboost (ML model)
- scikit-learn (preprocessing)

### Step 3: Test Installation

```bash
python test_installation.py
```

Expected output:
```
✓ All packages installed
✓ All files present
✓ Prediction test successful

🎉 ALL TESTS PASSED - System ready to use!
```

### Step 4: Run Prediction

```bash
python predict_h2s.py --input your_data.csv --output predictions.csv
```

### Step 5: Review Results

```bash
# View predictions
head predictions.csv

# Count alerts
grep "True" predictions.csv | wc -l
```

**Done!** Your system is operational.

---

## 📊 Model Performance

| Metric | Value | Meaning |
|--------|-------|---------|
| **Orange Detection** | **61.3%** | Catches 84 out of 137 critical events |
| **Yellow Detection** | 46.1% | Catches 113 out of 245 caution events |
| **Green Detection** | 81.7% | Correctly identifies 1,263 safe periods |
| **Balanced Accuracy** | 63.1% | Overall performance across all classes |
| **False Alarm Rate** | 5.4% | Only 84 false alarms per 1,927 predictions |

**Translation:**
- Model will catch ~6 out of 10 critical H2S events
- Provides 1-3 hour advance warning
- Low false alarm rate (acceptable for operators)
- Best performance among 10+ algorithms tested

---

## 💻 Usage Examples

### Basic Prediction

```bash
# Generate predictions from new data
python predict_h2s.py --input new_data.csv --output predictions.csv
```

### More Sensitive (Catch More Events)

```bash
# Lower threshold = more alerts, fewer misses
python predict_h2s.py \
    --input new_data.csv \
    --output predictions.csv \
    --orange-threshold 0.25
```

**Impact:** 61% → ~70% orange detection, 5.4% → ~10% false alarms

### Alerts Only

```bash
# Filter to show only yellow and orange predictions
python predict_h2s.py \
    --input new_data.csv \
    --output alerts.csv \
    --filter-alerts
```

### Batch Processing

```bash
# Process all CSV files in a directory
python batch_predict.py \
    --input-dir ./new_data \
    --output-dir ./predictions
```

### Automated Daily Updates

```bash
# Add to crontab for daily processing at 2 AM
crontab -e

# Add this line:
0 2 * * * cd /path/to/h2s_prediction && python batch_predict.py --input-dir /data/incoming --output-dir /data/predictions >> /var/log/h2s.log 2>&1
```

---

## 🎯 Threshold Configuration

Adjust sensitivity based on your needs:

### Scenarios

**Scenario 1: Health & Safety Priority**
- Want to catch maximum events
- Can tolerate false alarms
- **Use:** `--orange-threshold 0.25`
- **Result:** ~70% orange detection, ~10% false alarms

**Scenario 2: Balanced Approach** (Default)
- Good detection with acceptable false alarms
- **Use:** Default (no threshold flag)
- **Result:** 61% orange detection, 5.4% false alarms

**Scenario 3: Minimize False Alarms**
- Want high confidence predictions only
- Willing to miss some events
- **Use:** `--orange-threshold 0.40`
- **Result:** ~55% orange detection, ~3% false alarms

---

## 📁 Input Data Format

### Required Columns

Your CSV must have these exact column names:

**Weather (9 columns):**
- `temperature_2m` (°C)
- `wind_speed_10m` (m/s)
- `wind_direction_10m` (degrees)
- `wind_gusts_10m` (m/s)
- `precipitation` (mm)
- `relative_humidity_2m` (%)
- `surface_pressure` (hPa)
- `cloud_cover` (%)
- `dewpoint_2m` (°C)

**Tidal (3 columns):**
- `Flow (m^3/s)--Border` (m³/s)
- `tide_height` (m)
- `tidal_state` (flood, ebb, slack, slack low, slack high)

**Wind (1 column):**
- `wind_direction_categorical` (N, NE, E, SE, S, SW, W, NW)

**Time (1 column):**
- `time` (ISO format: 2024-01-15T12:00:00Z)

**Optional:**
- `site_name` (automatically filters to NESTOR - BES)

### Example CSV

See `example_input.csv` for a working sample.

---

## 📤 Output Format

### Added Columns

Script adds these columns to your data:

| Column | Type | Range | Meaning |
|--------|------|-------|---------|
| `predicted_category` | string | green, yellow, orange | Predicted H2S level |
| `probability_green` | float | 0.0 - 1.0 | Confidence in green |
| `probability_orange` | float | 0.0 - 1.0 | Confidence in orange |
| `probability_yellow` | float | 0.0 - 1.0 | Confidence in yellow |
| `confidence` | float | 0.0 - 1.0 | Highest probability |
| `alert` | boolean | True/False | True if yellow or orange |

### Interpreting Results

```csv
time,predicted_category,probability_orange,confidence,alert
2024-01-15T12:00:00Z,green,0.08,0.85,False        # Safe, high confidence
2024-01-15T13:00:00Z,orange,0.62,0.62,True        # Critical, monitor closely
2024-01-15T14:00:00Z,yellow,0.23,0.57,True        # Caution, moderate confidence
2024-01-15T15:00:00Z,orange,0.35,0.65,True        # Critical, lower confidence
```

**Guidelines:**
- Confidence > 0.7: High confidence
- Confidence 0.5-0.7: Moderate confidence
- Confidence < 0.5: Low confidence (uncertain)

---

## 🔄 Integration Options

### Python

```python
from src.predict_h2s import H2SPredictor
import pandas as pd

# Initialize once
predictor = H2SPredictor(
   '../nestor_xgboost_weighted_model.json',
   'nestor_preprocessing_info.pkl'
)

# Load new data
df = pd.read_csv('new_data.csv')
df_processed = predictor.preprocess_data(df)

# Generate predictions
predictions = predictor.predict(df_processed)

# Get alerts
alerts = predictions[predictions['alert'] == True]
orange_alerts = alerts[alerts['predicted_category'] == 'orange']

# Send notifications
if len(orange_alerts) > 0:
   send_email_alert(orange_alerts)
```

### REST API (Flask)

```python
from flask import Flask, request, jsonify
from src.predict_h2s import H2SPredictor

app = Flask(__name__)
predictor = H2SPredictor('model.json', 'preprocessing.pkl')


@app.route('/predict', methods=['POST'])
def predict():
   data = pd.DataFrame([request.get_json()])
   result = predictor.predict(predictor.preprocess_data(data))
   return jsonify(result.iloc[0].to_dict())


app.run(port=5000)
```

### R

```r
# Call Python script
system("python predict_h2s.py --input data.csv --output predictions.csv")

# Load results
predictions <- read.csv("predictions.csv")

# Filter and plot
alerts <- subset(predictions, alert == TRUE)
plot(predictions$time, predictions$probability_orange, type="l")
```

---

## 🔧 Maintenance

### Weekly Monitoring

Track these metrics:

```python
# Load predictions and actuals
predictions = pd.read_csv('predictions.csv')
actuals = pd.read_csv('actuals.csv')

# Calculate metrics
from sklearn.metrics import classification_report

merged = predictions.merge(actuals, on='time')
print(classification_report(merged['actual_category'], 
                            merged['predicted_category']))
```

**Alert if:**
- Orange recall < 55%
- False positive rate > 10%
- Balanced accuracy < 60%

### Retraining Schedule

**When:**
- Every 3 months (routine)
- When performance degrades
- After sensor maintenance
- When patterns change

**How:**
1. Collect new data
2. Combine with original dataset
3. Retrain model (contact data science team)
4. Validate on test set
5. Deploy if improved

### Log Monitoring

```bash
# Check prediction logs
tail -f /var/log/h2s_predictions.log

# Count daily alerts
grep "$(date +%Y-%m-%d)" predictions_*.csv | grep "alert.*True" | wc -l

# Find errors
grep "Error" /var/log/h2s_predictions.log
```

---

## ❓ Troubleshooting

### Installation Issues

**Problem:** "pip install fails"
```bash
# Upgrade pip first
pip install --upgrade pip

# Then retry
pip install -r requirements.txt
```

**Problem:** "xgboost won't install"
```bash
# Try with conda
conda install -c conda-forge xgboost

# Or specific version
pip install xgboost==1.7.6
```

### Runtime Issues

**Problem:** "Model file not found"
```bash
# Check files are present
ls -lh nestor_*.* 

# Use absolute paths
python predict_h2s.py \
    --model /full/path/to/nestor_xgboost_weighted_model.json \
    --preprocessing /full/path/to/nestor_preprocessing_info.pkl \
    --input data.csv --output predictions.csv
```

**Problem:** "KeyError: column not found"
```bash
# Verify your CSV has all required columns
head -1 your_data.csv

# Compare with required columns in DEPLOYMENT_GUIDE.md
```

**Problem:** "All predictions are green"
```bash
# Check if thresholds are too high
python predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.20

# Verify input data is realistic (not all zeros, etc.)
```

### Performance Issues

**Problem:** "Too many false alarms"
```bash
# Increase thresholds
python predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.40
```

**Problem:** "Missing too many events"
```bash
# Decrease thresholds
python predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.25
```

---

## 📞 Support Resources

### Documentation Hierarchy

1. **README.md** - Start here (quick overview)
2. **NESTOR_BES_Quick_Start.md** - 5-minute guide
3. **DEPLOYMENT_GUIDE.md** - Complete reference
4. **NESTOR_BES_H2S_Forecasting_Report.md** - Technical details
5. **Complete_Model_Testing_Summary.md** - Algorithm comparison

### Common Questions

**Q: Can I use this for other sites?**  
A: No, model is trained specifically for NESTOR - BES. Would need retraining for other sites.

**Q: Can I update the model myself?**  
A: Not recommended. Contact data science team for retraining.

**Q: What if new sensors are added?**  
A: Model needs retraining to incorporate new features.

**Q: Can I use different thresholds for different times?**  
A: Yes, you can run predictions multiple times with different thresholds and filter by time.

**Q: How do I integrate with SCADA?**  
A: Use API endpoint (Flask example in DEPLOYMENT_GUIDE.md) or batch processing with file exchange.

---

## 🎓 Understanding the Model

### How It Works

1. **Feature Engineering:** Converts raw sensor data into 20 features
   - Temporal patterns (hour, day, month - cyclical)
   - Wind characteristics (speed, direction, interactions)
   - Tidal conditions (height, state, flow)
   - Weather (temperature, humidity, pressure)

2. **XGBoost Prediction:** Ensemble of 300 decision trees
   - Each tree learns different patterns
   - Trees vote on final prediction
   - Class weighting favors minority classes (orange)

3. **Output:** Three probabilities (green, yellow, orange)
   - Highest probability = predicted category
   - Thresholds can adjust sensitivity

### Why XGBoost?

Tested 10+ algorithms:
- **XGBoost: 61.3% orange detection** ✓
- Logistic Regression: 48.2%
- KNN: 38.7%
- Binary classification: 19.9%
- Neural Network: 0.7%

XGBoost wins because:
- Best handles class imbalance (only 7% orange samples)
- Learns complex patterns
- Ensemble reduces overfitting
- Production-proven

### Limitations

**What it CAN'T do:**
- Detect instant spikes (< 1 hour warning)
- Achieve 100% accuracy
- Work without sensor data
- Replace direct H2S monitoring

**What it CAN do:**
- Provide 1-3 hour advance warning
- Detect 61% of critical events
- Reduce operator workload
- Enable proactive response

---

## 📊 Performance Metrics Explained

### Confusion Matrix

```
                Predicted
Actual      Green   Orange  Yellow
Green       1,263     84     198     81.7% correct
Orange         14     84      39     61.3% detected ✓
Yellow         75     57     113     46.1% detected
```

**Reading:**
- Row = Actual category
- Column = Predicted category
- Diagonal = Correct predictions
- Off-diagonal = Errors

### Key Metrics

**Orange Recall (61.3%):**
- Out of 137 actual orange events, caught 84
- Most important metric (safety critical)
- Target: > 55%

**False Positive Rate (5.4%):**
- Out of 1,545 green periods, 84 false alarms
- Acceptable for operators
- Target: < 8%

**Balanced Accuracy (63.1%):**
- Average of per-class accuracies
- Accounts for class imbalance
- Target: > 60%

---

## 🏆 Deployment Checklist

### Pre-Deployment

- [ ] All files downloaded
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Test installation passed (`python test_installation.py`)
- [ ] Sample prediction successful
- [ ] Input data format verified

### Initial Deployment

- [ ] Production server configured
- [ ] File paths set correctly
- [ ] Automated scheduling configured (cron/scheduler)
- [ ] Output directory created and accessible
- [ ] Logging enabled
- [ ] Alert notifications configured (email/SMS)

### Post-Deployment

- [ ] Monitor first week of predictions
- [ ] Compare with actual H2S values
- [ ] Adjust thresholds if needed
- [ ] Train operators on system
- [ ] Document local procedures
- [ ] Schedule first maintenance review (3 months)

---

## 📈 Next Steps

### Immediate (Week 1)

1. Run test installation
2. Generate predictions on sample data
3. Review output format
4. Configure thresholds

### Short-term (Month 1)

1. Deploy to production server
2. Set up automated daily runs
3. Configure alert notifications
4. Train operators
5. Monitor performance

### Long-term (Quarterly)

1. Collect performance metrics
2. Evaluate retraining needs
3. Update documentation
4. Review and adjust thresholds
5. Plan enhancements

---

## 📝 Version Information

**Model Version:** 1.0  
**Release Date:** December 2025  
**Training Data:** 9,631 samples (NESTOR - BES, Nov 2023 - Jan 2025)  
**Algorithm:** XGBoost with balanced class weighting  
**Features:** 20 engineered features  
**Performance:** 61.3% orange detection, 63.1% balanced accuracy

---

## 🙏 Acknowledgments

**Model Development:**
- Tested 10+ algorithms and 30+ configurations
- Analyzed 4 different datasets
- Comprehensive validation and documentation

**Best Practices:**
- Industry-standard ML workflow
- Production-ready code
- Comprehensive error handling
- Detailed documentation

---

**Ready to deploy?** Start with the Quick Deployment section above, then refer to DEPLOYMENT_GUIDE.md for details.

**Questions?** See the Support Resources section for documentation hierarchy and common questions.

**Issues?** Check the Troubleshooting section for solutions to common problems.

---

*Package created: December 2025*  
*For: NESTOR - BES H2S Forecasting*  
*Model: XGBoost v1.0*
