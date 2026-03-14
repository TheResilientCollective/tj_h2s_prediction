
# H2S Prediction System for Tijuana River Valley
presently, limited to single site: NESTOR - BES

This has been converted into a dagster project for automated execution.

These are in project/h2s


# The review of the original code:

Production-ready system for forecasting H2S levels using machine learning.

## 🎯 Performance

- **Orange Detection:** 61.3% (catches 84 out of 137 critical events)
- **Yellow Detection:** 46.1%
- **Balanced Accuracy:** 63.1%
- **False Alarm Rate:** 5.4%

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate predictions
python predict_h2s.py --input your_data.csv --output predictions.csv

# 3. View results
head predictions.csv
```

## 📁 Files

| File | Purpose |
|------|---------|
| `predict_h2s.py` | Main prediction script |
| `batch_predict.py` | Process multiple files |
| `nestor_xgboost_weighted_model.json` | Trained XGBoost model |
| `nestor_preprocessing_info.pkl` | Feature preprocessing |
| `DEPLOYMENT_GUIDE.md` | Complete documentation |
| `requirements.txt` | Python dependencies |

## 💡 Usage Examples

### Single File
```bash
python predict_h2s.py --input new_data.csv --output predictions.csv
```

### Batch Processing
```bash
python batch_predict.py --input-dir ./new_data --output-dir ./predictions
```

### More Sensitive (catch more events)
```bash
python predict_h2s.py --input data.csv --output predictions.csv --orange-threshold 0.25
```

### Alerts Only (filter out green)
```bash
python predict_h2s.py --input data.csv --output alerts.csv --filter-alerts
```

## 📊 Input Requirements

Your CSV must include:
- Weather: temperature, wind speed/direction, humidity, pressure, etc.
- Tidal: flow rate, tide height, tidal state
- Time: timestamp for each record

See `DEPLOYMENT_GUIDE.md` for complete column list.

## 📈 Output Format

Adds these columns to your data:
- `predicted_category`: green, yellow, or orange
- `probability_orange`: Confidence (0-1)
- `probability_yellow`: Confidence (0-1)
- `probability_green`: Confidence (0-1)
- `confidence`: Highest probability
- `alert`: True if yellow or orange

## 🔄 Automated Updates

### Daily Cron Job
```bash
# Add to crontab: daily at 2 AM
0 2 * * * python3 /path/to/batch_predict.py --input-dir /data --output-dir /predictions
```

### Python Integration

```python
from src.predict_h2s import H2SPredictor

predictor = H2SPredictor('nestor_xgboost_weighted_model.json',
                         'nestor_preprocessing_info.pkl')
predictions = predictor.predict(your_dataframe)
alerts = predictions[predictions['alert'] == True]
```

## 🎚️ Threshold Tuning

| Setting | Orange Threshold | Expected Orange Recall | False Positives |
|---------|------------------|------------------------|-----------------|
| Default | 0.33 | 61% | 5.4% |
| Sensitive | 0.25 | 70% | 10% |
| Very Sensitive | 0.20 | 75% | 15% |
| Conservative | 0.40 | 55% | 3% |

## 📖 Documentation

- **Quick Start:** `NESTOR_BES_Quick_Start.md`
- **Deployment:** `DEPLOYMENT_GUIDE.md` (this file)
- **Technical Report:** `NESTOR_BES_H2S_Forecasting_Report.md`
- **Model Testing:** `Complete_Model_Testing_Summary.md`

## ⚠️ Important Notes

**What it does:**
- ✅ Provides 1-3 hour advance warning
- ✅ Detects 61% of critical events
- ✅ Processes data automatically
- ✅ Low false alarm rate

**What it doesn't do:**
- ❌ Replace H2S sensors
- ❌ Catch 100% of events
- ❌ Work without sensor data
- ❌ Detect instant spikes

**Use for:** Early warning, planning, trend analysis
**Don't use for:** Emergency response, regulatory compliance alone

## 🔧 Troubleshooting

**Problem:** "Model file not found"
**Solution:** Ensure .json and .pkl files are in same directory

**Problem:** "Column not found"
**Solution:** Check your CSV has all required columns (see DEPLOYMENT_GUIDE.md)

**Problem:** "Too many false alarms"
**Solution:** Increase thresholds (--orange-threshold 0.40)

**Problem:** "Missing too many events"
**Solution:** Decrease thresholds (--orange-threshold 0.25)

## 📞 Support

See `DEPLOYMENT_GUIDE.md` for:
- Complete API reference
- Integration examples (Python, R, Flask)
- Monitoring setup
- Retraining procedures

## 📊 Model Details

- **Algorithm:** XGBoost with class weighting
- **Features:** 20 engineered features
- **Training Data:** 9,631 samples from NESTOR - BES
- **Time Period:** November 2023 - January 2025
- **Version:** 1.0 (December 2025)

## 🏆 Why This Model?

Tested 10+ algorithms including:
- KNN, Random Forest, Neural Networks, Gradient Boosting, etc.
- Binary vs 3-class classification
- Multiple datasets and configurations

**Result:** XGBoost with 3-class classification is the clear winner.

---

**Ready to deploy?** Start with `DEPLOYMENT_GUIDE.md` for detailed instructions.
