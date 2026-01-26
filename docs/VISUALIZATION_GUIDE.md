# Generating Model Visualizations

## Three Key Visualizations

The H2S prediction system can generate three important visualizations:

1. **Feature Importance** - Shows which variables matter most for predictions
2. **Confusion Matrix** - Shows prediction accuracy in detail
3. **Model Comparison** - Compares performance metrics

---

## Method 1: Use Pre-Generated Visualizations

The model training already generated these visualizations. They are included in your package:

| Visualization | Filename | What It Shows |
|---------------|----------|---------------|
| Feature Importance | `nestor_feature_importance.png` | Top 15 most important features |
| Confusion Matrix | `nestor_confusion_matrices.png` | Detailed error analysis (4 models) |
| Model Comparison | `nestor_model_comparison.png` | Performance across metrics |

**To use:** Simply view these existing PNG files - they show the model's performance on the test set.

---

## Method 2: Generate New Visualizations (Requires Predictions + Actuals)

If you want to generate new visualizations based on recent predictions:

### Prerequisites

```bash
pip install matplotlib seaborn xgboost pandas numpy scikit-learn
```

### Usage

```bash
# Feature importance only (no predictions needed)
python generate_visualizations.py

# All three plots (requires predictions and actuals)
python generate_visualizations.py \
    --predictions your_predictions.csv \
    --actuals your_actuals.csv \
    --output-dir ./reports
```

### Required CSV Formats

**Predictions CSV** (output from predict_h2s.py):
```csv
time,predicted_category,probability_orange,...
2024-01-15T12:00:00Z,green,0.08,...
2024-01-15T13:00:00Z,orange,0.62,...
```

Must have:
- `time` column (ISO format)
- `predicted_category` column (green, yellow, or orange)

**Actuals CSV** (your measured H2S values):
```csv
time,H2S
2024-01-15T12:00:00Z,2.5
2024-01-15T13:00:00Z,18.3
```

Must have:
- `time` column (ISO format, matching predictions)
- `H2S` column (measured ppb values)

---

## Method 3: Python Integration

```python
from src.generate_visualizations import generate_all_visualizations

# Generate all three visualizations
results = generate_all_visualizations(
    predictions_path='predictions.csv',
    actuals_path='actuals.csv',
    model_path='../nestor_xgboost_weighted_model.json',
    preprocessing_path='../nestor_preprocessing_info.pkl',
    output_dir='./reports'
)

# Access results
print(f"Balanced Accuracy: {results['metrics']['balanced_accuracy']:.1%}")
print(f"Orange Recall: {results['metrics']['recalls']['orange']:.1%}")
```

---

## What Each Visualization Shows

### 1. Feature Importance

**File:** `feature_importance.png`

Shows the top 15 features that influence predictions, ranked by importance.

**Interpretation:**
- Longer bar = more important feature
- Top features drive most predictions
- Helps understand what the model "looks at"

**Example findings:**
- Flow rate is #1 (most important)
- Relative humidity is #2
- Tide height is #3
- Hour of day matters (temporal patterns)

**Use for:**
- Understanding model behavior
- Identifying which sensors are critical
- Prioritizing sensor maintenance

---

### 2. Confusion Matrix

**File:** `confusion_matrix.png` or `confusion_matrices.png`

Shows how predictions compare to actual values in a grid.

**Reading the Matrix:**
```
                Predicted
Actual      Green   Orange  Yellow
Green         1263     84     198     ← 1263 correct, 282 errors
Orange          14     84      39     ← 84 caught, 53 missed
Yellow          75     57     113     ← 113 caught, 132 missed
```

**Key Numbers:**
- Diagonal (green cells) = Correct predictions
- Off-diagonal = Errors
- Orange row, Orange column = 84 critical events caught ✓
- Orange row, Green column = 14 critical events missed ✗

**Use for:**
- Detailed performance analysis
- Understanding types of errors
- Identifying patterns in misclassifications

---

### 3. Model Comparison

**File:** `model_comparison.png`

Multi-panel visualization showing:
- Overall balanced accuracy
- Per-class recall (detection rates)
- Precision vs recall trade-offs
- Confusion matrix

**Interpretation:**
- Panel 1: Overall score (target: >60%)
- Panel 2: How well each category is detected
- Panel 3: Precision-recall balance
- Panel 4: Detailed confusion matrix

**Use for:**
- Quick performance overview
- Comparing model versions
- Reporting to stakeholders
- Monitoring over time

---

## Generating Updated Visualizations

### When to Update

Generate new visualizations:
- **Monthly**: Monitor model performance
- **After retraining**: Compare old vs new model
- **When patterns change**: Understand new behavior
- **For reporting**: Show current performance

### Example Workflow

```bash
# 1. Generate predictions for last month
python predict_h2s.py --input december_data.csv --output december_predictions.csv

# 2. Collect actual H2S values (from sensors)
# Create december_actuals.csv with columns: time, H2S

# 3. Generate visualizations
python generate_visualizations.py \
    --predictions december_predictions.csv \
    --actuals december_actuals.csv \
    --output-dir ./reports/december

# 4. Review
ls ./reports/december/*.png
```

---

## Customizing Visualizations

### Adjusting Output Size

Edit `generate_visualizations.py`:

```python
# For feature importance
plt.figure(figsize=(10, 8))  # Change to (12, 10) for larger

# For confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))  # Change to (14, 12)
```

### Changing Colors

```python
# Feature importance
plt.barh(..., color='steelblue')  # Change to 'darkblue', 'forestgreen', etc.

# Confusion matrix
sns.heatmap(..., cmap='Blues')  # Change to 'Greens', 'Reds', 'YlOrRd'
```

### Adding More Features

Show top 20 instead of 15:

```python
importance_df = importance_df.sort_values('importance', ascending=False).head(20)  # Was 15
```

---

## Troubleshooting

### "No module named 'xgboost'"

```bash
pip install xgboost
```

### "No module named 'matplotlib'"

```bash
pip install matplotlib seaborn
```

### "No matching timestamps"

Check that:
- Time formats match exactly
- Both CSVs use same timezone
- Column names are correct (`time` and `H2S`)

```python
# Debug: Check timestamps
import pandas as pd
pred = pd.read_csv('predictions.csv')
act = pd.read_csv('actuals.csv')
print(pred['time'].head())
print(act['time'].head())
```

### "FileNotFoundError: model file"

Ensure you're in the correct directory:

```bash
ls -lh nestor_xgboost_weighted_model.json
# If not found, use --model with full path
python generate_visualizations.py --model /full/path/to/model.json
```

---

## Integration with Reports

### PowerPoint/Presentations

```bash
# Generate high-res images
python generate_visualizations.py ... --output-dir ./presentation

# Insert the PNG files into your slides
# Files are 300 DPI, suitable for printing
```

### Automated Reports

```python
from src.generate_visualizations import generate_all_visualizations
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Generate plots
results = generate_all_visualizations(
    predictions_path='current_predictions.csv',
    actuals_path='current_actuals.csv',
    output_dir='./reports'
)

# Email with attachments
msg = MIMEMultipart()
msg['Subject'] = f"H2S Model Performance - {date.today()}"

for filename in ['feature_importance.png', 'confusion_matrix.png']:
    with open(f'./reports/{filename}', 'rb') as f:
        img = MIMEImage(f.read())
        img.add_header('Content-Disposition', 'attachment',
                       filename=filename)
        msg.attach(img)

# Send email
# ... (add your email sending code)
```

### Dashboard Integration

Save metrics to database:

```python
results = generate_all_visualizations(...)

# Extract metrics
metrics = {
    'date': datetime.now(),
    'balanced_accuracy': results['metrics']['balanced_accuracy'],
    'orange_recall': results['metrics']['recalls']['orange'],
    'yellow_recall': results['metrics']['recalls']['yellow'],
    'green_recall': results['metrics']['recalls']['green']
}

# Save to database
import sqlite3
conn = sqlite3.connect('h2s_performance.db')
pd.DataFrame([metrics]).to_sql('performance_history', conn, if_exists='append')
```

---

## Performance Tracking Over Time

### Create Time Series

```python
import pandas as pd
import matplotlib.pyplot as plt

# Generate visualizations for each month
months = ['january', 'february', 'march']
metrics_history = []

for month in months:
    results = generate_all_visualizations(
        predictions_path=f'{month}_predictions.csv',
        actuals_path=f'{month}_actuals.csv'
    )
    metrics_history.append({
        'month': month,
        'balanced_accuracy': results['metrics']['balanced_accuracy'],
        'orange_recall': results['metrics']['recalls']['orange']
    })

# Plot trends
df = pd.DataFrame(metrics_history)
plt.figure(figsize=(10, 6))
plt.plot(df['month'], df['orange_recall'], marker='o', label='Orange Recall')
plt.plot(df['month'], df['balanced_accuracy'], marker='s', label='Balanced Accuracy')
plt.xlabel('Month')
plt.ylabel('Performance')
plt.title('Model Performance Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_trend.png', dpi=300)
```

---

## Summary

**Quick Answer:** 
The visualizations are already included as PNG files in your package. You can view them directly.

**For New Data:**
Use `generate_visualizations.py` script with your predictions and actual measurements to create updated plots.

**Three Files Generated:**
1. `feature_importance.png` - What drives predictions
2. `confusion_matrix.png` - Detailed accuracy
3. `model_comparison.png` - Overall performance

**Requirements:**
- matplotlib, seaborn (for plotting)
- xgboost (for feature importance)
- Your predictions + actual H2S values (for confusion matrix)

---

*See `generate_visualizations.py` for full script*
