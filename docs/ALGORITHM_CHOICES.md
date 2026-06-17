# Algorithm Choices in Model Archives

Every archived model version — both production (`station_model_archive`) and
walk-forward backfill (`backfill_station_models`) — carries an `algorithm_choices`
field inside `archive_metadata.json`. This documents which algorithm
`train_and_select()` auto-selected for each task/variant combination at training
time.

## Field structure

```json
"algorithm_choices": {
  "evidence": {
    "regression":  "XGBoost",
    "clf_5ppb":    "Ensemble (XGB+RF, w=0.60/0.40)",
    "clf_10ppb":   "RandomForest",
    "clf_30ppb":   "XGBoost"
  },
  "lean": {
    "regression":  "RandomForest",
    "clf_5ppb":    "XGBoost",
    "clf_10ppb":   "XGBoost",
    "clf_30ppb":   "Ensemble (XGB+RF, w=0.55/0.45)"
  }
}
```

### Possible values

| Value | Meaning |
|-------|---------|
| `"XGBoost"` | XGBoost model scored higher on the selection metric |
| `"RandomForest"` | Random Forest model scored higher |
| `"Ensemble"` | Scores were within `ensemble_margin` of each other; both are combined as a weighted average. Weights shown in the string, e.g. `w=0.60/0.40` (RF/XGB). |

## Selection logic

`train_and_select()` in `h2s/training/multi_station_trainer.py` trains both RF
and XGBoost on the same train/val split and picks using:

- **Regression tasks** (`regression`): `recall_30` — recall at 30 ppb, the
  operational orange-alert threshold. R² rewards bulk fit on the heavy-tailed
  H2S series but hides gaps in extreme-event recall, so `recall_30` was chosen
  as the primary selection metric.
- **Classifier tasks** (`clf_5ppb`, `clf_10ppb`, `clf_30ppb`): `AUC` (ROC
  area under the curve).

If the two models' scores differ by less than `ensemble_margin` (default `0.01`),
they are combined: the ensemble weight for each model is proportional to its score
on the selection metric. This avoids arbitrary coin-flip decisions when models
perform nearly identically.

## Where it lives

### Production archives
```
STATION_MODELS_ARCHIVE_BASE/{station_key}/{version_tag}/archive_metadata.json
```
`version_tag` format: `YYYYMMDDTHHMMSSZ-{gitsha}`

### Backfill archives
```
STATION_MODELS_ARCHIVE_BASE/{station_key}/backfill_{month_key}_{timestamp}-{sha}/archive_metadata.json
```

Both production and backfill archives also contain `training_report.json`, which
holds the full in-sample validation metrics per task/variant plus the feature
lists (`features.evidence`, `features.lean`). `features_{variant}.json` is **not**
written separately — the feature lists live inside `training_report.json`.

## Reading algorithm choices programmatically

```python
import json
from h2s.resources.minio import S3Resource

s3 = S3Resource(...)
meta_bytes = s3.getFile(
    path="tijuana/forecast/models/archive/stations/NESTOR__BES/"
         "20260612T213000Z-a1b2c3d/archive_metadata.json",
    bucket="resilentpublic",
)
meta = json.loads(meta_bytes)
choices = meta["algorithm_choices"]
print(choices["evidence"]["regression"])  # e.g. "XGBoost"
```
