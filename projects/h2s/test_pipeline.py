"""End-to-end H2S pipeline test — local execution, no S3 required.

Steps:
  1. Fetch hourly weather forecast from Open-Meteo (free, no key)
  2. Train xgboost_base + xgboost_smote on last available training data (Dec 2025)
  3. Predict for this month (March 2026) and tomorrow using all three models:
       - production model  (nestor_xgboost_weighted_model.json)
       - xgboost_base      (freshly trained, class weights only)
       - xgboost_smote     (freshly trained, SMOTE on hazard classes)

Run from projects/h2s/:
    uv run python test_pipeline.py
"""

import os
import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data")
TRAINING_CSV = os.path.join(DATA_DIR, "modeldata_h2s_nofill.csv")
PROD_MODEL = os.path.join(ROOT, "nestor_xgboost_weighted_model.json")
PROD_PREP = os.path.join(ROOT, "nestor_preprocessing_info.json")

# NESTOR - BES site coords (Imperial Beach / Tijuana border area)
LATITUDE = 32.545
LONGITUDE = -117.128
SITE = "NESTOR - BES"

# 8 basic features used by the retrained models
TRAIN_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "precipitation",
    "surface_pressure", "cloud_cover", "wind_speed_10m",
    "wind_direction_10m", "wind_gusts_10m"
  #  ,"Flow (m^3/s)--Border", "tide_height", "tide_state"
]

warnings.filterwarnings("ignore")


# ===========================================================================
# STEP 1: Fetch weather forecast from Open-Meteo
# ===========================================================================

def fetch_weather() -> pd.DataFrame:
    """Fetch past 14 days + next 3 days of hourly weather for NESTOR-BES."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
            "precipitation", "surface_pressure", "cloud_cover",
            "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
        ]),
        "timezone": "UTC",
        "past_days": 14,
        "forecast_days": 3,
    }
    print("Fetching weather from Open-Meteo...", end=" ", flush=True)
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "date":                pd.to_datetime(h["time"], utc=True),
        "temperature_2m":      h["temperature_2m"],
        "relative_humidity_2m": h["relative_humidity_2m"],
        "dewpoint_2m":         h["dewpoint_2m"],
        "precipitation":       h["precipitation"],
        "surface_pressure":    h["surface_pressure"],
        "cloud_cover":         h["cloud_cover"],
        "wind_speed_10m":      h["wind_speed_10m"],
        "wind_direction_10m":  h["wind_direction_10m"],
        "wind_gusts_10m":      h["wind_gusts_10m"],
    })
    df["site_name"] = SITE
    print(f"{len(df)} rows ({df['date'].dt.date.min()} → {df['date'].dt.date.max()})")
    return df


# ===========================================================================
# STEP 2: Train fresh models on Dec 2025 training data
# ===========================================================================

def load_training_data() -> pd.DataFrame:
    from h2s.training.relabeling import apply_categorization

    df = pd.read_csv(TRAINING_CSV)
    df = df[df["site_name"] == SITE].copy()
    df["time"] = pd.to_datetime(df["D"])
    df = apply_categorization(df, h2s_column="H2S")

    avail = [f for f in TRAIN_FEATURES if f in df.columns]
    df = df.dropna(subset=avail + ["h2s_category"])
    print(f"Training data: {len(df)} rows ({df['time'].min().date()} → {df['time'].max().date()})")
    dist = df["h2s_category"].value_counts()
    for cat in ["green", "yellow", "orange"]:
        print(f"  {cat:8s}: {dist.get(cat, 0)}")
    return df


def train_variant(train_df: pd.DataFrame, use_smote: bool):
    from h2s.training.model_trainer import train_model_with_cv

    avail = [f for f in TRAIN_FEATURES if f in train_df.columns]
    X = train_df[avail].reset_index(drop=True)
    y = train_df["h2s_category"].reset_index(drop=True)

    unique = sorted(y.unique())
    label_map = {c: i for i, c in enumerate(unique)}
    n_classes = len(unique)
    # need at least n_classes samples per fold — use 3 folds for small data
    n_folds = min(3, len(X) // (n_classes * 2))
    n_folds = max(2, n_folds)

    name = "xgboost_smote" if use_smote else "xgboost_base"
    print(f"  Training {name} ({len(X)} samples, {n_folds} folds)...", end=" ", flush=True)
    model, cv = train_model_with_cv(
        X_train=X, y_train=y, label_map=label_map,
        n_folds=n_folds, n_estimators=100, max_depth=5,
        learning_rate=0.1, use_class_weights=True,
        use_smote=use_smote, random_state=42,
    )
    from h2s.training.model_trainer import calculate_cv_summary
    summary = calculate_cv_summary(cv)
    bal_acc = summary.get("balanced_accuracy_mean", float("nan"))
    print(f"balanced_acc={bal_acc:.3f}")
    return model, label_map, avail, unique


# ===========================================================================
# STEP 3: Predict
# ===========================================================================

def predict_production(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Use the pre-trained production model (local JSON, 20 features)."""
    from h2s.predictor.h2s_predictor import H2SPredictor

    predictor = H2SPredictor.from_local(PROD_MODEL, PROD_PREP)
    df = forecast_df.copy()
    # month is a feature but commented out in preprocess_data — add manually
    df["month"] = df["date"].dt.month
    df_proc = predictor.preprocess_data(df)
    return predictor.predict(df_proc)


def predict_fresh(forecast_df: pd.DataFrame, model, label_map, features) -> pd.DataFrame:
    """Use a freshly trained model (8 basic features)."""
    reverse = {v: k for k, v in label_map.items()}
    X = forecast_df[features].fillna(0.0)
    y_proba = model.predict_proba(X.values)
    y_pred = np.argmax(y_proba, axis=1)  # multi:softprob returns proba matrix from predict()

    result = forecast_df.copy()
    result["predicted_category"] = [reverse[p] for p in y_pred]
    result["probability_green"]  = y_proba[:, label_map.get("green", 0)]  if "green"  in label_map else 0
    result["probability_orange"] = y_proba[:, label_map.get("orange", 1)] if "orange" in label_map else 0
    result["probability_yellow"] = y_proba[:, label_map.get("yellow", 2)] if "yellow" in label_map else 0
    result["confidence"]         = y_proba.max(axis=1)
    result["alert"]              = result["predicted_category"].isin(["orange", "yellow"])
    return result


# ===========================================================================
# STEP 4: Display
# ===========================================================================

CAT_EMOJI = {"orange": "🟠", "yellow": "🟡", "green": "🟢"}

def print_summary(preds: pd.DataFrame, model_name: str, label: str):
    n = len(preds)
    if n == 0:
        print(f"  [{model_name}] {label}: no data")
        return
    cats = preds["predicted_category"].value_counts()
    alerts = int(preds["alert"].sum())
    parts = [
        f"{CAT_EMOJI.get(c, '?')} {c}: {cats.get(c, 0)} ({cats.get(c, 0)/n*100:.0f}%)"
        for c in ["orange", "yellow", "green"]
    ]
    alert_str = f"  ⚠ {alerts} alert hrs" if alerts else "  ✓ no alerts"
    print(f"  [{model_name:28s}] {' | '.join(parts)}{alert_str}")


def print_hourly(preds: pd.DataFrame, label: str):
    print(f"\n  Hourly detail — {label}:")
    print(f"  {'Hour':5s} {'Temp':6s} {'Wind':6s} {'Precip':7s} {'Category':8s} {'Conf':5s}")
    print(f"  {'-'*55}")
    for _, row in preds.iterrows():
        h = row["date"].strftime("%H:00")
        print(
            f"  {h:5s} {row['temperature_2m']:5.1f}°C {row['wind_speed_10m']:5.1f}km/h "
            f"{row['precipitation']:6.1f}mm  "
            f"{CAT_EMOJI.get(row['predicted_category'], '?')} {row['predicted_category']:8s} "
            f"{row['confidence']:.2f}"
        )


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    tomorrow = today + timedelta(days=1)
    month_start = today.replace(day=1)

    print("=" * 65)
    print("  H2S PREDICTION PIPELINE — END-TO-END TEST")
    print(f"  Today: {today}   Tomorrow: {tomorrow}")
    print("=" * 65)

    # --- Step 1: Weather ---
    print("\n[1/3] WEATHER FORECAST")
    forecast = fetch_weather()

    this_month = forecast[
        (forecast["date"].dt.date >= month_start) &
        (forecast["date"].dt.date <= today)
    ].copy()
    tomorrow_df = forecast[forecast["date"].dt.date == tomorrow].copy()
    print(f"  This month (so far): {len(this_month)} hrs | Tomorrow: {len(tomorrow_df)} hrs")

    # --- Step 2: Train ---
    print("\n[2/3] TRAINING MODELS (Dec 2025 data)")
    train_df = load_training_data()
    model_base,  lm_base,  feat_base,  cls_base  = train_variant(train_df, use_smote=False)
    model_smote, lm_smote, feat_smote, cls_smote = train_variant(train_df, use_smote=True)

    # --- Step 3 & 4: Predict + display ---
    print("\n[3/3] PREDICTIONS")
    for label, window in [("This month (so far)", this_month), (f"Tomorrow ({tomorrow})", tomorrow_df)]:
        if len(window) == 0:
            print(f"\n  {label}: no forecast data in window")
            continue
        print(f"\n  {label}  ({len(window)} hours)")
        print_summary(predict_production(window),             "production model",    label)
        print_summary(predict_fresh(window, model_base,  lm_base,  feat_base),  "xgboost_base (retrained)",  label)
        print_summary(predict_fresh(window, model_smote, lm_smote, feat_smote), "xgboost_smote (retrained)", label)

    # Hourly breakdown for tomorrow using production model
    if len(tomorrow_df) > 0:
        print_hourly(predict_production(tomorrow_df), f"Tomorrow ({tomorrow}) — production model")

    print("\n✓ Done")
