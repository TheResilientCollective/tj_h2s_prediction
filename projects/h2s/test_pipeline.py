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

import argparse
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
TRAINING_CSV = os.path.join(DATA_DIR, "modeldata_h2s_nofill.parquet")

# Select production model variant via H2S_MODEL_VARIANT env var.
# Supported values: random_forest (default), xgboost_base, xgboost_smote
# Each variant lives under data/startmodels/<variant>/
_VARIANT_EXTS = {
    "random_forest": "model.joblib",
    "xgboost_base": "model.json",
    "xgboost_smote": "model.json",
}
PROD_VARIANT = os.environ.get("H2S_MODEL_VARIANT", "random_forest")
if PROD_VARIANT not in _VARIANT_EXTS:
    raise SystemExit(
        f"Unknown H2S_MODEL_VARIANT={PROD_VARIANT!r}. "
        f"Choose from: {', '.join(_VARIANT_EXTS)}"
    )
_variant_dir = os.path.join(DATA_DIR, "startmodels", PROD_VARIANT)
PROD_MODEL = os.path.join(_variant_dir, _VARIANT_EXTS[PROD_VARIANT])
PROD_PREP  = os.path.join(_variant_dir, "nestor_preprocessing_info.json")

# NESTOR - BES site coords (Imperial Beach / Tijuana border area)
LATITUDE = 32.545
LONGITUDE = -117.128
SITE = "NESTOR - BES"

# Features for retrained models — weather + derived + tidal
TRAIN_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dewpoint_2m",
    "precipitation", "surface_pressure", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "wind_direction_sin", "wind_direction_cos",
    "wind_speed_10m_avg_2h", "wind_speed_10m_avg_3h", "wind_speed_10m_avg_4h",
    "wind_gusts_10m_max_2h", "wind_gusts_10m_max_3h", "wind_gusts_10m_max_4h",
    "wind_temp_interaction", "humidity_temp_interaction",
    "tide_height", "tidal_state_encoded", "Flow (m^3/s)--Border",
]

# NOAA CO-OPS station: San Diego, CA (closest to NESTOR-BES)
NOAA_STATION = "9410170"

# Encoding from training parquet: slack=-1, low=0, flood=1, high=2, ebb=3
TIDAL_ENCODING = {"slack": -1, "low": 0, "flood": 1, "high": 2, "ebb": 3}

# Median dry-weather Tijuana River flow (m³/s) — used when live USGS data unavailable
FLOW_BASELINE_M3S = 1.35

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
    df = _add_derived_features(df, time_col="date")

    # Merge tidal data
    tides = fetch_tides(df["date"].dt.date.min(), df["date"].dt.date.max())
    df = df.merge(tides, on="date", how="left")
    df["tide_height"] = df["tide_height"].ffill().bfill()
    df["tidal_state_encoded"] = df["tidal_state_encoded"].ffill().bfill().fillna(-1).astype(int)
    df["Flow (m^3/s)--Border"] = df["Flow (m^3/s)--Border"].ffill().bfill()

    print(f"{len(df)} rows ({df['date'].dt.date.min()} → {df['date'].dt.date.max()})")
    return df


def fetch_tides(date_min, date_max) -> pd.DataFrame:
    """Fetch NOAA tidal predictions and estimate Tijuana River flow.

    Returns DataFrame with columns: date, tide_height, tidal_state_encoded,
    Flow (m^3/s)--Border.
    """
    begin = date_min.strftime("%Y%m%d")
    end   = date_max.strftime("%Y%m%d")

    print("Fetching tides from NOAA CO-OPS...", end=" ", flush=True)
    url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    params = {
        "station":   NOAA_STATION,
        "product":   "predictions",
        "datum":     "MLLW",
        "time_zone": "GMT",
        "units":     "metric",
        "interval":  "h",
        "format":    "json",
        "begin_date": begin,
        "end_date":   end,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    raw = resp.json()["predictions"]

    df = pd.DataFrame({
        "date":        pd.to_datetime([p["t"] for p in raw], utc=True),
        "tide_height": [float(p["v"]) for p in raw],
    }).sort_values("date").reset_index(drop=True)

    # Derive tidal state from hourly delta
    delta = df["tide_height"].diff()
    prev_delta = delta.shift(1)
    states = []
    for i in range(len(df)):
        d = delta.iloc[i] if not pd.isna(delta.iloc[i]) else 0.0
        p = prev_delta.iloc[i] if not pd.isna(prev_delta.iloc[i]) else 0.0
        if p > 0.02 and d <= 0:       # just turned — local high
            state = "high"
        elif p < -0.02 and d >= 0:    # just turned — local low
            state = "low"
        elif d > 0.02:
            state = "flood"
        elif d < -0.02:
            state = "ebb"
        else:
            state = "slack"
        states.append(state)
    df["tidal_state_encoded"] = pd.Series(states).map(TIDAL_ENCODING).fillna(-1).astype(int)

    # Flow: try USGS 11013500 (Tijuana River near Nestor); fall back to baseline
    flow_series = _fetch_usgs_flow(begin, end)
    if flow_series is not None:
        df = df.merge(flow_series, on="date", how="left")
        df["Flow (m^3/s)--Border"] = df["Flow (m^3/s)--Border"].ffill().fillna(FLOW_BASELINE_M3S)
        print(f"{len(df)} hrs, USGS flow merged")
    else:
        df["Flow (m^3/s)--Border"] = FLOW_BASELINE_M3S
        print(f"{len(df)} hrs, flow=baseline ({FLOW_BASELINE_M3S} m³/s)")

    return df[["date", "tide_height", "tidal_state_encoded", "Flow (m^3/s)--Border"]]


def _fetch_usgs_flow(begin: str, end: str) -> pd.DataFrame | None:
    """Fetch Tijuana River flow from USGS; returns None if unavailable."""
    try:
        url = "https://waterservices.usgs.gov/nwis/iv/"
        params = {
            "sites": "11013500",
            "parameterCd": "00060",   # discharge in cfs
            "startDT": f"{begin[:4]}-{begin[4:6]}-{begin[6:]}",
            "endDT":   f"{end[:4]}-{end[4:6]}-{end[6:]}",
            "format": "json",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        ts = resp.json()["value"]["timeSeries"]
        if not ts:
            return None
        vals = ts[0]["values"][0]["value"]
        if not vals:
            return None
        df = pd.DataFrame({
            "date": pd.to_datetime([v["dateTime"] for v in vals], utc=True),
            "Flow (m^3/s)--Border": [float(v["value"]) * 0.028316847 for v in vals],
        })
        # Resample to hourly
        df = df.set_index("date").resample("1h").mean().interpolate().reset_index()
        return df
    except Exception:
        return None


def _add_derived_features(df: pd.DataFrame, time_col: str = "date") -> pd.DataFrame:
    """Compute wind sin/cos, rolling windows, and interaction features."""
    df = df.copy().sort_values(time_col).reset_index(drop=True)
    rad = np.deg2rad(df["wind_direction_10m"].fillna(0))
    df["wind_direction_sin"] = np.sin(rad)
    df["wind_direction_cos"] = np.cos(rad)
    for h in (2, 3, 4):
        df[f"wind_speed_10m_avg_{h}h"]  = df["wind_speed_10m"].rolling(h, min_periods=1).mean()
        df[f"wind_gusts_10m_max_{h}h"]  = df["wind_gusts_10m"].rolling(h, min_periods=1).max()
    df["wind_temp_interaction"]      = df["wind_speed_10m"] * df["temperature_2m"]
    df["humidity_temp_interaction"]  = df["relative_humidity_2m"] * df["temperature_2m"]
    return df


# ===========================================================================
# STEP 2: Train fresh models on Dec 2025 training data
# ===========================================================================

def load_training_data(filter_zero_h2s: bool = True) -> pd.DataFrame:
    from h2s.training.relabeling import apply_categorization

    df = pd.read_parquet(TRAINING_CSV)
    df = df[df["site_name"] == SITE].copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(None)

    if filter_zero_h2s:
        before = len(df)
        df = df[df["H2S"].notna() & (df["H2S"] > 0)].copy()
        print(f"  --filter-zero-h2s: dropped {before - len(df)} rows with H2S=0 or NaN")

    df = apply_categorization(df, h2s_column="H2S")

    avail = [f for f in TRAIN_FEATURES if f in df.columns]
    df = df.dropna(subset=avail + ["h2s_category"])
    print(f"Training data: {len(df)} rows ({df['time'].min().date()} → {df['time'].max().date()})")
    dist = df["h2s_category"].value_counts()
    for cat in ["green", "yellow", "orange"]:
        print(f"  {cat:8s}: {dist.get(cat, 0)}")
    return df


def train_variant(train_df: pd.DataFrame, variant: str):
    from h2s.training.model_trainer import (
        train_model_with_cv, train_random_forest_with_cv, calculate_cv_summary,
    )

    avail = [f for f in TRAIN_FEATURES if f in train_df.columns]
    X = train_df[avail].reset_index(drop=True)
    y = train_df["h2s_category"].reset_index(drop=True)

    unique = sorted(y.unique())
    label_map = {c: i for i, c in enumerate(unique)}
    n_classes = len(unique)
    n_folds = max(2, min(3, len(X) // (n_classes * 2)))

    use_smote = variant == "xgboost_smote"
    print(f"  Training {variant} ({len(X)} samples, {n_folds} folds)...", end=" ", flush=True)

    if variant == "random_forest":
        model, cv = train_random_forest_with_cv(
            X_train=X, y_train=y, label_map=label_map,
            n_folds=n_folds, n_estimators=300,
            use_class_weights=True, use_smote=True, random_state=42,
        )
    else:
        model, cv = train_model_with_cv(
            X_train=X, y_train=y, label_map=label_map,
            n_folds=n_folds, n_estimators=100, max_depth=5,
            learning_rate=0.1, use_class_weights=True,
            use_smote=use_smote, random_state=42,
        )

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
CATS = ["green", "yellow", "orange"]


def print_feature_importance(model, features: list, name: str, top_n: int = 10):
    importances = model.feature_importances_
    ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)[:top_n]
    bar_max = 30
    print(f"\n  Feature importance — {name} (top {top_n}):")
    for feat, imp in ranked:
        bar = "█" * int(bar_max * imp / max(importances))
        print(f"    {feat:<38s} {bar:<{bar_max}s}  {imp:.4f}")


def print_confusion_matrix(y_true, y_pred, model_name: str):
    cats = [c for c in CATS if c in set(y_true)]
    n = len(y_true)
    # Build counts grid
    grid = {a: {p: 0 for p in cats} for a in cats}
    for a, p in zip(y_true, y_pred):
        if a in grid and p in grid:
            grid[a][p] += 1

    col_w = 8
    header = f"{'':14s}" + "".join(f"{c:>{col_w}s}" for c in cats) + f"{'recall':>{col_w}s}"
    print(f"\n  Confusion matrix — {model_name}  (N={n})")
    print(f"  {'':14s}" + "  Predicted →")
    print(f"  {header}")
    print(f"  {'Actual ↓':<14s}" + "-" * (col_w * (len(cats) + 1)))
    for actual in cats:
        row_total = sum(grid[actual].values())
        recall = grid[actual].get(actual, 0) / row_total if row_total else 0.0
        row = f"  {CAT_EMOJI.get(actual,'')} {actual:<11s}"
        row += "".join(f"{grid[actual][p]:>{col_w}d}" for p in cats)
        row += f"  {recall:>5.0%}"
        print(row)
    # Per-class precision footer
    print(f"  {'precision':<14s}", end="")
    for pred in cats:
        col_total = sum(grid[a][pred] for a in cats)
        prec = grid[pred].get(pred, 0) / col_total if col_total else 0.0
        print(f"{prec:>{col_w}.0%}", end="")
    print()

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


def print_week_vs_actuals(week_df: pd.DataFrame, models: dict):
    """Day-by-day table comparing actual H2S categories to each model's predictions.

    week_df must have columns: time (datetime), h2s_category, plus all model features.
    models: {display_name: (model, label_map, features)}
    """
    if len(week_df) == 0:
        print("  (no labelled data for this period)")
        return

    model_names = list(models.keys())
    col_w = 14

    # Header
    print(f"  {'Date':<12s}", end="")
    print(f"  {'Actual':<{col_w}s}", end="")
    for name in model_names:
        print(f"  {name:<{col_w}s}", end="")
    print()
    print(f"  {'':─<12s}", end="")
    print(f"  {'🟢/🟡/🟠  acc':>{col_w}s}", end="")
    for _ in model_names:
        print(f"  {'🟢/🟡/🟠  acc':>{col_w}s}", end="")
    print()

    # Precompute predictions for each model over the whole week
    preds_by_model = {}
    for name, (model, label_map, features) in models.items():
        avail = [f for f in features if f in week_df.columns]
        X = week_df[avail].fillna(0.0)
        reverse = {v: k for k, v in label_map.items()}
        raw = model.predict_proba(X.values) if hasattr(model, "predict_proba") else None
        if raw is not None and raw.ndim == 2:
            indices = np.argmax(raw, axis=1)
        else:
            indices = model.predict(X.values)
        preds_by_model[name] = np.array([reverse[int(i)] for i in indices])

    # Day rows
    week_df = week_df.copy()
    week_df["_date"] = week_df["time"].dt.date
    total_correct = {n: 0 for n in model_names}
    total_rows = 0

    for day, grp in week_df.groupby("_date"):
        idx = grp.index
        actual = grp["h2s_category"].values
        g = int((actual == "green").sum())
        y = int((actual == "yellow").sum())
        o = int((actual == "orange").sum())
        actual_str = f"{g}/{y}/{o}"

        print(f"  {str(day):<12s}", end="")
        print(f"  {actual_str:>{col_w}s}", end="")

        for name in model_names:
            day_preds = preds_by_model[name][week_df["_date"].values == day]
            acc = (day_preds == actual).mean()
            total_correct[name] += (day_preds == actual).sum()
            pg = int((day_preds == "green").sum())
            py = int((day_preds == "yellow").sum())
            po = int((day_preds == "orange").sum())
            pred_str = f"{pg}/{py}/{po}  {acc:3.0%}"
            print(f"  {pred_str:>{col_w}s}", end="")
        print()
        total_rows += len(idx)

    # Totals row
    print(f"  {'─'*12}", end="")
    print(f"  {'total acc':>{col_w}s}", end="")
    for name in model_names:
        overall = total_correct[name] / total_rows if total_rows else 0
        print(f"  {overall:>{col_w-1}.1%} ", end="")
    print()

    # Per-class confusion matrices
    y_true_all = week_df["h2s_category"].values
    for name in model_names:
        print_confusion_matrix(y_true_all, preds_by_model[name], name)


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
    parser = argparse.ArgumentParser(description="H2S end-to-end pipeline test")
    parser.add_argument(
        "--no-zero-h2s",
        dest="filter_zero_h2s",
        action="store_true",
        default=True,
        help="Exclude H2S=0/NaN rows from training (default: on)",
    )
    parser.add_argument(
        "--with-zero-h2s",
        dest="filter_zero_h2s",
        action="store_false",
        help="Include H2S=0/NaN rows in training (overrides default)",
    )
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    today = now_utc.date()
    tomorrow = today + timedelta(days=1)
    month_start = today.replace(day=1)

    print("=" * 65)
    print("  H2S PREDICTION PIPELINE — END-TO-END TEST")
    print(f"  Today: {today}   Tomorrow: {tomorrow}")
    print(f"  Production model: {PROD_VARIANT}  (H2S_MODEL_VARIANT)")
    if not args.filter_zero_h2s:
        print("  Mode: --with-zero-h2s (H2S=0/NaN rows included in training)")
    print("=" * 65)

    # --- Step 1: Weather ---
    print("\n[1/3] WEATHER FORECAST")
    forecast = fetch_weather()

    week_start = today - timedelta(days=7)

    this_month = forecast[
        (forecast["date"].dt.date >= month_start) &
        (forecast["date"].dt.date <= today)
    ].copy()
    today_df    = forecast[forecast["date"].dt.date == today].copy()
    tomorrow_df = forecast[forecast["date"].dt.date == tomorrow].copy()
    print(f"  This month: {len(this_month)} hrs | Today: {len(today_df)} hrs | Tomorrow: {len(tomorrow_df)} hrs")

    # --- Step 2: Train ---
    print("\n[2/3] TRAINING MODELS")
    train_df = load_training_data(filter_zero_h2s=args.filter_zero_h2s)
    model_base,  lm_base,  feat_base,  _  = train_variant(train_df, "xgboost_base")
    model_smote, lm_smote, feat_smote, _  = train_variant(train_df, "xgboost_smote")
    model_rf,    lm_rf,    feat_rf,    _  = train_variant(train_df, "random_forest")

    print_feature_importance(model_base,  feat_base,  "xgboost_base")
    print_feature_importance(model_smote, feat_smote, "xgboost_smote")
    print_feature_importance(model_rf,    feat_rf,    "random_forest")

    # Validation slices from training parquet (have actual H2S labels)
    val_df       = train_df[train_df["time"] >= pd.Timestamp(month_start)].copy()
    last_week_df = train_df[
        (train_df["time"].dt.date >= week_start) &
        (train_df["time"].dt.date < today)
    ].copy()

    retrained_models = {
        "xgboost_base":  (model_base,  lm_base,  feat_base),
        "xgboost_smote": (model_smote, lm_smote, feat_smote),
        "random_forest": (model_rf,    lm_rf,    feat_rf),
    }

    # --- Step 3 & 4: Predict + display ---
    print("\n[3/3] PREDICTIONS")
    for label, window in [
        ("This month (so far)", this_month),
        (f"Today ({today})",    today_df),
        (f"Tomorrow ({tomorrow})", tomorrow_df),
    ]:
        if len(window) == 0:
            print(f"\n  {label}: no forecast data in window")
            continue
        print(f"\n  {label}  ({len(window)} hours)")
        print_summary(predict_production(window),                               "production model",         label)
        for name, (m, lm, ft) in retrained_models.items():
            print_summary(predict_fresh(window, m, lm, ft), f"{name} (retrained)", label)

    # Hourly breakdown for today and tomorrow (production model)
    for df_, label_ in [(today_df, f"Today ({today})"), (tomorrow_df, f"Tomorrow ({tomorrow})")]:
        if len(df_) > 0:
            print_hourly(predict_production(df_), f"{label_} — production model")

    # Previous week: day-by-day predictions vs actuals
    print(f"\n  ── Previous week vs actuals ({week_start} → {today - timedelta(days=1)}) ──")
    print_week_vs_actuals(last_week_df, retrained_models)

    # Confusion matrices on labelled month data
    if len(val_df) > 0:
        print(f"\n  ── Month confusion matrices ({month_start} → {val_df['time'].max().date()}, N={len(val_df)}) ──")
        y_true = val_df["h2s_category"].values
        for name, (m, lm, ft) in retrained_models.items():
            avail = [f for f in ft if f in val_df.columns]
            X = val_df[avail].fillna(0.0)
            reverse = {v: k for k, v in lm.items()}
            preds = [reverse[int(i)] for i in np.argmax(m.predict_proba(X.values), axis=1)]
            print_confusion_matrix(y_true, preds, name)

    print("\n✓ Done")
