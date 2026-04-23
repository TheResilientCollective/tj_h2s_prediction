#!/usr/bin/env python3
"""Backfill validation metrics from S3 predictions or hindcast from observations.

Three modes:
  1. **Forecast mode** (default): Load existing hourly predictions from S3,
     compare against observations, write metrics.json.
  2. **Hindcast mode** (--hindcast): Load the production hourly model, run
     predictions against historical observations, compare against actual H2S.
     NESTOR-BES only.
  3. **Daily station mode** (--daily-station): Load per-station models (all 3
     stations), run against historical observations, write multi-site
     daily_station/metrics.json. This populates accuracy reports for all stations.

Usage:
    cd projects/h2s
    source .env

    # Forecast-based validation (uses existing predictions on S3)
    uv run python scripts/backfill_validation.py

    # Hindcast — hourly model against historical observations (NESTOR only)
    uv run python scripts/backfill_validation.py --hindcast

    # Daily station — per-station models, all 3 stations
    uv run python scripts/backfill_validation.py --daily-station
    uv run python scripts/backfill_validation.py --daily-station --start 2025-01-01 --end 2026-04-23

    # Dry run / overwrite
    uv run python scripts/backfill_validation.py --daily-station --dry-run
    uv run python scripts/backfill_validation.py --daily-station --overwrite
"""

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from h2s.constants import (
    HOURLY_PREDICTIONS_PATH,
    MODEL_PATH,
    STATION_MODELS_S3_BASE,
    STATIONS,
    VALIDATION_PATH,
    H2S_CLASS_NAMES,
    H2S_CLASS_TO_INT,
    RISK_TO_3CLASS,
    classify_risk,
    MODEL_FEATURES,
)
from h2s.resources.minio import S3Resource
from h2s.training.validation import calculate_metrics, calculate_false_alarm_rate

OBS_DATA_PATH = "latest/tijuana/forecast_data/modeldata_h2s.parquet"


def make_s3() -> S3Resource:
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
        S3_PORT=os.environ.get("S3_PORT", "443"),
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def metrics_exist(s3: S3Resource, date_str: str) -> bool:
    """Check if usable metrics.json already exists for this date."""
    import urllib.request

    url = s3.publicUrl(path=f"{VALIDATION_PATH}/{date_str}/metrics.json")
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        sites = data.get("sites", {})
        if sites and any(s.get("confusion_matrix") for s in sites.values()):
            return True
        if data.get("confusion_matrix") and data.get("n_matched", 0) > 0:
            return True
    except Exception:
        pass
    return False


def get_predictions(s3: S3Resource, date_str: str) -> pd.DataFrame | None:
    """Load hourly predictions for a date from the hive-partitioned path."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    y, m, d = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")

    dfs = []
    for hour in ["00", "06", "12", "18"]:
        path = f"{HOURLY_PREDICTIONS_PATH}/model=nestor_xgboost/year={y}/month={m}/day={d}/hour={hour}/h2s_predictions.csv"
        try:
            url = s3.publicUrl(path=path)
            df = pd.read_csv(url)
            df["run_hour"] = hour
            dfs.append(df)
        except Exception:
            pass

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    if "time" not in df.columns and "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"])

    df["time"] = pd.to_datetime(df["time"])
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")
    else:
        df["time"] = df["time"].dt.tz_convert("UTC")
    df["time"] = df["time"].dt.round("h")

    # Keep only predictions whose timestamps fall on the requested date
    df = df[df["time"].dt.strftime("%Y-%m-%d") == date_str].copy()
    if len(df) == 0:
        return None

    # Deduplicate: keep latest run for each hour
    df = df.sort_values("run_hour", ascending=False).drop_duplicates(subset=["time"], keep="first")
    return df


def categorize_h2s(value):
    if pd.isna(value):
        return None
    if value < 5:
        return "green"
    if value < 30:
        return "yellow"
    return "orange"


def compute_and_write_metrics(
    s3: S3Resource,
    date_str: str,
    preds: pd.DataFrame,
    obs: pd.DataFrame,
    pipeline_label: str = "hourly",
) -> bool:
    """Compute validation metrics and write to S3. Returns True on success."""
    merged = preds.merge(obs[["time", "H2S"]], on="time", how="inner")
    merged["actual_category"] = merged["H2S"].apply(categorize_h2s)
    merged = merged[merged["actual_category"].notna()].copy()

    if len(merged) == 0:
        print(f"    0 matched observations — skipping")
        return False

    y_true = merged["actual_category"].map(H2S_CLASS_TO_INT)
    y_pred = merged["predicted_category"].map(H2S_CLASS_TO_INT)

    metrics_dict = calculate_metrics(y_true=y_true, y_pred=y_pred, class_names=H2S_CLASS_NAMES)
    far = calculate_false_alarm_rate(
        y_true=(merged["actual_category"] == "orange").astype(int),
        y_pred=(merged["predicted_category"] == "orange").astype(int),
        positive_class=1,
    )

    site_metrics = {
        "n_predictions": len(preds),
        "n_matched_observations": len(merged),
        "match_rate": float(len(merged) / len(preds)),
        "balanced_accuracy": float(metrics_dict["balanced_accuracy"]),
        "false_alarm_rate": float(far),
        "confusion_matrix": metrics_dict["confusion_matrix"],
        "class_metrics": {
            cls: {
                "precision": float(metrics_dict.get(f"precision_{cls}", 0.0)),
                "recall": float(metrics_dict.get(f"recall_{cls}", 0.0)),
                "f1_score": float(metrics_dict.get(f"f1_{cls}", 0.0)),
            }
            for cls in ["green", "yellow", "orange"]
        },
        "regression_metrics": None,
    }

    payload = {
        "schema_version": 2,
        "date": date_str,
        "pipeline": pipeline_label,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sites": {"NESTOR__BES": site_metrics},
        "horizon_metrics": None,
    }

    body = json.dumps(payload, indent=2).encode("utf-8")
    base = f"{VALIDATION_PATH}/{date_str}"
    for path in [f"{base}/metrics.json", f"{base}/{pipeline_label}/metrics.json"]:
        s3.putFile(body, path, bucket=s3.S3_BUCKET, content_type="application/json")

    ba = site_metrics["balanced_accuracy"]
    print(f"    matched={len(merged)}  ba={ba:.3f}  far={far:.3f}")
    return True


# ---------------------------------------------------------------------------
# Hindcast: run model against historical observations
# ---------------------------------------------------------------------------

def load_predictor(s3: S3Resource):
    """Load the production H2SPredictor from S3."""
    from h2s.predictor.h2s_predictor import H2SPredictor

    return H2SPredictor.from_s3(
        s3,
        f"{MODEL_PATH}/nestor_xgboost_weighted_model.json",
        f"{MODEL_PATH}/nestor_preprocessing_info.json",
        model_name="nestor_xgboost",
    )


def hindcast_predictions(predictor, day_obs: pd.DataFrame) -> pd.DataFrame | None:
    """Run the model against historical observations to generate predictions.

    Keeps H2S in the dataframe during preprocessing so that lag features
    (h2s_lag_1h, etc.) are computed from actual measurements rather than zeros.
    This gives a realistic picture of model accuracy with observed inputs.
    """
    if len(day_obs) == 0 or day_obs["H2S"].isna().all():
        return None

    df = day_obs.copy()

    # Preprocess — H2S column is present so lags will be computed from actuals
    preprocessed = predictor.preprocess_data(df)

    # Predict
    result = predictor.predict(preprocessed)
    return result


# ---------------------------------------------------------------------------
# Daily station hindcast: per-station models against historical observations
# ---------------------------------------------------------------------------

def load_station_models(s3: S3Resource) -> dict:
    """Load per-station models (regression + clf_5ppb + clf_10ppb) from S3."""
    import pickle

    artifacts = {}
    tasks = ["regression", "clf_5ppb", "clf_10ppb"]

    for site_name, info in STATIONS.items():
        station_key = info["key"]
        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        station_models = {}

        for task in tasks:
            s3_path = f"{base_path}/{task}.pkl"
            try:
                model_bytes = s3.getFile(path=s3_path, bucket=s3.S3_BUCKET)
                station_models[task] = pickle.loads(model_bytes)
            except Exception as e:
                print(f"    Warning: {station_key}/{task}: {e}")

        # Load feature list
        try:
            feat_bytes = s3.getFile(path=f"{base_path}/features.json", bucket=s3.S3_BUCKET)
            station_models["_feature_cols"] = json.loads(feat_bytes.decode("utf-8"))
        except Exception:
            pass

        if station_models:
            artifacts[site_name] = station_models

    return artifacts


def daily_station_hindcast(
    station_models: dict,
    day_obs: pd.DataFrame,
    all_obs: pd.DataFrame,
) -> pd.DataFrame | None:
    """Run per-station models against historical observations.

    Returns DataFrame with columns: time, station, site_key, h2s_pred, risk, predicted_category.
    """
    from h2s.defs.h2s_daily_pipeline import _engineer_forecast_features

    if len(day_obs) == 0:
        return None

    results = []
    for site_name, info in STATIONS.items():
        if site_name not in station_models:
            continue

        station_key = info["key"]
        models = station_models[site_name]
        if "regression" not in models:
            continue

        feature_cols = models.get("_feature_cols", MODEL_FEATURES)

        # Get site observations for this day
        site_obs = day_obs[day_obs["site_name"].isin([site_name, station_key])].copy()
        if len(site_obs) == 0:
            continue

        # Get last known state from all_obs for lag initialization
        all_site = all_obs[all_obs["site_name"].isin([site_name, station_key])].sort_values("time")
        before_day = all_site[all_site["time"] < site_obs["time"].min()]
        if len(before_day) > 0:
            last_state = {
                "h2s": float(before_day.iloc[-1]["H2S"]),
                "h2s_6h": float(before_day.tail(6)["H2S"].mean()),
                "h2s_24h": float(before_day.tail(24)["H2S"].mean()),
                "flow": float(before_day.iloc[-1].get("flow_rate_cms", 2.0) or 2.0),
                "flow_24h": float(before_day.tail(24).get("flow_rate_cms", pd.Series([2.0])).mean()),
            }
        else:
            last_state = {"h2s": 0, "h2s_6h": 0, "h2s_24h": 0, "flow": 2.0, "flow_24h": 2.0}

        # Engineer features (reuse daily pipeline logic)
        sfc = site_obs.copy().reset_index(drop=True)
        sfc["site_name"] = site_name

        import numpy as np

        try:
            sfc = _engineer_forecast_features(sfc, last_state)
        except Exception:
            continue

        # Fill missing features with 0
        for col in feature_cols:
            if col not in sfc.columns:
                sfc[col] = 0.0

        X = sfc[feature_cols].fillna(0).values

        reg = models["regression"]
        clf5 = models.get("clf_5ppb")
        clf10 = models.get("clf_10ppb")

        h2s_pred = np.clip(reg.predict(X), 0, None)
        prob_5 = clf5.predict_proba(X)[:, 1] * 100 if clf5 else np.zeros(len(X))
        prob_10 = clf10.predict_proba(X)[:, 1] * 100 if clf10 else np.zeros(len(X))

        for i in range(len(sfc)):
            risk = classify_risk(prob_5[i] / 100, prob_10[i] / 100, h2s_pred[i])
            results.append({
                "time": sfc["time"].iloc[i],
                "station": site_name,
                "site_key": station_key,
                "h2s_pred": round(float(h2s_pred[i]), 1),
                "risk": risk,
                "predicted_category": RISK_TO_3CLASS[risk],
            })

    if not results:
        return None
    return pd.DataFrame(results)


def compute_and_write_station_metrics(
    s3: S3Resource,
    date_str: str,
    preds: pd.DataFrame,
    obs: pd.DataFrame,
) -> bool:
    """Compute per-station validation metrics and write to S3. Returns True on success."""
    sites: dict = {}

    for site_name, info in STATIONS.items():
        station_key = info["key"]
        site_preds = preds[preds["site_key"] == station_key].copy()
        site_obs = obs[obs["site_name"].isin([site_name, station_key])].copy()

        if len(site_preds) == 0:
            continue

        merged = site_preds.merge(site_obs[["time", "H2S"]], on="time", how="inner")
        merged["actual_category"] = merged["H2S"].apply(categorize_h2s)
        merged = merged[merged["actual_category"].notna()].copy()

        if len(merged) == 0:
            continue

        y_true = merged["actual_category"].map(H2S_CLASS_TO_INT)
        y_pred = merged["predicted_category"].map(H2S_CLASS_TO_INT)

        metrics_dict = calculate_metrics(y_true=y_true, y_pred=y_pred, class_names=H2S_CLASS_NAMES)
        far = calculate_false_alarm_rate(
            y_true=(merged["actual_category"] == "orange").astype(int),
            y_pred=(merged["predicted_category"] == "orange").astype(int),
            positive_class=1,
        )

        sites[station_key] = {
            "n_predictions": len(site_preds),
            "n_matched_observations": len(merged),
            "match_rate": float(len(merged) / len(site_preds)),
            "balanced_accuracy": float(metrics_dict["balanced_accuracy"]),
            "false_alarm_rate": float(far),
            "confusion_matrix": metrics_dict["confusion_matrix"],
            "class_metrics": {
                cls: {
                    "precision": float(metrics_dict.get(f"precision_{cls}", 0.0)),
                    "recall": float(metrics_dict.get(f"recall_{cls}", 0.0)),
                    "f1_score": float(metrics_dict.get(f"f1_{cls}", 0.0)),
                }
                for cls in ["green", "yellow", "orange"]
            },
            "regression_metrics": None,
        }
        ba = sites[station_key]["balanced_accuracy"]
        n = len(merged)
        print(f"      {station_key}: matched={n}  ba={ba:.3f}  far={far:.3f}")

    if not sites:
        print(f"    No stations had matched observations — skipping")
        return False

    payload = {
        "schema_version": 2,
        "date": date_str,
        "pipeline": "daily_station",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sites": sites,
        "horizon_metrics": None,
    }

    body = json.dumps(payload, indent=2).encode("utf-8")
    path = f"{VALIDATION_PATH}/{date_str}/daily_station/metrics.json"
    s3.putFile(body, path, bucket=s3.S3_BUCKET, content_type="application/json")
    print(f"    Wrote {len(sites)} stations to daily_station/metrics.json")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_all_observations(s3: S3Resource, station_filter: list[str] | None = None) -> pd.DataFrame:
    """Load and normalize the full observation parquet.

    If *station_filter* is provided, only keep those site_names.
    Otherwise returns all 3 stations.
    """
    public_bucket = os.environ.get("PUBLIC_BUCKET", s3.S3_BUCKET)
    obs_url = s3.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
    all_obs = pd.read_parquet(obs_url)

    # Keep known stations
    known_names = set()
    for name, info in STATIONS.items():
        known_names.add(name)
        known_names.add(info["key"])
    all_obs = all_obs[all_obs["site_name"].isin(known_names)].copy()

    if station_filter:
        all_obs = all_obs[all_obs["site_name"].isin(station_filter)].copy()

    all_obs["time"] = pd.to_datetime(all_obs["time"])
    if all_obs["time"].dt.tz is None:
        all_obs["time"] = all_obs["time"].dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC")
    else:
        all_obs["time"] = all_obs["time"].dt.tz_convert("UTC")
    all_obs["time"] = all_obs["time"].dt.round("h")

    if "H2S" not in all_obs.columns:
        h2s_cols = [c for c in all_obs.columns if c.upper() == "H2S" or "h2s" in c.lower()]
        if h2s_cols:
            all_obs["H2S"] = all_obs[h2s_cols[0]]

    return all_obs


def main():
    parser = argparse.ArgumentParser(
        description="Backfill validation metrics (forecast, hindcast, or daily-station mode)"
    )
    parser.add_argument("--start", default="2026-03-01")
    parser.add_argument("--end", default="2026-04-23")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metrics")
    parser.add_argument(
        "--hindcast",
        action="store_true",
        help="Run hourly model against historical observations (NESTOR only)",
    )
    parser.add_argument(
        "--daily-station",
        action="store_true",
        help="Run per-station models against historical observations (all 3 stations)",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    s3 = make_s3()

    # Load observations once (all stations for daily-station, NESTOR for hourly)
    print("Loading observation data...")
    if args.daily_station:
        all_obs = load_all_observations(s3)
        stations_loaded = list(all_obs["site_name"].unique())
        print(f"  {len(all_obs)} observations loaded ({len(stations_loaded)} stations)")
    else:
        all_obs = load_all_observations(s3, station_filter=["NESTOR__BES", "NESTOR - BES"])
        print(f"  {len(all_obs)} NESTOR-BES observations loaded")
    print(f"  Date range: {all_obs['time'].min()} to {all_obs['time'].max()}")

    # Load models
    predictor = None
    station_models = None

    if args.hindcast:
        print("Loading production hourly model from S3...")
        predictor = load_predictor(s3)
        print(f"  Model loaded: {len(predictor.feature_cols)} features")

    if args.daily_station:
        print("Loading per-station models from S3...")
        station_models = load_station_models(s3)
        print(f"  Loaded models for {len(station_models)} stations: {list(station_models.keys())}")

    # Scan dates and collect work
    to_backfill: list[tuple[str, str]] = []  # (date_str, mode)
    cur = start
    while cur <= end:
        ds = cur.isoformat()

        day_obs = all_obs[all_obs["time"].dt.strftime("%Y-%m-%d") == ds]
        has_h2s = day_obs["H2S"].notna().sum() > 0 if len(day_obs) > 0 else False

        if not has_h2s:
            cur += timedelta(days=1)
            continue

        if args.daily_station:
            # Check daily_station metrics existence
            if not args.overwrite:
                import urllib.request
                url = s3.publicUrl(path=f"{VALIDATION_PATH}/{ds}/daily_station/metrics.json")
                try:
                    with urllib.request.urlopen(url, timeout=5):
                        cur += timedelta(days=1)
                        continue
                except Exception:
                    pass
            to_backfill.append((ds, "daily_station"))
        elif args.hindcast:
            if not args.overwrite and metrics_exist(s3, ds):
                cur += timedelta(days=1)
                continue
            to_backfill.append((ds, "hindcast"))
        else:
            if not args.overwrite and metrics_exist(s3, ds):
                cur += timedelta(days=1)
                continue
            preds = get_predictions(s3, ds)
            if preds is not None:
                to_backfill.append((ds, "forecast"))
        cur += timedelta(days=1)

    mode_label = "daily_station" if args.daily_station else ("hindcast" if args.hindcast else "forecast")
    print(f"\n{len(to_backfill)} dates to backfill ({mode_label} mode)")

    if args.dry_run:
        for ds, mode in to_backfill:
            day_obs = all_obs[all_obs["time"].dt.strftime("%Y-%m-%d") == ds]
            n_h2s = day_obs["H2S"].notna().sum()
            stations = day_obs["site_name"].nunique() if "site_name" in day_obs.columns else 1
            print(f"  {ds}  {mode}  obs_with_h2s={n_h2s}  stations={stations}")
        return

    # Backfill
    success = 0
    for ds, mode in to_backfill:
        day_obs = all_obs[all_obs["time"].dt.strftime("%Y-%m-%d") == ds].copy()
        print(f"  {ds} ({mode}):")

        if mode == "daily_station":
            preds = daily_station_hindcast(station_models, day_obs, all_obs)
            if preds is None:
                print(f"    No predictions generated — skipping")
                continue
            if compute_and_write_station_metrics(s3, ds, preds, day_obs):
                success += 1
        elif mode == "hindcast":
            preds = hindcast_predictions(predictor, day_obs)
            if preds is None:
                print(f"    No H2S data for hindcast — skipping")
                continue
            if compute_and_write_metrics(s3, ds, preds, day_obs, pipeline_label="hindcast"):
                success += 1
        else:
            preds = get_predictions(s3, ds)
            if preds is None:
                print(f"    No predictions on S3 — skipping")
                continue
            if compute_and_write_metrics(s3, ds, preds, day_obs, pipeline_label="hourly"):
                success += 1

    print(f"\nBackfilled {success}/{len(to_backfill)} dates.")
    if success > 0:
        print("\nNow re-run the accuracy reports backfill:")
        print("  uv run python scripts/backfill_accuracy_reports.py")


if __name__ == "__main__":
    main()
