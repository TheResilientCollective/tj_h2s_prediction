#!/usr/bin/env python3
"""Sweep PROB_30_ALERT thresholds and report metrics tradeoffs.

Generates predictions ONCE then evaluates each threshold against actuals.
Reports BA, orange recall, FAR, and confusion matrix per threshold.

Usage:
    cd projects/h2s
    env $(grep -v '^#' ../../.env | xargs) uv run python scripts/sweep_prob_30_threshold.py
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from h2s.resources.minio import S3Resource
from h2s.constants import (
    STATIONS, STATION_MODELS_S3_BASE, MODEL_FEATURES, OBS_DATA_PATH,
    H2S_CLASS_NAMES, H2S_CLASS_TO_INT,
    PROB_5_CAUTION, PROB_5_ALERT, PROB_10_ALERT,
    H2S_THRESHOLD_LOW, H2S_THRESHOLD_MED, H2S_THRESHOLD_HIGH,
    RISK_GREEN, RISK_YELLOW_LOW, RISK_YELLOW_HIGH, RISK_ORANGE,
    RISK_TO_3CLASS,
)
from h2s.defs.h2s_daily_pipeline import _engineer_forecast_features
from h2s.training.validation import calculate_metrics, calculate_false_alarm_rate


def classify_risk_with_threshold(prob_5, prob_10, h2s_pred, prob_30, prob_30_alert):
    """Apply classify_risk logic with a configurable PROB_30_ALERT."""
    orange_by_prob = (prob_30 > prob_30_alert) if prob_30 > 0.0 else (prob_10 > PROB_10_ALERT)
    if orange_by_prob or h2s_pred > H2S_THRESHOLD_HIGH:
        return RISK_ORANGE
    elif prob_5 > PROB_5_ALERT or h2s_pred > H2S_THRESHOLD_MED:
        return RISK_YELLOW_HIGH
    elif prob_5 > PROB_5_CAUTION or h2s_pred > H2S_THRESHOLD_LOW:
        return RISK_YELLOW_LOW
    return RISK_GREEN


def categorize_h2s(value):
    if pd.isna(value):
        return None
    if value < 5:
        return "green"
    if value < 30:
        return "yellow"
    return "orange"


def make_s3():
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ["S3_ADDRESS"],
        S3_PORT=os.environ["S3_PORT"],
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def load_station_models(s3):
    artifacts = {}
    tasks = ["regression", "clf_5ppb", "clf_10ppb", "clf_30ppb"]

    for site_name, info in STATIONS.items():
        station_key = info["key"]
        base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"
        station_models = {}

        for task in tasks:
            try:
                model_bytes = s3.getFile(path=f"{base_path}/{task}_evidence.pkl", bucket=s3.S3_BUCKET)
                station_models[task] = pickle.loads(model_bytes)
            except Exception:
                pass

        try:
            feat_bytes = s3.getFile(path=f"{base_path}/features_evidence.json", bucket=s3.S3_BUCKET)
            station_models["_feature_cols"] = json.loads(feat_bytes.decode("utf-8"))
        except Exception:
            station_models["_feature_cols"] = MODEL_FEATURES

        artifacts[site_name] = station_models

    return artifacts


def load_observations(s3, start_date, end_date):
    """Load real measured observations from production parquet."""
    public_bucket = os.environ.get("PUBLIC_BUCKET", s3.S3_BUCKET)
    obs_url = s3.publicUrl(path=OBS_DATA_PATH, bucket=public_bucket)
    df = pd.read_parquet(obs_url)
    df = df[df["h2s_measured"] == True].copy()  # noqa: E712
    df = df[df["H2S"] <= 500].copy()
    df["H2S"] = df["H2S"].clip(lower=0)

    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("UTC").dt.round("h")
    df = df[(df["time"] >= start_date) & (df["time"] < end_date)].copy()

    return df.sort_values("time").reset_index(drop=True)


def generate_predictions(station_models, all_obs):
    """Generate raw probabilities and h2s_pred for all stations × times once."""
    results = []

    for site_name, info in STATIONS.items():
        if site_name not in station_models:
            continue
        station_key = info["key"]
        models = station_models[site_name]
        if "regression" not in models:
            continue

        feature_cols = models["_feature_cols"]
        site_obs = all_obs[all_obs["site_name"].isin([site_name, station_key])].copy()
        if len(site_obs) == 0:
            continue

        sfc = site_obs.copy().reset_index(drop=True)
        sfc["site_name"] = site_name

        # Preserve real lag features
        real_lag_cols = ["h2s_lag_1h", "h2s_lag_3h", "h2s_lag_6h",
                          "h2s_rolling_6h", "h2s_rolling_24h"]
        preserved = {col: sfc[col].copy() for col in real_lag_cols if col in sfc.columns}

        last_state = {"h2s": 0, "h2s_6h": 0, "h2s_24h": 0, "flow": 2.0, "flow_24h": 2.0}
        try:
            sfc = _engineer_forecast_features(sfc, last_state)
        except Exception as e:
            print(f"  Skipping {station_key}: {e}")
            continue

        # Restore real lag features
        for col, real_vals in preserved.items():
            sfc[col] = real_vals.values

        for col in feature_cols:
            if col not in sfc.columns:
                sfc[col] = 0.0

        X = sfc[feature_cols].fillna(0).values

        reg = models["regression"]
        clf5 = models.get("clf_5ppb")
        clf10 = models.get("clf_10ppb")
        clf30 = models.get("clf_30ppb")

        h2s_pred = np.clip(reg.predict(X), 0, None)
        prob_5 = clf5.predict_proba(X)[:, 1] if clf5 else np.zeros(len(X))
        prob_10 = clf10.predict_proba(X)[:, 1] if clf10 else np.zeros(len(X))

        # Experiment C: use clf_30ppb for ALL stations (NESTOR-only guard removed)
        # so we can probe whether a lower threshold recovers recall on the
        # high-AUC SAN_YSIDRO / IB_CIVIC_CTR models.
        prob_30 = clf30.predict_proba(X)[:, 1] if clf30 else np.zeros(len(X))

        for i in range(len(sfc)):
            results.append({
                "time": sfc["time"].iloc[i],
                "station": site_name,
                "site_key": station_key,
                "h2s_pred": float(h2s_pred[i]),
                "prob_5": float(prob_5[i]),
                "prob_10": float(prob_10[i]),
                "prob_30": float(prob_30[i]),
                "H2S_actual": float(site_obs["H2S"].iloc[i]),
            })

    return pd.DataFrame(results)


def evaluate_threshold(preds_df, prob_30_alert):
    """Apply classify_risk with given threshold, return metrics."""
    df = preds_df.copy()
    df["risk"] = df.apply(
        lambda r: classify_risk_with_threshold(
            r["prob_5"], r["prob_10"], r["h2s_pred"], r["prob_30"], prob_30_alert
        ),
        axis=1,
    )
    df["predicted_category"] = df["risk"].map(RISK_TO_3CLASS)
    df["actual_category"] = df["H2S_actual"].apply(categorize_h2s)
    df = df[df["actual_category"].notna()].copy()

    y_true = df["actual_category"].map(H2S_CLASS_TO_INT)
    y_pred = df["predicted_category"].map(H2S_CLASS_TO_INT)

    metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, class_names=H2S_CLASS_NAMES)
    far = calculate_false_alarm_rate(
        y_true=(df["actual_category"] == "orange").astype(int),
        y_pred=(df["predicted_category"] == "orange").astype(int),
        positive_class=1,
    )

    cm = np.array(metrics["confusion_matrix"])
    return {
        "threshold": prob_30_alert,
        "n_matched": len(df),
        "balanced_accuracy": metrics["balanced_accuracy"],
        "orange_recall": metrics.get("recall_orange", 0),
        "orange_precision": metrics.get("precision_orange", 0),
        "yellow_recall": metrics.get("recall_yellow", 0),
        "green_recall": metrics.get("recall_green", 0),
        "false_alarm_rate": far,
        "confusion_matrix": cm,
    }


# Acceptance gate (per the retrain brief): usable orange detection at deploy.
GATE_RECALL = 0.50
GATE_FAR = 0.10
DEPLOY_THRESHOLD = 0.25


def sweep_station(preds_df, thresholds, label):
    """Run the threshold sweep for one already-filtered prediction frame.

    Prints a per-threshold table and returns the list of result dicts.
    """
    n_orange = int((preds_df["H2S_actual"] >= 30).sum())
    n_total = len(preds_df)
    print("\n" + "=" * 100)
    print(f"STATION: {label}   (n={n_total} matched, orange actuals={n_orange})")
    print("=" * 100)

    if n_total == 0 or n_orange == 0:
        print("  No data / no orange actuals for this station — cannot evaluate recall.")
        return []

    print(f"{'Thresh':<8} {'BA':<8} {'OrangeR':<10} {'OrangeP':<10} {'YellowR':<10} "
          f"{'GreenR':<10} {'FAR':<8} {'TP/FP/FN':<15} {'GATE':<6}")
    print("-" * 100)

    results = []
    for thresh in thresholds:
        r = evaluate_threshold(preds_df, thresh)
        cm = r["confusion_matrix"]
        tp = cm[2, 2]
        fp = cm[0, 2] + cm[1, 2]
        fn = cm[2, 0] + cm[2, 1]
        passes = r["orange_recall"] >= GATE_RECALL and r["false_alarm_rate"] <= GATE_FAR
        gate = "PASS" if passes else ""
        marker = " <-- deploy" if abs(thresh - DEPLOY_THRESHOLD) < 1e-9 else ""
        print(f"{thresh:<8.2f} {r['balanced_accuracy']:<8.3f} {r['orange_recall']:<10.3f} "
              f"{r['orange_precision']:<10.3f} {r['yellow_recall']:<10.3f} {r['green_recall']:<10.3f} "
              f"{r['false_alarm_rate']:<8.3f} {tp}/{fp}/{fn:<11} {gate:<6}{marker}")
        r["_passes_gate"] = passes
        results.append(r)
    return results


def summarize_station(label, results):
    """Print the gate verdict for one station and return a one-line summary dict."""
    if not results:
        print(f"  {label}: NO DATA")
        return {"station": label, "verdict": "NO DATA"}

    at_deploy = next((r for r in results if abs(r["threshold"] - DEPLOY_THRESHOLD) < 1e-9), None)
    passing = [r for r in results if r["_passes_gate"]]

    # The formal acceptance gate is at the *single global* deploy threshold (0.25).
    deploy_verdict = "PASS" if (at_deploy and at_deploy["_passes_gate"]) else "FAIL"

    print(f"\n  {label}:")
    if at_deploy:
        print(f"    @ deploy threshold {DEPLOY_THRESHOLD}: recall={at_deploy['orange_recall']:.3f} "
              f"FAR={at_deploy['false_alarm_rate']:.3f}  -> ACCEPTANCE GATE {deploy_verdict}")
    if passing:
        # A lower per-station threshold may rescue recall (Experiment C escape hatch,
        # but per-station thresholds are out-of-scope for deployment).
        best = max(passing, key=lambda r: r["orange_recall"])
        passing_thresholds = ", ".join("%.2f" % r["threshold"] for r in passing)
        print(f"    gate passes at thresholds: {passing_thresholds}  (requires per-station threshold)")
        print(f"    best gate-passing: threshold={best['threshold']:.2f} "
              f"recall={best['orange_recall']:.3f} FAR={best['false_alarm_rate']:.3f}")
        rescued = True
    else:
        best_recall = max(results, key=lambda r: r["orange_recall"])
        print(f"    no threshold satisfies recall>={GATE_RECALL} AND FAR<={GATE_FAR}.")
        print(f"    best recall seen: {best_recall['orange_recall']:.3f} "
              f"@ threshold={best_recall['threshold']:.2f} (FAR={best_recall['false_alarm_rate']:.3f})")
        rescued = False
    return {
        "station": label,
        "deploy_verdict": deploy_verdict,
        "rescued_by_lower_threshold": rescued,
        "deploy_recall": at_deploy["orange_recall"] if at_deploy else None,
        "deploy_far": at_deploy["false_alarm_rate"] if at_deploy else None,
    }


def main():
    s3 = make_s3()
    print(f"Loading models from {s3.S3_BUCKET}...")
    station_models = load_station_models(s3)
    print(f"  Loaded {len(station_models)} stations\n")

    print("Loading observations (May 1 - June 19, 2026)...")
    all_obs = load_observations(s3, "2026-05-01", "2026-06-19")
    print(f"  {len(all_obs)} real measured observations\n")

    print("Generating predictions (once)...")
    preds_df = generate_predictions(station_models, all_obs)
    print(f"  {len(preds_df)} predictions\n")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]

    # Per-station sweeps (Experiment C). Stations of interest first.
    station_order = ["SAN_YSIDRO", "IB_CIVIC_CTR", "NESTOR__BES"]
    present = list(preds_df["station"].unique())
    ordered = [s for s in station_order if s in present] + \
              [s for s in present if s not in station_order]

    summaries = []
    for station in ordered:
        sub = preds_df[preds_df["station"] == station].copy()
        results = sweep_station(sub, thresholds, station)
        summaries.append((station, results))

    # Also the aggregate (all stations) for reference / parity with old script.
    agg_results = sweep_station(preds_df.copy(), thresholds, "ALL STATIONS (aggregate)")

    # ---- Verdicts against the acceptance gate ----
    print("\n" + "=" * 100)
    print(f"ACCEPTANCE GATE: orange recall >= {GATE_RECALL} AND FAR <= {GATE_FAR} "
          f"(deploy threshold PROB_30_ALERT = {DEPLOY_THRESHOLD})")
    print("=" * 100)
    verdicts = [summarize_station(st, res) for st, res in summaries]
    summarize_station("ALL STATIONS (aggregate)", agg_results)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for v in verdicts:
        dr = f"{v['deploy_recall']:.3f}" if v.get("deploy_recall") is not None else "n/a"
        df_ = f"{v['deploy_far']:.3f}" if v.get("deploy_far") is not None else "n/a"
        rescue = "lower-threshold rescue available" if v.get("rescued_by_lower_threshold") else "no rescue"
        print(f"  {v['station']:<16} gate@0.25={v['deploy_verdict']:<5} "
              f"(recall={dr} FAR={df_})  [{rescue}]")


if __name__ == "__main__":
    main()
