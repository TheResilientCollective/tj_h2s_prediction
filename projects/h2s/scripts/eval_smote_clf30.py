#!/usr/bin/env python3
"""Walk-forward evaluation of SMOTE on clf_30ppb (Step 2 of the retrain brief).

Trains per-station models on pre-cutoff data and evaluates OOS on post-cutoff
data, comparing the clf_30ppb baseline against a BorderlineSMOTE-oversampled
clf_30ppb. regression / clf_5ppb / clf_10ppb are trained once and shared, so
the only thing that differs between the two arms is the 30 ppb classifier —
isolating the SMOTE effect on the deployed orange call.

Scoring uses the production `classify_risk` at PROB_30_ALERT (0.25), so the
"orange recall" and "FAR" reported here are exactly the acceptance-gate
quantities:  recall@0.25 >= 0.50  AND  FAR <= 0.10.

This exercises the real `train_and_select(..., use_smote_on_minority=...)`
code path added for Step 2, on the Evidence (33-feat) production variant.

Usage:
    cd projects/h2s
    env $(grep -v '^#' ../../.env | xargs) uv run python scripts/eval_smote_clf30.py
    # optional: --cutoff 2026-01-01  --stations "SAN YSIDRO,IB CIVIC CTR,NESTOR - BES"
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from h2s.constants import (  # noqa: E402
    MODEL_FEATURES,
    OBS_DATA_PATH,
    PROB_30_ALERT,
    RISK_TO_3CLASS,
    classify_risk,
)
from h2s.resources.minio import S3Resource  # noqa: E402
from h2s.training.multi_station_trainer import (  # noqa: E402
    prepare_multi_station_features,
    train_and_select,
)

GATE_RECALL = 0.50
GATE_FAR = 0.10


def make_s3():
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ["S3_ADDRESS"],
        S3_PORT=os.environ["S3_PORT"],
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def categorize(v):
    if v < 5:
        return "green"
    if v < 30:
        return "yellow"
    return "orange"


def deployed_orange_metrics(prob_5, prob_10, h2s_pred, prob_30, actual):
    """Orange recall / FAR using the production classify_risk at PROB_30_ALERT."""
    pred_cat = np.array([
        RISK_TO_3CLASS[classify_risk(p5, p10, hp, p30)]
        for p5, p10, hp, p30 in zip(prob_5, prob_10, h2s_pred, prob_30)
    ])
    act_cat = np.array([categorize(a) for a in actual])

    act_orange = act_cat == "orange"
    pred_orange = pred_cat == "orange"
    n_orange = int(act_orange.sum())
    tp = int((pred_orange & act_orange).sum())
    fp = int((pred_orange & ~act_orange).sum())
    fn = int((~pred_orange & act_orange).sum())
    n_non_orange = int((~act_orange).sum())
    recall = tp / n_orange if n_orange else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    far = fp / n_non_orange if n_non_orange else float("nan")
    return {"n_orange": n_orange, "tp": tp, "fp": fp, "fn": fn,
            "recall": recall, "precision": precision, "far": far}


def prob_call_recall(prob_30, actual, p_thresh=0.5):
    """Backtest-style classifier flag recall (p30 > p_thresh vs actual >= 30)."""
    pred = prob_30 > p_thresh
    act = actual >= 30
    tp = int((pred & act).sum())
    recall = tp / int(act.sum()) if act.sum() else float("nan")
    precision = tp / int(pred.sum()) if pred.sum() else float("nan")
    return recall, precision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-01-01",
                    help="Walk-forward cutoff (train < cutoff, eval >= cutoff)")
    ap.add_argument("--stations", default="SAN YSIDRO,IB CIVIC CTR,NESTOR - BES")
    args = ap.parse_args()

    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    stations = [s.strip() for s in args.stations.split(",")]
    features = MODEL_FEATURES  # Evidence (33-feat) production variant

    s3 = make_s3()
    bucket = os.environ.get("PUBLIC_BUCKET", s3.S3_BUCKET)
    print(f"Loading obs data from {bucket}/{OBS_DATA_PATH} ...")
    raw = pd.read_parquet(io.BytesIO(s3.getFile(path=OBS_DATA_PATH, bucket=bucket)))
    df = prepare_multi_station_features(raw)
    print(f"  {len(df):,} feature rows; walk-forward cutoff = {cutoff.date()}\n")

    print("=" * 100)
    print(f"WALK-FORWARD SMOTE EVALUATION (Evidence variant, clf_30ppb)")
    print(f"Deploy scoring at PROB_30_ALERT={PROB_30_ALERT}  |  "
          f"gate: orange recall>={GATE_RECALL} AND FAR<={GATE_FAR}")
    print("=" * 100)

    summary = []
    for station in stations:
        sdf = (df[df["site_name"] == station]
               .sort_values("time").reset_index(drop=True))
        pre = sdf[sdf["time"] < cutoff]
        oos = sdf[sdf["time"] >= cutoff]
        if len(pre) < 100 or len(oos) < 10:
            print(f"\n{station}: insufficient data (pre={len(pre)}, oos={len(oos)}) — skip")
            continue

        Xtr = pre[features].values
        Xte = oos[features].values
        ytr_30 = pre["exceed_30"].values
        n_pos_tr = int(ytr_30.sum())

        # Shared models (regression / clf_5 / clf_10) — train once.
        reg, _, _ = train_and_select(Xtr, Xte, pre["H2S"].values, oos["H2S"].values, "regression")
        clf5, _, _ = train_and_select(Xtr, Xte, pre["exceed_5"].values, oos["exceed_5"].values, "clf_5ppb")
        clf10, _, _ = train_and_select(Xtr, Xte, pre["exceed_10"].values, oos["exceed_10"].values, "clf_10ppb")

        h2s_pred = np.clip(reg.predict(Xte), 0, None)
        prob_5 = clf5.predict_proba(Xte)[:, 1]
        prob_10 = clf10.predict_proba(Xte)[:, 1]
        actual = oos["H2S"].values

        print(f"\n{'='*100}\n{station}  "
              f"(pre-cutoff rows={len(pre):,}, 30ppb pos={n_pos_tr} | OOS rows={len(oos):,})")
        print("-" * 100)
        print(f"{'arm':<10} {'clf30 choice':<28} {'AUC':<7} {'OrR@0.25':<9} {'OrP@0.25':<9} "
              f"{'FAR':<7} {'TP/FP/FN':<12} {'pcall_R@0.5':<11} {'GATE':<5}")

        row = {"station": station, "n_pos_tr": n_pos_tr, "n_orange_oos": None}
        for arm, use_smote in [("baseline", False), ("smote", True)]:
            clf30, choice, info = train_and_select(
                Xtr, Xte, ytr_30, oos["exceed_30"].values, "clf_30ppb",
                use_smote_on_minority=use_smote,
            )
            prob_30 = clf30.predict_proba(Xte)[:, 1]
            dm = deployed_orange_metrics(prob_5, prob_10, h2s_pred, prob_30, actual)
            pcall_r, _ = prob_call_recall(prob_30, actual, 0.5)
            auc = info["RF"]["AUC"] if choice == "RandomForest" else info.get("XGB", {}).get("AUC", info["RF"]["AUC"])
            # use selection_value as the headline AUC (selected model)
            auc = info.get("selection_value_xgb") if choice == "XGBoost" else info.get("selection_value_rf")
            passes = (dm["recall"] >= GATE_RECALL) and (dm["far"] <= GATE_FAR)
            row["n_orange_oos"] = dm["n_orange"]
            print(f"{arm:<10} {choice:<28} {auc or 0:<7.3f} {dm['recall']:<9.3f} {dm['precision']:<9.3f} "
                  f"{dm['far']:<7.3f} {dm['tp']}/{dm['fp']}/{dm['fn']:<8} {pcall_r:<11.3f} "
                  f"{'PASS' if passes else 'FAIL':<5}")
            row[arm] = {"recall": dm["recall"], "far": dm["far"], "passes": passes,
                        "smote_applied": info["smote_applied"]}
        summary.append(row)

    print("\n" + "=" * 100)
    print("SUMMARY (deploy scoring @ PROB_30_ALERT=0.25)")
    print("=" * 100)
    for r in summary:
        b, s = r["baseline"], r["smote"]
        print(f"  {r['station']:<14} OOS orange={r['n_orange_oos']:<4} | "
              f"baseline recall={b['recall']:.3f} FAR={b['far']:.3f} ({'PASS' if b['passes'] else 'FAIL'})  ->  "
              f"smote recall={s['recall']:.3f} FAR={s['far']:.3f} ({'PASS' if s['passes'] else 'FAIL'})")


if __name__ == "__main__":
    main()
