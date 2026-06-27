#!/usr/bin/env python3
"""Experiment: Pooled vs per-station clf_30ppb (orange / >=30 ppb).

Question: SAN YSIDRO (40 train positives) and IB CIVIC CTR (51) are too
positive-starved for a reliable per-station >=30 classifier. Does pooling all
three stations into ONE model (435 combined positives) with a station feature
recover per-station orange recall — or should we just run >=30 for NESTOR only?

Compares, per station, on the SAME per-station held-out test fold:
  - per_station : model trained on that station's 80% only (baseline)
  - pooled      : one model trained on all stations' 80%, station one-hot added
  - pooled_nostn: one pooled model WITHOUT station feature (does station identity
                  even matter, or is it just more rows?)

Lean 19-feature subset. Evaluated at the default decision threshold AND at the
operational PROB_30_ALERT=0.25 cut so the threshold-sensitive low-count
stations are judged at their real operating point.

Usage:
  cd projects/h2s
  uv run python scripts/experiment_pooled_clf30.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from h2s.constants import MODEL_FEATURES_LEAN, STATION_PARTITION_MAP
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)

ALERT_THR = 0.25  # operational PROB_30_ALERT


def time_split(df_station: pd.DataFrame):
    """80/20 split preserving temporal order within a single station."""
    n = int(len(df_station) * TRAIN_FRACTION)
    return df_station.iloc[:n].copy(), df_station.iloc[n:].copy()


def metrics_at(y_true, y_prob, thr):
    y_pred = (y_prob > thr).astype(int)
    return {
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'n_pred_pos': int(y_pred.sum()),
    }


def evaluate(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        'auc': float(auc),
        'default': metrics_at(y_true, y_prob, 0.5),
        'alert_0.25': metrics_at(y_true, y_prob, ALERT_THR),
    }


def main():
    print("Loading + preparing data...")
    df = prepare_multi_station_features(pd.read_parquet('../../data/modeldata_h2s_nofill.parquet'))

    stations = list(STATION_PARTITION_MAP.values())  # display names
    feats = MODEL_FEATURES_LEAN

    # Per-station train/test folds (temporal). Pooled train = concat of train folds,
    # so a station's test rows never leak into the pooled training set.
    train_parts, test_parts = {}, {}
    for st in stations:
        s = df[df['site_name'] == st].sort_values('time')
        tr, te = time_split(s)
        train_parts[st], test_parts[st] = tr, te

    pooled_train = pd.concat(train_parts.values(), ignore_index=True)

    # Station one-hot (stable column order across train/test)
    stn_cols = [f'stn_{st}' for st in stations]

    def add_stn_onehot(frame):
        out = frame.copy()
        for st in stations:
            out[f'stn_{st}'] = (out['site_name'] == st).astype(int)
        return out

    pooled_train_oh = add_stn_onehot(pooled_train)

    print(f"\nTrain positives (>=30 ppb): "
          + ", ".join(f"{st}={int(train_parts[st]['exceed_30'].sum())}" for st in stations)
          + f"  | pooled={int(pooled_train['exceed_30'].sum())}")

    # ---- Train the three models once ----
    print("\nTraining per-station models...")
    per_station_models = {}
    for st in stations:
        tr = train_parts[st]
        if tr['exceed_30'].nunique() < 2:
            per_station_models[st] = None
            continue
        m, choice, _ = train_and_select(
            tr[feats], test_parts[st][feats],
            tr['exceed_30'].values, test_parts[st]['exceed_30'].values,
            task='clf_30ppb',
        )
        per_station_models[st] = m

    print("Training pooled model (with station feature)...")
    # Use NESTOR test as the throwaway eval set for selection (train_and_select
    # needs one); selection metric is AUC and doesn't affect the fitted model we
    # later score per-station.
    sel_X = add_stn_onehot(test_parts['NESTOR - BES'])
    pooled_m, _, _ = train_and_select(
        pooled_train_oh[feats + stn_cols], sel_X[feats + stn_cols],
        pooled_train_oh['exceed_30'].values, sel_X['exceed_30'].values,
        task='clf_30ppb',
    )

    print("Training pooled model (NO station feature)...")
    pooled_nostn_m, _, _ = train_and_select(
        pooled_train[feats], test_parts['NESTOR - BES'][feats],
        pooled_train['exceed_30'].values, test_parts['NESTOR - BES']['exceed_30'].values,
        task='clf_30ppb',
    )

    # ---- Evaluate every model on each station's held-out test fold ----
    results = {}
    for st in stations:
        te = test_parts[st]
        y = te['exceed_30'].values
        te_oh = add_stn_onehot(te)

        entry = {'n_test': int(len(te)), 'n_pos_test': int(y.sum())}
        if per_station_models[st] is not None:
            entry['per_station'] = evaluate(y, per_station_models[st].predict_proba(te[feats])[:, 1])
        entry['pooled'] = evaluate(y, pooled_m.predict_proba(te_oh[feats + stn_cols])[:, 1])
        entry['pooled_nostn'] = evaluate(y, pooled_nostn_m.predict_proba(te[feats])[:, 1])
        results[st] = entry

    # ---- Report ----
    print("\n" + "=" * 100)
    print("POOLED vs PER-STATION clf_30ppb  —  recall / precision / f1 (auc)")
    print("=" * 100)
    for st in stations:
        e = results[st]
        print(f"\n{st}  (test n={e['n_test']}, >=30 positives={e['n_pos_test']}):")
        print(f"  {'model':<14}{'@thr':<8}{'recall':<9}{'prec':<8}{'f1':<8}{'auc':<7}{'n_pred_pos'}")
        for model_name in ('per_station', 'pooled', 'pooled_nostn'):
            if model_name not in e:
                continue
            m = e[model_name]
            for thr_name in ('default', 'alert_0.25'):
                d = m[thr_name]
                print(f"  {model_name:<14}{thr_name.replace('alert_',''):<8}"
                      f"{d['recall']:<9.3f}{d['precision']:<8.3f}{d['f1']:<8.3f}"
                      f"{m['auc']:<7.3f}{d['n_pred_pos']}")

    out = Path('experiments/underfitting_results/pooled_clf30_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Saved {out}")


if __name__ == '__main__':
    main()
