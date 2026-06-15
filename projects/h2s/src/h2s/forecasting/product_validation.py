"""Forecast-vs-actual validation logic for the nowcast/nearcast/forecast
products (Phase 5) — pure pandas, no Dagster / S3.

Two pieces, both rebuildable from the immutable product-run parquets:

- :func:`build_validation_rows` — join every stored product row to the H2S
  actually measured at its target hour. One row per
  (run_ts, product, station, variant, lead_hour, time) that has a matching
  observation; the same target hour appears at many lead hours (forecast 1 h
  ahead in one run, 6 h ahead in an earlier run), which is exactly what a
  skill-vs-lead-hour curve needs.
- :func:`skill_curves` — per (product, variant, lead_hour): Spearman + MAE +
  recall/precision at the calibration thresholds. Two flavours of recall:
  *magnitude* (the calibration harness — cut h2s_pred and actual at k) and
  *probability-call* (did the classifier flag it: p_k > 0.5 vs actual ≥ k).

Honest scope: this measures how fast magnitude skill decays with lead hour
(expected — recursion compounds error toward the exogenous ceiling). It is the
substrate that tells the cascade where its probability cutoffs should sit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from h2s.training.calibration_eval import recall_at_threshold, spearman_rank

# Calibration thresholds (ppb). 5/10/30 have deployed classifiers; 100 is
# magnitude-only (extreme — no clf_100 task).
THRESHOLDS = (5, 10, 30, 100)
PROB_FOR_THRESHOLD = {5: "p5", 10: "p10", 30: "p30"}

_VALIDATION_COLS = [
    "run_ts", "product", "station", "variant", "model_version", "lead_hour",
    "time", "h2s_pred", "p5", "p10", "p30",
    "actual_h2s", "error", "abs_error",
    "actual_ge5", "actual_ge10", "actual_ge30", "actual_ge100",
]


def build_validation_rows(
    products_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    station_col: str = "site_name",
    actual_col: str = "H2S",
) -> pd.DataFrame:
    """Join product rows to measured H2S at their target hour.

    Returns one row per product row that has a matching observation, with
    ``actual_h2s``, ``error`` (= h2s_pred − actual), ``abs_error`` and
    ``actual_ge{5,10,30,100}`` flags. Unmatched rows (future hours not yet
    observed, other stations) are dropped. Empty inputs return the empty
    schema rather than raising.
    """
    if products_df is None or len(products_df) == 0:
        return pd.DataFrame(columns=_VALIDATION_COLS)

    p = products_df.copy()
    p["time"] = pd.to_datetime(p["time"], utc=True)
    for col in ("p5", "p10", "p30", "model_version", "run_ts"):
        if col not in p.columns:
            p[col] = np.nan

    o = obs_df[[station_col, "time", actual_col]].copy()
    o["time"] = pd.to_datetime(o["time"], utc=True)
    o = o.rename(columns={station_col: "station", actual_col: "actual_h2s"})
    o = o.drop_duplicates(["station", "time"], keep="last")

    merged = p.merge(o, on=["station", "time"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=_VALIDATION_COLS)

    merged["error"] = merged["h2s_pred"] - merged["actual_h2s"]
    merged["abs_error"] = merged["error"].abs()
    for k in THRESHOLDS:
        merged[f"actual_ge{k}"] = merged["actual_h2s"] >= k

    return (
        merged[_VALIDATION_COLS]
        .sort_values(["product", "variant", "lead_hour", "time"])
        .reset_index(drop=True)
    )


def _prob_call_metrics(group: pd.DataFrame, k: int, cutoff: float) -> dict:
    """Recall/precision of the classifier's exceedance call (p_k > cutoff)
    against the measured actual ≥ k, over rows with a non-NaN probability."""
    prob_col = PROB_FOR_THRESHOLD.get(k)
    out = {f"prob_recall_{k}": None, f"prob_precision_{k}": None, f"n_prob_{k}": 0}
    if prob_col is None or prob_col not in group.columns:
        return out
    mask = group[prob_col].notna()
    if not mask.any():
        return out
    pred = group.loc[mask, prob_col] > cutoff
    act = group.loc[mask, f"actual_ge{k}"].astype(bool)
    tp = int((pred & act).sum())
    out[f"n_prob_{k}"] = int(mask.sum())
    out[f"prob_recall_{k}"] = (tp / int(act.sum())) if act.sum() else None
    out[f"prob_precision_{k}"] = (tp / int(pred.sum())) if pred.sum() else None
    return out


def skill_curves(
    validation_df: pd.DataFrame,
    thresholds=THRESHOLDS,
    prob_cutoff: float = 0.5,
) -> pd.DataFrame:
    """Per (product, variant, lead_hour) skill summary.

    Columns: n, spearman (actual vs h2s_pred), mae, and per threshold k —
    magnitude recall/precision (calibration harness, cut at k) plus, for k with
    a classifier, the probability-call recall/precision (p_k > cutoff vs
    actual ≥ k).
    """
    if validation_df is None or len(validation_df) == 0:
        return pd.DataFrame()

    records = []
    for (product, variant, lead), g in validation_df.groupby(
        ["product", "variant", "lead_hour"], sort=True
    ):
        rec: dict = {
            "product": product,
            "variant": variant,
            "lead_hour": int(lead),
            "n": int(len(g)),
            "spearman": spearman_rank(g["actual_h2s"], g["h2s_pred"]),
            "mae": float(g["abs_error"].mean()),
        }
        for k in thresholds:
            mag = recall_at_threshold(g["actual_h2s"], g["h2s_pred"], k)
            rec[f"mag_recall_{k}"] = mag["recall"]
            rec[f"mag_precision_{k}"] = mag["precision"]
            rec[f"n_actual_ge{k}"] = int(g[f"actual_ge{k}"].sum())
            rec.update(_prob_call_metrics(g, k, prob_cutoff))
        records.append(rec)

    return (
        pd.DataFrame(records)
        .sort_values(["product", "variant", "lead_hour"])
        .reset_index(drop=True)
    )
