"""Forecast performance report with an asymmetric, temporally-tolerant rubric.

Standard accuracy treats every mistake equally. For an H2S health-warning
forecast the *direction* of the error and its *timing* matter far more than
raw agreement, so this module scores predicted-vs-actual the way an analyst
would read it (see RUBRIC below). It is pure pandas — no Dagster / S3 — and
runs off the Phase-5 validation rows (``build_validation_rows`` output: one
row per product row joined to the H2S measured at its target hour).

RUBRIC (predicted → actual), starting weights — tune with health officials:

  match            predicted tier == actual tier .................. good (0.0)
  yellow_band      yellow-low vs yellow-high mix-up ............... minor (0.25)
  early_warning_ok over-predicted, but the actual reaches the
                   predicted tier within ±TOLERANCE_HOURS ........ ok (0.25)
  expected_low     actual yellow-low (5–10), predicted green;
                   plausibly fine when readings are low .......... low (0.5)
  soft_miss        actual orange, predicted yellow — under by a
                   tier but still flagged a hazard ............... (1.0)
  false_alarm      over-predicted with NO nearby exceedance ...... (1.5)
  smell_miss       actual yellow-high (≥10, residents smell it),
                   predicted green — the level we must get right . (2.5)
  dangerous_miss   actual orange (≥30), predicted green .......... worst (5.0)

The headline numbers are ``tolerant_accuracy`` (match + yellow_band +
early_warning_ok), ``dangerous_miss_rate`` and ``smell_miss_rate``, plus a
mean cost per row. Lower cost is better.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from h2s.constants import (
    CATEGORY_GREEN,
    CATEGORY_ORANGE,
    CATEGORY_ORDER,
    CATEGORY_YELLOW_HIGH,
    CATEGORY_YELLOW_LOW,
    H2S_THRESHOLD_MED,
    h2s_category,
)
from h2s.training.calibration_eval import spearman_rank

# Severity rank per 4-tier category (worst = highest).
SEVERITY = {
    CATEGORY_GREEN: 0,
    CATEGORY_YELLOW_LOW: 1,
    CATEGORY_YELLOW_HIGH: 2,
    CATEGORY_ORANGE: 3,
}

# Over-prediction is forgiven when the actual reaches the predicted tier within
# this many hours either side of the target hour (the "valid within 1–2 h" rule).
TOLERANCE_HOURS = 2

# Verdict → cost weight (lower is better). Starting points; documented above.
VERDICT_COST = {
    "match": 0.0,
    "yellow_band": 0.25,
    "early_warning_ok": 0.25,
    "expected_low": 0.5,
    "soft_miss": 1.0,
    "false_alarm": 1.5,
    "smell_miss": 2.5,
    "dangerous_miss": 5.0,
}

# Verdicts counted as acceptable for the tolerant-accuracy headline.
_ACCEPTABLE = {"match", "yellow_band", "early_warning_ok"}


def _observed_series(validation_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Per-station unique (time, actual_h2s) sorted by time, for tolerance checks."""
    out: dict[str, pd.DataFrame] = {}
    cols = ["station", "time", "actual_h2s"]
    obs = validation_df[cols].dropna(subset=["actual_h2s"]).copy()
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    obs = obs.drop_duplicates(["station", "time"], keep="last")
    for station, g in obs.groupby("station"):
        out[station] = g.sort_values("time").reset_index(drop=True)
    return out


def _reaches_within(
    series: pd.DataFrame, target_time: pd.Timestamp, min_severity: int, hours: int
) -> bool:
    """Did the observed H2S reach ``min_severity`` within ±hours of target_time?"""
    if series is None or series.empty:
        return False
    lo = target_time - pd.Timedelta(hours=hours)
    hi = target_time + pd.Timedelta(hours=hours)
    window = series[(series["time"] >= lo) & (series["time"] <= hi)]
    if window.empty:
        return False
    sev = window["actual_h2s"].map(lambda v: SEVERITY[h2s_category(v)])
    return bool((sev >= min_severity).any())


def _verdict(pred_cat: str, act_cat: str, actual_h2s: float, early_warning: bool) -> str:
    """Classify one predicted→actual pair per the rubric."""
    ps, as_ = SEVERITY[pred_cat], SEVERITY[act_cat]

    if ps == as_:
        return "match"
    # Both yellow tiers — a band mix-up, not a hazard-direction error.
    if {pred_cat, act_cat} <= {CATEGORY_YELLOW_LOW, CATEGORY_YELLOW_HIGH}:
        return "yellow_band"

    if ps < as_:  # under-prediction — the dangerous direction
        if act_cat == CATEGORY_ORANGE and pred_cat == CATEGORY_GREEN:
            return "dangerous_miss"
        if act_cat == CATEGORY_ORANGE:  # predicted some yellow — still flagged
            return "soft_miss"
        if act_cat == CATEGORY_YELLOW_HIGH and pred_cat == CATEGORY_GREEN:
            return "smell_miss"  # residents smell it; we said green
        if act_cat == CATEGORY_YELLOW_LOW and pred_cat == CATEGORY_GREEN:
            return "expected_low"
        return "soft_miss"

    # ps > as_ — over-prediction: forgiven if the actual reaches it nearby.
    return "early_warning_ok" if early_warning else "false_alarm"


def annotate_verdicts(
    validation_df: pd.DataFrame, tolerance_hours: int = TOLERANCE_HOURS
) -> pd.DataFrame:
    """Add ``pred_cat``, ``actual_cat``, ``verdict`` and ``cost`` columns."""
    if validation_df is None or len(validation_df) == 0:
        return pd.DataFrame(
            columns=list(getattr(validation_df, "columns", []))
            + ["pred_cat", "actual_cat", "verdict", "cost"]
        )

    df = validation_df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["pred_cat"] = df["h2s_pred"].map(h2s_category)
    df["actual_cat"] = df["actual_h2s"].map(h2s_category)

    series = _observed_series(df)
    verdicts = []
    for row in df.itertuples(index=False):
        pred_cat, act_cat = row.pred_cat, row.actual_cat
        early = False
        if SEVERITY[pred_cat] > SEVERITY[act_cat]:
            early = _reaches_within(
                series.get(row.station), row.time, SEVERITY[pred_cat], tolerance_hours
            )
        verdicts.append(_verdict(pred_cat, act_cat, row.actual_h2s, early))

    df["verdict"] = verdicts
    df["cost"] = df["verdict"].map(VERDICT_COST).astype(float)
    return df


def confusion_counts(annotated: pd.DataFrame) -> pd.DataFrame:
    """4×4 predicted (rows) × actual (cols) category count matrix."""
    cats = CATEGORY_ORDER
    mat = pd.DataFrame(0, index=cats, columns=cats, dtype=int)
    if annotated is None or annotated.empty:
        return mat
    for pred_cat, act_cat in zip(annotated["pred_cat"], annotated["actual_cat"]):
        if pred_cat in mat.index and act_cat in mat.columns:
            mat.loc[pred_cat, act_cat] += 1
    return mat


def correlation_by(annotated: pd.DataFrame, by: str) -> pd.DataFrame:
    """Spearman(actual, pred) + MAE + n grouped by a column (e.g. lead_hour)."""
    if annotated is None or annotated.empty or by not in annotated.columns:
        return pd.DataFrame(columns=[by, "n", "spearman", "mae"])
    records = []
    for key, g in annotated.groupby(by, sort=True):
        records.append({
            by: key,
            "n": int(len(g)),
            "spearman": spearman_rank(g["actual_h2s"], g["h2s_pred"]),
            "mae": float((g["h2s_pred"] - g["actual_h2s"]).abs().mean()),
        })
    return pd.DataFrame(records)


def correlation_by_hour_of_day(annotated: pd.DataFrame) -> pd.DataFrame:
    """Spearman/MAE by UTC hour-of-day (0–23) of the target time."""
    if annotated is None or annotated.empty:
        return pd.DataFrame(columns=["hour_of_day", "n", "spearman", "mae"])
    df = annotated.copy()
    df["hour_of_day"] = pd.to_datetime(df["time"], utc=True).dt.hour
    out = correlation_by(df, "hour_of_day")
    return out.rename(columns={out.columns[0]: "hour_of_day"})


def summarize(
    annotated: pd.DataFrame, variant: Optional[str] = None, max_examples: int = 6
) -> dict:
    """Headline rubric metrics + verdict tally + worst-miss examples (narrative-ready)."""
    if variant is not None and not annotated.empty and "variant" in annotated.columns:
        annotated = annotated[annotated["variant"] == variant]

    n = int(len(annotated))
    base = {
        "variant": variant,
        "n": n,
        "tolerant_accuracy": None,
        "dangerous_miss_rate": None,
        "smell_miss_rate": None,
        "false_alarm_rate": None,
        "mean_cost": None,
        "overall_spearman": None,
        "verdict_counts": {},
        "worst_examples": [],
    }
    if n == 0:
        return base

    vc = annotated["verdict"].value_counts().to_dict()
    base["verdict_counts"] = {k: int(v) for k, v in vc.items()}
    base["tolerant_accuracy"] = round(
        sum(vc.get(v, 0) for v in _ACCEPTABLE) / n, 4
    )
    base["dangerous_miss_rate"] = round(vc.get("dangerous_miss", 0) / n, 4)
    base["smell_miss_rate"] = round(vc.get("smell_miss", 0) / n, 4)
    base["false_alarm_rate"] = round(vc.get("false_alarm", 0) / n, 4)
    base["mean_cost"] = round(float(annotated["cost"].mean()), 4)
    base["overall_spearman"] = spearman_rank(
        annotated["actual_h2s"], annotated["h2s_pred"]
    )

    worst = annotated[annotated["verdict"].isin(["dangerous_miss", "smell_miss"])]
    worst = worst.sort_values("cost", ascending=False).head(max_examples)
    base["worst_examples"] = [
        {
            "station": r.station,
            "time": pd.to_datetime(r.time, utc=True).isoformat(),
            "lead_hour": int(r.lead_hour),
            "predicted_ppb": round(float(r.h2s_pred), 1),
            "actual_ppb": round(float(r.actual_h2s), 1),
            "verdict": r.verdict,
        }
        for r in worst.itertuples(index=False)
    ]
    return base


def build_report(
    validation_df: pd.DataFrame,
    variant: Optional[str] = None,
    tolerance_hours: int = TOLERANCE_HOURS,
) -> dict:
    """Full structured performance report (the LLM-narrative payload).

    Returns the annotated rows plus summary, confusion matrix, and both
    correlation breakdowns — everything the charts and the narrative need.
    """
    annotated = annotate_verdicts(validation_df, tolerance_hours=tolerance_hours)
    if variant is not None and "variant" in annotated.columns and not annotated.empty:
        scope = annotated[annotated["variant"] == variant]
    else:
        scope = annotated

    return {
        "annotated": annotated,
        "summary": summarize(annotated, variant=variant),
        "confusion": confusion_counts(scope),
        "by_lead_hour": correlation_by(scope, "lead_hour"),
        "by_hour_of_day": correlation_by_hour_of_day(scope),
        "by_product": correlation_by(scope, "product"),
        "tolerance_hours": tolerance_hours,
        "smell_threshold_ppb": H2S_THRESHOLD_MED,
    }
