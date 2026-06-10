"""Calibration-aligned evaluation harness for H2S predictions.

Built from the findings in ../tj_calibration. On this heavy-tailed
series the headline metrics are Spearman + recall at fixed thresholds,
stratified by site and atmospheric regime (`stable_atm`). Persistence
(`h2s_lag_1h` → class) is the floor — XGBoost must beat it to earn its
keep.

See tj_calibration/tijuana-dispersion-experiments/docs/calibration_status.md
for the reasoning (entries 2026-05-15 emission_driver_attribution,
2026-05-16 box_driver_calibration, 2026-05-16 event_trigger_magnitude).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score


def spearman_rank(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """NaN-safe Spearman rank correlation. Returns nan if <3 paired points."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if mask.sum() < 3:
        return float("nan")
    rho, _ = spearmanr(yt[mask], yp[mask])
    return float(rho)


def recall_at_threshold(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    threshold: float,
) -> dict[str, float | int]:
    """Binary recall/precision when both y_true and y_pred are cut at `threshold`.

    Designed for heavy-tailed targets where the binary cut at a
    clinically meaningful value (30 ppb watch, 100 ppb critical) is
    what matters. Returns recall and precision relative to that cut.
    """
    yt_arr = np.asarray(y_true, dtype=float)
    yp_arr = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(yt_arr) | np.isnan(yp_arr))
    yt = (yt_arr[mask] > threshold).astype(int)
    yp = (yp_arr[mask] > threshold).astype(int)
    return {
        "threshold": float(threshold),
        "n_positives": int(yt.sum()),
        "n_predicted_positive": int(yp.sum()),
        "recall": float(recall_score(yt, yp, zero_division=0.0)),
        "precision": float(precision_score(yt, yp, zero_division=0.0)),
    }


def persistence_prediction(
    df: pd.DataFrame,
    target_col: str = "H2S",
    site_col: str = "site_name",
    time_col: str = "time",
    lag_hours: int = 1,
) -> pd.Series:
    """Naive forecast: y_pred[t] = y_obs[t − lag] within each site.

    The autoregressive ceiling per calibration finding #8 (h2s_lag_1h
    Spearman ≈ 0.70, recall@100 ≈ 0.21). Any model that does not beat
    this is not earning its keep.

    The shift is by row count, not clock time — so gaps in the input
    behave the same way `h2s_lag_1h` does in `multi_station_trainer.py`.

    Returns a Series aligned to df.index. NaN for the first `lag_hours`
    rows of each site.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, sub in df.groupby(site_col, sort=False):
        s = sub.sort_values(time_col)
        lagged = s[target_col].shift(lag_hours)
        out.loc[lagged.index] = lagged.values
    return out


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    time_col: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split — no shuffling.

    Calibration uses 70/30 chronological splits. Random shuffling leaks
    future into past on time-series targets and inflates scores.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    sorted_df = df.sort_values(time_col).reset_index(drop=True)
    cut = int(len(sorted_df) * train_fraction)
    return sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()


@dataclass
class CalibrationReport:
    """Calibration-aligned scoreboard for one set of predictions.

    `overall`, `per_site`, and `per_regime` each have the same shape:
        {
          "n": <count>,
          "spearman": <float>,
          "thr_30":  {"recall": ..., "precision": ..., "n_positives": ..., ...},
          "thr_100": {"recall": ..., "precision": ..., "n_positives": ..., ...},
        }
    """

    overall: dict
    per_site: dict[str, dict]
    per_regime: dict[str, dict]


def calibration_report(
    df: pd.DataFrame,
    y_pred: ArrayLike,
    target_col: str = "H2S",
    site_col: str = "site_name",
    regime_col: str = "stable_atm",
    thresholds: Iterable[float] = (30.0, 100.0),
) -> CalibrationReport:
    """Build the calibration-aligned scoreboard for one set of predictions.

    Args:
        df: Must contain `target_col` and (optionally) `site_col`,
            `regime_col`. Index aligned to `y_pred`.
        y_pred: Predicted H2S concentration (ppb), `len == len(df)`.
        thresholds: ppb cuts for recall@/precision@. Defaults: 30 (watch)
            and 100 (extreme — calibration's headline).

    Returns:
        CalibrationReport with overall, per-site, per-regime numbers.

    Raises:
        ValueError: if len(df) != len(y_pred).
    """
    y_true_arr = df[target_col].to_numpy(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(f"len(df)={len(y_true_arr)} != len(y_pred)={len(y_pred_arr)}")

    thresholds_list = list(thresholds)

    def _block(mask: np.ndarray) -> dict:
        yt = y_true_arr[mask]
        yp = y_pred_arr[mask]
        return {
            "n": int(mask.sum()),
            "spearman": spearman_rank(yt, yp),
            **{
                f"thr_{int(t)}": recall_at_threshold(yt, yp, t)
                for t in thresholds_list
            },
        }

    overall = _block(np.ones(len(df), dtype=bool))

    per_site: dict[str, dict] = {}
    if site_col in df.columns:
        for site in df[site_col].unique():
            site_mask = (df[site_col] == site).to_numpy()
            per_site[str(site)] = _block(site_mask)

    per_regime: dict[str, dict] = {}
    if regime_col in df.columns:
        calm_mask = (df[regime_col] == 1).to_numpy()
        per_regime["calm_stable_atm_1"] = _block(calm_mask)
        per_regime["windy_stable_atm_0"] = _block(~calm_mask)

    return CalibrationReport(overall=overall, per_site=per_site, per_regime=per_regime)
