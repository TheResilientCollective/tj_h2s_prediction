"""Recursive inference engine for the nowcast / nearcast / forecast products.

One recursive pass produces all three products (docs/feature/rename.md);
they differ only in how far the recursion has drifted from observed data:

- **nowcast** (leads 1–3): recursion seeded at the last actual. Lead 1's
  features are entirely observed; by lead 3 the 1–2 h lags are the model's
  own predictions, but the longer lags and most of the rolling windows are
  still actuals.
- **nearcast** (leads 4–6): the mid-window — lag_3h crosses into
  predictions at lead 4.
- **forecast** (leads 7–24): by lead 7 every lag ≤ 6 h is a prediction and
  the rolling windows are mostly predictions — "all forecasted h2s as
  features".

Honest scope (inherited from the tj_calibration arc): recursion compounds
error, and at the forecast tier magnitude skill is bounded by the exogenous
ceiling (Spearman ≈ 0.33 on calm-night extremes). The forecast product is a
risk-ranker at that horizon; the Phase-5 validation store measures exactly
how fast skill decays per lead hour.

Mechanics: a value series ordered oldest → newest where ``series[-1]`` is
the value one hour before the hour being predicted. Each hour's prediction
is appended to the series before the next hour is scored.
``autoregressive_features`` reads lags and rolling means off the series
tail, clamping to the oldest value when history is short (training drops
NaN-lag rows; inference cannot, so the clamp mirrors rolling(min_periods=1)
behaviour).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from h2s.constants import (
    PRODUCT_FORECAST,
    PRODUCT_HORIZONS_H,
    PRODUCT_NEARCAST,
    PRODUCT_NOWCAST,
)

# The autoregressive columns the engine owns. Everything else in the
# feature frame (met, tide, flow lags, interactions) is exogenous
# passthrough prepared by the caller.
H2S_FEATURE_COLS = (
    "h2s_lag_1h", "h2s_lag_3h", "h2s_lag_6h",
    "h2s_rolling_6h", "h2s_rolling_24h",
)


@dataclass
class VariantModels:
    """One feature-variant's model set. Classifiers are optional — a missing
    classifier yields NaN probabilities rather than an error (e.g. clf_30ppb
    before a station's first post-Phase-1 deployment)."""

    regression: Any
    clf_5ppb: Any | None = None
    clf_10ppb: Any | None = None
    clf_30ppb: Any | None = None


def autoregressive_features(series: Sequence[float]) -> dict[str, float]:
    """Lag + rolling features for the hour after the end of ``series``.

    ``series`` is ordered oldest → newest; ``series[-1]`` is the value one
    hour before the prediction hour. Short series clamp lags to the oldest
    value and shrink rolling windows (min_periods=1 semantics).
    """
    s = list(series)
    n = len(s)

    def lag(k: int) -> float:
        return s[n - k] if n >= k else s[0]

    return {
        "h2s_lag_1h": lag(1),
        "h2s_lag_3h": lag(3),
        "h2s_lag_6h": lag(6),
        "h2s_rolling_6h": float(np.mean(s[-6:])),
        "h2s_rolling_24h": float(np.mean(s[-24:])),
    }


def _predict_one(
    models: VariantModels,
    feature_frame: pd.DataFrame,
    lead_idx: int,
    ar_features: dict[str, float],
    feature_cols: list[str],
) -> tuple[float, float, float, float]:
    """Score one lead hour. Returns (h2s_pred, p5, p10, p30)."""
    row = feature_frame.iloc[lead_idx].copy()
    for col, value in ar_features.items():
        row[col] = value

    X = pd.DataFrame([row])[feature_cols].astype(float).to_numpy()
    h2s_pred = float(np.clip(models.regression.predict(X)[0], 0.0, None))

    def proba(clf) -> float:
        if clf is None:
            return float("nan")
        return float(clf.predict_proba(X)[0, 1])

    return h2s_pred, proba(models.clf_5ppb), proba(models.clf_10ppb), proba(models.clf_30ppb)


def _product_for_lead(lead: int) -> str | None:
    """Map a lead hour to its product window (start exclusive, end inclusive,
    except nowcast which starts at lead 1)."""
    for product in (PRODUCT_NOWCAST, PRODUCT_NEARCAST, PRODUCT_FORECAST):
        start, end = PRODUCT_HORIZONS_H[product]
        if start < lead <= end:
            return product
    return None


def run_products(
    feature_frame: pd.DataFrame,
    h2s_history: Sequence[float],
    models: VariantModels,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Run all three products for one station × one variant.

    Args:
        feature_frame: engineered exogenous features for leads 1..N, one row
            per lead hour, including a ``time`` column. The five
            H2S_FEATURE_COLS are overwritten per lead by the engine; any
            values present (e.g. the daily pipeline's decay heuristic) are
            ignored.
        h2s_history: actual H₂S values oldest → newest; ``[-1]`` is the last
            observation (t0). Ideally ≥ 24 values; shorter histories clamp.
        models: the variant's regression + classifiers.
        feature_cols: the variant's feature schema (column order matters —
            must match training).

    Returns:
        DataFrame with one row per lead hour 1..N:
        [lead_hour, time, product, h2s_pred, p5, p10, p30].
        Leads beyond the forecast window (> 24 by default) are not emitted.
    """
    if len(h2s_history) == 0:
        raise ValueError("h2s_history is empty — need at least the last actual")
    n_hours = len(feature_frame)

    rows: list[dict] = []

    # Single recursive pass, seeded at the last actual. Every lead's
    # prediction joins the series before the next lead is scored; product
    # labels are just window slices of the same recursion.
    series = list(h2s_history)
    for lead_idx in range(n_hours):
        lead = lead_idx + 1
        ar = autoregressive_features(series)
        h2s_pred, p5, p10, p30 = _predict_one(models, feature_frame, lead_idx, ar, feature_cols)
        series.append(h2s_pred)

        product = _product_for_lead(lead)
        if product is not None:
            rows.append({
                "lead_hour": lead,
                "time": feature_frame.iloc[lead_idx]["time"],
                "product": product,
                "h2s_pred": round(h2s_pred, 2),
                "p5": p5, "p10": p10, "p30": p30,
            })

    out = pd.DataFrame(rows).sort_values("lead_hour").reset_index(drop=True)
    return out
