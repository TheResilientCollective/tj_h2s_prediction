"""Tier definitions, horizon specs, hard gates, score function, nesting check."""

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
import yaml


class Horizon(str, Enum):
    NOWCAST   = "nowcast"
    NEAR      = "near"
    MID       = "mid"
    DAY_AHEAD = "day_ahead"


HORIZON_WINDOWS_H: dict[str, tuple[int, int]] = {
    Horizon.NOWCAST:   (0,  3),
    Horizon.NEAR:      (3,  6),
    Horizon.MID:       (6, 12),
    Horizon.DAY_AHEAD: (12, 24),
}

HORIZON_LABELS: dict[str, str] = {
    Horizon.NOWCAST:   "Nowcast (0–3h)",
    Horizon.NEAR:      "Near (3–6h)",
    Horizon.MID:       "Mid (6–12h)",
    Horizon.DAY_AHEAD: "Day-ahead (12–24h)",
}

# Ordered list for display
HORIZON_ORDER = [Horizon.NOWCAST, Horizon.NEAR, Horizon.MID, Horizon.DAY_AHEAD]


class TierNestingError(Exception):
    """Raised when Tier 3 fires in a horizon without Tier 2 and Tier 1 firing."""


@dataclass(frozen=True)
class TierResult:
    tier: str                          # "tier_1" | "tier_2" | "tier_3"
    label: str                         # "PLANT-SIGNAL" | "MULTI-SITE-RISK" | "EXCEEDANCE-RISK"
    horizon: str                       # Horizon enum value
    evaluated_at: pd.Timestamp
    window: tuple                      # (window_start, window_end) absolute timestamps
    gate_passed: bool
    score: float                       # 0–0.95 (saturation clip)
    n_stations_passing_gate: int       # for Tier 2's ≥2 requirement
    contributing_features: dict        # name -> (value, z, weight)
    daytime_horizon: bool              # True if <75% of window is night hours
    degraded: bool                     # True if NB fallback to IB was used
    fire: bool                         # gate_passed AND score >= 0.5


# Per-horizon Tier 3 acceptance criteria (design §6.1)
TIER3_TARGETS: dict[str, tuple[float, float]] = {
    "nowcast":   (0.65, 0.80),
    "near":      (0.60, 0.75),
    "mid":       (0.55, 0.70),
    "day_ahead": (0.50, 0.65),
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "configs" / "tiered_alerts.yaml"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hard gate functions
# ---------------------------------------------------------------------------

def _gate_tier1_single(row: dict) -> bool:
    # Data-availability check: gate passes when the station has valid met data.
    # Previously gated on SBIWTP deficit (flow < 23 MGD, anomaly < 0), but in April 2026
    # SBIWTP flow jumped to 30–34 MGD while H2S events continued, driven purely by
    # atmospheric conditions (stable_atm d=+1.48, wind d=−1.37 on Apr-May 2026 events).
    # SBIWTP still contributes to the score function via its weights.
    wind = row.get("wind_speed_10m")
    return wind is not None and not pd.isna(wind)


def gate_tier1(rows_by_station: dict[str, dict]) -> dict[str, bool]:
    """Return per-station gate results for Tier 1."""
    return {s: _gate_tier1_single(row) for s, row in rows_by_station.items()}


def gate_tier2(
    rows_by_station: dict[str, dict],
    tier1_by_station: dict[str, bool],
) -> tuple[dict[str, bool], int]:
    """Return per-station gate results for Tier 2 and count of stations passing Tier 1."""
    n_t1 = sum(tier1_by_station.values())
    results: dict[str, bool] = {}
    for station, row in rows_by_station.items():
        if not tier1_by_station.get(station, False):
            results[station] = False
            continue
        wind = row.get("wind_speed_10m")
        if wind is None or pd.isna(wind):
            results[station] = False
            continue
        results[station] = (n_t1 >= 2) and (float(wind) < 4.0)
    return results, n_t1


def gate_tier3(
    rows_by_station: dict[str, dict],
    tier2_by_station: dict[str, bool],
) -> dict[str, bool]:
    """Return per-station gate results for Tier 3."""
    results: dict[str, bool] = {}
    for station, row in rows_by_station.items():
        if not tier2_by_station.get(station, False):
            results[station] = False
            continue
        temp = row.get("temp_min", row.get("temperature_2m"))
        dew  = row.get("dewpoint_2m")
        stable = row.get("stable_atm_fraction", row.get("stable_atm"))
        if temp is None or dew is None or stable is None:
            results[station] = False
            continue
        t_f, d_f, s_f = float(temp), float(dew), float(stable)
        if pd.isna(t_f) or pd.isna(d_f) or pd.isna(s_f):
            results[station] = False
            continue
        results[station] = t_f > 13.0 and d_f > 11.0 and s_f > 0.6
    return results


# ---------------------------------------------------------------------------
# Score function
# ---------------------------------------------------------------------------

def compute_score(
    features: dict,
    weights: dict,
    stats: dict,
) -> tuple[float, dict]:
    """Sigmoid of weighted sum of standardized features.

    Returns (score, contributing_features) where contributing_features maps
    feature name → (value, z_score, weight).

    # TODO(weights): retrain logistic regression on labeled nights + daytime
    # recalibration (see design §8.6).
    """
    weighted_sum = 0.0
    contributing: dict[str, tuple[float, float, float]] = {}

    for feat, weight in weights.items():
        val = features.get(feat)
        if val is None or pd.isna(val):
            continue
        feat_stats = stats.get(feat, {})
        q_mean = feat_stats.get("mean", 0.0)
        q_std  = feat_stats.get("std", 1.0)
        if q_std == 0:
            continue
        val = float(val)
        z = (val - q_mean) / q_std
        weighted_sum += weight * z
        contributing[feat] = (val, z, weight)

    score = 1.0 / (1.0 + math.exp(-weighted_sum))
    score = min(0.95, score)
    return score, contributing


# ---------------------------------------------------------------------------
# Nesting invariant check
# ---------------------------------------------------------------------------

def check_nesting(results: list[TierResult]) -> None:
    """Raise TierNestingError if Tier 3 fires without Tier 2 and Tier 1 in same horizon."""
    by_horizon: dict[str, dict[str, TierResult]] = {}
    for r in results:
        by_horizon.setdefault(r.horizon, {})[r.tier] = r

    for horizon, tier_map in by_horizon.items():
        t3 = tier_map.get("tier_3")
        if t3 and t3.fire:
            t2 = tier_map.get("tier_2")
            t1 = tier_map.get("tier_1")
            if not (t2 and t2.fire):
                raise TierNestingError(
                    f"Tier 3 fire in {horizon} without Tier 2 fire (nesting invariant violated)"
                )
            if not (t1 and t1.fire):
                raise TierNestingError(
                    f"Tier 3 fire in {horizon} without Tier 1 fire (nesting invariant violated)"
                )
