"""Historical replay of tiered alert system against modeldata.

Usage:
    cd projects/h2s
    uv run python -m h2s.defs.tiered_alerts.backtest \\
        --data /mnt/data/modeldata_h2s_nofill.parquet \\
        --output ./output/tier_backtest/ \\
        [--emit-stats]

Falls back to the public S3 URL if --data path doesn't exist.
Exits non-zero if per-horizon Tier 3 precision/recall miss design §6.1 targets.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from h2s.constants import (
    ALERT_SBIWTP_BASELINE_MGD,
    ALERT_LOCAL_TZ,
    STATIONS,
)
from .features import compute_horizon_features, _NIGHT_HOURS
from .tiers import (
    TierResult,
    check_nesting,
    compute_score,
    gate_tier1,
    gate_tier2,
    gate_tier3,
    load_config,
    HORIZON_ORDER,
    HORIZON_WINDOWS_H,
    TierNestingError,
)

_FALLBACK_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
)

# Per-horizon Tier 3 acceptance criteria (design §6.1)
_TARGETS: dict[str, tuple[float, float]] = {
    "nowcast":   (0.65, 0.80),
    "near":      (0.60, 0.75),
    "mid":       (0.55, 0.70),
    "day_ahead": (0.50, 0.65),
}


def _load_data(path: str | None) -> pd.DataFrame:
    if path and Path(path).exists():
        df = pd.read_parquet(path)
    else:
        print(f"Local path not found — loading from {_FALLBACK_URL}")
        df = pd.read_parquet(_FALLBACK_URL)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def _ensure_sbiwtp_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    if "sbiwtp_anomaly" not in df.columns and "sbiwtp_flow_mgd" in df.columns:
        df = df.copy()
        df["sbiwtp_anomaly"] = df["sbiwtp_flow_mgd"] - ALERT_SBIWTP_BASELINE_MGD
    return df


def _is_above_threshold(df: pd.DataFrame, site_name: str, t_start: pd.Timestamp, t_end: pd.Timestamp, ppb: float) -> bool:
    mask = (
        (df["site_name"] == site_name) &
        (df["time"] >= t_start) &
        (df["time"] < t_end) &
        (df["H2S"] >= ppb)
    )
    return bool(mask.any())


def _n_stations_above(df: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp, ppb: float) -> int:
    count = 0
    for site in STATIONS:
        if _is_above_threshold(df, site, t_start, t_end, ppb):
            count += 1
    return count


def _site_max_h2s(df: pd.DataFrame, site_name: str, t_start: pd.Timestamp, t_end: pd.Timestamp) -> float:
    sub = df[
        (df["site_name"] == site_name) &
        (df["time"] >= t_start) &
        (df["time"] < t_end)
    ]
    if sub.empty or "H2S" not in sub.columns:
        return 0.0
    v = sub["H2S"].max()
    return float(v) if not pd.isna(v) else 0.0


def run_backtest(
    df: pd.DataFrame,
    config: dict,
    emit_stats: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Step through history hourly, evaluate all tier × horizon cells.

    Returns (records_df, stats_dict).
    """
    # Require H2S column for labeling
    if "H2S" not in df.columns:
        raise ValueError("modeldata must have H2S column for backtest labels")
    if "sbiwtp_flow_mgd" not in df.columns:
        raise ValueError("modeldata must have sbiwtp_flow_mgd column")

    df = _ensure_sbiwtp_anomaly(df)

    # Evaluation timestamps: hourly, start from first complete day
    t_min = df["time"].min().ceil("1h")
    t_max = df["time"].max().floor("1h") - pd.Timedelta(hours=24)

    eval_times = pd.date_range(t_min, t_max, freq="1h", tz="UTC")

    records = []
    quiet_night_rows: list[dict] = []

    for t in eval_times:
        cell_features, _ = compute_horizon_features(df, t)

        # Evaluate all three tiers for all horizons
        tier_results: dict[str, list] = {}

        for tier_key in ("tier_1", "tier_2", "tier_3"):
            tier_horizon_results = []
            for horizon in HORIZON_ORDER:
                rows_by_station = {
                    site: cell_features.get((horizon, site), {})
                    for site in STATIONS
                }

                t1_gates = gate_tier1(rows_by_station, ALERT_SBIWTP_BASELINE_MGD)

                if tier_key == "tier_1":
                    gates = t1_gates
                    n_stations = sum(t1_gates.values())
                elif tier_key == "tier_2":
                    gates, n_stations = gate_tier2(rows_by_station, t1_gates)
                else:
                    _, n_t1 = gate_tier2(rows_by_station, t1_gates)
                    t2_gates, _ = gate_tier2(rows_by_station, t1_gates)
                    gates = gate_tier3(rows_by_station, t2_gates)
                    n_stations = n_t1

                gate_passed = any(gates.values())
                bw_features = (
                    cell_features.get((horizon, "NESTOR - BES"), {}) or
                    cell_features.get((horizon, "IB CIVIC CTR"), {}) or
                    {}
                )
                weights = config["tiers"][tier_key]["score_weights"]
                stats = config["quiet_night_stats"]
                score, contributing = compute_score(bw_features, weights, stats)
                fire = gate_passed and score >= 0.5

                t_start = t + pd.Timedelta(hours=HORIZON_WINDOWS_H[horizon][0])
                t_end   = t + pd.Timedelta(hours=HORIZON_WINDOWS_H[horizon][1])

                # Ground-truth labels from actual observations
                threshold_map = {"tier_1": 5.0, "tier_2": 10.0, "tier_3": 30.0}
                ppb = threshold_map[tier_key]

                n_exc = _n_stations_above(df, t_start, t_end, ppb)
                nb_max = df[
                    (df["site_name"] == "NESTOR - BES") &
                    (df["time"] >= t_start) & (df["time"] < t_end)
                ]["H2S"].max() if "H2S" in df.columns else float("nan")

                nb_max = float(nb_max) if not pd.isna(nb_max) else 0.0

                # Lead time: hours from evaluation to first exceedance in window
                exc_sub = df[
                    (df["H2S"] >= ppb) &
                    (df["time"] >= t_start) & (df["time"] < t_end)
                ]["time"]
                if exc_sub.empty:
                    lead_time = float("nan")
                else:
                    first_exc = exc_sub.min()
                    lead_time = float((first_exc - t).total_seconds() / 3600)

                # daytime_horizon
                sample = cell_features.get((horizon, "NESTOR - BES"), {})
                daytime = bool(sample.get("_daytime_horizon", False))

                records.append({
                    "evaluated_at":     t.isoformat(),
                    "tier":             tier_key,
                    "horizon":          horizon,
                    "gate_passed":      gate_passed,
                    "score":            round(score, 4),
                    "fired":            fire,
                    "daytime_horizon":  daytime,
                    "actual_max_h2s_nb": nb_max,
                    "actual_max_h2s_ib": _site_max_h2s(df, "IB CIVIC CTR", t_start, t_end),
                    "actual_max_h2s_sy": _site_max_h2s(df, "SAN YSIDRO", t_start, t_end),
                    "n_stations_exceeding": n_exc,
                    "lead_time_hours":  lead_time,
                })
                tier_horizon_results.append((horizon, fire, gate_passed))

            tier_results[tier_key] = tier_horizon_results

        # Nesting check per horizon (die loudly on violation)
        for horizon in HORIZON_ORDER:
            t3 = next((f for h, f, _ in tier_results.get("tier_3", []) if h == horizon), False)
            t2 = next((f for h, f, _ in tier_results.get("tier_2", []) if h == horizon), False)
            t1 = next((f for h, f, _ in tier_results.get("tier_1", []) if h == horizon), False)
            if t3 and not (t2 and t1):
                raise TierNestingError(
                    f"Nesting invariant violated at {t.isoformat()} horizon={horizon}"
                )

        # Collect quiet-night rows for --emit-stats
        if emit_stats:
            local_t = t.tz_convert(ALERT_LOCAL_TZ)
            if local_t.hour in _NIGHT_HOURS:
                nb = cell_features.get(("nowcast", "NESTOR - BES"), {})
                nb_max_val = nb_max  # type: ignore[name-defined]
                if nb_max_val < 1.0 and nb:
                    quiet_night_rows.append(nb)

    records_df = pd.DataFrame(records)

    # Compute per-horizon Tier 3 precision / recall / F1
    stats: dict = {}
    if not records_df.empty:
        t3 = records_df[records_df["tier"] == "tier_3"].copy()
        t3["actual"] = t3["n_stations_exceeding"] >= 1

        for horizon in HORIZON_ORDER:
            h_df = t3[t3["horizon"] == horizon]
            if h_df.empty:
                continue
            tp = int(( h_df["fired"] &  h_df["actual"]).sum())
            fp = int(( h_df["fired"] & ~h_df["actual"]).sum())
            fn = int((~h_df["fired"] &  h_df["actual"]).sum())
            prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            lt    = h_df.loc[h_df["fired"] & h_df["actual"], "lead_time_hours"]
            stats[horizon] = {
                "precision": round(prec, 3),
                "recall":    round(rec, 3),
                "f1":        round(f1, 3),
                "mean_lead_time_h": round(float(lt.mean()), 2) if not lt.empty else float("nan"),
                "n_events":  tp + fn,
                "n_fires":   tp + fp,
            }

    if emit_stats and quiet_night_rows:
        qn_df = pd.DataFrame(quiet_night_rows)
        print("\n--- Quiet-night feature statistics (for tiered_alerts.yaml) ---")
        for col in qn_df.select_dtypes(include="number").columns:
            print(f"  {col}: mean={qn_df[col].mean():.3f}  std={qn_df[col].std():.3f}")

    return records_df, stats


def main():
    parser = argparse.ArgumentParser(description="Backtest tiered H2S alert system")
    parser.add_argument("--data", help="Path to modeldata_h2s_nofill.parquet")
    parser.add_argument("--output", default="./output/tier_backtest", help="Output directory")
    parser.add_argument("--emit-stats", action="store_true", help="Print quiet-night feature stats")
    args = parser.parse_args()

    print("Loading data...")
    df = _load_data(args.data)
    print(f"  {len(df):,} rows, {df['time'].min()} – {df['time'].max()}")

    # Apply feature engineering
    try:
        from h2s.training.feature_builder import ensure_base_features
        df = ensure_base_features(df)
    except Exception as e:
        print(f"Warning: could not apply ensure_base_features: {e}")
    df = _ensure_sbiwtp_anomaly(df)

    config = load_config()
    print("Running backtest (hourly evaluation)...")
    records_df, stats = run_backtest(df, config, emit_stats=args.emit_stats)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_df.to_parquet(out_dir / "tier_backtest_records.parquet", index=False)
    print(f"\nSaved {len(records_df):,} records → {out_dir / 'tier_backtest_records.parquet'}")

    print("\n--- Per-horizon Tier 3 results ---")
    all_pass = True
    for horizon, s in stats.items():
        target_prec, target_rec = _TARGETS[horizon]
        prec_ok = s["precision"] >= target_prec
        rec_ok  = s["recall"]    >= target_rec
        status  = "✓" if prec_ok and rec_ok else "✗"
        print(
            f"  {status} {horizon:<12} "
            f"prec={s['precision']:.3f} (target≥{target_prec})  "
            f"rec={s['recall']:.3f} (target≥{target_rec})  "
            f"F1={s['f1']:.3f}  events={s['n_events']}  "
            f"lead_time={s['mean_lead_time_h']:.1f}h"
        )
        if not (prec_ok and rec_ok):
            all_pass = False

    if not all_pass:
        print("\n✗ Some horizons missed acceptance targets (design §6.1).")
        sys.exit(1)

    print("\n✓ All Tier 3 horizons meet acceptance targets.")


if __name__ == "__main__":
    main()
