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
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from h2s.constants import (
    ALERT_SBIWTP_BASELINE_MGD,
    ALERT_LOCAL_TZ,
    STATIONS,
)
from .features import compute_horizon_features, _NIGHT_HOURS
from .tiers import (
    compute_score,
    gate_tier1,
    gate_tier2,
    gate_tier3,
    load_config,
    HORIZON_ORDER,
    HORIZON_WINDOWS_H,
    TIER3_TARGETS,
    TierNestingError,
)

_FALLBACK_URL = (
    "https://oss.resilientservice.mooo.com/resilentpublic/"
    "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"
)


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
    v: object = sub["H2S"].max()
    try:
        fv = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if np.isnan(fv) else fv


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

    _threshold_map = {"tier_1": 5.0, "tier_2": 10.0, "tier_3": 30.0}

    for t in eval_times:
        cell_features, _ = compute_horizon_features(df, t)
        tier_results: dict[str, list] = {}

        # Process horizons first so we can enforce tier nesting (T3 fires only if T2 fired).
        # Tiers use different score weights; without this ordering Tier 3's extra
        # temp/dewpoint weights could push it above 0.5 while Tier 2 stays below.
        for horizon in HORIZON_ORDER:
            rows_by_station = {
                site: cell_features.get((horizon, site), {})
                for site in STATIONS
            }
            t1_gates = gate_tier1(rows_by_station)
            t2_gates, n_t1 = gate_tier2(rows_by_station, t1_gates)
            t3_gates = gate_tier3(rows_by_station, t2_gates)

            bw_features = (
                cell_features.get((horizon, "NESTOR - BES"), {}) or
                cell_features.get((horizon, "IB CIVIC CTR"), {}) or
                {}
            )

            # Single-station mode: use summer-calibrated stats to avoid seasonal bias
            # (single-station periods cluster in Jun–Sep when temps are 2–4 °C above baseline).
            # _h2s_active is set by compute_horizon_features before data replication,
            # so it correctly reflects genuine H2S sensor coverage.
            n_h2s_active = sum(1 for row in rows_by_station.values() if row.get("_h2s_active", False))
            single_station_mode = (n_h2s_active <= 1)
            if single_station_mode and "single_station_quiet_night_stats" in config:
                stats = config["single_station_quiet_night_stats"]
                score_threshold = config.get("score_thresholds", {}).get("single_station", 0.5)
            else:
                stats = config["quiet_night_stats"]
                score_threshold = config.get("score_thresholds", {}).get("multi_station", 0.5)

            t_start = t + pd.Timedelta(hours=HORIZON_WINDOWS_H[horizon][0])
            t_end   = t + pd.Timedelta(hours=HORIZON_WINDOWS_H[horizon][1])

            _nb_raw = (
                df.loc[
                    (df["site_name"] == "NESTOR - BES") &
                    (df["time"] >= t_start) & (df["time"] < t_end),
                    "H2S",
                ].max()
                if "H2S" in df.columns else float("nan")
            )
            try:
                _nb_f = float(_nb_raw)  # type: ignore[arg-type]
                nb_max: float = 0.0 if np.isnan(_nb_f) else _nb_f
            except (TypeError, ValueError):
                nb_max = 0.0

            ib_max  = _site_max_h2s(df, "IB CIVIC CTR", t_start, t_end)
            sy_max  = _site_max_h2s(df, "SAN YSIDRO",   t_start, t_end)
            sample  = cell_features.get((horizon, "NESTOR - BES"), {})
            daytime = bool(sample.get("_daytime_horizon", False))

            parent_fired = True  # Tier 1 has no parent constraint
            for tier_key, tier_gates in [
                ("tier_1", t1_gates),
                ("tier_2", t2_gates),
                ("tier_3", t3_gates),
            ]:
                n_stations = sum(t1_gates.values()) if tier_key == "tier_1" else n_t1
                ppb = _threshold_map[tier_key]

                gate_passed = any(tier_gates.values())
                weights = config["tiers"][tier_key]["score_weights"]
                score, _ = compute_score(bw_features, weights, stats)
                # Nesting enforced at fire time: Tier N fires only if Tier N-1 fired
                fire = gate_passed and score >= score_threshold and parent_fired

                n_exc = _n_stations_above(df, t_start, t_end, ppb)
                exc_sub: pd.Series = df.loc[
                    (df["H2S"] >= ppb) &
                    (df["time"] >= t_start) & (df["time"] < t_end),
                    "time",
                ]
                if exc_sub.empty:
                    lead_time = float("nan")
                else:
                    first_exc = exc_sub.min()
                    lead_time = float((first_exc - t).total_seconds() / 3600)

                records.append({
                    "evaluated_at":         t.isoformat(),
                    "tier":                 tier_key,
                    "horizon":              horizon,
                    "gate_passed":          gate_passed,
                    "score":                round(score, 4),
                    "fired":                fire,
                    "daytime_horizon":      daytime,
                    "single_station_mode":  single_station_mode,
                    "actual_max_h2s_nb":    nb_max,
                    "actual_max_h2s_ib":    ib_max,
                    "actual_max_h2s_sy":    sy_max,
                    "n_stations_exceeding": n_exc,
                    "lead_time_hours":      lead_time,
                })
                tier_results.setdefault(tier_key, []).append((horizon, fire, gate_passed))
                parent_fired = fire

        # Nesting is guaranteed by construction above; keep check as a safety assertion
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
                nc_start = t
                nc_end   = t + pd.Timedelta(hours=3)
                _nc_raw  = (
                    df.loc[
                        (df["site_name"] == "NESTOR - BES") &
                        (df["time"] >= nc_start) & (df["time"] < nc_end),
                        "H2S",
                    ].max()
                    if "H2S" in df.columns else float("nan")
                )
                try:
                    _nc_f = float(_nc_raw)  # type: ignore[arg-type]
                    nowcast_nb_max: float = 0.0 if np.isnan(_nc_f) else _nc_f
                except (TypeError, ValueError):
                    nowcast_nb_max = 0.0
                if nowcast_nb_max < 1.0 and nb:
                    quiet_night_rows.append(nb)

    records_df = pd.DataFrame(records)

    # Compute per-horizon Tier 3 precision / recall / F1
    stats: dict = {}
    if not records_df.empty:
        t3 = records_df[records_df["tier"] == "tier_3"].copy()
        t3["actual"] = t3["n_stations_exceeding"] >= 1

        for horizon in HORIZON_ORDER:
            h_df: pd.DataFrame = t3.loc[t3["horizon"] == horizon]
            if h_df.empty:
                continue
            tp = int(( h_df["fired"] &  h_df["actual"]).sum())
            fp = int(( h_df["fired"] & ~h_df["actual"]).sum())
            fn = int((~h_df["fired"] &  h_df["actual"]).sum())
            prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            lt: pd.Series = h_df.loc[h_df["fired"] & h_df["actual"], "lead_time_hours"]
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


def _compute_metrics(group: pd.DataFrame) -> dict:
    tp = int(( group["fired"] &  group["actual"]).sum())
    fp = int(( group["fired"] & ~group["actual"]).sum())
    fn = int((~group["fired"] &  group["actual"]).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    lt   = group.loc[group["fired"] & group["actual"], "lead_time_hours"]
    return {
        "precision":        round(prec, 3),
        "recall":           round(rec, 3),
        "f1":               round(f1, 3),
        "mean_lead_time_h": round(float(lt.mean()), 2) if not lt.empty else float("nan"),
        "n_events":         tp + fn,
        "n_fires":          tp + fp,
        "n_rows":           len(group),
    }


def generate_report(records_df: pd.DataFrame, output_dir: Path | None = None) -> pd.DataFrame:
    """Compute monthly × horizon × day/night metrics from a backtest records DataFrame.

    Works on the parquet saved by run_backtest — no re-evaluation needed.
    Returns a tidy DataFrame and optionally saves CSV + per-tier summary prints.
    """
    df = records_df.copy()
    df["evaluated_at"] = pd.to_datetime(df["evaluated_at"], utc=True)
    df["month"] = df["evaluated_at"].dt.to_period("M").astype(str)
    df["period"] = df["daytime_horizon"].map(lambda x: "day" if x else "night")
    df["actual"] = df["n_stations_exceeding"] >= 1

    rows = []
    for tier in ("tier_1", "tier_2", "tier_3"):
        tier_df: pd.DataFrame = df.loc[df["tier"] == tier].copy()
        threshold_map = {"tier_1": 5.0, "tier_2": 10.0, "tier_3": 30.0}
        ppb = threshold_map[tier]
        tier_df["actual"] = tier_df["actual_max_h2s_nb"] >= ppb

        for keys3, grp3 in tier_df.groupby(["month", "horizon", "period"]):
            k3 = tuple(keys3)  # type: ignore[arg-type]
            month_k, horizon_k, period_k = str(k3[0]), str(k3[1]), str(k3[2])
            m = _compute_metrics(grp3)  # type: ignore[arg-type]
            rows.append({"tier": tier, "month": month_k, "horizon": horizon_k, "period": period_k, **m})

        # Overall (all months, day+night) per tier × horizon
        for keys1, grp1 in tier_df.groupby("horizon"):
            horizon_k = str(keys1)
            m = _compute_metrics(grp1)  # type: ignore[arg-type]
            rows.append({"tier": tier, "month": "ALL", "horizon": horizon_k, "period": "all", **m})

    report_df = pd.DataFrame(rows).sort_values(["tier", "month", "horizon", "period"])

    # Print summary for Tier 3 (the acceptance-criteria tier)
    print("\n=== Monthly Tier 3 report (day / night) ===")
    t3: pd.DataFrame = report_df.loc[
        (report_df["tier"] == "tier_3") & (report_df["month"] != "ALL")
    ]
    months_list: list[str] = sorted(t3["month"].unique().tolist())
    for month in months_list:
        print(f"\n  {month}")
        for horizon in HORIZON_ORDER:
            for period in ("day", "night"):
                row: pd.DataFrame = t3.loc[
                    (t3["month"] == month) &
                    (t3["horizon"] == horizon) &
                    (t3["period"] == period)
                ]
                if row.empty:
                    continue
                r = row.iloc[0]
                target_prec, target_rec = TIER3_TARGETS[horizon]
                ok = "✓" if r["precision"] >= target_prec and r["recall"] >= target_rec else "✗"
                print(
                    f"    {ok} {horizon:<12} {period:<6}  "
                    f"prec={r['precision']:.3f}  rec={r['recall']:.3f}  "
                    f"F1={r['f1']:.3f}  events={r['n_events']}  "
                    f"lead={r['mean_lead_time_h']:.1f}h"
                )

    print("\n=== Overall Tier 3 (all months) ===")
    t3_all: pd.DataFrame = report_df.loc[
        (report_df["tier"] == "tier_3") & (report_df["month"] == "ALL")
    ]
    for _, r in t3_all.iterrows():
        horizon_str = str(r["horizon"])
        target_prec, target_rec = TIER3_TARGETS.get(horizon_str, (0.0, 0.0))
        ok = "✓" if r["precision"] >= target_prec and r["recall"] >= target_rec else "✗"
        print(
            f"  {ok} {horizon_str:<12}  "
            f"prec={r['precision']:.3f} (≥{target_prec})  "
            f"rec={r['recall']:.3f} (≥{target_rec})  "
            f"F1={r['f1']:.3f}  events={r['n_events']}  "
            f"lead={r['mean_lead_time_h']:.1f}h"
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "tier_backtest_report.csv"
        report_df.to_csv(out_path, index=False)
        print(f"\nReport saved → {out_path}")

    return report_df


def generate_html_report(records_df: pd.DataFrame, output_dir: Path) -> Path:
    """Build a self-contained HTML report with embedded charts from backtest records."""
    from .report import build_html_report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_df = generate_report(records_df, output_dir=None)
    html = build_html_report(records_df, report_df)
    out_path = output_dir / "tier_backtest_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"HTML report saved → {out_path}")
    return out_path


def generate_monthly_site(
    records_df: pd.DataFrame,
    output_dir: Path,
    months_back: int = 12,
) -> Path:
    """Write one HTML page per month + index.html into output_dir/monthly/."""
    from .report import generate_static_site
    report_df = generate_report(records_df, output_dir=None)
    site_dir = output_dir / "monthly"
    return generate_static_site(records_df, report_df, site_dir, months_back=months_back)


def main():
    parser = argparse.ArgumentParser(description="Backtest tiered H2S alert system")
    parser.add_argument("--data", help="Path to modeldata_h2s_nofill.parquet")
    parser.add_argument("--output", default="./output/tier_backtest", help="Output directory")
    parser.add_argument("--emit-stats", action="store_true", help="Print quiet-night feature stats")
    parser.add_argument(
        "--report-only",
        metavar="RECORDS_PARQUET",
        help="Skip backtest; load existing records parquet and generate monthly report",
    )
    parser.add_argument("--html", action="store_true", help="Also write HTML report with charts")
    parser.add_argument("--monthly-html", action="store_true",
                        help="Write one HTML page per month + index (past 12 months)")
    parser.add_argument("--months-back", type=int, default=12,
                        help="How many months to include in --monthly-html (default: 12)")
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.report_only:
        print(f"Loading records from {args.report_only} ...")
        records_df = pd.read_parquet(args.report_only)
        print(f"  {len(records_df):,} records")
        if args.monthly_html:
            generate_monthly_site(records_df, out_dir, months_back=args.months_back)
        if args.html:
            generate_html_report(records_df, output_dir=out_dir)
        if not args.monthly_html and not args.html:
            generate_report(records_df, output_dir=out_dir)
        return

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

    out_dir.mkdir(parents=True, exist_ok=True)
    records_df.to_parquet(out_dir / "tier_backtest_records.parquet", index=False)
    print(f"\nSaved {len(records_df):,} records → {out_dir / 'tier_backtest_records.parquet'}")

    # Snapshot the weights and gates used for this run so results are reproducible.
    snapshot = {
        "run_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config_source": str(load_config.__module__),
        "gates": config.get("gates"),
        "tiers": config.get("tiers"),
        "quiet_night_stats": config.get("quiet_night_stats"),
    }
    snapshot_path = out_dir / "weights_snapshot.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    print(f"Saved weights snapshot → {snapshot_path}")

    # Overall Tier 3 acceptance check
    print("\n--- Per-horizon Tier 3 results ---")
    all_pass = True
    for horizon, s in stats.items():
        target_prec, target_rec = TIER3_TARGETS[horizon]
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

    # Monthly + day/night report
    if args.monthly_html:
        generate_monthly_site(records_df, out_dir, months_back=args.months_back)
    if args.html:
        generate_html_report(records_df, output_dir=out_dir)
    if not args.monthly_html and not args.html:
        generate_report(records_df, output_dir=out_dir)

    if not all_pass:
        print("\n✗ Some horizons missed acceptance targets (design §6.1).")
        sys.exit(1)

    print("\n✓ All Tier 3 horizons meet acceptance targets.")


if __name__ == "__main__":
    main()
