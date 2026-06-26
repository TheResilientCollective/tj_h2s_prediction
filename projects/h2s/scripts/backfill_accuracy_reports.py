#!/usr/bin/env python3
"""Backfill accuracy reporting scorecards (daily / rolling / monthly + alert perf).

Two sources, selected with ``--source``:

  metrics (default) — HISTORICAL HINDCAST. Aggregates scorecards from the
    per-date ``daily_station`` ``metrics.json`` files written by
    ``backfill_validation.py --daily-station``. Covers the full observation
    history. Probability-calibration metrics (Brier / ECE / prob-recall) are
    unavailable in this mode — ``metrics.json`` carries confusion matrices, not
    the per-row p5/p10/p30 columns — so those fields are ``null``.

  store — FORWARD / OPERATIONAL. Builds scorecards from the parquet forecast
    validation store (``FORECAST_VALIDATION_STORE_PATH``) via the production
    builder ``build_period_scorecard``. Includes calibration metrics, but only
    covers dates that have real product runs (populated by
    ``station_forecast_validation_rebuild_job``).

Both write the same ``PeriodScorecard`` JSON schema to
``tijuana/forecast/accuracy_reports/{daily,rolling,monthly}/.../scorecard.json``
(+ ``latest.json`` and ``alert_performance/30d.json``), so a single dashboard
reads either. Run ``metrics`` for history, then ``store`` so recent product-run
days are overwritten with the calibrated, authoritative scorecards.

Usage:
    cd projects/h2s
    set -a && source .env && set +a

    # Full-history hindcast scorecards from metrics.json
    uv run python scripts/backfill_accuracy_reports.py --source metrics \
        --start 2023-11-20 --end 2026-06-24

    # Forward scorecards from the parquet validation store (product runs)
    uv run python scripts/backfill_accuracy_reports.py --source store

    # Dry run (scan only, no writes)
    uv run python scripts/backfill_accuracy_reports.py --dry-run
"""

import argparse
import json
import os
import urllib.request
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

# Add project to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from h2s.constants import H2S_CLASS_NAMES, VALIDATION_PATH, date_path
from h2s.defs.accuracy_reporting_pipeline import (
    ACCURACY_PREFIX,
    CATEGORIES,
    ROLLING_WINDOWS_DAYS,
    AccuracyStore,
    PeriodScorecard,
    SiteScorecard,
    balanced_accuracy,
    build_period_scorecard,
    class_precision_recall,
    combine_confusion_matrices,
    false_alarm_rate_for_orange,
    overall_from_sites,
)
from h2s.resources.minio import S3Resource

# metrics.json confusion matrices use H2S_CLASS_NAMES axis order
# (green/orange/yellow), whereas the accuracy module's helpers assume
# CATEGORIES order (green/yellow/orange). _METRICS_TO_CATEGORIES maps a
# metrics.json class index → the CATEGORIES index.
_METRICS_TO_CATEGORIES = [CATEGORIES.index(c) for c in H2S_CLASS_NAMES]


def make_s3() -> S3Resource:
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
        S3_PORT=os.environ.get("S3_PORT", "443"),
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# metrics.json source (historical hindcast)
# ---------------------------------------------------------------------------

def _remap_cm(cm: list[list[int]]) -> list[list[int]]:
    """Reorder a metrics.json confusion matrix into CATEGORIES axis order."""
    src = np.asarray(cm, dtype=int)
    out = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for i, ni in enumerate(_METRICS_TO_CATEGORIES):
        for j, nj in enumerate(_METRICS_TO_CATEGORIES):
            out[ni, nj] = src[i, j]
    return out.tolist()


def read_day_metrics(s3: S3Resource, day: date) -> dict[str, Any] | None:
    """Read the daily_station metrics.json for a date (root copy as fallback)."""
    dp = date_path(day)
    for path in (
        f"{VALIDATION_PATH}/{dp}/daily_station/metrics.json",
        f"{VALIDATION_PATH}/{dp}/metrics.json",
    ):
        try:
            with urllib.request.urlopen(s3.publicUrl(path=path), timeout=10) as resp:
                data = json.loads(resp.read())
        except Exception:
            continue
        sites = data.get("sites")
        if sites and any(s.get("confusion_matrix") for s in sites.values()):
            return data
    return None


def _site_card_from_cm(site: str, cm: list[list[int]]) -> SiteScorecard:
    """Build a SiteScorecard from a confusion matrix (no probability metrics)."""
    orange_precision, orange_recall = class_precision_recall(cm, "orange")
    n = int(np.asarray(cm).sum())
    return SiteScorecard(
        site=site,
        n_predictions=n,
        n_matched_observations=n,
        balanced_accuracy=balanced_accuracy(cm),
        orange_recall=orange_recall,
        orange_precision=orange_precision,
        false_alarm_rate=false_alarm_rate_for_orange(cm),
        brier_score_p5=None,
        brier_score_p10=None,
        brier_score_p30=None,
        ece_p5=None,
        ece_p10=None,
        ece_p30=None,
        prob_recall_p5=None,
        prob_recall_p10=None,
        prob_recall_p30=None,
        confusion_matrix=cm,
    )


def build_metrics_scorecard(
    s3: S3Resource, start: date, end: date, scope: str
) -> PeriodScorecard | None:
    """Aggregate per-station metrics.json confusion matrices over [start, end]."""
    site_cms: dict[str, list[list[list[int]]]] = {}
    day = start
    while day <= end:
        m = read_day_metrics(s3, day)
        if m:
            for site, sm in m.get("sites", {}).items():
                cm = sm.get("confusion_matrix")
                if cm:
                    site_cms.setdefault(site, []).append(_remap_cm(cm))
        day += timedelta(days=1)

    if not site_cms:
        return None

    cards = [
        _site_card_from_cm(site, combine_confusion_matrices(cms))
        for site, cms in sorted(site_cms.items())
    ]
    return PeriodScorecard(
        scope=scope,
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        generated_at=_now_iso(),
        sites={c.site: asdict(c) for c in cards},
        overall=overall_from_sites(cards),
    )


# ---------------------------------------------------------------------------
# parquet store source (forward / operational)
# ---------------------------------------------------------------------------

def build_store_scorecard(
    store: AccuracyStore, start: date, end: date, scope: str
) -> PeriodScorecard | None:
    """Build a scorecard from the parquet validation store; None when empty."""
    try:
        return build_period_scorecard(store, start, end, scope)
    except Exception:
        # build_period_scorecard raises dg.Failure when the window has no rows.
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backfill accuracy report scorecards")
    parser.add_argument("--start", default="2026-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-23", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--source",
        choices=["metrics", "store"],
        default="metrics",
        help="Scorecard data source: 'metrics' (historical hindcast from "
        "metrics.json) or 'store' (parquet validation store from product runs)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan only; no writes")
    parser.add_argument(
        "--skip-daily",
        action="store_true",
        help="Skip per-date daily scorecards (write only rolling/monthly/latest/alert aggregates)",
    )
    parser.add_argument(
        "--skip-aggregates",
        action="store_true",
        help="Skip rolling/monthly/latest/alert aggregates (write only per-date daily scorecards). "
        "Use when a sparse source (e.g. --source store) should refresh recent daily cards "
        "without clobbering the rich historical aggregates.",
    )
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    s3 = make_s3()
    store = AccuracyStore(s3)

    # Bind the per-window builder for the chosen source.
    if args.source == "store":
        builder: Callable[[date, date, str], PeriodScorecard | None] = (
            lambda a, b, scope: build_store_scorecard(store, a, b, scope)
        )
    else:
        builder = lambda a, b, scope: build_metrics_scorecard(s3, a, b, scope)

    print(f"Source: {args.source}  bucket={s3.S3_BUCKET}")

    # Phase 1: find usable days (a single-day scorecard builds successfully).
    print(f"Scanning {start} to {end} for usable validation data...")
    usable_days: list[date] = []
    cur = start
    while cur <= end:
        card = builder(cur, cur, "daily")
        if card is not None:
            usable_days.append(cur)
            n = card.overall.get("n_matched_observations", 0)
            print(f"  {cur}  sites={len(card.sites)}  matched={n}")
        cur += timedelta(days=1)

    if not usable_days:
        hint = (
            "Run backfill_validation.py --daily-station first."
            if args.source == "metrics"
            else "Run station_forecast_validation_rebuild_job first."
        )
        print(f"\nNo usable validation data found. {hint}")
        return

    print(f"\nFound {len(usable_days)} days with usable data.")

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        print(f"  - {len(usable_days)} daily scorecards")
        print(f"  - Rolling scorecards: {', '.join(f'{w}d' for w in ROLLING_WINDOWS_DAYS)}")
        print("  - Monthly scorecard for previous complete month")
        print("  - latest.json + alert_performance/30d.json")
        return

    # Phase 2: daily scorecards
    if args.skip_daily:
        print("\nSkipping daily scorecards (--skip-daily).")
    else:
        print("\nGenerating daily scorecards...")
        daily_count = 0
        for day in usable_days:
            card = builder(day, day, "daily")
            if card is None:
                continue
            key = f"{ACCURACY_PREFIX}/daily/{date_path(day)}/scorecard.json"
            store.write_json(key, card.to_dict())
            ba = card.overall.get("balanced_accuracy")
            print(f"  {day}  ba={ba:.3f}" if ba is not None else f"  {day}  written")
            daily_count += 1
        print(f"\nWrote {daily_count} daily scorecards.")

    if args.skip_aggregates:
        print("\nSkipping rolling/monthly/latest/alert aggregates (--skip-aggregates).")
        print("\nDone! Check S3 at: tijuana/forecast/accuracy_reports/")
        return

    # Phase 3: rolling scorecards (anchored at today)
    print("\nGenerating rolling scorecards...")
    today = datetime.now(timezone.utc).date()
    summary: dict[str, Any] = {"generated_at": _now_iso(), "windows": {}}
    for window in ROLLING_WINDOWS_DAYS:
        card = builder(today - timedelta(days=window), today, "rolling")
        if card is None:
            print(f"  {window}d: no data")
            continue
        key = f"{ACCURACY_PREFIX}/rolling/{window}d/scorecard.json"
        store.write_json(key, card.to_dict())
        ba = card.overall.get("balanced_accuracy")
        n_matched = card.overall.get("n_matched_observations", 0)
        print(f"  {window}d: ba={ba:.3f}, matched={n_matched}" if ba is not None else f"  {window}d: written")
        summary["windows"][f"{window}d"] = {
            "key": key,
            "overall": card.overall,
            "n_sites": len(card.sites),
        }

    if summary["windows"]:
        store.write_json(f"{ACCURACY_PREFIX}/latest.json", summary)
        print(f"\nWrote latest.json with {len(summary['windows'])} windows.")

    # Phase 4: monthly scorecard for the previous complete month
    print("\nGenerating monthly scorecard...")
    first_of_this_month = today.replace(day=1)
    last_of_prev = first_of_this_month - timedelta(days=1)
    month_start = last_of_prev.replace(day=1)
    period = f"{month_start:%Y-%m}"
    card = builder(month_start, last_of_prev, "monthly")
    if card is None:
        print(f"  {period}: no data")
    else:
        key = f"{ACCURACY_PREFIX}/monthly/{period}/scorecard.json"
        store.write_json(key, card.to_dict())
        ba = card.overall.get("balanced_accuracy")
        n_matched = card.overall.get("n_matched_observations", 0)
        print(f"  {period}: ba={ba:.3f}, matched={n_matched}" if ba is not None else f"  {period}: written")

    # Phase 5: alert performance (30d rolling, per-level precision/recall)
    print("\nGenerating alert performance (30d)...")
    card = builder(today - timedelta(days=30), today, "rolling")
    if card is None:
        print("  no data")
    else:
        overall_cm = card.overall.get("confusion_matrix") or [[0] * len(CATEGORIES) for _ in CATEGORIES]
        by_level: dict[str, dict[str, float | None]] = {}
        for cls in CATEGORIES:
            precision, recall = class_precision_recall(overall_cm, cls)
            f1 = None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            by_level[cls] = {"precision": precision, "recall": recall, "f1": f1}
        payload = {
            "generated_at": _now_iso(),
            "window": "30d",
            "by_level": by_level,
            "overall": card.overall,
        }
        store.write_json(f"{ACCURACY_PREFIX}/alert_performance/30d.json", payload)
        print(f"  Written. Orange recall={by_level['orange'].get('recall')}")

    print("\nDone! Check S3 at: tijuana/forecast/accuracy_reports/")


if __name__ == "__main__":
    main()
