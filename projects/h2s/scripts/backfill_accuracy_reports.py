#!/usr/bin/env python3
"""Backfill accuracy reporting scorecards for dates that have validation metrics.

This script:
1. Scans S3 for dates with usable metrics.json files (v1 or v2 schema)
2. Generates daily scorecards for each date
3. Generates rolling (7d/30d/90d) scorecards
4. Generates monthly scorecard for the previous complete month

Usage:
    cd projects/h2s
    uv run python scripts/backfill_accuracy_reports.py
    uv run python scripts/backfill_accuracy_reports.py --start 2026-03-01 --end 2026-04-23
    uv run python scripts/backfill_accuracy_reports.py --dry-run
"""

import argparse
import json
import os
import urllib.request
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

# Add project to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from h2s.defs.accuracy_reporting_pipeline import (
    ACCURACY_PREFIX,
    VALIDATION_PREFIX,
    ROLLING_WINDOWS_DAYS,
    AccuracyStore,
    build_period_scorecard,
    PeriodScorecard,
)
from h2s.resources.minio import S3Resource


def make_s3() -> S3Resource:
    return S3Resource(
        S3_BUCKET=os.environ.get("S3_BUCKET", "test"),
        S3_ADDRESS=os.environ.get("S3_ADDRESS", "oss.resilientservice.mooo.com"),
        S3_PORT=os.environ.get("S3_PORT", "443"),
        S3_USE_SSL=os.environ.get("S3_USE_SSL", "true").lower() == "true",
        S3_ACCESS_KEY=os.environ["S3_ACCESS_KEY"],
        S3_SECRET_KEY=os.environ["S3_SECRET_KEY"],
    )


def check_metrics(store: AccuracyStore, day: date) -> dict[str, Any] | None:
    """Read and validate metrics for a date. Returns metrics if usable, None otherwise."""
    m = store.read_day_metrics(day)
    if m is None:
        return None

    # v2 schema: check sites dict has confusion matrices
    sites = m.get("sites")
    if sites:
        has_cm = any(s.get("confusion_matrix") is not None for s in sites.values())
        if has_cm:
            return m
        return None

    # v1 schema: check flat confusion_matrix
    if m.get("confusion_matrix") is not None:
        return m

    return None


def main():
    parser = argparse.ArgumentParser(description="Backfill accuracy report scorecards")
    parser.add_argument("--start", default="2026-03-01", help="Start date")
    parser.add_argument("--end", default="2026-04-23", help="End date")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without writing")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    s3 = make_s3()
    store = AccuracyStore(s3)

    # Phase 1: Find all dates with usable metrics
    print(f"Scanning {start} to {end} for usable validation metrics...")
    usable_days = []
    cur = start
    while cur <= end:
        m = check_metrics(store, cur)
        if m is not None:
            usable_days.append(cur)
            # Show brief info
            sites = m.get("sites", {})
            if sites:
                total_matched = sum(s.get("n_matched_observations", 0) for s in sites.values())
                print(f"  {cur}  v2  sites={len(sites)}  matched={total_matched}")
            else:
                print(f"  {cur}  v1  site={m.get('site','?')}  matched={m.get('n_matched', '?')}")
        cur += timedelta(days=1)

    if not usable_days:
        print("\nNo usable metrics found. Run the validation backfill first.")
        return

    print(f"\nFound {len(usable_days)} days with usable metrics.")

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        print(f"  - {len(usable_days)} daily scorecards")
        print(f"  - Rolling scorecards: 7d, 30d, 90d")
        print(f"  - Monthly scorecard for previous complete month")
        print(f"  - latest.json")
        return

    # Phase 2: Generate daily scorecards
    print("\nGenerating daily scorecards...")
    daily_count = 0
    for day in usable_days:
        try:
            card = build_period_scorecard(store, day, day, scope="daily")
            key = f"{ACCURACY_PREFIX}/daily/{day.isoformat()}/scorecard.json"
            store.write_json(key, card.to_dict())
            ba = card.overall.get("balanced_accuracy")
            print(f"  {day}  ba={ba:.3f}" if ba else f"  {day}  written")
            daily_count += 1
        except Exception as e:
            print(f"  {day}  FAILED: {e}")

    print(f"\nWrote {daily_count} daily scorecards.")

    # Phase 3: Generate rolling scorecards
    print("\nGenerating rolling scorecards...")
    today = datetime.now(timezone.utc).date()
    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "windows": {},
    }
    for window in ROLLING_WINDOWS_DAYS:
        window_start = today - timedelta(days=window)
        try:
            card = build_period_scorecard(store, window_start, today, scope="rolling")
            key = f"{ACCURACY_PREFIX}/rolling/{window}d/scorecard.json"
            store.write_json(key, card.to_dict())
            ba = card.overall.get("balanced_accuracy")
            n_matched = card.overall.get("n_matched_observations", 0)
            print(f"  {window}d: ba={ba:.3f}, matched={n_matched}" if ba else f"  {window}d: no data")
            summary["windows"][f"{window}d"] = {
                "key": key,
                "overall": card.overall,
                "n_sites": len(card.sites),
            }
        except Exception as e:
            print(f"  {window}d: FAILED — {e}")

    # Write latest.json
    if summary["windows"]:
        store.write_json(f"{ACCURACY_PREFIX}/latest.json", summary)
        print(f"\nWrote latest.json with {len(summary['windows'])} windows.")

    # Phase 4: Generate monthly scorecard for previous complete month
    print("\nGenerating monthly scorecard...")
    first_of_this_month = today.replace(day=1)
    last_of_prev = first_of_this_month - timedelta(days=1)
    month_start = last_of_prev.replace(day=1)
    period = f"{month_start:%Y-%m}"
    try:
        card = build_period_scorecard(store, month_start, last_of_prev, scope="monthly")
        key = f"{ACCURACY_PREFIX}/monthly/{period}/scorecard.json"
        store.write_json(key, card.to_dict())
        ba = card.overall.get("balanced_accuracy")
        n_matched = card.overall.get("n_matched_observations", 0)
        print(f"  {period}: ba={ba:.3f}, matched={n_matched}" if ba else f"  {period}: written")
    except Exception as e:
        print(f"  {period}: FAILED — {e}")

    # Phase 5: Generate alert performance
    print("\nGenerating alert performance (30d)...")
    from h2s.defs.accuracy_reporting_pipeline import (
        CATEGORIES,
        class_precision_recall,
    )
    try:
        card = build_period_scorecard(store, today - timedelta(days=30), today, scope="rolling")
        overall_cm = card.overall.get("confusion_matrix") or [[0] * 3 for _ in range(3)]
        by_level: dict[str, dict[str, float | None]] = {}
        for cls in CATEGORIES:
            precision, recall = class_precision_recall(overall_cm, cls)
            f1 = None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            by_level[cls] = {"precision": precision, "recall": recall, "f1": f1}
        payload = {
            "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "window": "30d",
            "by_level": by_level,
            "overall": card.overall,
        }
        store.write_json(f"{ACCURACY_PREFIX}/alert_performance/30d.json", payload)
        print(f"  Written. Orange recall={by_level['orange'].get('recall')}")
    except Exception as e:
        print(f"  FAILED — {e}")

    print("\nDone! Check S3 at: tijuana/forecast/accuracy_reports/")


if __name__ == "__main__":
    main()
