#!/usr/bin/env python3
"""Diagnose which validation metrics.json files exist on S3 and whether they contain usable data.

Usage:
    cd projects/h2s
    uv run python scripts/check_validation_data.py
    uv run python scripts/check_validation_data.py --start 2026-01-01 --end 2026-04-23
"""

import argparse
import json
import urllib.request
from datetime import date, timedelta


S3_BASE = "https://oss.resilientservice.mooo.com"
BUCKET = "test"
VALIDATION_PREFIX = "tijuana/forecast/validation"
HOURLY_PREFIX = "tijuana/forecast/hourly"


def check_url(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def check_url_exists(url: str) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Check S3 validation data inventory")
    parser.add_argument("--start", default="2026-03-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-23", help="End date (YYYY-MM-DD)")
    parser.add_argument("--bucket", default=BUCKET)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    bucket = args.bucket

    base = f"{S3_BASE}/{bucket}"

    usable = []
    empty = []
    missing = []
    has_predictions = []

    cur = start
    while cur <= end:
        ds = cur.isoformat()

        # Check root metrics.json (what accuracy_reporting reads)
        root_url = f"{base}/{VALIDATION_PREFIX}/{ds}/metrics.json"
        root_data = check_url(root_url)

        # Check hourly subdir
        hourly_url = f"{base}/{VALIDATION_PREFIX}/{ds}/hourly/metrics.json"
        hourly_data = check_url(hourly_url)

        # Check if predictions exist for any hour
        y, m, d = cur.strftime("%Y"), cur.strftime("%m"), cur.strftime("%d")
        pred_exists = any(
            check_url_exists(
                f"{base}/{HOURLY_PREFIX}/model=nestor_xgboost/year={y}/month={m}/day={d}/hour={h}/h2s_predictions.csv"
            )
            for h in ["00", "06", "12", "18"]
        )
        if pred_exists:
            has_predictions.append(ds)

        data = root_data or hourly_data
        source = "root" if root_data else ("hourly" if hourly_data else None)

        if data is None:
            missing.append(ds)
            status = "MISSING"
            detail = f"  predictions={'YES' if pred_exists else 'no'}"
        else:
            sites = data.get("sites") or {}
            n_sites = len(sites)
            has_cm = any(
                s.get("confusion_matrix") is not None for s in sites.values()
            )
            total_matched = sum(
                s.get("n_matched_observations", 0) for s in sites.values()
            )
            if has_cm and total_matched > 0:
                usable.append(ds)
                status = "USABLE"
                ba = next(
                    (s.get("balanced_accuracy") for s in sites.values() if s.get("balanced_accuracy")),
                    None,
                )
                detail = f"  source={source} sites={n_sites} matched={total_matched} ba={ba:.3f}" if ba else f"  source={source} sites={n_sites} matched={total_matched}"
            else:
                empty.append(ds)
                status = "EMPTY"
                detail = f"  source={source} sites={n_sites} matched={total_matched} cm={'yes' if has_cm else 'NO'}"

        print(f"{ds}  {status}{detail}")
        cur += timedelta(days=1)

    print("\n" + "=" * 60)
    print(f"Date range: {start} to {end} ({(end - start).days + 1} days)")
    print(f"  Usable metrics:    {len(usable)}")
    print(f"  Empty metrics:     {len(empty)}")
    print(f"  Missing metrics:   {len(missing)}")
    print(f"  Has predictions:   {len(has_predictions)}")
    print(f"  Backfillable:      {len(set(has_predictions) - set(usable) - set(empty))}")

    if missing and has_predictions:
        backfillable = sorted(set(has_predictions) & set(missing))
        if backfillable:
            print(f"\nDates with predictions but no metrics (need validation backfill):")
            for ds in backfillable:
                print(f"  uv run dg launch --job daily_validation_job --partition {ds}")

    if usable:
        print(f"\nDates with usable metrics (need accuracy reporting backfill):")
        for ds in usable:
            print(f"  uv run dg launch --job accuracy_reporting_job --partition {ds}")


if __name__ == "__main__":
    main()
