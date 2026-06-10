"""Calibration-aligned baseline report for the production H2S system.

Loads the historical H2S parquet, applies a chronological 70/30 split per
site, computes the persistence baseline (h2s_lag_1h → ppb), and reports
Spearman + recall@30 + recall@100 per site and per regime
(`stable_atm` ∈ {0, 1}) — the metrics the calibration arc identified as
the headline measures on this heavy-tailed series.

Calibration findings folded in (see
tj_calibration/tijuana-dispersion-experiments/docs/calibration_status.md):
- Spearman is the headline (Pearson under-reports on heavy tails).
- Persistence at lag-1h is the autoregressive ceiling — any model must
  beat it to earn its keep.
- recall@100 is bounded ≈ 0.21 even autoregressively; exogenous = 0.00.
- Regime stratification matters: rank skill differs ≈ 2× between
  stable_atm=1 (calm) and stable_atm=0 (windy) subsets.
- Berry (NESTOR-BES) carries the operational stakes (242 hours >100 ppb).

Usage:
    cd projects/h2s
    uv run python scripts/calibration_baseline_report.py [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from h2s.constants import H2S_THRESHOLD_EXTREME, H2S_THRESHOLD_HIGH
from h2s.training.calibration_eval import (
    calibration_report,
    chronological_split,
    persistence_prediction,
)


def _filter_measured(df: pd.DataFrame, max_ppb: float = 500.0) -> pd.DataFrame:
    """Match the multi-station trainer's data hygiene: measured rows, capped, sorted."""
    out = df[(df["h2s_measured"] == True) & (df["H2S"] <= max_ppb)].copy()  # noqa: E712
    out["H2S"] = out["H2S"].clip(lower=0)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    return out.sort_values(["site_name", "time"]).reset_index(drop=True)


def _per_site_chronological_split(
    df: pd.DataFrame, train_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological 70/30 within each site so every site gets a holdout window."""
    trains, tests = [], []
    for _, sub in df.groupby("site_name", sort=False):
        tr, te = chronological_split(sub, train_fraction)
        trains.append(tr)
        tests.append(te)
    return pd.concat(trains).reset_index(drop=True), pd.concat(tests).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_parquet = repo_root / "data" / "modeldata_h2s_nofill.parquet"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", type=Path, default=default_parquet)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON here; default stdout")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--lag-hours", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.parquet.exists():
        print(f"ERROR: parquet not found at {args.parquet}", file=sys.stderr)
        return 1

    df_raw = pd.read_parquet(args.parquet)
    df = _filter_measured(df_raw)
    train, test = _per_site_chronological_split(df, args.train_fraction)

    # Persistence is computed within the test slice — predictions at row t
    # use the previous test-slice observation at the same site. The very
    # first row per site is dropped (no lag available).
    test_sorted = test.sort_values(["site_name", "time"]).reset_index(drop=True)
    pred = persistence_prediction(test_sorted, lag_hours=args.lag_hours)
    eval_mask = ~pred.isna()
    test_eval = test_sorted.loc[eval_mask].reset_index(drop=True)
    pred_eval = pred.loc[eval_mask].reset_index(drop=True)

    report = calibration_report(
        test_eval,
        pred_eval.to_numpy(),
        thresholds=(H2S_THRESHOLD_HIGH, H2S_THRESHOLD_EXTREME),
    )

    payload = {
        "config": {
            "parquet": str(args.parquet),
            "train_fraction": args.train_fraction,
            "lag_hours": args.lag_hours,
            "thresholds_ppb": [H2S_THRESHOLD_HIGH, H2S_THRESHOLD_EXTREME],
            "n_train": len(train),
            "n_test": len(test),
            "n_test_eval": len(test_eval),
            "train_time_range": [str(train["time"].min()), str(train["time"].max())],
            "test_time_range": [str(test["time"].min()), str(test["time"].max())],
        },
        "persistence_baseline": {
            "overall": report.overall,
            "per_site": report.per_site,
            "per_regime": report.per_regime,
        },
    }

    output = json.dumps(payload, indent=2, default=str)
    if args.out:
        args.out.write_text(output)
        print(f"Wrote {len(output)} bytes to {args.out}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
