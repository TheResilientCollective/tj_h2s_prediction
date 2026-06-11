"""Preview-forecast NESTOR-BES H2S using the Evidence and Lean candidate sets.

For each candidate feature set:
  1. Train a production-config XGBoost regressor on the historical 70 %
     (same chronological split as feature_ablation.py).
  2. Evaluate on the held-out 30 % so the manifest carries honest holdout
     metrics.
  3. Apply the trained model to the latest 15-min forecast input (data/
     modeldata_forecast_15min.csv by default) to produce a time-series
     forecast.

Writes:
  output/forecast_evidence.csv   — per-timestep predicted ppb + category
  output/forecast_lean.csv
  output/run_manifest.json       — self-describing record of the run

The manifest is the contract with downstream consumers (e.g. a dashboard
that displays the forecast alongside model provenance). It records the
git SHA, feature list per candidate, hyperparameters, input/output paths,
and holdout metrics — so a reader can answer "what produced this snapshot"
without re-running the script or reading the source.

The output/ directory is tracked in git so each commit captures a
snapshot of (code + data → output). When the code changes, re-run, commit,
and the diff shows what shifted.

Usage:
  cd projects/h2s
  uv run python ../../experiments/2026-06-10_feature_trim_berry/forecast_candidates.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from h2s.constants import (
    FLOW_COL,
    H2S_THRESHOLD_EXTREME,
    H2S_THRESHOLD_HIGH,
    H2S_THRESHOLD_LOW,
    H2S_THRESHOLD_MED,
    MODEL_FEATURES_EVIDENCE,
    MODEL_FEATURES_LEAN,
)
from h2s.training.calibration_eval import chronological_split
from h2s.training.feature_builder import ensure_base_features
from h2s.training.multi_station_trainer import (
    eval_regressor,
    get_xgb_regressor,
    prepare_multi_station_features,
)


SCHEMA_VERSION = "1.0"
STATION = "NESTOR - BES"
EVAL_THRESHOLDS = (
    H2S_THRESHOLD_LOW,        # 5  — green/yellow_low
    H2S_THRESHOLD_MED,        # 10 — yellow_low/yellow_high
    H2S_THRESHOLD_HIGH,       # 30 — yellow_high/orange (watch)
    H2S_THRESHOLD_EXTREME,    # 100 — extreme/critical
)


@dataclass
class Candidate:
    code: str
    name: str
    slug: str          # filename-safe identifier
    features: list[str]


CANDIDATES: list[Candidate] = [
    Candidate("C", "Evidence-only", "evidence", list(MODEL_FEATURES_EVIDENCE)),
    Candidate("B", "Lean", "lean", list(MODEL_FEATURES_LEAN)),
]


# --------------------------------------------------------------------------
# Git provenance
# --------------------------------------------------------------------------


def _git_provenance(repo_root: Path) -> dict:
    """Capture git SHA, branch, and source-clean state for the manifest.

    `git_dirty` reflects whether *source* files (anything outside an
    `output/` directory in any experiment) have uncommitted modifications
    relative to HEAD. The output/ directories themselves contain the
    very files this script is about to overwrite, so they don't count —
    they're the artifact, not the source. Untracked files don't count either.

    The reproducibility contract: `git_dirty=False` means "checking out
    `git_sha` reproduces the producing source exactly".
    """
    def _run(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, cwd=repo_root, text=True).strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        # Tracked diff vs HEAD, excluding experiment outputs (the artifact)
        diff = _run([
            "git", "diff", "--name-only", "HEAD", "--",
            ".", ":(exclude)experiments/*/output/*",
        ])
        return {"git_sha": sha, "git_branch": branch, "git_dirty": bool(diff)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_sha": None, "git_branch": None, "git_dirty": None}


# --------------------------------------------------------------------------
# Categorization
# --------------------------------------------------------------------------


def _categorize(ppb: float) -> str:
    """Map predicted ppb to the operational category label."""
    if ppb < H2S_THRESHOLD_LOW:
        return "green"
    if ppb < H2S_THRESHOLD_MED:
        return "yellow_low"
    if ppb < H2S_THRESHOLD_HIGH:
        return "yellow_high"
    if ppb < H2S_THRESHOLD_EXTREME:
        return "orange"
    return "critical"


# --------------------------------------------------------------------------
# Forecast input
# --------------------------------------------------------------------------


def _load_forecast_input(path: Path, station: str) -> pd.DataFrame:
    """Load the 15-min forecast input, engineer features, filter to station."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[df["site_name"] == station].copy()
    df = df.sort_values("time").reset_index(drop=True)
    # The 15-min input already ships most of the engineered features,
    # but ensure_base_features is idempotent — fills any missing.
    df = ensure_base_features(df, flow_col=FLOW_COL)
    # Site-specific h2s lag fields aren't in the forecast input by design
    # (the future has no observations). Set them to 0 so the model sees a
    # consistent feature set; the regressor's autoregressive features
    # carry whatever seed values the parquet ships with.
    for col in ("h2s_lag_1h", "h2s_lag_3h", "h2s_lag_6h",
                "h2s_rolling_6h", "h2s_rolling_24h"):
        if col not in df.columns:
            df[col] = 0.0
    return df


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--training-parquet", type=Path,
        default=repo_root / "data" / "modeldata_h2s_nofill.parquet",
    )
    parser.add_argument(
        "--forecast-input", type=Path,
        default=repo_root / "data" / "modeldata_forecast_15min.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args(argv)

    if not args.training_parquet.exists():
        print(f"ERROR: training parquet not found at {args.training_parquet}", file=sys.stderr)
        return 1
    if not args.forecast_input.exists():
        print(f"ERROR: forecast input not found at {args.forecast_input}", file=sys.stderr)
        return 1

    # Capture git state *before* we write any outputs — otherwise the
    # just-created files appear as untracked changes and make the manifest
    # report git_dirty=True even when the producing source is clean.
    git = _git_provenance(repo_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Historical training data ----------------------------------------
    df_raw = pd.read_parquet(args.training_parquet)
    train_eng = prepare_multi_station_features(df_raw, station=STATION)
    train_df, test_df = chronological_split(train_eng, args.train_fraction)
    print(f"Training data: {len(train_eng)} rows {STATION!r}")
    print(f"  train n={len(train_df)} ({train_df['time'].min()} → {train_df['time'].max()})")
    print(f"  test  n={len(test_df)} ({test_df['time'].min()} → {test_df['time'].max()})")

    # --- Forecast input ---------------------------------------------------
    forecast_df = _load_forecast_input(args.forecast_input, STATION)
    print(f"Forecast input: {len(forecast_df)} rows "
          f"({forecast_df['time'].min()} → {forecast_df['time'].max()})")

    y_train = train_df["H2S"]
    y_test = test_df["H2S"]

    # --- Train + forecast each candidate ----------------------------------
    candidate_records: list[dict] = []
    xgb_hyperparams = get_xgb_regressor().get_xgb_params()

    for cand in CANDIDATES:
        print(f"\nCandidate {cand.code}: {cand.name} ({len(cand.features)} features)")

        missing_train = set(cand.features) - set(train_df.columns)
        missing_fc = set(cand.features) - set(forecast_df.columns)
        if missing_train:
            print(f"  ERROR: training data missing columns: {missing_train}", file=sys.stderr)
            return 1
        if missing_fc:
            print(f"  ERROR: forecast input missing columns: {missing_fc}", file=sys.stderr)
            return 1

        model = get_xgb_regressor()
        model.fit(
            train_df[cand.features], y_train,
            eval_set=[(test_df[cand.features], y_test)], verbose=False,
        )
        holdout = eval_regressor(model, test_df[cand.features], y_test)
        print(f"  holdout: Spearman? (see manifest)  R²={holdout['R2']:.3f}  "
              f"recall@30={holdout['recall_30']:.3f}  recall@100={holdout['recall_100']:.3f}")

        # Forecast
        forecast_preds = np.clip(model.predict(forecast_df[cand.features]), 0, None)
        forecast_out = pd.DataFrame({
            "time": forecast_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "predicted_ppb": np.round(forecast_preds, 3),
            "predicted_category": [_categorize(p) for p in forecast_preds],
        })
        forecast_csv = args.output_dir / f"forecast_{cand.slug}.csv"
        forecast_out.to_csv(forecast_csv, index=False)
        print(f"  forecast → {forecast_csv.name} (n={len(forecast_out)})")

        # Forecast summary for the manifest
        category_counts = (
            forecast_out["predicted_category"].value_counts().to_dict()
        )
        candidate_records.append({
            "code": cand.code,
            "name": cand.name,
            "n_features": len(cand.features),
            "features": cand.features,
            "hyperparameters": {
                k: (float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in xgb_hyperparams.items()
                if not k.startswith("_")
            },
            "holdout_metrics": {
                "mae": round(holdout["MAE"], 4),
                "rmse": round(holdout["RMSE"], 4),
                "r2": round(holdout["R2"], 4),
                "recall_5": round(holdout["recall_5"], 4),
                "recall_10": round(holdout["recall_10"], 4),
                "recall_30": round(holdout["recall_30"], 4),
                "recall_100": round(holdout["recall_100"], 4),
                "n_positives_5": holdout["n_positives_5"],
                "n_positives_10": holdout["n_positives_10"],
                "n_positives_30": holdout["n_positives_30"],
                "n_positives_100": holdout["n_positives_100"],
                "n_test": len(test_df),
            },
            "forecast_output": forecast_csv.name,
            "forecast_summary": {
                "n_predictions": len(forecast_out),
                "max_predicted_ppb": float(forecast_preds.max()),
                "mean_predicted_ppb": float(forecast_preds.mean()),
                "category_counts": category_counts,
            },
        })

    # --- Manifest ---------------------------------------------------------
    # produced_at uses the forecast input's max time as the anchor so the
    # manifest is reproducible (no wall-clock dependency in the file).
    produced_at = forecast_df["time"].max().isoformat()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "produced_at": produced_at,
            **git,
            "station": STATION,
        },
        "inputs": {
            "training_parquet": {
                "path": str(args.training_parquet.relative_to(repo_root)),
                "rows_filtered": len(train_eng),
                "time_range": [str(train_eng["time"].min()), str(train_eng["time"].max())],
            },
            "forecast_input": {
                "path": str(args.forecast_input.relative_to(repo_root)),
                "rows_filtered": len(forecast_df),
                "time_range": [str(forecast_df["time"].min()), str(forecast_df["time"].max())],
                "cadence_minutes": 15,
            },
        },
        "split": {
            "method": "chronological",
            "train_fraction": args.train_fraction,
            "train_n": len(train_df),
            "test_n": len(test_df),
            "test_time_range": [str(test_df["time"].min()), str(test_df["time"].max())],
        },
        "categorical_thresholds_ppb": {
            "yellow_low_min": int(H2S_THRESHOLD_LOW),
            "yellow_high_min": int(H2S_THRESHOLD_MED),
            "orange_min": int(H2S_THRESHOLD_HIGH),
            "critical_min": int(H2S_THRESHOLD_EXTREME),
        },
        "eval_thresholds_ppb": [int(t) for t in EVAL_THRESHOLDS],
        "candidates": candidate_records,
    }

    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nManifest → {manifest_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
