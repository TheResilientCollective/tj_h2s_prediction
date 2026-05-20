"""Head-to-head: does adding wind_x_stable_atm improve Berry's regression model?

Calibration finding #3 motivates the feature — `f_volatilization ∝ wind²`
is anti-skilled in the trapped regime (`stable_atm=1`), so the tree
needs an explicit handle on wind-in-calm-regime to learn the sign flip.

Procedure:
1. Load modeldata_h2s_nofill.parquet, filter to NESTOR - BES (Berry).
2. Chronological 70/30 split.
3. Train `train_and_select` regression with the OLD 43-feature set and
   the NEW 44-feature set (just wind_x_stable_atm added).
4. Score each model + the persistence baseline against the same test
   slice via calibration_report() — Spearman, recall@30, recall@100,
   regime-stratified.
5. Print a side-by-side scoreboard.

Honest about scope:
- This is a single-task (regression) per-station validation, not a full
  re-training of every model. If the new feature helps here, promote to
  the multi_station_training_job + retrain everything.
- We do NOT train the hourly XGBoost classifier here (different code path).

Usage:
    cd projects/h2s
    uv run python scripts/retrain_compare_nestor.py [--out json_path]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from h2s.constants import H2S_THRESHOLD_EXTREME, H2S_THRESHOLD_HIGH, MODEL_FEATURES
from h2s.training.calibration_eval import (
    calibration_report,
    chronological_split,
    persistence_prediction,
)
from h2s.training.multi_station_trainer import (
    eval_regressor,
    get_xgb_regressor,
    prepare_multi_station_features,
    train_and_select,
)


NEW_FEATURE = "wind_x_stable_atm"


def _importance_for_features(model, feature_names: list[str]) -> dict[str, float]:
    """Feature importance keyed by the *actual* feature list used to train.

    The multi_station_trainer helper assumes MODEL_FEATURES order — wrong
    when we train on a subset (the OLD 43-feature side), so re-do it here.
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp_arr = np.asarray(imp)
    return {name: round(float(imp_arr[i]), 4) for i, name in enumerate(feature_names)}


def _train_xgb_regressor(X_train, X_test, y_train, y_test, feature_names) -> tuple:
    """Train production-config XGBoost regressor (no ensembling, no auto-select).

    Matches `get_xgb_regressor` hyperparameters from multi_station_trainer so
    the result is directly comparable to the per-station XGB models that
    ship to S3.
    """
    model = get_xgb_regressor()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    metrics = eval_regressor(model, X_test, y_test)
    fi = _importance_for_features(model, feature_names)
    return model, {"XGB": metrics, "selected": "XGBoost", "feature_importance": fi}


def _format_block(block: dict) -> str:
    """One-line summary of a {'spearman', 'thr_30', 'thr_100'} block."""
    spearman = block["spearman"]
    r30 = block["thr_30"]["recall"]
    r100 = block["thr_100"]["recall"]
    n30 = block["thr_30"]["n_positives"]
    n100 = block["thr_100"]["n_positives"]
    return (
        f"Spearman={spearman:6.3f}  "
        f"recall@30={r30:.3f} (n={n30:3d})  "
        f"recall@100={r100:.3f} (n={n100:3d})  "
        f"N={block['n']}"
    )


def _print_scoreboard(report_persistence, report_old, report_new) -> None:
    """Side-by-side block per slice (overall / per-site / per-regime)."""
    def _print_section(name: str, p_block: dict, old_block: dict, new_block: dict):
        print(f"\n  {name}")
        print(f"    persistence    {_format_block(p_block)}")
        print(f"    OLD (43 feat)  {_format_block(old_block)}")
        print(f"    NEW (44 feat)  {_format_block(new_block)}")

    print("\n=== HEAD-TO-HEAD SCOREBOARD ===")
    _print_section(
        "OVERALL",
        report_persistence.overall,
        report_old.overall,
        report_new.overall,
    )

    for site in sorted(report_persistence.per_site.keys()):
        _print_section(
            f"site: {site}",
            report_persistence.per_site[site],
            report_old.per_site[site],
            report_new.per_site[site],
        )

    for regime in ("calm_stable_atm_1", "windy_stable_atm_0"):
        if regime not in report_persistence.per_regime:
            continue
        _print_section(
            f"regime: {regime}",
            report_persistence.per_regime[regime],
            report_old.per_regime[regime],
            report_new.per_regime[regime],
        )


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_parquet = repo_root / "data" / "modeldata_h2s_nofill.parquet"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", type=Path, default=default_parquet)
    parser.add_argument("--station", default="NESTOR - BES")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument(
        "--mode",
        choices=("xgb", "auto"),
        default="xgb",
        help="'xgb': force production-config XGBoost. 'auto': use train_and_select.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON dump")
    args = parser.parse_args(argv)

    if not args.parquet.exists():
        print(f"ERROR: parquet not found at {args.parquet}", file=sys.stderr)
        return 1

    # --- Data prep ---------------------------------------------------------
    df_raw = pd.read_parquet(args.parquet)
    df = prepare_multi_station_features(df_raw, station=args.station)
    print(
        f"Loaded {len(df)} rows for site={args.station!r}, "
        f"time {df['time'].min()} → {df['time'].max()}"
    )

    if NEW_FEATURE not in df.columns:
        print(f"ERROR: {NEW_FEATURE!r} missing — feature_builder change not picked up?",
              file=sys.stderr)
        return 1

    train_df, test_df = chronological_split(df, args.train_fraction)
    print(f"Chronological 70/30: train n={len(train_df)} ({train_df['time'].min()} → "
          f"{train_df['time'].max()}), test n={len(test_df)} ({test_df['time'].min()} → "
          f"{test_df['time'].max()})")

    new_features = list(MODEL_FEATURES)
    old_features = [f for f in MODEL_FEATURES if f != NEW_FEATURE]
    assert len(new_features) - len(old_features) == 1
    print(f"Feature sets: OLD={len(old_features)}  NEW={len(new_features)}")

    y_train = train_df["H2S"]
    y_test = test_df["H2S"]

    # --- Train both regressors --------------------------------------------
    print(f"\nMode: {args.mode}")
    print("\nTraining OLD regressor (43 features) ...")
    if args.mode == "xgb":
        model_old, metrics_old = _train_xgb_regressor(
            train_df[old_features], test_df[old_features], y_train, y_test, old_features
        )
        choice_old = "XGBoost"
        print(f"  selected={choice_old}  XGB={metrics_old['XGB']}")
    else:
        model_old, choice_old, metrics_old = train_and_select(
            train_df[old_features], test_df[old_features], y_train, y_test, task="regression"
        )
        print(f"  selected={choice_old}  RF={metrics_old.get('RF')}  XGB={metrics_old.get('XGB')}")

    print("\nTraining NEW regressor (44 features) ...")
    if args.mode == "xgb":
        model_new, metrics_new = _train_xgb_regressor(
            train_df[new_features], test_df[new_features], y_train, y_test, new_features
        )
        choice_new = "XGBoost"
        print(f"  selected={choice_new}  XGB={metrics_new['XGB']}")
    else:
        model_new, choice_new, metrics_new = train_and_select(
            train_df[new_features], test_df[new_features], y_train, y_test, task="regression"
        )
        print(f"  selected={choice_new}  RF={metrics_new.get('RF')}  XGB={metrics_new.get('XGB')}")

    # --- Predictions on the test slice ------------------------------------
    pred_old = np.clip(model_old.predict(test_df[old_features]), 0, None)
    pred_new = np.clip(model_new.predict(test_df[new_features]), 0, None)

    # Persistence on the same test slice (within-site, lag-1h)
    test_sorted = test_df.sort_values("time").reset_index(drop=True)
    persistence = persistence_prediction(test_sorted, lag_hours=1)
    # For persistence, drop the first row (no lag available)
    p_mask = ~persistence.isna()
    test_for_persistence = test_sorted.loc[p_mask].reset_index(drop=True)
    persistence = persistence.loc[p_mask].reset_index(drop=True)

    # --- Calibration-aligned scoreboards ----------------------------------
    thresholds = (H2S_THRESHOLD_HIGH, H2S_THRESHOLD_EXTREME)
    report_old = calibration_report(test_df, pred_old, thresholds=thresholds)
    report_new = calibration_report(test_df, pred_new, thresholds=thresholds)
    report_persistence = calibration_report(
        test_for_persistence, persistence.to_numpy(), thresholds=thresholds
    )

    _print_scoreboard(report_persistence, report_old, report_new)

    # Feature importance — does the tree actually use the new feature?
    print("\n=== FEATURE IMPORTANCE — TOP 15 (NEW model) ===")
    feat_imp = metrics_new["feature_importance"]
    for i, (name, score) in enumerate(
        sorted(feat_imp.items(), key=lambda kv: kv[1], reverse=True)[:15], 1
    ):
        marker = " <-- NEW" if name == NEW_FEATURE else ""
        print(f"  {i:2d}. {name:35s} {score:.4f}{marker}")

    if args.out:
        payload = {
            "config": {
                "station": args.station,
                "train_fraction": args.train_fraction,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "feature_added": NEW_FEATURE,
                "n_features_old": len(old_features),
                "n_features_new": len(new_features),
            },
            "train_and_select_metrics": {
                "old": {"selected": choice_old, **metrics_old},
                "new": {"selected": choice_new, **metrics_new},
            },
            "calibration_scoreboard": {
                "persistence": {
                    "overall": report_persistence.overall,
                    "per_site": report_persistence.per_site,
                    "per_regime": report_persistence.per_regime,
                },
                "old_model": {
                    "overall": report_old.overall,
                    "per_site": report_old.per_site,
                    "per_regime": report_old.per_regime,
                },
                "new_model": {
                    "overall": report_new.overall,
                    "per_site": report_new.per_site,
                    "per_regime": report_new.per_regime,
                },
            },
        }
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote full JSON to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
