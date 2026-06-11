"""Feature ablation on Berry — does a trimmed feature set match the 44-feature baseline?

Trains a production-config XGBoost regressor on Berry (NESTOR - BES) for each of
four candidate feature sets, then scores all four (plus the persistence floor)
against the calibration-aligned harness from PR #26.

Candidates (defined in `h2s.constants`):
  D - MODEL_FEATURES          (44 features, control)
  C - MODEL_FEATURES_EVIDENCE (33 features, drops calibration-dismissed only)
  B - MODEL_FEATURES_LEAN     (19 features, plus drops redundant interactions/rollings)
  A - MODEL_FEATURES_MINIMAL  (11 features, calibration's load-bearing core only)

Acceptance criteria for each X < D:
  recall@30(X)  ≥ recall@30(D)  - 0.02   (≤ 2 pp drop)
  recall@100(X) ≥ recall@100(D) - 0.05   (≤ 5 pp drop)
  Spearman(X)   ≥ Spearman(D)   - 0.02

Decision: smallest winning set ships. If none wins, keep D and document why.

The ablation runs `get_xgb_regressor()` directly — bypassing `train_and_select`'s
selector so the comparison isolates feature-set effects, not selector effects.

Usage:
    cd projects/h2s
    uv run python ../../experiments/2026-06-10_feature_trim_berry/feature_ablation.py \
        --out ../../experiments/2026-06-10_feature_trim_berry/output/ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from h2s.constants import (
    H2S_THRESHOLD_EXTREME,
    H2S_THRESHOLD_HIGH,
    H2S_THRESHOLD_LOW,
    H2S_THRESHOLD_MED,
    MODEL_FEATURES,
    MODEL_FEATURES_EVIDENCE,
    MODEL_FEATURES_LEAN,
    MODEL_FEATURES_MINIMAL,
)


# Score at all four categorical boundaries (green<5 / yellow_low / yellow_high /
# orange / extreme). 5 ppb and 10 ppb added per complaint-rate analysis:
# complaints rise in the 5-10 ppb band (yellow_low), with 8 ppb identified
# as a known complaint-trigger level.
EVAL_THRESHOLDS = (
    H2S_THRESHOLD_LOW,        # 5
    H2S_THRESHOLD_MED,        # 10
    H2S_THRESHOLD_HIGH,       # 30
    H2S_THRESHOLD_EXTREME,    # 100
)
from h2s.training.calibration_eval import (
    CalibrationReport,
    calibration_report,
    chronological_split,
    persistence_prediction,
)
from h2s.training.multi_station_trainer import (
    eval_regressor,
    get_xgb_regressor,
    prepare_multi_station_features,
)


# Acceptance thresholds, matching the plan
RECALL_30_TOLERANCE = 0.02
RECALL_100_TOLERANCE = 0.05
SPEARMAN_TOLERANCE = 0.02


@dataclass
class Candidate:
    code: str            # "A" / "B" / "C" / "D"
    name: str            # human label
    features: list[str]


CANDIDATES: list[Candidate] = [
    Candidate("D", "Baseline (44 feat)", list(MODEL_FEATURES)),
    Candidate("C", "Evidence-only (33 feat)", list(MODEL_FEATURES_EVIDENCE)),
    Candidate("B", "Lean (19 feat)", list(MODEL_FEATURES_LEAN)),
    Candidate("A", "Minimal (11 feat)", list(MODEL_FEATURES_MINIMAL)),
]


def _importance_for_features(model, feature_names: list[str]) -> dict[str, float]:
    """Feature importance keyed by the actual feature list the model was trained on."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp_arr = np.asarray(imp)
    return {name: round(float(imp_arr[i]), 4) for i, name in enumerate(feature_names)}


def _train_and_score(
    cand: Candidate,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[dict, CalibrationReport, dict[str, float]]:
    """Train production-config XGBoost on cand.features, score against test slice."""
    X_train = train_df[cand.features]
    X_test = test_df[cand.features]
    y_train = train_df["H2S"]
    y_test = test_df["H2S"]

    model = get_xgb_regressor()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    fit_metrics = eval_regressor(model, X_test, y_test)
    preds = np.clip(model.predict(X_test), 0, None)
    report = calibration_report(test_df, preds, thresholds=EVAL_THRESHOLDS)
    importance = _importance_for_features(model, cand.features)
    return fit_metrics, report, importance


def _evaluate_acceptance(
    cand_report: CalibrationReport, baseline_report: CalibrationReport
) -> tuple[bool, dict]:
    """Check acceptance criteria vs baseline. Returns (passed, per-criterion deltas)."""
    cb = baseline_report.overall
    ca = cand_report.overall
    drop_r30 = cb["thr_30"]["recall"] - ca["thr_30"]["recall"]
    drop_r100 = cb["thr_100"]["recall"] - ca["thr_100"]["recall"]
    drop_spearman = cb["spearman"] - ca["spearman"]
    crit = {
        "recall_30_drop": round(drop_r30, 4),
        "recall_30_drop_allowed": RECALL_30_TOLERANCE,
        "recall_30_pass": drop_r30 <= RECALL_30_TOLERANCE,
        "recall_100_drop": round(drop_r100, 4),
        "recall_100_drop_allowed": RECALL_100_TOLERANCE,
        "recall_100_pass": drop_r100 <= RECALL_100_TOLERANCE,
        "spearman_drop": round(drop_spearman, 4),
        "spearman_drop_allowed": SPEARMAN_TOLERANCE,
        "spearman_pass": drop_spearman <= SPEARMAN_TOLERANCE,
    }
    passed = bool(crit["recall_30_pass"] and crit["recall_100_pass"] and crit["spearman_pass"])
    return passed, crit


def _format_overall(block: dict) -> str:
    parts = [f"Spearman={block['spearman']:6.3f}"]
    for thr in EVAL_THRESHOLDS:
        recall = block[f"thr_{int(thr)}"]["recall"]
        parts.append(f"r@{int(thr)}={recall:.3f}")
    parts.append(f"N={block['n']}")
    return "  ".join(parts)


def _print_scoreboard(
    persistence_report: CalibrationReport,
    candidate_results: list[dict],
) -> None:
    print("\n=== ABLATION SCOREBOARD (overall) ===")
    print(f"  persistence  {_format_overall(persistence_report.overall)}")
    for r in candidate_results:
        marker = " ✓" if r["passed"] else "  "
        print(f"  {r['code']} {marker} {r['name']:25s}  {_format_overall(r['report'].overall)}")

    print("\n=== REGIME SPLIT — calm (stable_atm=1) ===")
    print(f"  persistence  {_format_overall(persistence_report.per_regime['calm_stable_atm_1'])}")
    for r in candidate_results:
        print(f"  {r['code']}    {r['name']:25s}  {_format_overall(r['report'].per_regime['calm_stable_atm_1'])}")

    print("\n=== REGIME SPLIT — windy (stable_atm=0) ===")
    print(f"  persistence  {_format_overall(persistence_report.per_regime['windy_stable_atm_0'])}")
    for r in candidate_results:
        print(f"  {r['code']}    {r['name']:25s}  {_format_overall(r['report'].per_regime['windy_stable_atm_0'])}")


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    default_parquet = repo_root / "data" / "modeldata_h2s_nofill.parquet"

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", type=Path, default=default_parquet)
    parser.add_argument("--station", default="NESTOR - BES")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.parquet.exists():
        print(f"ERROR: parquet not found at {args.parquet}", file=sys.stderr)
        return 1

    # --- Data prep ---------------------------------------------------------
    df_raw = pd.read_parquet(args.parquet)
    df = prepare_multi_station_features(df_raw, station=args.station)
    print(f"Loaded {len(df)} rows for site={args.station!r}, "
          f"time {df['time'].min()} → {df['time'].max()}")

    train_df, test_df = chronological_split(df, args.train_fraction)
    print(f"Chronological 70/30: train n={len(train_df)}, test n={len(test_df)}")
    print(f"  train: {train_df['time'].min()} → {train_df['time'].max()}")
    print(f"  test:  {test_df['time'].min()} → {test_df['time'].max()}")

    # Verify each candidate's features all exist in the engineered DataFrame.
    for cand in CANDIDATES:
        missing = set(cand.features) - set(df.columns)
        if missing:
            print(f"ERROR: candidate {cand.code} ({cand.name}) missing columns: {missing}",
                  file=sys.stderr)
            return 1

    # --- Persistence floor ------------------------------------------------
    test_sorted = test_df.sort_values("time").reset_index(drop=True)
    pers_pred = persistence_prediction(test_sorted, lag_hours=1)
    p_mask = ~pers_pred.isna()
    persistence_test_df = test_sorted.loc[p_mask].reset_index(drop=True)
    persistence_pred = pers_pred.loc[p_mask].reset_index(drop=True)
    persistence_report = calibration_report(
        persistence_test_df,
        persistence_pred.to_numpy(),
        thresholds=EVAL_THRESHOLDS,
    )

    # --- Train each candidate ---------------------------------------------
    candidate_results: list[dict] = []
    baseline_report = None  # set after training D
    for cand in CANDIDATES:
        print(f"\nTraining {cand.code}: {cand.name} ({len(cand.features)} features) ...")
        fit, report, importance = _train_and_score(cand, train_df, test_df)
        if cand.code == "D":
            baseline_report = report
            passed = True  # baseline always "passes"
            crit = {"note": "baseline (reference)"}
        else:
            assert baseline_report is not None
            passed, crit = _evaluate_acceptance(report, baseline_report)
        candidate_results.append({
            "code": cand.code,
            "name": cand.name,
            "n_features": len(cand.features),
            "features": cand.features,
            "fit_metrics": fit,
            "report": report,
            "feature_importance": importance,
            "acceptance": crit,
            "passed": passed,
        })

    # --- Print scoreboard + decision --------------------------------------
    _print_scoreboard(persistence_report, candidate_results)

    print("\n=== ACCEPTANCE GATE (vs baseline D) ===")
    for r in candidate_results:
        if r["code"] == "D":
            continue
        ac = r["acceptance"]
        verdict = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['code']} {r['name']:25s}  {verdict}  "
            f"Δspearman={ac['spearman_drop']:+.3f}  "
            f"Δrecall@30={ac['recall_30_drop']:+.3f}  "
            f"Δrecall@100={ac['recall_100_drop']:+.3f}"
        )

    print("\n=== DECISION ===")
    winners = [r for r in candidate_results if r["code"] != "D" and r["passed"]]
    if not winners:
        print("  No candidate met acceptance criteria. Recommendation: keep MODEL_FEATURES (D).")
        winner_code = "D"
    else:
        # smallest winning set
        winners.sort(key=lambda r: r["n_features"])
        winner = winners[0]
        winner_code = winner["code"]
        print(f"  Smallest winning set: {winner_code} - {winner['name']} "
              f"({winner['n_features']} features)")

    # --- Persist JSON -----------------------------------------------------
    if args.out:
        # Convert CalibrationReport dataclasses to dicts for JSON
        def _report_to_dict(rep: CalibrationReport) -> dict:
            return {"overall": rep.overall, "per_site": rep.per_site, "per_regime": rep.per_regime}

        payload = {
            "config": {
                "station": args.station,
                "parquet": str(args.parquet),
                "train_fraction": args.train_fraction,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_test_evaluated": len(persistence_test_df),
                "train_time_range": [str(train_df["time"].min()), str(train_df["time"].max())],
                "test_time_range": [str(test_df["time"].min()), str(test_df["time"].max())],
                "thresholds_ppb": [int(t) for t in EVAL_THRESHOLDS],
                "acceptance_tolerances": {
                    "recall_30_drop": RECALL_30_TOLERANCE,
                    "recall_100_drop": RECALL_100_TOLERANCE,
                    "spearman_drop": SPEARMAN_TOLERANCE,
                },
            },
            "persistence_baseline": _report_to_dict(persistence_report),
            "candidates": [
                {
                    "code": r["code"],
                    "name": r["name"],
                    "n_features": r["n_features"],
                    "features": r["features"],
                    "fit_metrics": r["fit_metrics"],
                    "calibration_report": _report_to_dict(r["report"]),
                    "feature_importance": r["feature_importance"],
                    "acceptance": r["acceptance"],
                    "passed": r["passed"],
                }
                for r in candidate_results
            ],
            "decision": {"winner": winner_code},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote JSON report to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
