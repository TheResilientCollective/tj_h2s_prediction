"""Multi-Station H2S Model Training Pipeline.

Replaces the single-model monthly training pipeline with per-station,
per-task auto-selected models (RF vs XGBoost vs Ensemble), trained for
each of the two feature variants (Evidence, Lean) per cycle.

Produces 18 pickle files per cycle: 3 stations × 3 tasks × 2 variants
(regression, >5ppb, >10ppb × evidence, lean). Each file carries an
explicit variant suffix so a future variant slots in without renames.
Uploaded to S3 at:
  tijuana/forecast/models/stations/{station_key}/{task}_{variant}.pkl
"""

import io
import json
import os
import pickle
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import dagster as dg
import numpy as np
import pandas as pd

from h2s.constants import (
    MODEL_FEATURES,
    MODEL_FEATURES_LEAN,
    STATION_MODELS_ARCHIVE_BASE,
    STATION_MODELS_S3_BASE,
    STATION_PARTITION_MAP,
    STATIONS,
    TRAINING_SNAPSHOTS_PATH,
)
from h2s.training.multi_station_trainer import (
    TRAIN_FRACTION,
    prepare_multi_station_features,
    train_and_select,
)

# Per-station training trains parallel variants per cycle, each tagged with
# an explicit suffix on every artifact name so a future third variant slots
# in by adding one entry here (no renames):
#   - "evidence" (33 features, current production default — suffix `_evidence`)
#   - "lean"     (19 features, parallel "not overdetermined" demonstration — suffix `_lean`)
# Files in the deployment dict, S3 keys, and `features_<variant>.json` schemas
# all carry the variant suffix. Consumers (daily pipeline, predictor) pick
# the variant explicitly by suffix — there is no unsuffixed default.
_VARIANTS: dict[str, list[str]] = {
    "evidence": MODEL_FEATURES,
    "lean": MODEL_FEATURES_LEAN,
}

STATION_PARTITIONS = dg.StaticPartitionsDefinition(
    partition_keys=list(STATION_PARTITION_MAP.keys())  # san_ysidro, nestor_bes, ib_civic_ctr
)

_KEY = lambda name: dg.AssetKey(["h2s", name])

_TASKS = ('regression', 'clf_5ppb', 'clf_10ppb', 'clf_30ppb')


# ==============================================================================
# Model versioning helpers
# ==============================================================================

def _git_short_sha() -> str:
    """Short git sha of the running code, or a fallback for git-less deploys."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("GIT_SHA", "unknown")[:12] or "unknown"


def make_version_tag(now: datetime | None = None, sha: str | None = None) -> str:
    """Lexically-sortable model version: 20260612T213000Z-a1b2c3d.

    Sorting version tags as strings sorts them chronologically, which is what
    `station_model_promotion` relies on to resolve "latest".
    """
    now = now or datetime.now(timezone.utc)
    sha = sha or _git_short_sha()
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{sha}"


def _station_artifact_names() -> list[str]:
    """Every file that constitutes one station's deployed model set."""
    names = [f"{task}_{variant}.pkl" for variant in _VARIANTS for task in _TASKS]
    names += [f"features_{variant}.json" for variant in _VARIANTS]
    return names


# Headline metrics for the new-vs-production promotion comparison.
# (metric_path, higher_is_better); regression recalls are the
# calibration-aligned alert metrics, AUCs cover the classifier ladder.
_COMPARISON_METRICS = [
    ("regression.R2", True),
    ("regression.recall_5", True),
    ("regression.recall_10", True),
    ("regression.recall_30", True),
    ("regression.recall_100", True),
    ("clf_5ppb.AUC", True),
    ("clf_10ppb.AUC", True),
    ("clf_30ppb.AUC", True),
]

# A metric "regresses" if it drops more than this vs production.
_PROMOTION_TOLERANCE = 0.02


def _metric_lookup(report_tasks: dict, variant: str, path: str):
    """Fetch tasks[variant][task][metric] from a training report, or None."""
    task, metric = path.split(".", 1)
    try:
        value = report_tasks[variant][task][metric]
    except (KeyError, TypeError):
        return None
    return float(value) if value is not None else None


def compare_training_reports(new_report: dict, prod_report: dict | None,
                             variant: str = "evidence") -> dict:
    """Compare a fresh training report against the production one.

    Returns {"metrics": [{name, new, prod, delta}], "n_improved",
    "n_regressed", "recommendation", "reason"}. With no production baseline
    the recommendation is to promote (nothing to lose).
    """
    if prod_report is None:
        return {
            "metrics": [],
            "n_improved": 0,
            "n_regressed": 0,
            "recommendation": "promote",
            "reason": "no production baseline found — nothing to compare against",
        }

    rows = []
    n_improved = 0
    n_regressed = 0
    for path, _higher_better in _COMPARISON_METRICS:
        new_v = _metric_lookup(new_report.get("tasks", {}), variant, path)
        prod_v = _metric_lookup(prod_report.get("tasks", {}), variant, path)
        if new_v is None or prod_v is None:
            continue
        delta = new_v - prod_v
        rows.append({"name": path, "new": round(new_v, 4),
                     "prod": round(prod_v, 4), "delta": round(delta, 4)})
        if delta > 0:
            n_improved += 1
        if delta < -_PROMOTION_TOLERANCE:
            n_regressed += 1

    if not rows:
        recommendation, reason = "review", "no comparable metrics between reports"
    elif n_regressed == 0:
        recommendation = "promote"
        reason = (f"{n_improved}/{len(rows)} headline metrics improved, "
                  f"none regressed beyond {_PROMOTION_TOLERANCE}")
    else:
        recommendation = "review"
        worst = min(rows, key=lambda r: r["delta"])
        reason = (f"{n_regressed} metric(s) regressed beyond {_PROMOTION_TOLERANCE} "
                  f"(worst: {worst['name']} {worst['delta']:+.3f}) — human judgement needed")

    return {"metrics": rows, "n_improved": n_improved,
            "n_regressed": n_regressed,
            "recommendation": recommendation, "reason": reason}


def build_promotion_message(site_name: str, partition: str, version_tag: str,
                            comparison: dict, env_label: str = "") -> str:
    """Slack text for the post-training promotion decision."""
    label = f" [{env_label}]" if env_label else ""
    lines = [
        f"*H2S model training complete{label}* — {site_name}",
        f"Archived as version `{version_tag}`",
        "",
    ]
    if comparison["metrics"]:
        lines.append("*New vs production (evidence variant):*")
        for row in comparison["metrics"]:
            arrow = "▲" if row["delta"] > 0 else ("▼" if row["delta"] < 0 else "→")
            lines.append(
                f"  {arrow} {row['name']}: {row['new']:.3f} (prod {row['prod']:.3f}, {row['delta']:+.3f})"
            )
        lines.append("")
    lines.append(f"*Recommendation: {comparison['recommendation'].upper()}* — {comparison['reason']}")
    lines.append("")
    lines.append("To promote this version to production:")
    lines.append(
        "```uv run dg launch --job promote_station_models_job "
        f"--partition {partition} "
        "--config-json '{\"ops\":{\"h2s__station_model_promotion\":"
        f"{{\"config\":{{\"version_tag\":\"{version_tag}\"}}}}}}'```"
    )
    return "\n".join(lines)


# ==============================================================================
# Asset 1: Load and prepare training data (unpartitioned — shared across stations)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Multi-station training dataset loaded from S3, filtered and feature-engineered",
    config_schema={
        "s3_bucket": dg.Field(
            str,
            default_value="resilentpublic",
            description="S3 bucket for training data (resilentpublic or test)",
        ),
    },
)
def multi_station_training_data(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Load training parquet from S3, filter to measured rows, engineer features."""
    s3 = context.resources.s3
    bucket = context.op_config["s3_bucket"]
    s3_path = "latest/tijuana/forecast_data/modeldata_h2s_nofill.parquet"

    # Load training data bytes from S3 (so we can snapshot the exact input used)
    raw_bytes = s3.getFile(path=s3_path, bucket=bucket)
    raw_df = pd.read_parquet(io.BytesIO(raw_bytes))
    context.log.info(f"✓ Loaded training data from S3 ({bucket}/{s3_path}): {len(raw_df)} rows")

    # Snapshot the exact parquet used for this training run to S3
    snapshot_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    snapshot_path = f"{TRAINING_SNAPSHOTS_PATH}/{snapshot_ts}/modeldata_h2s_nofill.parquet"
    s3.putFile(
        raw_bytes,
        snapshot_path,
        bucket=s3.S3_BUCKET,
        content_type="application/octet-stream",
    )
    context.log.info(f"✓ Wrote training data snapshot to S3: {snapshot_path}")

    df = prepare_multi_station_features(raw_df)
    df.attrs["training_snapshot_s3_path"] = snapshot_path
    df.attrs["training_snapshot_bucket"] = s3.S3_BUCKET
    df.attrs["training_snapshot_source_bucket"] = bucket
    df.attrs["training_snapshot_source_path"] = s3_path
    df.attrs["training_snapshot_timestamp"] = snapshot_ts

    context.log.info(f"✓ Feature engineering complete: {len(df)} clean rows")
    for site in df['site_name'].unique():
        ss = df[df['site_name'] == site]
        context.log.info(f"  {site}: {len(ss)} rows, >5ppb={ss['exceed_5'].mean()*100:.1f}%")

    context.add_output_metadata({
        "row_count": len(df),
        "stations": list(df['site_name'].unique()),
        "features": len(MODEL_FEATURES),
        "date_min": str(df['time'].min()),
        "date_max": str(df['time'].max()),
        "training_snapshot_s3_path": snapshot_path,
        "training_snapshot_bucket": s3.S3_BUCKET,
    })
    return df


# ==============================================================================
# Asset 2: Train per-station models (partitioned by station)
# ==============================================================================

def _train_one_variant(
    context: dg.AssetExecutionContext,
    sdf: pd.DataFrame,
    features: list[str],
    split: int,
    ensemble_margin: float,
    variant_label: str,
) -> dict[str, tuple]:
    """Train regression + clf_5ppb + clf_10ppb + clf_30ppb for one feature set.

    Returns dict[task → (model, choice_str)] for the current station.
    The full metrics dict is recomputed downstream by station_training_report
    using the per-variant feature slice (so importance keys are correct).
    """
    X = sdf[features].values
    y_cont = sdf['H2S'].values
    y_5 = sdf['exceed_5'].values
    y_10 = sdf['exceed_10'].values
    y_30 = sdf['exceed_30'].values

    Xtr, Xte = X[:split], X[split:]
    ytr_c, yte_c = y_cont[:split], y_cont[split:]
    ytr_5, yte_5 = y_5[:split], y_5[split:]
    ytr_10, yte_10 = y_10[:split], y_10[split:]
    ytr_30, yte_30 = y_30[:split], y_30[split:]

    result: dict[str, tuple] = {}
    for task, ytr_, yte_ in [
        ('regression', ytr_c, yte_c),
        ('clf_5ppb',   ytr_5, yte_5),
        ('clf_10ppb',  ytr_10, yte_10),
        ('clf_30ppb',  ytr_30, yte_30),
    ]:
        context.log.info(f"  [{variant_label}] Training {task}...")
        model, choice, _ = train_and_select(
            Xtr, Xte, ytr_, yte_, task, ensemble_margin=ensemble_margin
        )
        context.log.info(f"    [{variant_label}] {task} → {choice}")
        result[task] = (model, choice)
    return result


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    kinds={"python", "ml"},
    description="Auto-trained Evidence (33 feat) + Lean (19 feat) models per station",
    ins={"multi_station_training_data": dg.AssetIn(key=_KEY("multi_station_training_data"))},
    config_schema={
        "ensemble_margin": dg.Field(
            float,
            default_value=0.01,
            description="AUC margin for ensembling classifiers (R² margin = 2×)",
        ),
    },
)
def per_station_trained_models(
    context: dg.AssetExecutionContext,
    multi_station_training_data: pd.DataFrame,
) -> dict:
    """Train every variant in _VARIANTS for the current station partition.

    Returns a flat dict, one entry per (task, variant). Tasks: regression,
    clf_5ppb, clf_10ppb, clf_30ppb. Both variants today:
      {task}_evidence ← 33 feat, production
      {task}_lean     ← 19 feat, parallel

    clf_30ppb exists to give the tiered alert cascade a calibrated
    P(H2S > 30 ppb) — Tier 3 in docs/feature/rename_workplan.md. Positives
    at 30 ppb are sparse outside NESTOR-BES; per-site AUC/recall in the
    training report is the honest gauge.

    Both variants are deployed in parallel so reviewers can load either
    model from S3 and reproduce the comparison; see RESULTS.md for the
    not-overdetermined argument. Adding a future variant is one entry in
    `_VARIANTS` — no renames; all consumers select variant by suffix.
    """
    partition = context.partition_key  # e.g. 'san_ysidro'
    site_name = STATION_PARTITION_MAP[partition]
    ensemble_margin = context.op_config["ensemble_margin"]

    context.log.info(f"Training models for station: {site_name} (partition: {partition})")

    sdf = multi_station_training_data[
        multi_station_training_data['site_name'] == site_name
    ].copy().sort_values('time').reset_index(drop=True)

    if len(sdf) < 100:
        raise ValueError(f"Insufficient data for {site_name}: {len(sdf)} rows")

    split = int(len(sdf) * TRAIN_FRACTION)
    y_5 = sdf['exceed_5'].values
    y_10 = sdf['exceed_10'].values
    y_30 = sdf['exceed_30'].values
    context.log.info(f"  Records: {len(sdf):,} (train: {split:,}, test: {len(sdf)-split:,})")
    context.log.info(
        f"  Exceedance: >5={y_5.mean()*100:.1f}%, >10={y_10.mean()*100:.1f}%, "
        f">30={y_30.mean()*100:.1f}% (n={int(y_30.sum())})"
    )

    models: dict = {}
    choices: dict[str, dict[str, str]] = {}
    for variant, features in _VARIANTS.items():
        suffix = f"_{variant}"
        variant_results = _train_one_variant(
            context, sdf, features, split, ensemble_margin, variant
        )
        choices[variant] = {task: choice for task, (_, choice) in variant_results.items()}
        for task, (model, _) in variant_results.items():
            models[f"{task}{suffix}"] = model

    context.add_output_metadata({
        "station": site_name,
        "partition": partition,
        "n_train": int(split),
        "n_test": int(len(sdf) - split),
        "tasks": list(models.keys()),
        "variants": list(_VARIANTS.keys()),
        "algorithm_choices": choices,
    })
    return models


# ==============================================================================
# Asset 3: Station training report (partitioned by station)
# ==============================================================================

def _importance_for_features(model, feature_names: list[str], top_n: int = 10) -> dict:
    """Feature importance keyed by the variant's actual feature list.

    `multi_station_trainer.get_feature_importance` hardcodes MODEL_FEATURES,
    which would mis-label Lean models' importances. We re-do it here against
    the variant's own column order.
    """
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp)
    idx = np.argsort(imp)[::-1][:top_n]
    return {feature_names[i]: round(float(imp[i]), 4) for i in idx if i < len(feature_names)}


@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"json", "s3"},
    description="JSON training metrics report for both Evidence + Lean variants per station",
    ins={
        "multi_station_training_data": dg.AssetIn(key=_KEY("multi_station_training_data")),
        "per_station_trained_models": dg.AssetIn(key=_KEY("per_station_trained_models")),
    },
    config_schema={
        "ensemble_margin": dg.Field(float, default_value=0.01),
    },
)
def station_training_report(
    context: dg.AssetExecutionContext,
    multi_station_training_data: pd.DataFrame,
    per_station_trained_models: dict,
) -> dict:
    """Generate and upload training metrics report for both variants.

    Report shape:
      tasks: {
        evidence: { regression: {...}, clf_5ppb: {...}, clf_10ppb: {...} },
        lean:     { regression: {...}, clf_5ppb: {...}, clf_10ppb: {...} },
      }
      features: { evidence: [...33 cols...], lean: [...19 cols...] }
    """
    from h2s.training.multi_station_trainer import eval_regressor, eval_classifier

    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    ensemble_margin = context.op_config["ensemble_margin"]

    sdf = multi_station_training_data[
        multi_station_training_data['site_name'] == site_name
    ].sort_values('time').reset_index(drop=True)

    split = int(len(sdf) * TRAIN_FRACTION)
    yte_c = sdf['H2S'].values[split:]
    yte_5 = sdf['exceed_5'].values[split:]
    yte_10 = sdf['exceed_10'].values[split:]
    yte_30 = sdf['exceed_30'].values[split:]

    tasks_metrics: dict[str, dict] = {}
    for variant, features in _VARIANTS.items():
        suffix = f"_{variant}"
        Xte = sdf[features].values[split:]
        variant_metrics: dict[str, dict] = {}
        for task_base, yte in [('regression', yte_c), ('clf_5ppb', yte_5),
                               ('clf_10ppb', yte_10), ('clf_30ppb', yte_30)]:
            model = per_station_trained_models[f"{task_base}{suffix}"]
            if task_base == 'regression':
                m = eval_regressor(model, Xte, yte)
            else:
                m = eval_classifier(model, Xte, yte)
            variant_metrics[task_base] = {
                **m,
                'feature_importance': _importance_for_features(model, features),
            }
        tasks_metrics[variant] = variant_metrics

    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'n_records': len(sdf),
        'n_train': split,
        'n_test': len(sdf) - split,
        'features': {variant: features for variant, features in _VARIANTS.items()},
        'ensemble_margin': ensemble_margin,
        'tasks': tasks_metrics,
        'training_snapshot': {
            's3_path': multi_station_training_data.attrs.get('training_snapshot_s3_path'),
            'bucket': multi_station_training_data.attrs.get('training_snapshot_bucket'),
            'source_bucket': multi_station_training_data.attrs.get('training_snapshot_source_bucket'),
            'source_path': multi_station_training_data.attrs.get('training_snapshot_source_path'),
            'timestamp': multi_station_training_data.attrs.get('training_snapshot_timestamp'),
        },
    }

    # Upload to S3
    station_key = STATIONS[site_name]['key']
    report_path = f"{STATION_MODELS_S3_BASE}/{station_key}/training_report.json"
    try:
        report_bytes = json.dumps(report, indent=2, default=str).encode('utf-8')
        s3.putFile(report_bytes, report_path, bucket=s3.S3_BUCKET, content_type='application/json')
        context.log.info(f"✓ Uploaded training report to S3: {report_path}")
    except Exception as e:
        context.log.warning(f"Could not upload report to S3: {e}")

    context.add_output_metadata({
        "station": site_name,
        "evidence_regression_r2": tasks_metrics['evidence']['regression'].get('R2'),
        "evidence_clf5_auc": tasks_metrics['evidence']['clf_5ppb'].get('AUC'),
        "evidence_clf10_auc": tasks_metrics['evidence']['clf_10ppb'].get('AUC'),
        "evidence_clf30_auc": tasks_metrics['evidence']['clf_30ppb'].get('AUC'),
        "lean_regression_r2": tasks_metrics['lean']['regression'].get('R2'),
        "lean_clf5_auc": tasks_metrics['lean']['clf_5ppb'].get('AUC'),
        "lean_clf10_auc": tasks_metrics['lean']['clf_10ppb'].get('AUC'),
        "lean_clf30_auc": tasks_metrics['lean']['clf_30ppb'].get('AUC'),
    })
    return report


# ==============================================================================
# Asset 3b: Immutable model archive + Slack promotion report
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3", "slack"},
    kinds={"python", "s3"},
    description="Archive trained models to a versioned S3 prefix + Slack promotion report",
    ins={
        "per_station_trained_models": dg.AssetIn(key=_KEY("per_station_trained_models")),
        "station_training_report": dg.AssetIn(key=_KEY("station_training_report")),
    },
    config_schema={
        "post_to_slack": dg.Field(
            bool, default_value=True,
            description="Post the new-vs-production comparison to Slack",
        ),
    },
)
def station_model_archive(
    context: dg.AssetExecutionContext,
    per_station_trained_models: dict,
    station_training_report: dict,
) -> dict:
    """Write this training run to an immutable versioned archive.

    Every run lands at {STATION_MODELS_ARCHIVE_BASE}/{station_key}/{version}/
    regardless of whether it is ever promoted — so any past analysis can be
    replayed against the exact models that produced it. Production is NOT
    touched here; that happens via station_model_deployment (direct) or
    promote_station_models_job (from this archive, human-in-the-loop).

    Also posts the new-vs-production metric comparison to Slack with a
    promote/review recommendation and the exact promote command.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    station_key = STATIONS[site_name]['key']

    version_tag = make_version_tag()
    archive_base = f"{STATION_MODELS_ARCHIVE_BASE}/{station_key}/{version_tag}"
    context.log.info(f"Archiving {site_name} models as version {version_tag}")

    # Model pickles
    for name, model in per_station_trained_models.items():
        s3.putFile(pickle.dumps(model), f"{archive_base}/{name}.pkl",
                   bucket=s3.S3_BUCKET, content_type='application/octet-stream')
    # Feature schemas
    for variant, features in _VARIANTS.items():
        s3.putFile(json.dumps(features, indent=2).encode('utf-8'),
                   f"{archive_base}/features_{variant}.json",
                   bucket=s3.S3_BUCKET, content_type='application/json')
    # Training report (makes the version self-describing for replays)
    s3.putFile(json.dumps(station_training_report, indent=2, default=str).encode('utf-8'),
               f"{archive_base}/training_report.json",
               bucket=s3.S3_BUCKET, content_type='application/json')
    # Archive metadata
    archive_meta = {
        "model_version": version_tag,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_short_sha(),
        "station": site_name,
        "partition": partition,
        "artifacts": _station_artifact_names(),
    }
    s3.putFile(json.dumps(archive_meta, indent=2).encode('utf-8'),
               f"{archive_base}/archive_metadata.json",
               bucket=s3.S3_BUCKET, content_type='application/json')
    context.log.info(f"✓ Archived {len(per_station_trained_models) + 4} files → {archive_base}")

    # Compare against the production training report (if any)
    prod_report = None
    try:
        prod_bytes = s3.getFile(
            path=f"{STATION_MODELS_S3_BASE}/{station_key}/training_report.json",
            bucket=s3.S3_BUCKET)
        prod_report = json.loads(prod_bytes.decode('utf-8'))
    except Exception:
        context.log.info("No production training report found — first deployment?")

    comparison = compare_training_reports(station_training_report, prod_report)
    context.log.info(
        f"Comparison vs production: {comparison['recommendation'].upper()} — {comparison['reason']}"
    )

    message = build_promotion_message(
        site_name, partition, version_tag, comparison,
        env_label=os.environ.get("ENV_LABEL", ""),
    )
    if context.op_config["post_to_slack"]:
        try:
            slack = context.resources.slack
            slack.get_client().chat_postMessage(channel=slack.channel, text=message)
            context.log.info("✓ Posted promotion report to Slack")
        except Exception as e:
            context.log.warning(f"Could not post to Slack: {e}")
    else:
        context.log.info(f"Slack posting disabled; message would have been:\n{message}")

    context.add_output_metadata({
        "model_version": version_tag,
        "archive_prefix": archive_base,
        "recommendation": comparison["recommendation"],
        "n_improved": comparison["n_improved"],
        "n_regressed": comparison["n_regressed"],
    })
    return {
        "model_version": version_tag,
        "archive_prefix": archive_base,
        "comparison": comparison,
    }


# ==============================================================================
# Asset 4: Model deployment gate (manual approval + S3 upload)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Manual approval gate → upload station models to S3 production path",
    ins={
        "per_station_trained_models": dg.AssetIn(key=_KEY("per_station_trained_models")),
        "station_model_archive": dg.AssetIn(key=_KEY("station_model_archive")),
    },
    config_schema={
        "approve_deployment": dg.Field(
            bool,
            default_value=True,
            description=(
                "Default True: running station_deployment_job IS the approval — "
                "models are uploaded to S3. Set to False for a dry run that "
                "loads + validates the trained models without writing to S3."
            ),
        ),
    },
)
def station_model_deployment(
    context: dg.AssetExecutionContext,
    per_station_trained_models: dict,
    station_model_archive: dict,
) -> dict:
    """Upload trained station models to S3 (default) or dry-run.

    By default, running station_deployment_job uploads the trained models to
    S3 — the act of launching the job IS the approval. Pass
    `approve_deployment=False` in the asset config to do a dry run that
    validates the upstream models without writing to S3 (returns
    `{"status": "dry_run", ...}`).

    Models are written to: tijuana/forecast/models/stations/{station_key}/{task}.pkl

    Both Evidence (33-feat, the production default — no suffix) and Lean
    (19-feat, suffix `_lean`) variants are uploaded each cycle. Schema files
    `features.json` and `features_lean.json` describe each variant's column
    order so a consumer can load `regression{_lean}.pkl` + `features{_lean}.json`
    and produce inferences end-to-end.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    approved = context.op_config["approve_deployment"]

    station_key = STATIONS[site_name]['key']
    base_path = f"{STATION_MODELS_S3_BASE}/{station_key}"

    if not approved:
        context.log.warning(
            f"Dry-run for {site_name} (approve_deployment=False). "
            f"Skipping S3 upload."
        )
        return {"status": "dry_run", "station": site_name}

    context.log.info(f"Deploying models for {site_name} to S3: {base_path}")

    # Upload each variant's pickles (Evidence's filenames are unsuffixed,
    # Lean's carry `_lean`). The dict already contains both sets keyed
    # appropriately by per_station_trained_models.
    uploaded: dict[str, str] = {}
    for task, model in per_station_trained_models.items():
        s3_path = f"{base_path}/{task}.pkl"
        s3.putFile(pickle.dumps(model), s3_path, bucket=s3.S3_BUCKET,
                   content_type='application/octet-stream')
        context.log.info(f"  ✓ Uploaded {task} → {s3_path}")
        uploaded[task] = s3_path

    # Write per-variant feature schema files (used by inference to match
    # the variant's column order).
    feature_files: dict[str, str] = {}
    for variant, features in _VARIANTS.items():
        suffix = f"_{variant}"
        feat_path = f"{base_path}/features{suffix}.json"
        s3.putFile(
            json.dumps(features, indent=2).encode('utf-8'),
            feat_path,
            bucket=s3.S3_BUCKET,
            content_type='application/json',
        )
        context.log.info(f"  ✓ Uploaded features{suffix}.json ({len(features)} features)")
        feature_files[variant] = feat_path

    # Deployment metadata describes both variants under `variants` keys so
    # downstream consumers can pick either path without guessing filenames.
    variants_meta: dict[str, dict] = {}
    for variant, features in _VARIANTS.items():
        suffix = f"_{variant}"
        variants_meta[variant] = {
            'features_path': feature_files[variant],
            'n_features': len(features),
            'models': {task: uploaded[f"{task}{suffix}"]
                       for task in ('regression', 'clf_5ppb', 'clf_10ppb', 'clf_30ppb')},
        }

    model_version = station_model_archive.get("model_version", "unversioned")
    meta = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'model_version': model_version,
        'archive_prefix': station_model_archive.get("archive_prefix"),
        'variants': variants_meta,
    }
    meta_path = f"{base_path}/deployment_metadata.json"
    s3.putFile(
        json.dumps(meta, indent=2).encode('utf-8'),
        meta_path,
        bucket=s3.S3_BUCKET,
        content_type='application/json',
    )

    context.add_output_metadata({
        "status": "deployed",
        "station": site_name,
        "model_version": model_version,
        "models_uploaded": list(uploaded.keys()),
        "variants": list(_VARIANTS.keys()),
        "s3_base_path": base_path,
    })
    return {"status": "deployed", "station": site_name, "model_version": model_version,
            "models": uploaded, "variants": list(_VARIANTS.keys())}


# ==============================================================================
# Asset 5: Promote an archived version to production (human-in-the-loop)
# ==============================================================================

@dg.asset(
    key_prefix="h2s",
    group_name="h2s_multi_station_training",
    partitions_def=STATION_PARTITIONS,
    required_resource_keys={"s3"},
    kinds={"python", "s3"},
    description="Copy an archived model version to the production prefix",
    config_schema={
        "version_tag": dg.Field(
            str, default_value="",
            description=(
                "Archive version to promote (e.g. 20260612T213000Z-a1b2c3d). "
                "Empty = latest version in the archive for this station."
            ),
        ),
    },
)
def station_model_promotion(context: dg.AssetExecutionContext) -> dict:
    """Promote an archived model version to production.

    This is the human-in-the-loop approval for the monthly retrain flow:
    training archives every run and posts a comparison to Slack; a human
    reviews and runs promote_station_models_job with the version tag from
    the Slack message. Running this job IS the approval.

    Copies all model pickles + feature schemas + training_report.json from
    {STATION_MODELS_ARCHIVE_BASE}/{station_key}/{version}/ to the production
    prefix, then rewrites production deployment_metadata.json with the
    promoted model_version.
    """
    partition = context.partition_key
    site_name = STATION_PARTITION_MAP[partition]
    s3 = context.resources.s3
    station_key = STATIONS[site_name]['key']

    archive_root = f"{STATION_MODELS_ARCHIVE_BASE}/{station_key}/"
    version_tag = context.op_config["version_tag"]

    if not version_tag:
        # Resolve latest: version tags are lexically sortable by design
        prefixes = [
            obj.object_name for obj in s3.listPath(path=archive_root, bucket=s3.S3_BUCKET)
            if obj.object_name.endswith('/')
        ]
        versions = sorted(p.rstrip('/').rsplit('/', 1)[-1] for p in prefixes)
        if not versions:
            raise dg.Failure(f"No archived versions found under {archive_root}")
        version_tag = versions[-1]
        context.log.info(f"No version_tag configured — promoting latest: {version_tag}")

    archive_base = f"{archive_root}{version_tag}"
    prod_base = f"{STATION_MODELS_S3_BASE}/{station_key}"
    context.log.info(f"Promoting {site_name} {version_tag}: {archive_base} → {prod_base}")

    promoted: list[str] = []
    for name in _station_artifact_names() + ["training_report.json"]:
        try:
            data = s3.getFile(path=f"{archive_base}/{name}", bucket=s3.S3_BUCKET)
        except Exception as e:
            raise dg.Failure(
                f"Archive {version_tag} is missing {name} — refusing partial promotion: {e}"
            )
        content_type = 'application/json' if name.endswith('.json') else 'application/octet-stream'
        s3.putFile(data, f"{prod_base}/{name}", bucket=s3.S3_BUCKET, content_type=content_type)
        promoted.append(name)
        context.log.info(f"  ✓ {name}")

    # Rebuild production deployment metadata around the promoted version
    variants_meta: dict[str, dict] = {}
    for variant, features in _VARIANTS.items():
        variants_meta[variant] = {
            'features_path': f"{prod_base}/features_{variant}.json",
            'n_features': len(features),
            'models': {task: f"{prod_base}/{task}_{variant}.pkl" for task in _TASKS},
        }
    meta = {
        'deployed_at': datetime.now(timezone.utc).isoformat(),
        'station': site_name,
        'partition': partition,
        'model_version': version_tag,
        'archive_prefix': archive_base,
        'promoted': True,
        'variants': variants_meta,
    }
    s3.putFile(json.dumps(meta, indent=2).encode('utf-8'),
               f"{prod_base}/deployment_metadata.json",
               bucket=s3.S3_BUCKET, content_type='application/json')

    context.add_output_metadata({
        "station": site_name,
        "model_version": version_tag,
        "files_promoted": len(promoted),
        "archive_prefix": archive_base,
    })
    return {"status": "promoted", "station": site_name,
            "model_version": version_tag, "files": promoted}


# ==============================================================================
# Job definitions
# ==============================================================================

multi_station_training_job = dg.define_asset_job(
    name="multi_station_training_job",
    description="Train per-station H2S models for all stations",
    selection=dg.AssetSelection.assets(
        multi_station_training_data,
        per_station_trained_models,
        station_training_report,
        station_model_archive,
    ),
    partitions_def=STATION_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_multi_station_training"},
)

station_deployment_job = dg.define_asset_job(
    name="station_deployment_job",
    description=(
        "Deploy station models to S3 — running this job IS the approval. "
        "Pass approve_deployment=False in run config for a dry run."
    ),
    selection=dg.AssetSelection.assets(station_model_deployment),
    partitions_def=STATION_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_deployment"},
)

promote_station_models_job = dg.define_asset_job(
    name="promote_station_models_job",
    description=(
        "Promote an archived model version to production — running this job "
        "IS the approval. Configure version_tag (from the Slack training "
        "report) or leave empty for the latest archive."
    ),
    selection=dg.AssetSelection.assets(station_model_promotion),
    partitions_def=STATION_PARTITIONS,
    tags={"environment": "production", "pipeline": "h2s_deployment"},
)
