"""Accuracy reporting pipeline.

Aggregates the per-day `metrics.json` artifacts that
`daily_validation_schedule` writes into S3 and produces stakeholder-facing
rollups:

    s3://{bucket}/tijuana/forecast/accuracy_reports/
        daily/{YYYY-MM-DD}/scorecard.json
        rolling/{7d,30d,90d}/scorecard.json
        monthly/{YYYY-MM}/scorecard.json
        alert_performance/{period}.json
        lead_time/{period}.json
        calibration/{period}.json
        regime/{period}.json
        latest.json                 ← single source of truth for downstream UIs

Downstream consumers (Quarto report, Panel dashboard, geodemic Analytics page,
weekly Slack scorecard) read these rollups directly and stay dumb.

Pure computation functions live at the top of the file so they are unit-
testable without a Dagster context. Dagster assets at the bottom wire them
together and handle I/O.
"""


import json
import os
import shutil
import urllib.request
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext

from h2s.resources.minio import S3Resource

# ---------------------------------------------------------------------------
# S3 layout
# ---------------------------------------------------------------------------

VALIDATION_PREFIX = "tijuana/forecast/validation"
ACCURACY_PREFIX = "tijuana/forecast/accuracy_reports"
FORECAST_PREFIX = "tijuana/forecast/output"
OBSERVATIONS_KEY = "latest/tijuana/forecast_data/modeldata_h2s.csv"

ROLLING_WINDOWS_DAYS = (7, 30, 90)

# Category thresholds, mirrored from dashboard/constants.py.
H2S_GREEN_MAX = 5.0
H2S_YELLOW_MAX = 30.0
CATEGORIES = ("green", "yellow", "orange")


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------


@dataclass
class SiteScorecard:
    site: str
    n_predictions: int
    n_matched_observations: int
    balanced_accuracy: float | None
    orange_recall: float | None
    orange_precision: float | None
    false_alarm_rate: float | None
    brier_score: float | None
    expected_calibration_error: float | None
    confusion_matrix: list[list[int]]


@dataclass
class PeriodScorecard:
    scope: str  # "daily" | "rolling" | "monthly"
    period_start: str
    period_end: str
    generated_at: str
    sites: dict[str, dict[str, Any]]
    overall: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pure metric functions (no I/O)
# ---------------------------------------------------------------------------


def categorize(series: pd.Series) -> pd.Series:
    """Map H2S ppb → {green, yellow, orange}."""
    return pd.cut(
        series,
        bins=[-np.inf, H2S_GREEN_MAX, H2S_YELLOW_MAX, np.inf],
        labels=list(CATEGORIES),
    )


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str]) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    m = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m.tolist()


def _safe_div(num: float, den: float) -> float | None:
    return float(num) / float(den) if den else None


def balanced_accuracy(cm: list[list[int]]) -> float | None:
    m = np.asarray(cm)
    row_sums = m.sum(axis=1)
    recalls = [m[i, i] / row_sums[i] for i in range(len(m)) if row_sums[i] > 0]
    return float(np.mean(recalls)) if recalls else None


def class_precision_recall(
    cm: list[list[int]], cls: str
) -> tuple[float | None, float | None]:
    idx = CATEGORIES.index(cls)
    m = np.asarray(cm)
    tp = int(m[idx, idx])
    fn = int(m[idx, :].sum()) - tp
    fp = int(m[:, idx].sum()) - tp
    return _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)


def false_alarm_rate_for_orange(cm: list[list[int]]) -> float | None:
    """FPR for the orange class = FP / (FP + TN) where negatives are
    green+yellow."""
    idx = CATEGORIES.index("orange")
    m = np.asarray(cm)
    fp = int(m[:, idx].sum()) - int(m[idx, idx])
    tn = int(m.sum()) - int(m[idx, :].sum()) - fp
    return _safe_div(fp, fp + tn)


def brier_score(
    prob_orange: Iterable[float], y_true_orange: Iterable[int]
) -> float | None:
    p = np.asarray(list(prob_orange), dtype=float)
    y = np.asarray(list(y_true_orange), dtype=float)
    if len(p) == 0:
        return None
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    prob_orange: Iterable[float],
    y_true_orange: Iterable[int],
    n_bins: int = 10,
) -> float | None:
    p = np.asarray(list(prob_orange), dtype=float)
    y = np.asarray(list(y_true_orange), dtype=float)
    if len(p) == 0:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(p)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += (mask.sum() / total) * abs(conf - acc)
    return float(ece)


def reliability_bins(
    prob_orange: Iterable[float],
    y_true_orange: Iterable[int],
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Reliability diagram data — what Quarto + Panel render as a chart."""
    p = np.asarray(list(prob_orange), dtype=float)
    y = np.asarray(list(y_true_orange), dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float | int]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": 0,
                        "mean_pred": None, "observed_rate": None})
        else:
            out.append({
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": n,
                "mean_pred": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            })
    return out


def scorecard_from_predictions(
    preds: pd.DataFrame,
    obs: pd.DataFrame,
    site: str,
) -> SiteScorecard:
    """Build a SiteScorecard by joining predictions to observations on time.

    `preds` columns expected: time, site_name, predicted_category,
    probability_orange (optional).
    `obs` columns expected: time, site_name, H2S.
    """
    p = preds[preds["site_name"] == site].copy()
    o = obs[obs["site_name"] == site].copy()
    if p.empty:
        return SiteScorecard(site, 0, 0, None, None, None, None, None, None,
                             [[0] * len(CATEGORIES) for _ in CATEGORIES])

    joined = pd.merge_asof(
        p.sort_values("time"),
        o.sort_values("time"),
        on="time",
        tolerance=pd.Timedelta("30min"),
        direction="nearest",
    ).dropna(subset=["H2S"])

    if joined.empty:
        return SiteScorecard(site, len(p), 0, None, None, None, None, None, None,
                             [[0] * len(CATEGORIES) for _ in CATEGORIES])

    y_true = categorize(joined["H2S"]).astype(str).tolist()
    y_pred = joined["predicted_category"].astype(str).tolist()
    cm = confusion_matrix(y_true, y_pred)

    _, orange_recall = class_precision_recall(cm, "orange")
    orange_precision, _ = class_precision_recall(cm, "orange")
    far = false_alarm_rate_for_orange(cm)

    brier = ece = None
    if "probability_orange" in joined:
        y_true_orange = [1 if c == "orange" else 0 for c in y_true]
        brier = brier_score(joined["probability_orange"], y_true_orange)
        ece = expected_calibration_error(joined["probability_orange"], y_true_orange)

    return SiteScorecard(
        site=site,
        n_predictions=len(p),
        n_matched_observations=len(joined),
        balanced_accuracy=balanced_accuracy(cm),
        orange_recall=orange_recall,
        orange_precision=orange_precision,
        false_alarm_rate=far,
        brier_score=brier,
        expected_calibration_error=ece,
        confusion_matrix=cm,
    )


def combine_confusion_matrices(cms: Iterable[list[list[int]]]) -> list[list[int]]:
    arr = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)
    for cm in cms:
        arr = arr + np.asarray(cm, dtype=int)
    return arr.tolist()


def overall_from_sites(site_cards: list[SiteScorecard]) -> dict[str, Any]:
    if not site_cards:
        return {"balanced_accuracy": None}
    cm = combine_confusion_matrices(c.confusion_matrix for c in site_cards)
    _, orange_recall = class_precision_recall(cm, "orange")
    orange_precision, _ = class_precision_recall(cm, "orange")
    return {
        "n_sites": len(site_cards),
        "n_predictions": sum(c.n_predictions for c in site_cards),
        "n_matched_observations": sum(c.n_matched_observations for c in site_cards),
        "balanced_accuracy": balanced_accuracy(cm),
        "orange_recall": orange_recall,
        "orange_precision": orange_precision,
        "false_alarm_rate": false_alarm_rate_for_orange(cm),
        "confusion_matrix": cm,
    }


def regime_slices(
    preds: pd.DataFrame,
    obs: pd.DataFrame,
    regime_col: str,
) -> list[dict[str, Any]]:
    """Accuracy sliced by a regime column on the predictions frame
    (wind_regime, season, tide_phase, flow_quartile, ...)."""
    out: list[dict[str, Any]] = []
    if regime_col not in preds.columns:
        return out
    for val, group in preds.groupby(regime_col, dropna=True):
        cms = [
            scorecard_from_predictions(group, obs, site).confusion_matrix
            for site in group["site_name"].unique()
        ]
        cm = combine_confusion_matrices(cms)
        _, rec = class_precision_recall(cm, "orange")
        out.append({
            "regime": regime_col,
            "value": str(val),
            "balanced_accuracy": balanced_accuracy(cm),
            "orange_recall": rec,
            "false_alarm_rate": false_alarm_rate_for_orange(cm),
            "n": int(np.asarray(cm).sum()),
        })
    return out


# ---------------------------------------------------------------------------
# Thin S3 wrapper around S3Resource (MinIO-compatible)
# ---------------------------------------------------------------------------


class AccuracyStore:
    """Read historical `metrics.json` files and write rollup artifacts.

    Uses the project's :class:`S3Resource` (minio client) so that no extra
    dependency (boto3) is needed.
    """

    def __init__(self, s3: S3Resource) -> None:
        self._s3 = s3

    def read_json(self, key: str) -> dict[str, Any] | None:
        try:
            url = self._s3.publicUrl(path=key)
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                return json.loads(resp.read())
        except Exception:
            return None

    def write_json(self, key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, default=str, indent=2)
        self._s3.putFile_text(data=body, path=key, content_type="application/json")

    def read_csv(self, key: str) -> pd.DataFrame:
        url = self._s3.publicUrl(path=key)
        return pd.read_csv(url)

    def list_validation_days(
        self, since: date, until: date, pipeline: str | None = None,
    ) -> list[date]:
        """Return the set of validation days present under the validation
        prefix within [since, until].

        If *pipeline* is given (e.g. "hourly", "daily_station"), look under
        the per-pipeline subdirectory.  Otherwise fall back to the legacy
        root ``metrics.json``.
        """
        days: list[date] = []
        cur = since
        while cur <= until:
            if self.read_day_metrics(cur, pipeline=pipeline) is not None:
                days.append(cur)
            cur += timedelta(days=1)
        return days

    def read_day_metrics(
        self, day: date, pipeline: str | None = None,
    ) -> dict[str, Any] | None:
        if pipeline:
            key = f"{VALIDATION_PREFIX}/{day.isoformat()}/{pipeline}/metrics.json"
        else:
            key = f"{VALIDATION_PREFIX}/{day.isoformat()}/metrics.json"
        return self.read_json(key)


# ---------------------------------------------------------------------------
# Rollup builders
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _extract_sites_dict(m: dict[str, Any]) -> dict[str, Any] | None:
    """Normalise a metrics.json payload into a sites dict (v1 or v2 schema)."""
    sites_dict = m.get("sites")
    if sites_dict is not None:
        return sites_dict
    # v1 flat schema: wrap into sites dict for uniform processing
    if m.get("confusion_matrix") is not None:
        site_key = m.get("site", "NESTOR__BES")
        return {
            site_key: {
                "confusion_matrix": m["confusion_matrix"],
                "n_predictions": m.get("n_predictions", 0),
                "n_matched_observations": m.get("n_matched", 0),
            },
        }
    return None


# Pipeline subdirectories to scan for per-day metrics.
# Order matters: daily_station covers all 3 stations, so its metrics are
# preferred.  The root path (pipeline=None) is the legacy hourly-only fallback.
_PIPELINES_TO_SCAN: list[str | None] = ["daily_station", None]


def build_period_scorecard(
    store: AccuracyStore,
    start: date,
    end: date,
    scope: str,
) -> PeriodScorecard:
    """Aggregate per-day metrics.json into a single scorecard for [start, end].

    Scans multiple pipeline subdirectories (daily_station and the legacy
    root) for each day and merges all sites found.  When a site appears in
    more than one pipeline on the same day, daily_station takes precedence
    (it covers all 3 stations and uses per-station models).

    Raises ``dg.Failure`` when no validation days are found in the window.
    """
    site_cms: dict[str, list[list[list[int]]]] = {}
    site_pred_counts: dict[str, int] = {}
    site_match_counts: dict[str, int] = {}
    found_any = False

    cur = start
    while cur <= end:
        # Collect sites from all pipelines for this day.
        # Track which sites we've already seen so earlier pipelines in the
        # list take precedence (daily_station > root).
        seen_sites_today: set[str] = set()

        for pipeline in _PIPELINES_TO_SCAN:
            m = store.read_day_metrics(cur, pipeline=pipeline)
            if not m:
                continue
            sites_dict = _extract_sites_dict(m)
            if not sites_dict:
                continue

            for site, site_m in sites_dict.items():
                if site in seen_sites_today:
                    continue  # already covered by a higher-priority pipeline
                cm = site_m.get("confusion_matrix")
                if cm is None:
                    continue
                seen_sites_today.add(site)
                found_any = True
                site_cms.setdefault(site, []).append(cm)
                site_pred_counts[site] = site_pred_counts.get(site, 0) + int(
                    site_m.get("n_predictions", 0)
                )
                site_match_counts[site] = site_match_counts.get(site, 0) + int(
                    site_m.get("n_matched_observations", site_m.get("n_matched", 0))
                )

        cur += timedelta(days=1)

    if not found_any:
        raise dg.Failure(
            f"No validation metrics found for {scope} window "
            f"[{start.isoformat()} .. {end.isoformat()}]"
        )

    site_cards: list[SiteScorecard] = []
    for site, cms in site_cms.items():
        cm = combine_confusion_matrices(cms)
        _, orange_recall = class_precision_recall(cm, "orange")
        orange_precision, _ = class_precision_recall(cm, "orange")
        site_cards.append(
            SiteScorecard(
                site=site,
                n_predictions=site_pred_counts.get(site, 0),
                n_matched_observations=site_match_counts.get(site, 0),
                balanced_accuracy=balanced_accuracy(cm),
                orange_recall=orange_recall,
                orange_precision=orange_precision,
                false_alarm_rate=false_alarm_rate_for_orange(cm),
                brier_score=None,
                expected_calibration_error=None,
                confusion_matrix=cm,
            )
        )

    return PeriodScorecard(
        scope=scope,
        period_start=start.isoformat(),
        period_end=end.isoformat(),
        generated_at=_now_iso(),
        sites={c.site: asdict(c) for c in site_cards},
        overall=overall_from_sites(site_cards),
    )


# ---------------------------------------------------------------------------
# Dagster assets
# ---------------------------------------------------------------------------

daily_partitions = dg.DailyPartitionsDefinition(start_date="2024-01-01")


@dg.asset(
    partitions_def=daily_partitions,
    group_name="accuracy_reporting",
    required_resource_keys={"s3"},
    description="Daily scorecard aggregated from that day's metrics.json",
)
def daily_accuracy_scorecard(context: AssetExecutionContext) -> dict[str, Any]:
    day = date.fromisoformat(context.partition_key)
    store = AccuracyStore(context.resources.s3)
    metrics = store.read_day_metrics(day)
    if metrics is None:
        raise dg.Failure(
            f"No metrics.json found for {day.isoformat()} at "
            f"{VALIDATION_PREFIX}/{day.isoformat()}/metrics.json"
        )
    card = build_period_scorecard(store, day, day, scope="daily")
    store.write_json(
        f"{ACCURACY_PREFIX}/daily/{day.isoformat()}/scorecard.json",
        card.to_dict(),
    )
    context.log.info(
        "daily scorecard: %d sites, overall balanced_accuracy=%s",
        len(card.sites),
        card.overall.get("balanced_accuracy"),
    )
    return card.to_dict()


@dg.asset(
    group_name="accuracy_reporting",
    required_resource_keys={"s3"},
    description="Rolling 7d/30d/90d scorecards written to S3 and a combined "
                "`latest.json` pointer for downstream UIs.",
    deps=[daily_accuracy_scorecard],
)
def rolling_accuracy_scorecards(context: AssetExecutionContext) -> dict[str, Any]:
    store = AccuracyStore(context.resources.s3)
    today = datetime.now(timezone.utc).date()
    summary: dict[str, Any] = {"generated_at": _now_iso(), "windows": {}}
    for window in ROLLING_WINDOWS_DAYS:
        start = today - timedelta(days=window)
        card = build_period_scorecard(store, start, today, scope="rolling")
        key = f"{ACCURACY_PREFIX}/rolling/{window}d/scorecard.json"
        store.write_json(key, card.to_dict())
        summary["windows"][f"{window}d"] = {
            "key": key,
            "overall": card.overall,
            "n_sites": len(card.sites),
        }
    # Always-current pointer for UIs.
    store.write_json(f"{ACCURACY_PREFIX}/latest.json", summary)
    return summary


@dg.asset(
    group_name="accuracy_reporting",
    required_resource_keys={"s3"},
    description="Calendar-month scorecard for the previous complete month.",
    deps=[daily_accuracy_scorecard],
)
def monthly_accuracy_scorecard(context: AssetExecutionContext) -> dict[str, Any]:
    store = AccuracyStore(context.resources.s3)
    today = datetime.now(timezone.utc).date()
    first_of_this_month = today.replace(day=1)
    last_of_prev = first_of_this_month - timedelta(days=1)
    start = last_of_prev.replace(day=1)
    period = f"{start:%Y-%m}"
    card = build_period_scorecard(store, start, last_of_prev, scope="monthly")
    store.write_json(
        f"{ACCURACY_PREFIX}/monthly/{period}/scorecard.json", card.to_dict()
    )
    return {"period": period, "overall": card.overall}


@dg.asset(
    group_name="accuracy_reporting",
    required_resource_keys={"s3"},
    description="Alert-level precision/recall for green/yellow/orange over the "
                "rolling 30-day window.",
    deps=[daily_accuracy_scorecard],
)
def alert_performance(context: AssetExecutionContext) -> dict[str, Any]:
    store = AccuracyStore(context.resources.s3)
    today = datetime.now(timezone.utc).date()
    card = build_period_scorecard(
        store, today - timedelta(days=30), today, scope="rolling"
    )
    overall_cm = card.overall.get("confusion_matrix") or [[0] * 3 for _ in range(3)]
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
    return payload


# ---------------------------------------------------------------------------
# Stakeholder artifacts: monthly Quarto render + weekly Slack scorecard
# ---------------------------------------------------------------------------

# Co-located templates and helpers live under projects/h2s/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
QUARTO_TEMPLATE = _REPO_ROOT / "projects" / "h2s" / "reports" / "monthly_accuracy.qmd"


@dg.asset(
    group_name="accuracy_reporting",
    required_resource_keys={"s3"},
    description="Render the monthly Quarto accuracy report to HTML and upload "
                "to S3 under `accuracy_reports/monthly/{YYYY-MM}/`. Netlify "
                "publishing is handled by the generic netlify_triggers flow.",
    deps=[monthly_accuracy_scorecard],
)
def monthly_accuracy_report_html(context: AssetExecutionContext) -> dict[str, Any]:
    if not QUARTO_TEMPLATE.exists():
        raise dg.Failure(f"Quarto template not found: {QUARTO_TEMPLATE}")
    if shutil.which("quarto") is None:
        raise dg.Failure(
            "`quarto` CLI not found on PATH. Install Quarto in the Dagster "
            "code-server image before enabling this asset."
        )

    today = datetime.now(timezone.utc).date()
    first_of_this_month = today.replace(day=1)
    last_of_prev = first_of_this_month - timedelta(days=1)
    period = f"{last_of_prev:%Y-%m}"

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "_site"
        cmd = [
            "quarto", "render", str(QUARTO_TEMPLATE),
            "--to", "html",
            "--output-dir", str(out_dir),
        ]
        env = {**os.environ, "REPORT_PERIOD": period}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            context.log.error("quarto stderr: %s", result.stderr)
            raise dg.Failure(f"quarto render failed: {result.stderr[:400]}")

        html_path = out_dir / "monthly_accuracy.html"
        if not html_path.exists():  # fall back to whatever quarto wrote
            candidates = list(out_dir.glob("*.html"))
            if not candidates:
                raise dg.Failure("quarto produced no HTML output")
            html_path = candidates[0]

        s3 = context.resources.s3
        key = f"{ACCURACY_PREFIX}/monthly/{period}/report.html"
        s3.putFile(
            data=html_path.read_bytes(),
            path=key,
            content_type="text/html; charset=utf-8",
        )
        context.log.info("uploaded monthly report to s3://%s/%s", s3.S3_BUCKET, key)
        return {"period": period, "s3_key": key}


@dg.asset(
    group_name="accuracy_reporting",
    description="Post the weekly H2S accuracy scorecard to Slack. Requires "
                "SLACK_WEBHOOK_URL in the environment.",
    deps=[rolling_accuracy_scorecards],
)
def weekly_scorecard_post(context: AssetExecutionContext) -> dict[str, Any]:
    from h2s.reporting import weekly_scorecard  # local import — avoid at import time

    if "SLACK_WEBHOOK_URL" not in os.environ:
        raise dg.Failure("SLACK_WEBHOOK_URL is not set")
    weekly_scorecard.main()
    return {"posted_at": _now_iso()}


accuracy_reporting_job = dg.define_asset_job(
    name="accuracy_reporting_job",
    selection=[
        "daily_accuracy_scorecard",
        "rolling_accuracy_scorecards",
        "alert_performance",
    ],
)

monthly_accuracy_job = dg.define_asset_job(
    name="monthly_accuracy_job",
    selection=["monthly_accuracy_scorecard", "monthly_accuracy_report_html"],
)

weekly_scorecard_job = dg.define_asset_job(
    name="weekly_scorecard_job",
    selection=["weekly_scorecard_post"],
)


@dg.schedule(
    job=accuracy_reporting_job,
    cron_schedule="0 10 * * *",  # after all validation schedules (hourly 8AM, station 9AM, MH 9:30AM).
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def daily_accuracy_schedule(context: dg.ScheduleEvaluationContext):
    partition = (context.scheduled_execution_time.date() - timedelta(days=1)).isoformat()
    return dg.RunRequest(
        run_key=f"accuracy-{partition}",
        partition_key=partition,
    )


@dg.schedule(
    job=monthly_accuracy_job,
    cron_schedule="0 9 1 * *",  # first day of the month, after rollups settle.
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def monthly_accuracy_schedule(context: dg.ScheduleEvaluationContext):
    return dg.RunRequest(run_key=f"monthly-{context.scheduled_execution_time:%Y-%m}")


@dg.schedule(
    job=weekly_scorecard_job,
    cron_schedule="0 16 * * 1",  # Mondays 09:00 America/Los_Angeles = 16:00 UTC.
    execution_timezone="UTC",
    default_status=dg.DefaultScheduleStatus.RUNNING,
)
def weekly_scorecard_schedule(context: dg.ScheduleEvaluationContext):
    return dg.RunRequest(
        run_key=f"weekly-scorecard-{context.scheduled_execution_time:%Y-%m-%d}"
    )
