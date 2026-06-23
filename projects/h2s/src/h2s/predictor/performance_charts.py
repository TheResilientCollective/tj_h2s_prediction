"""Charts for the forecast performance report.

Renders the report from ``forecasting.performance_report`` as PNG ``BytesIO``
objects (matplotlib Agg) for S3 upload:

  generate_pred_vs_actual_scatter() predicted vs measured H2S, coloured by the
                                    asymmetric verdict, with the tier cut lines.
  generate_correlation_by_hour()    Spearman + MAE vs lead hour and vs UTC
                                    hour-of-day (the "performance by hours"
                                    correlation chart).
  generate_verdict_confusion()      4×4 category confusion grid + verdict tally
                                    bar, annotated with the headline metrics.
"""

from io import BytesIO
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from h2s.constants import (  # noqa: E402
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
    H2S_THRESHOLD_HIGH,
    H2S_THRESHOLD_LOW,
    H2S_THRESHOLD_MED,
)
from h2s.forecasting.performance_report import VERDICT_COST  # noqa: E402
from h2s.utils.chart_meta import stamp_generated  # noqa: E402

# Verdict → colour (green=good … red=dangerous), ordered best-to-worst.
_VERDICT_ORDER = [
    "match", "yellow_band", "early_warning_ok", "expected_low",
    "soft_miss", "false_alarm", "smell_miss", "dangerous_miss",
]
_VERDICT_COLORS = {
    "match": "#2ecc71",
    "yellow_band": "#a9dfbf",
    "early_warning_ok": "#5dade2",
    "expected_low": "#aeb6bf",
    "soft_miss": "#f7dc6f",
    "false_alarm": "#f0b27a",
    "smell_miss": "#e67e22",
    "dangerous_miss": "#e74c3c",
}


def _empty(message: str) -> BytesIO:
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    stamp_generated(fig)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _finish(fig) -> BytesIO:
    stamp_generated(fig)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_pred_vs_actual_scatter(
    annotated: pd.DataFrame, *, env_label: str = ""
) -> BytesIO:
    """Predicted vs measured H2S on a symlog grid, coloured by verdict."""
    if annotated is None or annotated.empty:
        return _empty("No matched forecast/observation pairs yet")

    label = f" [{env_label}]" if env_label else ""
    fig, ax = plt.subplots(figsize=(7.5, 7))
    fig.patch.set_facecolor("#f8f9fa")

    for verdict in _VERDICT_ORDER:
        g = annotated[annotated["verdict"] == verdict]
        if g.empty:
            continue
        ax.scatter(
            g["actual_h2s"], g["h2s_pred"], s=18, alpha=0.6,
            color=_VERDICT_COLORS.get(verdict, "#888"),
            label=f"{verdict} ({len(g)})", edgecolors="none",
        )

    hi = float(max(annotated["actual_h2s"].max(), annotated["h2s_pred"].max(), 30)) * 1.1
    ax.plot([0, hi], [0, hi], "--", color="#555", linewidth=1, label="perfect")
    for thr in (H2S_THRESHOLD_LOW, H2S_THRESHOLD_MED, H2S_THRESHOLD_HIGH):
        ax.axvline(thr, color="#bbb", linewidth=0.7, linestyle=":")
        ax.axhline(thr, color="#bbb", linewidth=0.7, linestyle=":")

    ax.set_xscale("symlog", linthresh=5)
    ax.set_yscale("symlog", linthresh=5)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("measured H2S (ppb)")
    ax.set_ylabel("predicted H2S (ppb)")
    ax.set_title(
        f"Predicted vs Measured H2S{label}\n"
        "below the diagonal = under-prediction (the hazardous direction)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    return _finish(fig)


def _corr_panel(ax, df: pd.DataFrame, xcol: str, xlabel: str):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    x = df[xcol].to_numpy()
    sp = pd.to_numeric(df["spearman"], errors="coerce").to_numpy(dtype=float)
    ax.plot(x, sp, "o-", color="#2980b9", label="Spearman ρ")
    ax.axhline(0, color="#999", linewidth=0.7)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Spearman ρ", color="#2980b9")
    ax.tick_params(axis="y", labelcolor="#2980b9")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x, df["mae"].to_numpy(dtype=float), "s--", color="#c0392b",
             alpha=0.7, label="MAE")
    ax2.set_ylabel("MAE (ppb)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")


def generate_correlation_by_hour(
    by_lead_hour: pd.DataFrame,
    by_hour_of_day: pd.DataFrame,
    *,
    env_label: str = "",
) -> BytesIO:
    """Two-panel correlation chart: skill vs lead hour and vs hour-of-day."""
    if (by_lead_hour is None or by_lead_hour.empty) and (
        by_hour_of_day is None or by_hour_of_day.empty
    ):
        return _empty("No matched pairs for correlation yet")

    label = f" [{env_label}]" if env_label else ""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#f8f9fa")
    _corr_panel(axl, by_lead_hour, "lead_hour", "lead hour (h ahead)")
    axl.set_title("Skill vs lead hour", fontsize=11, fontweight="bold")
    _corr_panel(axr, by_hour_of_day, "hour_of_day", "UTC hour-of-day")
    axr.set_title("Skill vs hour-of-day", fontsize=11, fontweight="bold")
    fig.suptitle(
        f"Forecast Performance by Hour{label}  (ρ = rank correlation of predicted vs measured)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _finish(fig)


def generate_verdict_confusion(
    confusion: pd.DataFrame,
    summary: dict,
    *,
    env_label: str = "",
) -> BytesIO:
    """Category confusion grid + verdict tally bar with headline metrics."""
    label = f" [{env_label}]" if env_label else ""
    fig, (axc, axb) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    fig.patch.set_facecolor("#f8f9fa")

    # --- Confusion grid (predicted rows × actual cols) ---------------------
    cats = list(CATEGORY_ORDER)
    mat = confusion.reindex(index=cats, columns=cats, fill_value=0).to_numpy(dtype=float)
    total = mat.sum() or 1
    axc.imshow(mat / total, cmap="Blues", vmin=0, vmax=max(0.25, (mat / total).max()))
    for i in range(len(cats)):
        for j in range(len(cats)):
            cnt = int(mat[i, j])
            axc.text(j, i, cnt, ha="center", va="center", fontsize=10,
                     color="white" if mat[i, j] / total > 0.2 else "#1a1a1a")
    axc.set_xticks(range(len(cats)))
    axc.set_xticklabels([CATEGORY_LABELS[c] for c in cats], rotation=20, fontsize=8)
    axc.set_yticks(range(len(cats)))
    axc.set_yticklabels([CATEGORY_LABELS[c] for c in cats], fontsize=8)
    axc.set_xlabel("measured")
    axc.set_ylabel("predicted")
    axc.set_title("Category confusion (counts)\nupper-right = missed hazard",
                  fontsize=10, fontweight="bold")
    # Box the worst cell: predicted green, measured orange (top-right).
    n_cat = len(cats)
    axc.add_patch(plt.Rectangle((n_cat - 1.5, -0.5), 1, 1, fill=False,
                                edgecolor="#e74c3c", linewidth=2.5, clip_on=False))

    # --- Verdict tally bar -------------------------------------------------
    vc = summary.get("verdict_counts", {})
    present = [v for v in _VERDICT_ORDER if vc.get(v, 0) > 0]
    counts = [vc[v] for v in present]
    colors = [_VERDICT_COLORS.get(v, "#888") for v in present]
    axb.barh(range(len(present)), counts, color=colors)
    axb.set_yticks(range(len(present)))
    axb.set_yticklabels([f"{v} (×{VERDICT_COST[v]:g})" for v in present], fontsize=8)
    axb.invert_yaxis()
    axb.set_xlabel("rows")
    axb.set_title("Verdict tally (cost weight)", fontsize=10, fontweight="bold")
    for i, c in enumerate(counts):
        axb.text(c, i, f" {c}", va="center", fontsize=8)

    ta = summary.get("tolerant_accuracy")
    dm = summary.get("dangerous_miss_rate")
    sm = summary.get("smell_miss_rate")
    mc = summary.get("mean_cost")
    headline = (
        f"n={summary.get('n', 0)}   "
        f"tolerant acc={ta:.0%}" if ta is not None else f"n={summary.get('n', 0)}"
    )
    if dm is not None:
        headline += f"   dangerous-miss={dm:.1%}   smell-miss={sm:.1%}   mean cost={mc:.2f}"
    fig.suptitle(
        f"Forecast Performance Rubric{label}\n{headline}",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _finish(fig)
