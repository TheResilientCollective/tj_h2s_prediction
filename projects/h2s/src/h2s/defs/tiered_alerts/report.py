"""Self-contained HTML report generator for tier backtest records.

Called from backtest.generate_html_report(); not intended for direct import
outside of this package.
"""

import base64
import io
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .tiers import HORIZON_ORDER, TIER3_TARGETS

# ── palette ──────────────────────────────────────────────────────────────────
_DAY_COLOR   = "#e07b39"   # amber
_NIGHT_COLOR = "#4a7fb5"   # steel blue
_PREC_ALPHA  = 1.0
_REC_ALPHA   = 0.55
_TIER_COLORS = {"tier_1": "#5aab61", "tier_2": "#e07b39", "tier_3": "#c0392b"}
_HORIZON_LABELS = {
    "nowcast":   "Nowcast (0–3h)",
    "near":      "Near (3–6h)",
    "mid":       "Mid (6–12h)",
    "day_ahead": "Day-ahead (12–24h)",
}


def _fig_to_b64(fig: matplotlib.figure.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _f(v: object) -> float:
    """Extract a Python float from an iterrows() cell (Pyright can't narrow these)."""
    return float(v)  # type: ignore[arg-type]


def _i(v: object) -> int:
    return int(v)  # type: ignore[arg-type]


def _lt(v: object) -> str:
    """Format a lead-time cell as '1.2h', or '—' if NaN."""
    try:
        fv = float(v)  # type: ignore[arg-type]
        return f"{fv:.1f}h" if not math.isnan(fv) else "—"
    except (TypeError, ValueError):
        return "—"


def _months_sorted(df: pd.DataFrame) -> list[str]:
    return sorted(df["month"].unique().tolist())  # type: ignore[union-attr]


# ── Chart 1: monthly Tier-3 precision & recall ───────────────────────────────

def _chart_precision_recall(report_df: pd.DataFrame) -> str:
    t3: pd.DataFrame = report_df.loc[
        (report_df["tier"] == "tier_3") & (report_df["month"] != "ALL")
    ].copy()
    months = _months_sorted(t3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    fig.suptitle("Tier 3 — Monthly Precision & Recall by Horizon", fontsize=13, fontweight="bold")

    for ax, horizon in zip(axes.flat, HORIZON_ORDER):
        tgt_p, tgt_r = TIER3_TARGETS.get(horizon, (0.0, 0.0))
        label = _HORIZON_LABELS.get(horizon, horizon)

        for period, base_color in (("day", _DAY_COLOR), ("night", _NIGHT_COLOR)):
            sub: pd.DataFrame = t3.loc[(t3["horizon"] == horizon) & (t3["period"] == period)]
            sub = sub.set_index("month").reindex(months)
            xs = range(len(months))
            ax.plot(xs, sub["precision"], color=base_color, alpha=_PREC_ALPHA,
                    marker="o", markersize=4, label=f"{period} prec", linewidth=1.5)
            ax.plot(xs, sub["recall"], color=base_color, alpha=_REC_ALPHA,
                    marker="s", markersize=4, linestyle="--", label=f"{period} rec", linewidth=1.5)

        ax.axhline(tgt_p, color="#555", linewidth=0.8, linestyle=":", label=f"target prec ≥{tgt_p}")
        ax.axhline(tgt_r, color="#999", linewidth=0.8, linestyle=":", label=f"target rec ≥{tgt_r}")
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _fig_to_b64(fig)


# ── Chart 2: monthly F1 heatmap (parameterised by tier) ──────────────────────

_TIER_LABELS = {"tier_1": "Tier 1 — Plant Signal", "tier_2": "Tier 2 — Multi-Site Risk", "tier_3": "Tier 3 — Exceedance Risk"}


def _chart_f1_heatmap(report_df: pd.DataFrame, tier: str = "tier_3") -> str:
    tier_df: pd.DataFrame = report_df.loc[
        (report_df["tier"] == tier) & (report_df["month"] != "ALL")
    ].copy()
    months = _months_sorted(tier_df)
    periods = ["day", "night"]

    title = f"{_TIER_LABELS.get(tier, tier)} — F1 Score Heatmap"
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(months) * 0.9 + 2), 3.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for ax, period in zip(axes, periods):
        matrix = np.full((len(HORIZON_ORDER), len(months)), np.nan)
        for r_idx, horizon in enumerate(HORIZON_ORDER):
            for c_idx, month in enumerate(months):
                row: pd.DataFrame = tier_df.loc[
                    (tier_df["horizon"] == horizon) &
                    (tier_df["period"] == period) &
                    (tier_df["month"] == month)
                ]
                if not row.empty:
                    matrix[r_idx, c_idx] = float(row.iloc[0]["f1"])

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(months)))
        ax.set_xticklabels(months, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(HORIZON_ORDER)))
        ax.set_yticklabels([_HORIZON_LABELS.get(h, h) for h in HORIZON_ORDER], fontsize=8)
        ax.set_title(f"{period.capitalize()}", fontsize=10)
        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                v = matrix[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black" if v > 0.4 else "white")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Chart 3: monthly event counts & fire rate ─────────────────────────────────

def _chart_event_counts(report_df: pd.DataFrame) -> str:
    t3_filtered: pd.DataFrame = report_df.loc[
        (report_df["tier"] == "tier_3") & (report_df["month"] != "ALL")
    ].copy()

    # Use 'all' period rows if available, else aggregate day+night
    has_all = (t3_filtered["period"] == "all").any()
    if has_all:
        base: pd.DataFrame = t3_filtered.loc[t3_filtered["period"] == "all"]
    else:
        base = (
            t3_filtered
            .groupby(["month", "horizon"], as_index=False)[["n_events", "n_fires", "n_rows"]]  # type: ignore[call-overload]
            .sum()
        )

    monthly: pd.DataFrame = pd.DataFrame(
        base
        .groupby("month", as_index=False)[["n_events", "n_fires"]]  # type: ignore[call-overload]
        .sum()
    ).sort_values("month")
    monthly["far"] = (
        (monthly["n_fires"] - monthly["n_events"].clip(upper=monthly["n_fires"]))
        / monthly["n_fires"].replace(0, np.nan)
    )

    months: list = monthly["month"].tolist()
    x = np.arange(len(months))
    w = 0.35

    fig, ax1 = plt.subplots(figsize=(max(7, len(months) * 0.8), 4))
    ax1.bar(x - w / 2, monthly["n_events"], w, label="True events",
            color=_TIER_COLORS["tier_3"], alpha=0.8)
    ax1.bar(x + w / 2, monthly["n_fires"],  w, label="Alerts fired", color="#888", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Count")
    ax1.set_title("Tier 3 — Monthly Event Counts & Alerts Fired",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, monthly["far"], color="#c0392b", marker="D", markersize=5,
             linewidth=1.5, label="False alarm rate")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax2.set_ylabel("False alarm rate", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Chart 4: lead-time distribution ──────────────────────────────────────────

def _chart_lead_time(records_df: pd.DataFrame) -> str:
    t3: pd.DataFrame = records_df.loc[records_df["tier"] == "tier_3"].copy()
    t3["actual"] = t3["actual_max_h2s_nb"] >= 30.0
    hits: pd.DataFrame = t3.loc[t3["fired"] & t3["actual"]].dropna(subset=["lead_time_hours"])

    fig, ax = plt.subplots(figsize=(9, 4))
    data_day = [
        hits.loc[hits["horizon"] == h, "lead_time_hours"].values
        for h in HORIZON_ORDER
    ]
    data_night = [
        hits.loc[(hits["horizon"] == h) & (~hits["daytime_horizon"]), "lead_time_hours"].values
        for h in HORIZON_ORDER
    ]

    positions_day   = [i * 3 for i in range(len(HORIZON_ORDER))]
    positions_night = [i * 3 + 1 for i in range(len(HORIZON_ORDER))]

    bp_day = ax.boxplot(data_day, positions=positions_day, widths=0.7, patch_artist=True,
                        boxprops=dict(facecolor=_DAY_COLOR, alpha=0.7),
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                        flierprops=dict(marker=".", markersize=3))
    bp_night = ax.boxplot(data_night, positions=positions_night, widths=0.7, patch_artist=True,
                          boxprops=dict(facecolor=_NIGHT_COLOR, alpha=0.7),
                          medianprops=dict(color="black", linewidth=2),
                          whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
                          flierprops=dict(marker=".", markersize=3))

    mid = [(d + n) / 2 for d, n in zip(positions_day, positions_night)]
    ax.set_xticks(mid)
    ax.set_xticklabels([_HORIZON_LABELS.get(h, h) for h in HORIZON_ORDER], fontsize=9)
    ax.set_ylabel("Lead time (hours)")
    ax.set_title("Tier 3 True-Positive Lead Time by Horizon (day vs night)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.legend([bp_day["boxes"][0], bp_night["boxes"][0]], ["Day", "Night"], fontsize=9)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── HTML assembly ─────────────────────────────────────────────────────────────

_CSS = """
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6fa; color: #222; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 0.5rem; border-bottom: 2px solid #ddd; padding-bottom: 0.3rem; }
.meta { color: #666; font-size: 0.88rem; margin-bottom: 1.5rem; }
.chart { background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.1);
         padding: 1rem; margin-bottom: 1.5rem; }
.chart img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
table { border-collapse: collapse; width: 100%; font-size: 0.82rem; background: #fff;
        border-radius: 6px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
th { background: #2c3e50; color: #fff; padding: 7px 10px; text-align: left; white-space: nowrap; }
td { padding: 5px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }
tr:last-child td { border-bottom: none; }
tr:nth-child(even) td { background: #f9f9fb; }
.pass { color: #27ae60; font-weight: bold; }
.fail { color: #c0392b; font-weight: bold; }
.tier-1 { color: #5aab61; }
.tier-2 { color: #e07b39; }
.tier-3 { color: #c0392b; }
"""


def _overall_table_html(report_df: pd.DataFrame) -> str:
    t3: pd.DataFrame = report_df.loc[
        (report_df["tier"] == "tier_3") & (report_df["month"] == "ALL")
    ]
    rows_html = ""
    for _, r in t3.iterrows():
        h = str(r["horizon"])
        tgt_p, tgt_r = TIER3_TARGETS.get(h, (0.0, 0.0))
        ok_class = "pass" if _f(r["precision"]) >= tgt_p and _f(r["recall"]) >= tgt_r else "fail"
        ok_sym   = "✓" if ok_class == "pass" else "✗"
        lt       = _lt(r["mean_lead_time_h"])
        rows_html += (
            f"<tr><td>{_HORIZON_LABELS.get(h, h)}</td>"
            f"<td class='{ok_class}'>{ok_sym}</td>"
            f"<td>{_f(r['precision']):.3f} <span style='color:#aaa'>(≥{tgt_p})</span></td>"
            f"<td>{_f(r['recall']):.3f} <span style='color:#aaa'>(≥{tgt_r})</span></td>"
            f"<td>{_f(r['f1']):.3f}</td>"
            f"<td>{_i(r['n_events'])}</td>"
            f"<td>{lt}</td></tr>\n"
        )
    return (
        "<table><thead><tr>"
        "<th>Horizon</th><th>Target</th><th>Precision</th>"
        "<th>Recall</th><th>F1</th><th>Events</th><th>Avg Lead</th>"
        "</tr></thead><tbody>" + rows_html + "</tbody></table>"
    )


def _monthly_table_html(report_df: pd.DataFrame) -> str:
    df: pd.DataFrame = report_df.loc[report_df["month"] != "ALL"].sort_values(
        ["tier", "month", "horizon", "period"]
    )
    rows_html = ""
    for _, r in df.iterrows():
        h = str(r["horizon"])
        tgt_p, tgt_r = TIER3_TARGETS.get(h, (0.0, 0.0))
        ok_class = "pass" if _f(r["precision"]) >= tgt_p and _f(r["recall"]) >= tgt_r else "fail"
        ok_sym   = "✓" if ok_class == "pass" else "✗"
        lt       = _lt(r["mean_lead_time_h"])
        tier_cls = str(r["tier"]).replace("_", "-")
        rows_html += (
            f"<tr><td class='{tier_cls}'>{r['tier']}</td>"
            f"<td>{r['month']}</td>"
            f"<td>{_HORIZON_LABELS.get(h, h)}</td>"
            f"<td>{r['period']}</td>"
            f"<td class='{ok_class}'>{ok_sym} {_f(r['precision']):.3f}</td>"
            f"<td class='{ok_class}'>{_f(r['recall']):.3f}</td>"
            f"<td>{_f(r['f1']):.3f}</td>"
            f"<td>{_i(r['n_events'])}</td>"
            f"<td>{_i(r['n_fires'])}</td>"
            f"<td>{lt}</td></tr>\n"
        )
    return (
        "<table><thead><tr>"
        "<th>Tier</th><th>Month</th><th>Horizon</th><th>Period</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th>"
        "<th>Events</th><th>Fires</th><th>Avg Lead</th>"
        "</tr></thead><tbody>" + rows_html + "</tbody></table>"
    )


_TIER_SHORT = {"tier_1": "T1 Plant-Signal", "tier_2": "T2 Multi-Site", "tier_3": "T3 Exceedance"}
_H_SHORT = {"nowcast": "0–3h", "near": "3–6h", "mid": "6–12h", "day_ahead": "12–24h"}


# ── Per-month F1 grid (3 tiers × 4 horizons, day | night) ─────────────────────

def _chart_month_grid(month_df: pd.DataFrame, month: str) -> str:
    tiers = ["tier_1", "tier_2", "tier_3"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    fig.suptitle(f"F1 by Tier & Horizon — {month}", fontsize=11, fontweight="bold")

    for ax, period in zip(axes, ["day", "night"]):
        matrix = np.full((len(tiers), len(HORIZON_ORDER)), np.nan)
        for ri, tier in enumerate(tiers):
            for ci, horizon in enumerate(HORIZON_ORDER):
                row: pd.DataFrame = month_df.loc[
                    (month_df["tier"] == tier) &
                    (month_df["horizon"] == horizon) &
                    (month_df["period"] == period)
                ]
                if not row.empty:
                    matrix[ri, ci] = float(row.iloc[0]["f1"])

        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(HORIZON_ORDER)))
        ax.set_xticklabels([_H_SHORT.get(h, h) for h in HORIZON_ORDER], fontsize=8)
        ax.set_yticks(range(len(tiers)))
        ax.set_yticklabels([_TIER_SHORT.get(t, t) for t in tiers], fontsize=8)
        ax.set_title(period.capitalize(), fontsize=9)
        for ri2 in range(matrix.shape[0]):
            for ci2 in range(matrix.shape[1]):
                v = matrix[ri2, ci2]
                if not np.isnan(v):
                    ax.text(ci2, ri2, f"{v:.2f}", ha="center", va="center",
                            fontsize=7.5, color="black" if v > 0.45 else "white")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── Monthly page ───────────────────────────────────────────────────────────────

def _month_metrics_table_html(month_df: pd.DataFrame) -> str:
    rows_html = ""
    for tier in ("tier_1", "tier_2", "tier_3"):
        tier_df: pd.DataFrame = month_df.loc[month_df["tier"] == tier]
        for _, r in tier_df.sort_values(["horizon", "period"]).iterrows():
            h = str(r["horizon"])
            tgt_p, tgt_r = TIER3_TARGETS.get(h, (0.0, 0.0)) if tier == "tier_3" else (0.0, 0.0)
            ok_class = (
                "pass" if tier == "tier_3" and _f(r["precision"]) >= tgt_p and _f(r["recall"]) >= tgt_r
                else ("pass" if tier != "tier_3" else "fail")
            )
            tier_cls = tier.replace("_", "-")
            rows_html += (
                f"<tr><td class='{tier_cls}'>{_TIER_SHORT.get(tier, tier)}</td>"
                f"<td>{_HORIZON_LABELS.get(h, h)}</td>"
                f"<td>{r['period']}</td>"
                f"<td class='{ok_class}'>{_f(r['precision']):.3f}</td>"
                f"<td class='{ok_class}'>{_f(r['recall']):.3f}</td>"
                f"<td>{_f(r['f1']):.3f}</td>"
                f"<td>{_i(r['n_events'])}</td>"
                f"<td>{_i(r['n_fires'])}</td>"
                f"<td>{_lt(r['mean_lead_time_h'])}</td></tr>\n"
            )
    return (
        "<table><thead><tr>"
        "<th>Tier</th><th>Horizon</th><th>Period</th>"
        "<th>Precision</th><th>Recall</th><th>F1</th>"
        "<th>Events</th><th>Fires</th><th>Avg Lead</th>"
        "</tr></thead><tbody>" + rows_html + "</tbody></table>"
    )


def _month_page_html(
    month: str,
    month_report: pd.DataFrame,
    all_months: list[str],
) -> str:
    idx = all_months.index(month)
    prev_link = f'<a href="{all_months[idx - 1]}.html">← {all_months[idx - 1]}</a>' if idx > 0 else ""
    next_link = f'<a href="{all_months[idx + 1]}.html">{all_months[idx + 1]} →</a>' if idx < len(all_months) - 1 else ""
    nav = f'<div class="nav">{prev_link} <a href="index.html">↑ Index</a> {next_link}</div>'

    img_grid = _chart_month_grid(month_report, month)
    metrics_tbl = _month_metrics_table_html(month_report)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tier Alert — {month}</title>
<style>{_CSS}
.nav {{ margin-bottom: 1.5rem; font-size: 0.9rem; display: flex; gap: 1.5rem; }}
.nav a {{ color: #2c3e50; text-decoration: none; font-weight: 600; }}
.nav a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="wrap">
  {nav}
  <h1>Tiered H2S Alert — {month}</h1>
  <p class="meta">Generated: {generated_at}</p>

  <h2>F1 Score Grid</h2>
  <div class="chart"><img src="data:image/png;base64,{img_grid}" alt="F1 grid"></div>

  <h2>Metrics Detail</h2>
  {metrics_tbl}

  {nav}
</div>
</body>
</html>
"""


# ── Index page ─────────────────────────────────────────────────────────────────

def _index_page_html(report_df: pd.DataFrame, recent_months: list[str]) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rows_html = ""
    for month in reversed(recent_months):  # newest first
        month_df: pd.DataFrame = report_df.loc[report_df["month"] == month]
        cells = ""
        for tier in ("tier_1", "tier_2", "tier_3"):
            tier_df: pd.DataFrame = month_df.loc[month_df["tier"] == tier]
            # overall F1 = mean across horizons for day+night combined
            f1_vals = tier_df.loc[tier_df["period"].isin(["day", "night"]), "f1"]
            f1_mean = _f(f1_vals.mean()) if not f1_vals.empty else float("nan")  # type: ignore[union-attr]
            color = "#27ae60" if f1_mean >= 0.7 else ("#e07b39" if f1_mean >= 0.5 else "#c0392b")
            cells += f'<td style="color:{color};font-weight:600">{f1_mean:.2f}</td>'

        t3_df: pd.DataFrame = month_df.loc[
            (month_df["tier"] == "tier_3") & (month_df["period"].isin(["day", "night"]))
        ]
        n_events = _i(t3_df["n_events"].sum()) if not t3_df.empty else 0  # type: ignore[union-attr]
        n_fires  = _i(t3_df["n_fires"].sum())  if not t3_df.empty else 0  # type: ignore[union-attr]

        rows_html += (
            f"<tr>"
            f'<td><a href="{month}.html"><strong>{month}</strong></a></td>'
            f"{cells}"
            f"<td>{n_events}</td><td>{n_fires}</td>"
            f'<td><a href="{month}.html">→</a></td>'
            f"</tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tiered Alert Backtest — Monthly Index</title>
<style>{_CSS}
td a {{ color: #2c3e50; text-decoration: none; font-weight: 600; }}
td a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Tiered H2S Alert — Monthly Backtest Index</h1>
  <p class="meta">
    {len(recent_months)} months &nbsp;·&nbsp; Generated: {generated_at}
  </p>
  <p class="meta" style="color:#888">
    F1 columns show mean F1 across horizons (day + night).
    Colors: <span style="color:#27ae60">■</span> ≥0.70 &nbsp;
            <span style="color:#e07b39">■</span> ≥0.50 &nbsp;
            <span style="color:#c0392b">■</span> &lt;0.50
  </p>
  <table>
    <thead><tr>
      <th>Month</th>
      <th>T1 F1</th><th>T2 F1</th><th>T3 F1</th>
      <th>T3 Events</th><th>T3 Fires</th><th></th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
</body>
</html>
"""


def generate_static_site(
    records_df: pd.DataFrame,
    report_df: pd.DataFrame,
    output_dir: Path,
    months_back: int = 12,
) -> Path:
    """Write one HTML page per month + index.html into output_dir.

    Returns the path to index.html.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_months: list[str] = sorted(
        report_df.loc[report_df["month"] != "ALL", "month"].unique().tolist()  # type: ignore[union-attr]
    )
    recent_months = all_months[-months_back:]

    for month in recent_months:
        month_report: pd.DataFrame = report_df.loc[report_df["month"] == month]
        html = _month_page_html(month, month_report, recent_months)
        (output_dir / f"{month}.html").write_text(html, encoding="utf-8")

    index_path = output_dir / "index.html"
    index_path.write_text(_index_page_html(report_df, recent_months), encoding="utf-8")

    print(f"Static site ({len(recent_months)} months) → {output_dir}/index.html")
    return index_path


def build_html_report(records_df: pd.DataFrame, report_df: pd.DataFrame) -> str:
    """Return a self-contained HTML string.

    Parameters
    ----------
    records_df : raw backtest records (one row per evaluated_at × tier × horizon)
    report_df  : aggregated metrics from generate_report()
    """
    rc = records_df.copy()
    rc["evaluated_at"] = pd.to_datetime(rc["evaluated_at"], utc=True)
    date_range = (
        f"{rc['evaluated_at'].min().strftime('%Y-%m-%d')} – "
        f"{rc['evaluated_at'].max().strftime('%Y-%m-%d')}"
    )
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_hours: int = _i(rc["evaluated_at"].nunique())

    img_pr   = _chart_precision_recall(report_df)
    img_f1_1 = _chart_f1_heatmap(report_df, "tier_1")
    img_f1_2 = _chart_f1_heatmap(report_df, "tier_2")
    img_f1_3 = _chart_f1_heatmap(report_df, "tier_3")
    img_cnt  = _chart_event_counts(report_df)
    img_lt   = _chart_lead_time(rc)

    overall_tbl = _overall_table_html(report_df)
    monthly_tbl = _monthly_table_html(report_df)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tier Alert Backtest Report</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>Tiered H2S Alert — Backtest Report</h1>
  <p class="meta">
    Period: <strong>{date_range}</strong> &nbsp;·&nbsp;
    {n_hours:,} evaluation hours &nbsp;·&nbsp;
    Generated: {generated_at}
  </p>

  <h2>Overall Tier 3 Performance (all months)</h2>
  {overall_tbl}

  <h2>Monthly Precision &amp; Recall (Tier 3)</h2>
  <div class="chart"><img src="data:image/png;base64,{img_pr}" alt="precision recall chart"></div>

  <h2>F1 Score Heatmap — Day vs Night (Tier 1)</h2>
  <div class="chart"><img src="data:image/png;base64,{img_f1_1}" alt="Tier 1 F1 heatmap"></div>

  <h2>F1 Score Heatmap — Day vs Night (Tier 2)</h2>
  <div class="chart"><img src="data:image/png;base64,{img_f1_2}" alt="Tier 2 F1 heatmap"></div>

  <h2>F1 Score Heatmap — Day vs Night (Tier 3)</h2>
  <div class="chart"><img src="data:image/png;base64,{img_f1_3}" alt="Tier 3 F1 heatmap"></div>

  <h2>Monthly Event Counts &amp; Fire Rate (Tier 3)</h2>
  <div class="chart"><img src="data:image/png;base64,{img_cnt}" alt="event counts chart"></div>

  <h2>True-Positive Lead Time Distribution (Tier 3)</h2>
  <div class="chart"><img src="data:image/png;base64,{img_lt}" alt="lead time chart"></div>

  <h2>Detailed Monthly Breakdown (All Tiers)</h2>
  {monthly_tbl}
</div>
</body>
</html>
"""
