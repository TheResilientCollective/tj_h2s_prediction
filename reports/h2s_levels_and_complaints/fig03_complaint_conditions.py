"""Figure 3 — the complainant's-eye view, and a cross-station robustness check.

Figure 2 asks "given a concentration, how likely is a complaint?". This asks the
inverse, which is the question a health officer actually has: **when somebody
calls, what is the monitor reading?**

The two are not redundant. The dose-response is steep, but low concentrations
are so much more common than high ones that most complaints are still filed at
concentrations the 4-tier scheme calls green. That is the number that decides
whether a 5 ppb alert threshold matches what residents experience.

Panel (c) repeats the dose-response at all three monitors. NESTOR-BES is used
throughout the report because it is the closest monitor to the point the county
files these calls at, but the result should not depend on that choice, and it
does not.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
from fig02_dose_response import binned


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return float(2 * 6371.0 * np.arcsin(np.sqrt(h)))


def main() -> None:
    C.style()

    panel = C.hourly_panel(C.NESTOR)
    # One row per complaint, carrying the concentration measured that hour.
    comp = C.load_complaints()
    lookup = panel["H2S"]
    comp = comp[comp["hour"].isin(lookup.index)].copy()
    comp["h2s"] = comp["hour"].map(lookup)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))

    # --- (a) distribution of concentration at complaint time ----------------
    ax = axes[0]
    edges = np.logspace(np.log10(0.1), np.log10(500), 34)
    ax.hist(comp["h2s"].clip(lower=0.1), bins=edges, color="#1f77b4", alpha=0.85)
    for t, col in ((C.T_YELLOW, C.TIER_COLORS["yellow"]),
                   (C.T_YELLOW_HIGH, C.TIER_COLORS["yellow-high"]),
                   (C.T_ORANGE, C.TIER_COLORS["orange"])):
        ax.axvline(t, color=col, lw=1.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("H₂S at NESTOR-BES when the call was filed (ppb)")
    ax.set_ylabel("complaints")
    med = comp["h2s"].median()
    ax.set_title(f"(a) Median reading at complaint time:\n{med:.1f} ppb")

    # --- (b) cumulative share of complaints ---------------------------------
    ax = axes[1]
    s = comp["h2s"].dropna().sort_values()
    cdf = 100 * np.arange(1, len(s) + 1) / len(s)
    ax.plot(s.clip(lower=0.1), cdf, color="#1f77b4", lw=2, label="complaints")
    hours = panel["H2S"].sort_values()
    ax.plot(hours.clip(lower=0.1), 100 * np.arange(1, len(hours) + 1) / len(hours),
            color="#9e9e9e", lw=1.4, ls="--", label="all measured hours")
    rows = []
    for t, col in ((C.T_YELLOW, C.TIER_COLORS["yellow"]),
                   (C.T_YELLOW_HIGH, C.TIER_COLORS["yellow-high"]),
                   (C.T_ORANGE, C.TIER_COLORS["orange"])):
        below = 100 * (s < t).mean()
        rows.append({"threshold_ppb": t, "pct_complaints_below": below,
                     "pct_hours_below": 100 * (panel["H2S"] < t).mean()})
        ax.axvline(t, color=col, lw=1.5, ls="--")
        ax.plot([t], [below], "o", color=col, ms=6, zorder=5)
        ax.annotate(f"{below:.0f}% of calls\nbelow {t:g} ppb", (t, below),
                    textcoords="offset points", xytext=(8, -22), fontsize=7.5,
                    color=col)
    ax.set_xscale("log")
    ax.set_xlabel("H₂S (ppb)")
    ax.set_ylabel("cumulative % at or below")
    ax.set_ylim(0, 102)
    ax.set_title("(b) Most calls are filed at concentrations\nthe 4-tier scheme calls green")
    ax.legend(loc="lower right")

    # --- (c) cross-station robustness ---------------------------------------
    ax = axes[2]
    locs = pd.read_csv(C.fetch("h2s_locations.csv"))
    dist = {}
    curves = []
    for station, colour in ((C.NESTOR, "#1f77b4"), (C.SAN_YSIDRO, "#d62728"),
                            (C.IB, "#2ca02c")):
        row = locs[locs["site_name"] == station].iloc[0]
        dist[station] = haversine_km(C.COMPLAINT_POINT, (row["lat"], row["lon"]))
        p = C.hourly_panel(station)
        b = binned(p)
        ax.plot(b["median_h2s"], 100 * b["p_any"], "-o", ms=3.5, color=colour,
                label=f"{station} ({dist[station]:.1f} km, n={len(p):,} h)")
        b.insert(0, "station", station)
        curves.append(b)
    for t, col in ((C.T_YELLOW, C.TIER_COLORS["yellow"]),
                   (C.T_ORANGE, C.TIER_COLORS["orange"])):
        ax.axvline(t, color=col, lw=1.2, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("H₂S at that monitor, same hour (ppb)")
    ax.set_ylabel("% of hours drawing ≥1 odour complaint")
    ax.set_title("(c) Same shape at every monitor\n(distance to the filing point in brackets)")
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    C.save(fig, "fig03_complaint_conditions")
    plt.close(fig)

    C.save_table(pd.DataFrame(rows), "tbl03_share_below_threshold", index=False)
    C.save_table(pd.concat(curves, ignore_index=True), "tbl03_by_station", index=False)

    tiers = comp["h2s"].map(C.h2s_category).value_counts()
    tiers = (100 * tiers / tiers.sum()).round(1)
    summary = pd.DataFrame(
        [
            ("complaints matched to a measured hour", len(comp)),
            ("median H2S at complaint time (ppb)", round(med, 2)),
            ("75th percentile (ppb)", round(comp["h2s"].quantile(0.75), 2)),
            ("95th percentile (ppb)", round(comp["h2s"].quantile(0.95), 2)),
            *[(f"% of complaints in tier '{k}'", v) for k, v in tiers.items()],
            *[(f"km from filing point to {k}", round(v, 2)) for k, v in dist.items()],
        ],
        columns=["quantity", "value"],
    )
    C.save_table(summary, "tbl03_complaint_conditions", index=False)
    print(summary.to_string(index=False))
    print()
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
