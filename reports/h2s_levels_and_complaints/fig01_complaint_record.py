"""Figure 1 — what the odour-complaint record actually is.

Three panels that establish the properties of the complaint data before any
inference is drawn from it:

  (a) monthly volume of Imperial Beach (ZIP 91932) odour complaints, against
      all other San Diego County complaints, showing how completely the record
      is dominated by this one ZIP;
  (b) the spatial collapse — the share of IB odour complaints filed at a single
      repeated coordinate, which is a records-management artifact and the
      reason no spatial analysis is attempted anywhere in this report;
  (c) hour-of-day of complaints against hour-of-day of measured exceedances,
      which is the confound every later panel has to control for: people file
      complaints when they are awake, so raw complaint counts mix odour
      intensity with reporting availability.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C


def main() -> None:
    C.style()

    all_c = C.load_complaints(odor_only=False, imperial_beach_only=False)
    ib = C.load_complaints()
    h2s = C.load_h2s(C.NESTOR)

    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.26)

    # --- (a) monthly volume -------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    month = all_c["time"].dt.to_period("M")
    ib_mask = (all_c["nature_of_complaint"].str.lower() == "odor") & all_c[
        "zip"
    ].str.startswith("91932", na=False)
    counts = pd.DataFrame(
        {
            "Imperial Beach odour (ZIP 91932)": all_c[ib_mask]
            .groupby(month[ib_mask])
            .size(),
            "All other county complaints": all_c[~ib_mask]
            .groupby(month[~ib_mask])
            .size(),
        }
    ).fillna(0)
    counts.index = counts.index.to_timestamp()
    ax.bar(
        counts.index,
        counts.iloc[:, 0],
        width=22,
        color=C.TIER_COLORS["orange"],
        label=counts.columns[0],
    )
    ax.bar(
        counts.index,
        counts.iloc[:, 1],
        width=22,
        bottom=counts.iloc[:, 0],
        color="#9e9e9e",
        label=counts.columns[1],
    )
    share = 100 * counts.iloc[:, 0].sum() / counts.sum().sum()
    ax.set_title(
        f"(a) San Diego APCD complaint volume — Imperial Beach odour is "
        f"{share:.0f}% of the whole county record"
    )
    ax.set_ylabel("complaints per month")
    ax.legend(loc="upper right")

    # --- (b) spatial collapse ----------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    pt = ib.groupby(["y_coordinate", "x_coordinate"]).size().sort_values(ascending=False)
    top_share = 100 * pt.iloc[0] / len(ib)
    ax.barh(
        ["single filing point", "all other coordinates"],
        [top_share, 100 - top_share],
        color=[C.TIER_COLORS["orange"], "#9e9e9e"],
        height=0.55,
    )
    for y, v in enumerate([top_share, 100 - top_share]):
        ax.text(v + 1.5, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of Imperial Beach odour complaints")
    ax.set_title(
        "(b) The county files nearly every call\n"
        f"at {C.COMPLAINT_POINT[0]:.4f}, {C.COMPLAINT_POINT[1]:.4f}"
    )
    ax.grid(axis="y", visible=False)

    # --- (c) hour of day ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    hours = np.arange(24)
    c_by_hour = ib["time"].dt.hour.value_counts().reindex(hours, fill_value=0)
    c_by_hour = 100 * c_by_hour / c_by_hour.sum()
    exc = h2s[h2s["H2S"] > C.T_YELLOW]["time"].dt.hour.value_counts().reindex(
        hours, fill_value=0
    )
    exc = 100 * exc / exc.sum()
    ax.plot(hours, c_by_hour, "-o", ms=3, color=C.TIER_COLORS["orange"], label="odour complaints")
    ax.plot(hours, exc, "-s", ms=3, color="#1f77b4", label="measured hours > 5 ppb")
    ax.set_xticks(range(0, 24, 4))
    ax.set_xlabel("hour of day (Pacific)")
    ax.set_ylabel("% of records")
    ax.set_title("(c) Complaints track exceedances,\nbut sag through the night")
    ax.legend(loc="upper center")

    C.save(fig, "fig01_complaint_record")
    plt.close(fig)

    summary = pd.DataFrame(
        [
            ("complaint records, all county", len(all_c)),
            ("with a real time of day", int(all_c["time"].notna().sum())),
            ("odour complaints, all county", int((all_c["nature_of_complaint"].str.lower() == "odor").sum())),
            ("odour complaints, ZIP 91932", len(ib)),
            ("filed at the single repeated point", int(pt.iloc[0])),
            ("distinct coordinates used", int(len(pt))),
            ("record starts", ib["time"].min().strftime("%Y-%m-%d")),
            ("record ends", ib["time"].max().strftime("%Y-%m-%d")),
        ],
        columns=["quantity", "value"],
    )
    C.save_table(summary, "tbl01_complaint_record", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
