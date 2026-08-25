"""Figure 7 — how well the deployed exceedance-probability models actually do.

Two numbers get quoted for these models and they are not the same number.

* **Held-out training metrics** (``training_report.json``, written by
  ``station_model_training_job``) score the classifier on a held-out slice of
  the training frame, where the H2S lag and rolling features are filled with
  *observed* H2S. That is a nowcast setting: the model is told what the monitor
  read an hour ago. AUC 0.97–0.98 is real, and it is the right number for
  "is there an event in progress".

* **Operational skill** (``forecast_skill_report.json``, rebuilt by
  ``station_forecast_validation_rebuild_job``) scores the deployed products
  against what was later measured. Past lead 6 the recursion is feeding on its
  own predictions, and skill collapses — at 7–24 h ahead the ≥30 ppb
  probability call currently catches nothing at all.

Reporting only the first number would overstate the system by a wide margin, so
this figure puts them side by side. Panels:

  (a) held-out classifier metrics, all three stations, primary (Lean) variant;
  (b) the same models' operational probability-call recall against lead hour;
  (c) top feature importances, which explain (b): the models lean on the H2S
      history, and the H2S history is exactly what a long-lead forecast does
      not have;
  (d) a closer look at the ≥30 ppb misses, because a recall of exactly zero
      usually means a broken cutoff rather than a broken model. Here it does
      not: on the hours that actually reached ≥30 ppb the emitted P(>30) had a
      median of 0.006, and lowering the cutoff to 0.05 lifts recall only to
      0.04 while pushing the false-alarm rate to 4.6%. The probability head is
      not mis-thresholded, it is uninformative at this horizon.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

STATION_KEYS = {
    "NESTOR__BES": C.NESTOR,
    "IB_CIVIC_CTR": C.IB,
    "SAN_YSIDRO": C.SAN_YSIDRO,
}
TASKS = ["clf_5ppb", "clf_10ppb", "clf_30ppb"]
VARIANTS = ["evidence", "lean"]


def holdout_table() -> pd.DataFrame:
    rows = []
    for key, name in STATION_KEYS.items():
        rep = C.load_json(f"{key}_training_report.json")
        meta = C.load_json(f"{key}_deployment_metadata.json")
        for variant in VARIANTS:
            for task in TASKS:
                m = rep["tasks"][variant][task]
                rows.append({
                    "station": name,
                    "variant": variant,
                    "task": task,
                    "AUC": m["AUC"],
                    "Brier": m["Brier"],
                    "F1": m["F1"],
                    "precision": m["Precision"],
                    "recall": m["Recall"],
                    "n_train": rep["n_train"],
                    "n_test": rep["n_test"],
                    "model_version": meta.get("model_version", ""),
                    "trained_at": rep["generated_at"][:10],
                })
            reg = rep["tasks"][variant]["regression"]
            rows.append({
                "station": name, "variant": variant, "task": "regression",
                "AUC": np.nan, "Brier": np.nan, "F1": np.nan,
                "precision": reg["precision_30"], "recall": reg["recall_30"],
                "n_train": rep["n_train"], "n_test": rep["n_test"],
                "model_version": meta.get("model_version", ""),
                "trained_at": rep["generated_at"][:10],
            })
    return pd.DataFrame(rows)


def positives_table() -> pd.DataFrame:
    """How many exceedance positives each station's held-out slice contained.

    This is the number behind the ``CLF_30PPB_STATIONS`` gate: P(>30) is only
    emitted for NESTOR-BES because the other two stations have too few ≥30
    positives to hold a fixed operating point.
    """
    rows = []
    for key, name in STATION_KEYS.items():
        rep = C.load_json(f"{key}_training_report.json")
        reg = rep["tasks"]["lean"]["regression"]
        rows.append({
            "station": name,
            "n_records": rep["n_records"],
            "n_test": rep["n_test"],
            "test_positives_5ppb": reg["n_positives_5"],
            "test_positives_10ppb": reg["n_positives_10"],
            "test_positives_30ppb": reg["n_positives_30"],
            "test_positives_100ppb": reg["n_positives_100"],
        })
    return pd.DataFrame(rows)


def skill_frame() -> pd.DataFrame:
    rep = C.load_json("forecast_skill_report.json")
    rows = []
    for h in rep["headline"]:
        for thr in (5, 10, 30):
            for lead, val in h[f"prob_recall_{thr}_by_lead"].items():
                rows.append({"product": h["product"], "variant": h["variant"],
                             "threshold": thr, "lead_hour": int(lead),
                             "prob_recall": val, "n": h["n"]})
    meta = {k: rep[k] for k in ("generated_at", "n_validation_rows",
                                "time_min", "time_max")}
    df = pd.DataFrame(rows)
    df.attrs["meta"] = meta
    return df


def main() -> None:
    C.style()
    hold = holdout_table()
    C.save_table(hold.round(4), "tbl07_holdout_metrics", index=False)
    pos = positives_table()
    C.save_table(pos, "tbl07_test_positives", index=False)
    skill = skill_frame()
    C.save_table(skill.round(4), "tbl07_operational_skill", index=False)

    val = C.load_validation()

    fig = plt.figure(figsize=(12.6, 8.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1.15],
                          height_ratios=[1, 0.92], wspace=0.32, hspace=0.46)

    # --- (a) held-out metrics ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lean = hold[(hold["variant"] == "lean") & hold["task"].isin(TASKS)]
    labels = ["> 5 ppb", "> 10 ppb", "> 30 ppb"]
    x = np.arange(len(TASKS))
    for i, (name, colour) in enumerate([(C.NESTOR, "#1f77b4"),
                                        (C.SAN_YSIDRO, "#d62728"),
                                        (C.IB, "#2ca02c")]):
        sub = lean[lean["station"] == name].set_index("task").loc[TASKS]
        ax.bar(x + (i - 1) * 0.27, sub["recall"], width=0.25, color=colour,
               label=name)
        for xi, (rec, prec) in enumerate(zip(sub["recall"], sub["precision"])):
            ax.plot([x[xi] + (i - 1) * 0.27], [prec], "_", color="k", ms=9, mew=1.6)
    ax.plot([], [], "_", color="k", ms=9, mew=1.6, label="precision")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("held-out recall (bar) / precision (tick)")
    ax.set_title("(a) Held-out classifier skill\n(Lean variant, observed H₂S lags present)")
    ax.legend(fontsize=7, ncol=2, loc="upper center")

    # --- (b) operational recall by lead hour --------------------------------
    ax = fig.add_subplot(gs[0, 1])
    prim = skill[skill["variant"] == "lean"]
    for thr, colour in ((5, C.TIER_COLORS["yellow"]),
                        (10, C.TIER_COLORS["yellow-high"]),
                        (30, C.TIER_COLORS["orange"])):
        sub = prim[prim["threshold"] == thr].sort_values("lead_hour")
        ax.plot(sub["lead_hour"], sub["prob_recall"], "-o", ms=3,
                color=colour, label=f"P(>{thr}) call")
    for (x0, x1, lbl), ytxt, shade in (((0.5, 3.5, "nowcast"), 0.97, 0.09),
                                       ((3.5, 6.5, "nearcast"), 0.90, 0.04),
                                       ((6.5, 24.5, "forecast"), 0.83, 0.0)):
        ax.axvspan(x0, x1, color="#000", alpha=shade, lw=0)
        ax.text((x0 + x1) / 2, ytxt, lbl, ha="center", va="top",
                fontsize=7.5, color="#666")
    ax.set_xlabel("forecast lead (hours ahead)")
    ax.set_ylabel("recall of the probability call")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.5, 24.5)
    meta = skill.attrs["meta"]
    ax.set_title("(b) Operational skill of the same models\n"
                 f"({meta['time_min'][:10]} – {meta['time_max'][:10]}, "
                 f"n={meta['n_validation_rows']:,} rows)")
    ax.legend(fontsize=7.5, loc="upper right")

    # --- (c) feature importance --------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    rep = C.load_json("NESTOR__BES_training_report.json")
    imp5 = pd.Series(rep["tasks"]["lean"]["clf_5ppb"]["feature_importance"])
    imp30 = pd.Series(rep["tasks"]["lean"]["clf_30ppb"]["feature_importance"])
    feats = list(dict.fromkeys(list(imp5.index[:8]) + list(imp30.index[:8])))[:11]
    y = np.arange(len(feats))
    ax.barh(y - 0.2, [imp5.get(f, 0) for f in feats], height=0.38,
            color=C.TIER_COLORS["yellow"], label="clf_5ppb")
    ax.barh(y + 0.2, [imp30.get(f, 0) for f in feats], height=0.38,
            color=C.TIER_COLORS["orange"], label="clf_30ppb")
    ax.set_yticks(y)
    ax.set_yticklabels([f"`{f}`".strip("`") for f in feats], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("gain importance")
    ax.set_title("(c) What the NESTOR-BES models lean on\n"
                 "— mostly recent H₂S, which long leads lack")
    ax.legend(fontsize=7.5, loc="lower right")

    # --- (d) why the ≥30 ppb call never fires -------------------------------
    ax = fig.add_subplot(gs[1, :2])
    p30 = val[val["p30"].notna()]
    pos = p30[p30["actual_ge30"]]
    neg = p30[~p30["actual_ge30"]]
    grid = np.logspace(-4, 0, 60)
    ax.plot(grid, [(pos["p30"] > g).mean() for g in grid], lw=2,
            color=C.TIER_COLORS["orange"],
            label=f"hours that DID reach ≥30 ppb (n={len(pos):,})")
    ax.plot(grid, [(neg["p30"] > g).mean() for g in grid], lw=2, ls="--",
            color="#9e9e9e", label=f"hours that did not (n={len(neg):,})")
    ax.axvline(0.5, color="#333", lw=1.2)
    ax.text(0.52, 0.45, "deployed\ncutoff", fontsize=7.5, color="#333")
    ax.set_xscale("log")
    ax.set_xlabel("P(>30 ppb) emitted by the model")
    ax.set_ylabel("fraction of hours above that probability")
    ax.set_title("(d) The ≥30 ppb probability barely separates the events from "
                 "everything else — no cutoff rescues it")
    ax.legend(loc="lower left", fontsize=7.5)

    # --- (e) magnitude head on the same hours --------------------------------
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(pos["actual_h2s"], pos["h2s_pred"].clip(lower=0.05), s=10,
               color=C.TIER_COLORS["orange"], alpha=0.55, edgecolor="none")
    lim = [0.05, max(pos["actual_h2s"].max(), 300)]
    ax.plot(lim, lim, "-", color="#999", lw=1, label="perfect")
    ax.axhline(C.T_ORANGE, color="#333", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured H₂S (ppb)")
    ax.set_ylabel("predicted H₂S (ppb)")
    ax.set_title("(e) …and the magnitude head\nmisses them too")
    ax.legend(fontsize=7.5, loc="upper left")

    C.save(fig, "fig07_model_performance")
    plt.close(fig)

    print(hold[hold["task"].isin(TASKS)]
          .pivot_table(index=["station", "task"], columns="variant",
                       values=["AUC", "recall", "precision"])
          .round(3).to_string())
    print()
    print(pos.to_string(index=False))
    print()
    print()
    print(f"validation rows with a P(>30) emitted: {len(p30):,}; "
          f"of those, {int(pos.shape[0]):,} hours actually reached 30 ppb")
    print("P(>30) on those hours: median "
          f"{pos['p30'].median():.4f}, 95th pct {pos['p30'].quantile(0.95):.4f}, "
          f"max {pos['p30'].max():.4f}")
    cut = pd.DataFrame([
        {"cutoff": c,
         "recall_30": (pos["p30"] > c).mean(),
         "false_alarm_rate": (neg["p30"] > c).mean()}
        for c in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02)
    ])
    C.save_table(cut.round(4), "tbl07_p30_cutoff_sweep", index=False)
    print(cut.round(4).to_string(index=False))
    print()
    for product in ("nowcast", "nearcast", "forecast"):
        sub = skill[(skill["product"] == product) & (skill["variant"] == "lean")]
        print(f"{product:9s} mean prob-recall: "
              + "  ".join(f">{t}: {sub[sub.threshold == t]['prob_recall'].mean():.3f}"
                          for t in (5, 10, 30)))


if __name__ == "__main__":
    main()
