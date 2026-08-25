"""Table 7 — the features the deployed exceedance-probability models use.

The feature lists are not hard-coded here. They are read out of the deployed
models' own ``training_report.json`` on S3, so the table can never drift from
what is actually in production. Only the prose descriptions live in this file.

Two variants are deployed side by side:

* **Evidence** (33 features) — the set promoted from the 2026-06 ablation, after
  the SBIWTP terms and the 4 h wind rolling aggregates were dropped.
* **Lean** (19 features) — Evidence minus the engineered interactions, the
  remaining wind rolling aggregates, four low-importance weather channels and
  the hour-of-day cyclicals. This is ``PRIMARY_VARIANT``: it drives the reports
  and the alert cascade, with Evidence shown alongside.

The emitted table marks which variant each feature belongs to, its family, and
where it comes from — which matters, because a feature the model leans on is
only as good as the feed behind it (see the ``Flow (m^3/s)--Border`` note).
"""

from __future__ import annotations

import pandas as pd

import common as C

#: family, source, description. Prose only — membership comes from S3.
DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # --- weather, as observed / forecast at the station --------------------
    "temperature_2m": ("weather", "Open-Meteo", "Air temperature at 2 m (°C). The strongest single exogenous predictor, but see §4 — it acts as a marker for the season rather than a lever within one."),
    "relative_humidity_2m": ("weather", "Open-Meteo", "Relative humidity at 2 m (%)."),
    "dewpoint_2m": ("weather", "Open-Meteo", "Dew point at 2 m (°C). Evidence only."),
    "surface_pressure": ("weather", "Open-Meteo", "Surface pressure (hPa). Evidence only."),
    "cloud_cover": ("weather", "Open-Meteo", "Total cloud cover (%), a proxy for overnight radiative cooling. Evidence only."),
    "precipitation": ("weather", "Open-Meteo", "Hourly precipitation (mm). Evidence only."),
    # --- wind -------------------------------------------------------------
    "wind_speed_10m": ("wind", "Open-Meteo", "Wind speed at 10 m (m/s). Calm nights ventilate the valley poorly; this is the dominant dispersion term."),
    "wind_gusts_10m": ("wind", "Open-Meteo", "Wind gusts at 10 m (m/s). Evidence only."),
    "wind_direction_sin": ("wind", "derived", "sin of wind direction. Direction is split into sin/cos so the model sees it as a circle rather than a number that jumps at 360°."),
    "wind_direction_cos": ("wind", "derived", "cos of wind direction, paired with the above."),
    "wind_speed_10m_avg_2h": ("wind", "derived", "2 h rolling mean wind speed — recent ventilation history. Evidence only."),
    "wind_speed_10m_avg_3h": ("wind", "derived", "3 h rolling mean wind speed. Evidence only."),
    "wind_gusts_10m_max_2h": ("wind", "derived", "2 h rolling maximum gust — a mixing-event marker. Evidence only."),
    "wind_gusts_10m_max_3h": ("wind", "derived", "3 h rolling maximum gust. Evidence only."),
    # --- interactions and regime -----------------------------------------
    "wind_temp_interaction": ("regime", "derived", "wind_speed × temperature. Evidence only; dropped from Lean on the argument that a tree learns the product implicitly."),
    "humidity_temp_interaction": ("regime", "derived", "humidity × temperature. Evidence only, same argument."),
    "stable_atm": ("regime", "derived", "Binary stable-atmosphere flag (calm and at night) — the nocturnal-inversion proxy that traps the plume near the ground."),
    "wind_x_stable_atm": ("regime", "derived", "wind_speed × stable_atm. Evidence only."),
    "is_night": ("regime", "derived", "1 between sunset and sunrise. Consistently a top-3 classifier feature, because the hazard is almost entirely nocturnal (Figure 4)."),
    "source_regime": ("regime", "derived", "Coarse source regime: night flag crossed with the wind-direction quadrant, i.e. which part of the valley is upwind."),
    # --- time -------------------------------------------------------------
    "hour_sin": ("time", "derived", "sin of hour-of-day. Evidence only; Lean drops it as redundant with is_night."),
    "hour_cos": ("time", "derived", "cos of hour-of-day. Evidence only."),
    "month_sin": ("time", "derived", "sin of month — the seasonal cycle documented in §4 enters the model through this pair."),
    "month_cos": ("time", "derived", "cos of month, paired with the above."),
    # --- water ------------------------------------------------------------
    "tide_height": ("water", "NOAA tides", "Tide height (m) at the estuary mouth."),
    "tidal_state_encoded": ("water", "derived", "Ordinal encoding of ebb / flood / high / low."),
    "flow_lag_6h": ("water", "IBWC border gauge", "Tijuana River flow 6 h earlier (m³/s) — travel time from the border reach to the monitor. **Stuck at a constant since 2026-01; see §3.4.**"),
    "flow_rolling_24h": ("water", "IBWC border gauge", "24 h rolling mean river flow (m³/s). **Same feed, same problem.**"),
    # --- autoregressive ---------------------------------------------------
    "h2s_lag_1h": ("H2S history", "SD APCD monitor", "H2S one hour earlier (ppb). The single most informative feature in every model; it is also why forecast skill decays once the recursion runs past the observed record."),
    "h2s_lag_3h": ("H2S history", "SD APCD monitor", "H2S three hours earlier (ppb)."),
    "h2s_lag_6h": ("H2S history", "SD APCD monitor", "H2S six hours earlier (ppb)."),
    "h2s_rolling_6h": ("H2S history", "SD APCD monitor", "6 h rolling mean H2S (ppb) — the event-in-progress signal, and the top feature for the ≥10 and ≥30 classifiers."),
    "h2s_rolling_24h": ("H2S history", "SD APCD monitor", "24 h rolling mean H2S (ppb) — the multi-night episode signal."),
}

FAMILY_ORDER = ["H2S history", "wind", "regime", "weather", "time", "water"]


def main() -> None:
    report = C.load_json("NESTOR__BES_training_report.json")
    evidence = report["features"]["evidence"]
    lean = set(report["features"]["lean"])

    missing = [f for f in evidence if f not in DESCRIPTIONS]
    if missing:
        raise SystemExit(
            f"Deployed model uses features with no description here: {missing}. "
            "Add them to DESCRIPTIONS rather than letting the table go stale."
        )

    rows = []
    for feat in evidence:
        family, source, desc = DESCRIPTIONS[feat]
        rows.append(
            {
                "family": family,
                "feature": feat,
                "in_lean": "yes" if feat in lean else "—",
                "source": source,
                "description": desc,
            }
        )
    table = pd.DataFrame(rows)
    table["_order"] = table["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    table = table.sort_values(["_order", "feature"]).drop(columns="_order")
    C.save_table(table, "tbl07_model_features", index=False)

    # Markdown, ready to paste into REPORT.md.
    lines = ["| Family | Feature | In Lean | Source | Description |",
             "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['family']} | `{r['feature']}` | {r['in_lean']} | {r['source']} | {r['description']} |"
        )
    md = "\n".join(lines)
    (C.TABLES / "tbl07_model_features.md").write_text(md + "\n")
    print(f"  wrote tables/tbl07_model_features.md")
    print(f"\nEvidence: {len(evidence)} features; Lean: {len(lean)} features")
    print(table.groupby("family").size().reindex(FAMILY_ORDER).to_string())


if __name__ == "__main__":
    main()
