"""Plain-language narrative of the forecast performance report.

Two outputs from a :func:`performance_report.build_report` result:

- :func:`build_llm_payload` — a JSON-serialisable dict (rubric framing +
  metrics + worst examples + correlation curves) to POST to the ResilientLLM
  webhook (``ResilientLLMResource.execute_with_data``).
- :func:`build_narrative_text` — a deterministic, template-based narrative that
  reads the rubric the way the user framed it. It needs no external service, so
  it is both the offline fallback and a sane default when the webhook is unset.

The rubric framing (shared by both) encodes the user's intent: under-predicting
a hazard is far worse than over-predicting it, getting the ≥10 ppb resident-
smell level right matters, and an over-prediction that the actual reaches within
a couple of hours is an acceptable early warning rather than a false alarm.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from h2s.forecasting.performance_report import TOLERANCE_HOURS, VERDICT_COST

# The interpretation guide handed to the LLM (and mirrored in the local text).
RUBRIC_GUIDANCE = [
    "Under-predicting a real ORANGE (>=30 ppb) event as GREEN is the worst "
    "outcome (a dangerous miss) — residents get no warning of a hazard.",
    "Under-predicting ORANGE as some YELLOW is acceptable: a hazard was still "
    "flagged, just under-stated.",
    "Predicting ORANGE when it is ORANGE is the goal.",
    "Missing the >=10 ppb YELLOW-HIGH (resident-smell) level as GREEN matters: "
    "residents smell it, so we want to get this level right.",
    "Predicting GREEN when the measurement is a low YELLOW (5-10 ppb) is often "
    "acceptable; confirm the low-end cut with health officials.",
    f"Over-predicting GREEN as YELLOW/ORANGE is acceptable when a real "
    f"YELLOW/ORANGE level actually occurs within {TOLERANCE_HOURS} hours "
    f"(an early warning) — only a persistent over-prediction is a false alarm.",
]


def build_llm_payload(report: dict, variant: Optional[str] = None) -> dict:
    """Structured payload for the ResilientLLM webhook."""
    summary = report.get("summary", {})
    by_lead = report.get("by_lead_hour")
    by_hod = report.get("by_hour_of_day")
    confusion = report.get("confusion")

    def _records(df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.where(pd.notna(df), None).to_dict(orient="records")
        return []

    return {
        "report_type": "h2s_forecast_performance",
        "variant": variant or summary.get("variant"),
        "rubric_guidance": RUBRIC_GUIDANCE,
        "verdict_cost_weights": VERDICT_COST,
        "tolerance_hours": report.get("tolerance_hours", TOLERANCE_HOURS),
        "smell_threshold_ppb": report.get("smell_threshold_ppb"),
        "summary": summary,
        "confusion_matrix": (
            confusion.to_dict() if isinstance(confusion, pd.DataFrame) else {}
        ),
        "correlation_by_lead_hour": _records(by_lead),
        "correlation_by_hour_of_day": _records(by_hod),
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _lead_skill_phrase(by_lead) -> str:
    if not isinstance(by_lead, pd.DataFrame) or by_lead.empty:
        return "Not enough matched pairs yet to chart skill against lead hour."
    d = by_lead.dropna(subset=["spearman"]).sort_values("lead_hour")
    if d.empty:
        return "Skill-vs-lead-hour correlation is not yet estimable."
    first, last = d.iloc[0], d.iloc[-1]
    trend = (
        " — skill decays toward longer leads as the recursion drifts from observed "
        "data, as expected."
        if last.spearman < first.spearman - 0.05
        else "."
    )
    return (
        f"Rank correlation with measurements is ~{first.spearman:.2f} at lead "
        f"{int(first.lead_hour)} h and ~{last.spearman:.2f} by lead "
        f"{int(last.lead_hour)} h{trend}"
    )


def build_narrative_text(report: dict, variant: Optional[str] = None) -> str:
    """Deterministic plain-language performance narrative (offline fallback)."""
    summary = report.get("summary", {})
    n = summary.get("n", 0)
    v = variant or summary.get("variant") or "evidence"
    if not n:
        return (
            f"No measured H2S has overlapped the {v} forecast horizon yet, so "
            "there is nothing to score. This fills in over the days after "
            "products start running."
        )

    vc = summary.get("verdict_counts", {})
    lines = [
        f"*{v.capitalize()} forecast performance* over {n} matched forecast-hours:",
        f"• Acceptable (match / early-warning / yellow-band) "
        f"{_pct(summary.get('tolerant_accuracy'))} of the time; "
        f"mean rubric cost {summary.get('mean_cost')}.",
    ]

    dm = vc.get("dangerous_miss", 0)
    sm = vc.get("smell_miss", 0)
    if dm:
        lines.append(
            f"• ⚠️ {dm} *dangerous miss(es)* — a real orange (>=30 ppb) event was "
            f"forecast as green ({_pct(summary.get('dangerous_miss_rate'))}). This "
            "is the outcome we most need to drive down."
        )
    else:
        lines.append("• No dangerous misses: no orange event was forecast as green.")
    if sm:
        lines.append(
            f"• {sm} *smell miss(es)* — the >=10 ppb resident-smell level read as "
            f"green ({_pct(summary.get('smell_miss_rate'))}); a level we want to "
            "get right."
        )

    fa = vc.get("false_alarm", 0)
    ew = vc.get("early_warning_ok", 0)
    lines.append(
        f"• Over-predictions split into {ew} acceptable early warnings (the actual "
        f"reached the level within {report.get('tolerance_hours', TOLERANCE_HOURS)} h) "
        f"and {fa} genuine false alarms."
    )
    soft = vc.get("soft_miss", 0)
    if soft:
        lines.append(
            f"• {soft} soft miss(es) — an orange event under-stated as yellow; a "
            "hazard was still flagged."
        )

    lines.append(f"• {_lead_skill_phrase(report.get('by_lead_hour'))}")

    examples = summary.get("worst_examples", [])
    if examples:
        lines.append("• Worst cases:")
        for ex in examples[:4]:
            lines.append(
                f"    – {ex['station']} {ex['time'][:16]} (lead {ex['lead_hour']}h): "
                f"predicted {ex['predicted_ppb']} ppb vs measured {ex['actual_ppb']} ppb "
                f"[{ex['verdict']}]"
            )
    return "\n".join(lines)
