"""Shared chart metadata helpers.

Every graphic the system produces must carry an applicable date so a reader
can tell when it was generated / what period it covers. ``stamp_generated``
adds a small UTC footer to a figure; call it once, right before ``savefig``.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _to_utc(when: datetime | str | None) -> datetime | None:
    """Coerce a datetime / ISO-ish string / run_tag / None to an aware UTC datetime.

    None → now (UTC). An unparseable string → None.
    """
    if when is None:
        return datetime.now(timezone.utc)
    if isinstance(when, str):
        import pandas as pd

        ts = pd.to_datetime(when, utc=True, errors="coerce")
        if ts is None or pd.isna(ts):
            return None
        return ts.to_pydatetime()
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def as_of_label(when: datetime | str | None = None, *, prefix: str = "as of") -> str:
    """Return a compact ``as of YYYY-MM-DD HH:MM UTC`` label for a chart title.

    Accepts a datetime, an ISO-ish string (e.g. a run_tag like
    ``2026-06-23T1400Z``), or None (→ now, UTC). Returns ``""`` when a string is
    given that can't be parsed, so callers can append it unconditionally.
    """
    ts = _to_utc(when)
    if ts is None:
        return ""
    return f"{prefix} {ts:%Y-%m-%d %H:%M UTC}"


def stamp_generated(fig, *, when: datetime | None = None, prefix: str = "Generated") -> None:
    """Stamp a small ``<prefix> YYYY-MM-DD HH:MM UTC`` footer on ``fig``.

    Placed at the bottom-right in muted grey so it never competes with the
    chart content. ``when`` defaults to now (UTC) — the time the graphic was
    produced — but pass an explicit timestamp (e.g. a run_ts or the last data
    point) when a more meaningful "as-of" date is available.
    """
    ts = when if when is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    fig.text(
        0.995, 0.005,
        f"{prefix} {ts.astimezone(timezone.utc):%Y-%m-%d %H:%M UTC}",
        ha="right", va="bottom", fontsize=6.5, color="#8a8a8a",
    )
