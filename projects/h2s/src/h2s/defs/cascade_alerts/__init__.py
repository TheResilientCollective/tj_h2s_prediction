"""Forecast cascade pre-alert system — Tiers 1–3, product-probability-driven.

Replaces the legacy met-regime gate + sigmoid-score tier system. Triggers now
read directly off the per-station classifier probabilities (p5 / p10 / p30)
emitted by the nowcast/nearcast/forecast products (Phase 3). The classifiers
carry their own ~0.9-AUC skill, so the cascade trusts them directly instead of
re-deriving risk from a hand-weighted met-feature score.
"""
