"""Tests for the candidate feature sets added for the Berry feature-trim ablation.

These pin the contracts the ablation script and downstream production decisions
depend on:
  - Counts (so an accidental edit to MODEL_FEATURES doesn't silently move them)
  - Subset relationship (MINIMAL ⊂ LEAN ⊂ EVIDENCE ⊂ MODEL_FEATURES)
  - Calibration-load-bearing features are in every candidate (never drop these)
  - No typos / no features outside the master list

Provenance for each candidate is in
experiments/2026-06-10_feature_trim_berry/RESULTS.md.
"""

from __future__ import annotations

from h2s.constants import (
    MODEL_FEATURES,
    MODEL_FEATURES_EVIDENCE,
    MODEL_FEATURES_LEAN,
    MODEL_FEATURES_LEGACY,
    MODEL_FEATURES_MINIMAL,
)


# Calibration's load-bearing core — these MUST be in every candidate.
# Touching this set requires an explicit re-read of calibration_status.md
# and a corresponding RESULTS.md update; it's not a casual edit.
_LOAD_BEARING = {
    "temperature_2m",     # finding #1
    "stable_atm",         # finding #2
    "h2s_lag_1h",         # finding #4
    "h2s_rolling_6h",     # consistently top-3 in training reports
}


class TestCounts:
    """The Phase 2 promotion (PR #28) set MODEL_FEATURES = MODEL_FEATURES_EVIDENCE.
    Pin the counts so a future edit can't silently move them.
    """

    def test_production_is_33(self):
        assert len(MODEL_FEATURES) == 33

    def test_evidence_is_33(self):
        # Equal to MODEL_FEATURES post-promotion; the EVIDENCE alias stays
        # for the ablation script and historical reference
        assert len(MODEL_FEATURES_EVIDENCE) == 33
        assert MODEL_FEATURES_EVIDENCE == MODEL_FEATURES

    def test_legacy_is_44(self):
        # The pre-promotion baseline, retained for backward compat with
        # deployed models whose preprocessing_info.json references the
        # 11 dropped features
        assert len(MODEL_FEATURES_LEGACY) == 44

    def test_lean_is_19(self):
        assert len(MODEL_FEATURES_LEAN) == 19

    def test_minimal_is_11(self):
        assert len(MODEL_FEATURES_MINIMAL) == 11


class TestSubsetRelations:
    """Each candidate must be a subset of MODEL_FEATURES_LEGACY (no typos, no new features)."""

    def test_production_subset_of_legacy(self):
        # The promotion drops 11 features from legacy — production must be
        # a strict subset of what was deployed before
        assert set(MODEL_FEATURES) < set(MODEL_FEATURES_LEGACY)
        assert len(set(MODEL_FEATURES_LEGACY) - set(MODEL_FEATURES)) == 11

    def test_lean_subset_of_production(self):
        # Lean is a strict refinement of the production set — anything
        # dropped from production stays dropped in Lean
        assert set(MODEL_FEATURES_LEAN) <= set(MODEL_FEATURES)

    def test_minimal_subset_of_legacy(self):
        # Minimal isn't required to be a subset of Lean (it's a re-curated
        # calibration-only set), but every feature must exist somewhere
        # in the legacy list
        assert set(MODEL_FEATURES_MINIMAL) <= set(MODEL_FEATURES_LEGACY)


class TestLoadBearingFeaturesPreserved:
    """Calibration's empirically-validated features must survive every trim."""

    def test_evidence_keeps_load_bearing(self):
        missing = _LOAD_BEARING - set(MODEL_FEATURES_EVIDENCE)
        assert not missing, f"Evidence is missing load-bearing features: {missing}"

    def test_lean_keeps_load_bearing(self):
        missing = _LOAD_BEARING - set(MODEL_FEATURES_LEAN)
        assert not missing, f"Lean is missing load-bearing features: {missing}"

    def test_minimal_keeps_load_bearing(self):
        missing = _LOAD_BEARING - set(MODEL_FEATURES_MINIMAL)
        assert not missing, f"Minimal is missing load-bearing features: {missing}"


class TestNoDuplicates:
    """Order matters for XGBoost feature_cols, and duplicates would break it."""

    def test_all_sets_dedup_clean(self):
        for name, feats in [
            ("MODEL_FEATURES_EVIDENCE", MODEL_FEATURES_EVIDENCE),
            ("MODEL_FEATURES_LEAN", MODEL_FEATURES_LEAN),
            ("MODEL_FEATURES_MINIMAL", MODEL_FEATURES_MINIMAL),
        ]:
            assert len(feats) == len(set(feats)), f"{name} contains duplicates"


class TestCalibrationDismissalsDropped:
    """SBIWTP + flow derivatives should be dropped from Evidence and below.

    These are the features calibration explicitly invalidated. If a future
    edit re-introduces them, the test will surface it.
    """

    _CALIBRATION_DISMISSED = {
        # SBIWTP — calibration's event-trigger sweep: recall@100=0.00 across 140 configs
        "sbiwtp_flow_mgd", "sbiwtp_anomaly", "sbiwtp_deficit",
        "sbiwtp_flow_x_temp", "sbiwtp_hourly_mgd", "sbiwtp_sli",
        # Flow derivatives — all calibration flow terms < 0.11 Spearman
        "flow_log", "flow_low", "flow_high",
    }

    def test_production_drops_all_calibration_dismissals(self):
        # The Phase 2 promotion: production MODEL_FEATURES must not contain
        # any feature calibration explicitly dismissed
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES)
        assert not kept, (
            f"MODEL_FEATURES (production) should drop calibration-dismissed "
            f"features: {kept}"
        )

    def test_evidence_drops_all_calibration_dismissals(self):
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES_EVIDENCE)
        assert not kept, f"Evidence should drop calibration-dismissed features: {kept}"

    def test_lean_drops_all_calibration_dismissals(self):
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES_LEAN)
        assert not kept, f"Lean should drop calibration-dismissed features: {kept}"

    def test_minimal_drops_all_calibration_dismissals(self):
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES_MINIMAL)
        assert not kept, f"Minimal should drop calibration-dismissed features: {kept}"


class TestForecastProducts:
    """Pins for the nowcast/nearcast/forecast product constants (Phase 1 of
    docs/feature/rename_workplan.md on the feature/rename_models branch).
    """

    def test_product_horizons_cover_0_to_24(self):
        from h2s.constants import (
            PRODUCT_FORECAST,
            PRODUCT_HORIZONS_H,
            PRODUCT_NEARCAST,
            PRODUCT_NOWCAST,
        )
        assert PRODUCT_HORIZONS_H[PRODUCT_NOWCAST] == (0, 3)
        assert PRODUCT_HORIZONS_H[PRODUCT_NEARCAST] == (3, 6)
        assert PRODUCT_HORIZONS_H[PRODUCT_FORECAST] == (6, 24)
        # Windows chain: each product starts where the previous ends
        assert PRODUCT_HORIZONS_H[PRODUCT_NOWCAST][1] == PRODUCT_HORIZONS_H[PRODUCT_NEARCAST][0]
        assert PRODUCT_HORIZONS_H[PRODUCT_NEARCAST][1] == PRODUCT_HORIZONS_H[PRODUCT_FORECAST][0]

    def test_cascade_triggers_ladder(self):
        from h2s.constants import (
            CASCADE_TRIGGERS,
            H2S_THRESHOLD_HIGH,
            H2S_THRESHOLD_LOW,
            H2S_THRESHOLD_MED,
            PRODUCT_FORECAST,
            PRODUCT_NEARCAST,
            PRODUCT_NOWCAST,
        )
        assert CASCADE_TRIGGERS["tier_1"] == {
            "product": PRODUCT_NOWCAST, "threshold_ppb": H2S_THRESHOLD_LOW, "prob_cutoff": 0.5}
        assert CASCADE_TRIGGERS["tier_2"] == {
            "product": PRODUCT_NEARCAST, "threshold_ppb": H2S_THRESHOLD_MED, "prob_cutoff": 0.5}
        assert CASCADE_TRIGGERS["tier_3"] == {
            "product": PRODUCT_FORECAST, "threshold_ppb": H2S_THRESHOLD_HIGH, "prob_cutoff": 0.5}
