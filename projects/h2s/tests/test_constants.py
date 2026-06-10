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
    """The plan specified each count explicitly — pin them so future edits surface."""

    def test_baseline_is_44(self):
        assert len(MODEL_FEATURES) == 44

    def test_evidence_is_33(self):
        assert len(MODEL_FEATURES_EVIDENCE) == 33

    def test_lean_is_19(self):
        assert len(MODEL_FEATURES_LEAN) == 19

    def test_minimal_is_11(self):
        assert len(MODEL_FEATURES_MINIMAL) == 11


class TestSubsetRelations:
    """Each candidate must be a subset of MODEL_FEATURES (no typos, no new features)."""

    def test_evidence_subset_of_baseline(self):
        assert set(MODEL_FEATURES_EVIDENCE) <= set(MODEL_FEATURES)

    def test_lean_subset_of_evidence(self):
        # Lean is a strict refinement of Evidence — anything dropped from Evidence
        # stays dropped in Lean
        assert set(MODEL_FEATURES_LEAN) <= set(MODEL_FEATURES_EVIDENCE)

    def test_minimal_subset_of_baseline(self):
        # Minimal isn't required to be a subset of Lean (it's a re-curated
        # calibration-only set), but every feature must exist in the master list
        assert set(MODEL_FEATURES_MINIMAL) <= set(MODEL_FEATURES)


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

    def test_evidence_drops_all_calibration_dismissals(self):
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES_EVIDENCE)
        assert not kept, f"Evidence should drop calibration-dismissed features: {kept}"

    def test_lean_drops_all_calibration_dismissals(self):
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES_LEAN)
        assert not kept, f"Lean should drop calibration-dismissed features: {kept}"

    def test_minimal_drops_all_calibration_dismissals(self):
        kept = self._CALIBRATION_DISMISSED & set(MODEL_FEATURES_MINIMAL)
        assert not kept, f"Minimal should drop calibration-dismissed features: {kept}"
