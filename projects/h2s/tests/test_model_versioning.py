"""Tests for the model-versioning helpers (Phase 2 of
docs/feature/rename_workplan.md): version tags, the new-vs-production
comparison, and the Slack promotion message.

Pure unit tests — no S3, no Dagster context.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import dagster as dg

from h2s.defs.h2s_multi_station_training import (
    _VARIANTS,
    _station_artifact_names,
    build_promotion_message,
    compare_training_reports,
    make_version_tag,
    station_model_promotion,
)


class TestVersionTag:
    def test_format(self):
        tag = make_version_tag(
            now=datetime(2026, 6, 12, 21, 30, 0, tzinfo=timezone.utc), sha="a1b2c3d"
        )
        assert tag == "20260612T213000Z-a1b2c3d"
        assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-z]+", tag)

    def test_lexical_sort_is_chronological(self):
        # promote "latest" relies on string sorting == time sorting
        earlier = make_version_tag(
            now=datetime(2026, 6, 12, 9, 0, 0, tzinfo=timezone.utc), sha="zzzzzzz"
        )
        later = make_version_tag(
            now=datetime(2026, 6, 12, 10, 0, 0, tzinfo=timezone.utc), sha="aaaaaaa"
        )
        assert sorted([later, earlier]) == [earlier, later]

    def test_falls_back_to_real_sha(self):
        # No sha argument → resolves from git (or env fallback); never empty
        tag = make_version_tag(now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert re.fullmatch(r"\d{8}T\d{6}Z-.+", tag)


class TestArtifactNames:
    def test_full_set(self):
        names = _station_artifact_names()
        # 4 tasks × 2 variants pickles + 2 feature schemas
        assert len(names) == 10
        assert "regression_evidence.pkl" in names
        assert "clf_30ppb_lean.pkl" in names
        assert "features_evidence.json" in names
        assert "features_lean.json" in names


def _report(r2=0.5, r5=0.85, r10=0.86, r30=0.79, r100=0.67,
            auc5=0.95, auc10=0.95, auc30=0.96) -> dict:
    return {
        "tasks": {
            "evidence": {
                "regression": {"R2": r2, "recall_5": r5, "recall_10": r10,
                               "recall_30": r30, "recall_100": r100},
                "clf_5ppb": {"AUC": auc5},
                "clf_10ppb": {"AUC": auc10},
                "clf_30ppb": {"AUC": auc30},
            },
        }
    }


class TestCompareTrainingReports:
    def test_no_baseline_recommends_promote(self):
        out = compare_training_reports(_report(), None)
        assert out["recommendation"] == "promote"
        assert "no production baseline" in out["reason"]

    def test_all_improved_recommends_promote(self):
        new = _report(r30=0.82, r100=0.70, auc30=0.97)
        prod = _report(r30=0.79, r100=0.67, auc30=0.96)
        out = compare_training_reports(new, prod)
        assert out["recommendation"] == "promote"
        assert out["n_regressed"] == 0
        assert out["n_improved"] >= 3

    def test_within_tolerance_still_promotes(self):
        # 1pp dips are inside the 2pp tolerance
        new = _report(r30=0.78, r100=0.66)
        prod = _report(r30=0.79, r100=0.67)
        out = compare_training_reports(new, prod)
        assert out["recommendation"] == "promote"

    def test_regression_beyond_tolerance_recommends_review(self):
        new = _report(r100=0.60)   # -7pp on recall@100
        prod = _report(r100=0.67)
        out = compare_training_reports(new, prod)
        assert out["recommendation"] == "review"
        assert "recall_100" in out["reason"]

    def test_missing_metrics_skipped_not_fatal(self):
        # Old production report without clf_30ppb (pre-Phase-1) still compares
        prod = _report()
        del prod["tasks"]["evidence"]["clf_30ppb"]
        out = compare_training_reports(_report(), prod)
        names = [m["name"] for m in out["metrics"]]
        assert "clf_30ppb.AUC" not in names
        assert len(names) == 7  # the other comparisons still happen


class TestPromotionMessage:
    def test_contains_essentials(self):
        comparison = compare_training_reports(_report(r100=0.70), _report(r100=0.67))
        msg = build_promotion_message(
            "NESTOR - BES", "nestor_bes", "20260612T213000Z-a1b2c3d",
            comparison, env_label="DEV",
        )
        assert "NESTOR - BES" in msg
        assert "[DEV]" in msg
        assert "20260612T213000Z-a1b2c3d" in msg
        assert "PROMOTE" in msg
        assert "promote_station_models_job" in msg
        assert "--partition nestor_bes" in msg
        # The command must carry the exact version tag, not "latest"
        assert '\\"version_tag\\":\\"20260612T213000Z-a1b2c3d\\"' in msg.replace('"', '\\"') or \
               '"version_tag":"20260612T213000Z-a1b2c3d"' in msg


class _FakeS3:
    """Minimal S3 stand-in. Backfill archives omit features_{variant}.json, so
    copyFile raises NoSuchKey for those keys; everything else is recorded."""

    def __init__(self, archive_features: dict[str, list[str]]):
        self.S3_BUCKET = "test"
        self._report = json.dumps({"features": archive_features}).encode("utf-8")
        self.copied: list[tuple[str, str]] = []
        self.put: dict[str, bytes] = {}

    def copyFile(self, src_path, dst_path, **kwargs):
        if src_path.endswith("features_evidence.json") or src_path.endswith("features_lean.json"):
            raise Exception(f"NoSuchKey: {src_path}")
        self.copied.append((src_path, dst_path))

    def getFile(self, path, bucket=None):
        assert path.endswith("training_report.json")
        return self._report

    def putFile(self, data, path, bucket=None, content_type=None):
        self.put[path] = data


class TestPromotionBackfillReconstruction:
    """A backfill archive has no features_{variant}.json — promotion must
    reconstruct them from training_report.json rather than refuse (regression
    test for the partial-promotion Failure on backfill_* versions)."""

    def test_features_reconstructed_from_report(self):
        feats = {v: [f"{v}_feat_{i}" for i in range(3)] for v in _VARIANTS}
        s3 = _FakeS3(feats)
        version = "backfill_2026-05_20260623T144248Z-96b650e"

        ctx = dg.build_asset_context(
            partition_key="ib_civic_ctr",
            resources={"s3": s3},
            asset_config={"version_tag": version},
        )
        result = station_model_promotion(ctx)

        assert result["status"] == "promoted"
        assert result["model_version"] == version
        # Pickles + training_report were server-side copied (not put).
        copied_names = {dst.rsplit("/", 1)[-1] for _, dst in s3.copied}
        assert "training_report.json" in copied_names
        assert "clf_30ppb_evidence.pkl" in copied_names
        # Both feature schemas were reconstructed and written to production.
        written = {p.rsplit("/", 1)[-1]: d for p, d in s3.put.items()}
        for variant in _VARIANTS:
            name = f"features_{variant}.json"
            assert name in written, f"{name} not reconstructed"
            assert json.loads(written[name]) == feats[variant]
        # deployment_metadata.json is rewritten around the promoted version.
        assert "deployment_metadata.json" in written
        assert json.loads(written["deployment_metadata.json"])["model_version"] == version
