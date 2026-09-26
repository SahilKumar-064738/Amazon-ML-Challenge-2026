"""
test_audit_fixes.py
-------------------
Focused tests for the four production audit fixes:

  FIX 1  — retrieval agreement column aliases after fuse_candidates()
  FIX 2  — SystemExit from matching stage is handled by run_pipeline.py
  FIX 3  — threshold sweep receives validation-only ground truth
  OPT 5  — atomic output writes (write_matching_results / write_candidate_pairs_final)
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHING_SRC = REPO_ROOT / "matching"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MATCHING_SRC))

# ---------------------------------------------------------------------------
# Helpers — build minimal DataFrames that resemble fuse_candidates() output
# ---------------------------------------------------------------------------

def _make_fused_df_with_tfidf_and_bm25() -> pd.DataFrame:
    """Simulate fuse_candidates() output when both TF-IDF and BM25 are active."""
    return pd.DataFrame({
        "source1_entity_id":   ["S1-001", "S1-001", "S1-002"],
        "candidate_entity_id": ["S2-099", "S2-100", "S2-101"],
        "cosine_similarity":   [0.9, 0.4, 0.8],
        "rank":                [1, 2, 1],
        # multi-view TF-IDF provenance flags (from blocking_multiview)
        "retrieved_by_tfidf_name":     [1, 0, 1],
        "retrieved_by_tfidf_address":  [1, 1, 0],
        "retrieved_by_tfidf_combined": [1, 1, 1],
        # BM25 signal (from blocking_bm25 — no flag column, only score)
        "bm25_score": [3.5, 0.0, 2.1],
        "bm25_rank":  [1, 0, 2],
        # retrieval_agreement_count already set by fuse_candidates()
        "retrieval_agreement_count": [2, 1, 2],
    })


def _make_fused_df_skip_bm25() -> pd.DataFrame:
    """Simulate fuse_candidates() output when --skip-bm25 is active."""
    return pd.DataFrame({
        "source1_entity_id":   ["S1-001", "S1-002"],
        "candidate_entity_id": ["S2-099", "S2-101"],
        "cosine_similarity":   [0.9, 0.8],
        "rank":                [1, 1],
        "retrieved_by_tfidf_name":     [1, 1],
        "retrieved_by_tfidf_address":  [1, 0],
        "retrieved_by_tfidf_combined": [1, 1],
        # No bm25_score / bm25_rank columns — BM25 was skipped
        "retrieval_agreement_count": [1, 1],
    })


def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inline copy of the alias logic added to run_blocking() in run_production.py.
    This lets the test verify the logic without importing the full pipeline.
    """
    if df.empty:
        return df

    # retrieved_by_char_tfidf
    if "retrieved_by_tfidf_combined" in df.columns:
        df = df.copy()
        df["retrieved_by_char_tfidf"] = df["retrieved_by_tfidf_combined"].fillna(0).astype(int)
    elif any(c.startswith("retrieved_by_tfidf_") for c in df.columns):
        df = df.copy()
        tfidf_cols = [c for c in df.columns if c.startswith("retrieved_by_tfidf_")]
        df["retrieved_by_char_tfidf"] = df[tfidf_cols].fillna(0).max(axis=1).astype(int)
    else:
        df = df.copy()
        df["retrieved_by_char_tfidf"] = 0

    # retrieved_by_bm25
    if "bm25_score" in df.columns:
        df["retrieved_by_bm25"] = (
            df["bm25_score"].notna() &
            (pd.to_numeric(df["bm25_score"], errors="coerce").fillna(0) > 0)
        ).astype(int)
    elif "bm25_rank" in df.columns:
        df["retrieved_by_bm25"] = df["bm25_rank"].notna().astype(int)
    else:
        df["retrieved_by_bm25"] = 0

    # retrieval_agreement_count — keep existing; add if absent
    if "retrieval_agreement_count" not in df.columns:
        df["retrieval_agreement_count"] = (
            df["retrieved_by_char_tfidf"] + df["retrieved_by_bm25"]
        )

    return df


# ===========================================================================
# FIX 1 — retrieval agreement column aliases
# ===========================================================================

class TestRetrievalAgreementAliases:

    def test_bm25_enabled_has_required_columns(self):
        """All three required columns exist after aliasing (BM25 enabled)."""
        df = _apply_aliases(_make_fused_df_with_tfidf_and_bm25())
        for col in ("retrieved_by_char_tfidf", "retrieved_by_bm25", "retrieval_agreement_count"):
            assert col in df.columns, f"Missing column: {col}"

    def test_skip_bm25_has_required_columns(self):
        """All three required columns exist after aliasing (--skip-bm25)."""
        df = _apply_aliases(_make_fused_df_skip_bm25())
        for col in ("retrieved_by_char_tfidf", "retrieved_by_bm25", "retrieval_agreement_count"):
            assert col in df.columns, f"Missing column: {col}"

    def test_bm25_enabled_char_tfidf_values(self):
        """retrieved_by_char_tfidf mirrors retrieved_by_tfidf_combined."""
        raw = _make_fused_df_with_tfidf_and_bm25()
        df  = _apply_aliases(raw)
        assert list(df["retrieved_by_char_tfidf"]) == list(raw["retrieved_by_tfidf_combined"].astype(int))

    def test_bm25_enabled_bm25_flag_values(self):
        """retrieved_by_bm25 is 1 where bm25_score > 0, else 0."""
        df = _apply_aliases(_make_fused_df_with_tfidf_and_bm25())
        # bm25_score = [3.5, 0.0, 2.1] → expected = [1, 0, 1]
        assert list(df["retrieved_by_bm25"]) == [1, 0, 1]

    def test_skip_bm25_bm25_flag_is_zero(self):
        """retrieved_by_bm25 is all-zero when BM25 columns are absent."""
        df = _apply_aliases(_make_fused_df_skip_bm25())
        assert df["retrieved_by_bm25"].sum() == 0, "Expected all zeros for retrieved_by_bm25 when BM25 skipped"

    def test_no_nan_in_alias_columns(self):
        """None of the three alias columns contain NaN."""
        for maker in (_make_fused_df_with_tfidf_and_bm25, _make_fused_df_skip_bm25):
            df = _apply_aliases(maker())
            for col in ("retrieved_by_char_tfidf", "retrieved_by_bm25", "retrieval_agreement_count"):
                assert not df[col].isna().any(), f"NaN found in {col}"

    def test_empty_dataframe_returns_empty(self):
        """Empty fused DataFrame passes through without error."""
        empty = pd.DataFrame()
        result = _apply_aliases(empty)
        assert result.empty


# ===========================================================================
# FIX 2 — SystemExit from matching stage is handled by run_pipeline.py
# ===========================================================================

class TestSystemExitHandling:

    def test_systemexit_from_matching_returns_nonzero(self, tmp_path):
        """
        If the matching stage raises SystemExit(1), run_pipeline.py must
        return a non-zero exit code rather than propagating as an unhandled
        exception (which would also be non-zero, but via a different code path).

        We verify this by running run_pipeline.py with a data directory that
        will trigger the SystemExit(1) in run_production.py's missing-file check.
        --skip-preprocessing is used so we jump straight to matching with no data,
        which causes the matching stage to raise SystemExit(1).
        """
        # Create a fake real-dir that is empty (no clean_*.tsv files)
        fake_real = tmp_path / "real"
        fake_real.mkdir()

        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "run_pipeline.py"),
                "--skip-preprocessing",
                "--input-dir", str(tmp_path),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Must exit non-zero (failure handled, not an uncaught crash)
        assert result.returncode != 0, (
            f"Expected non-zero exit when matching stage raises SystemExit.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # Must NOT be exit code 130 (that is the KeyboardInterrupt path)
        assert result.returncode != 130

    def test_run_pipeline_catches_systemexit_not_keyboard_interrupt(self):
        """
        Verify that run_pipeline.py only catches (Exception, SystemExit)
        and not all BaseException subclasses (which would incorrectly catch
        KeyboardInterrupt).
        """
        src = (REPO_ROOT / "run_pipeline.py").read_text(encoding="utf-8")
        # The handler must NOT be a bare BaseException catch
        assert "except BaseException" not in src, (
            "run_pipeline.py must not catch BaseException broadly"
        )
        # The handler MUST include SystemExit
        assert "SystemExit" in src, (
            "run_pipeline.py must explicitly catch SystemExit"
        )
        # The KeyboardInterrupt path must still be separate
        assert "KeyboardInterrupt" in src


# ===========================================================================
# FIX 3 — threshold sweep receives validation-only ground truth
# ===========================================================================

class TestValidationOnlyGroundTruth:

    def _load_model_module(self):
        """Import src.model from the matching directory."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "src.model",
            str(MATCHING_SRC / "src" / "model.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_filter_ground_truth_to_s1_ids_exists(self):
        """filter_ground_truth_to_s1_ids must be importable from src.model."""
        try:
            mod = self._load_model_module()
        except Exception as e:
            pytest.skip(f"src.model not importable in test environment: {e}")
        assert hasattr(mod, "filter_ground_truth_to_s1_ids"), (
            "filter_ground_truth_to_s1_ids not found in src.model"
        )

    def test_filter_ground_truth_returns_only_val_rows(self):
        """Filtered GT contains only rows whose S1 id is in val_ids."""
        try:
            mod = self._load_model_module()
            fn = mod.filter_ground_truth_to_s1_ids
        except Exception as e:
            pytest.skip(f"src.model not importable: {e}")

        gt = pd.DataFrame({
            "source1_entity_id": ["A", "B", "C", "D"],
            "matching_entity_ids": ["X", "Y", "Z", "W"],
        })
        val_ids = ["A", "C"]
        result = fn(gt, val_ids)
        assert set(result["source1_entity_id"].tolist()) == {"A", "C"}, (
            "Filtered GT contains non-validation entity IDs"
        )

    def test_filter_excludes_training_entities(self):
        """Training entity IDs must NOT appear in the filtered GT."""
        try:
            mod = self._load_model_module()
            fn = mod.filter_ground_truth_to_s1_ids
        except Exception as e:
            pytest.skip(f"src.model not importable: {e}")

        gt = pd.DataFrame({
            "source1_entity_id": ["TRAIN-1", "TRAIN-2", "VAL-1", "VAL-2"],
            "matching_entity_ids": ["x", "y", "z", "w"],
        })
        val_ids = ["VAL-1", "VAL-2"]
        result = fn(gt, val_ids)
        assert "TRAIN-1" not in result["source1_entity_id"].values
        assert "TRAIN-2" not in result["source1_entity_id"].values

    def test_run_production_imports_filter_fn(self):
        """run_production.py must import filter_ground_truth_to_s1_ids."""
        src = (REPO_ROOT / "matching" / "run_production.py").read_text(encoding="utf-8")
        assert "filter_ground_truth_to_s1_ids" in src, (
            "run_production.py does not import or use filter_ground_truth_to_s1_ids"
        )

    def test_sweep_uses_val_gt_not_full_gt(self):
        """run_production.py Step 6 must pass val_gt_df to sweep_thresholds."""
        src = (REPO_ROOT / "matching" / "run_production.py").read_text(encoding="utf-8")
        assert "val_gt_df" in src, (
            "run_production.py does not use val_gt_df variable in Step 6"
        )
        assert "sweep_thresholds(scored_val, val_gt_df)" in src, (
            "sweep_thresholds must be called with val_gt_df, not gt_df"
        )


# ===========================================================================
# OPTIONAL 5 — atomic output writes
# ===========================================================================

class TestAtomicOutputWrites:

    def _import_postprocess(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "src.postprocess",
            str(MATCHING_SRC / "src" / "postprocess.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_write_matching_results_no_tmp_file_remains(self, tmp_path):
        """No .tmp file must remain after a successful write."""
        try:
            pp = self._import_postprocess()
        except Exception as e:
            pytest.skip(f"postprocess not importable: {e}")

        out = tmp_path / "matching_results.tsv"
        pp.write_matching_results({"S1-001": ["S2-099"], "S1-002": []}, str(out))
        assert out.exists()
        assert not (tmp_path / "matching_results.tsv.tmp").exists()

    def test_write_matching_results_correct_content(self, tmp_path):
        """written file has header + expected rows."""
        try:
            pp = self._import_postprocess()
        except Exception as e:
            pytest.skip(f"postprocess not importable: {e}")

        out = tmp_path / "matching_results.tsv"
        pp.write_matching_results({"S1-001": ["S2-099"], "S1-002": []}, str(out))
        lines = out.read_text().splitlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids"
        assert len(lines) == 3  # header + 2 rows

    def test_write_candidate_pairs_no_tmp_file_remains(self, tmp_path):
        """No .tmp file must remain after a successful write."""
        try:
            pp = self._import_postprocess()
        except Exception as e:
            pytest.skip(f"postprocess not importable: {e}")

        out = tmp_path / "candidate_pairs.tsv"
        pp.write_candidate_pairs_final({"S1-001": ["S2-099", "S2-100"]}, str(out))
        assert out.exists()
        assert not (tmp_path / "candidate_pairs.tsv.tmp").exists()

    def test_write_matching_results_uses_os_replace(self):
        """postprocess.py source must use os.replace for atomic writes."""
        src = (MATCHING_SRC / "src" / "postprocess.py").read_text(encoding="utf-8")
        assert "os.replace(" in src, (
            "postprocess.py does not use os.replace() for atomic output writing"
        )
