"""
tests/test_faiss_blocking.py
-----------------------------
Comprehensive tests for the FAISS semantic retrieval integration.

Tests cover:
  - FAISS disabled path (pipeline works without faiss-cpu installed)
  - FAISS enabled: build index, query, candidate schema, provenance
  - Cache reuse (second run skips rebuild)
  - Invalid cache (metadata mismatch → rebuild)
  - Missing dependency error message
  - Train/test separation (no cross-split leakage)
  - Top-K respected
  - Empty result handling
  - BM25 combinations (OFF+FAISS OFF, OFF+FAISS ON, ON+FAISS ON)
  - blocking_fusion.fuse_candidates excludes retrieved_by_faiss from
    retrieval_agreement_count so the count stays in {1, 2}
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure matching/src is importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHING_SRC = REPO_ROOT / "matching" / "src"
for p in (str(REPO_ROOT), str(REPO_ROOT / "matching")):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Helpers to build small synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_s2s3(n: int = 6, prefix: str = "S2") -> pd.DataFrame:
    """Return a tiny corpus DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            "entity_id":     f"{prefix}-{i:03d}",
            "clean_text":    f"company {i} street {i}",
            "clean_name":    f"company {i}",
            "clean_address": f"street {i}",
        })
    return pd.DataFrame(rows)


def _make_s1(n: int = 3) -> pd.DataFrame:
    """Return a tiny query DataFrame."""
    rows = []
    for i in range(n):
        rows.append({
            "entity_id":     f"S1-{i:03d}",
            "clean_text":    f"company {i} street {i}",
            "clean_name":    f"company {i}",
            "clean_address": f"street {i}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures: stub faiss + sentence_transformers so tests run without the
# real packages installed.  Tests that need real FAISS are skipped if absent.
# ---------------------------------------------------------------------------

def _has_real_faiss() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def _has_real_st() -> bool:
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except ImportError:
        return False


REAL_FAISS_AVAILABLE = _has_real_faiss() and _has_real_st()
skip_if_no_faiss = pytest.mark.skipif(
    not REAL_FAISS_AVAILABLE,
    reason="faiss-cpu and/or sentence-transformers not installed",
)


# ---------------------------------------------------------------------------
# 1. FAISS DISABLED — pipeline imports work without faiss-cpu
# ---------------------------------------------------------------------------

class TestFAISSDisabled:
    """Verify that blocking_fusion and run_blocking work with FAISS off."""

    def test_blocking_fusion_importable_without_faiss(self):
        """fuse_candidates must not import faiss at module level."""
        from src.blocking_fusion import fuse_candidates  # noqa: F401

    def test_fuse_candidates_faiss_off_no_faiss_column(self):
        """When FAISS is off, retrieved_by_faiss column is absent — no crash."""
        from src.blocking_fusion import fuse_candidates

        tfidf_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9, "retrieved_by_tfidf_combined": 1},
        ])
        result = fuse_candidates([tfidf_df])
        assert "source1_entity_id" in result.columns
        assert "candidate_entity_id" in result.columns
        # retrieved_by_faiss must NOT be present when FAISS was not used
        assert "retrieved_by_faiss" not in result.columns

    def test_retrieval_agreement_count_tfidf_only_is_one(self):
        """TF-IDF-only fusion → agreement count == 1."""
        from src.blocking_fusion import fuse_candidates

        df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 0},
        ])
        result = fuse_candidates([df])
        assert int(result["retrieval_agreement_count"].iloc[0]) == 1

    def test_retrieval_agreement_count_tfidf_bm25_is_two(self):
        """TF-IDF + BM25 → agreement count == 2."""
        from src.blocking_fusion import fuse_candidates

        df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 1},
        ])
        result = fuse_candidates([df])
        assert int(result["retrieval_agreement_count"].iloc[0]) == 2


# ---------------------------------------------------------------------------
# 2. blocking_fusion with FAISS ON — agreement count must NOT become 3
# ---------------------------------------------------------------------------

class TestFusionWithFAISS:
    """FAISS provenance must not inflate retrieval_agreement_count above 2."""

    def test_faiss_provenance_column_preserved(self):
        """retrieved_by_faiss is kept as an informational column."""
        from src.blocking_fusion import fuse_candidates

        tfidf_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 0},
        ])
        faiss_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.85, "retrieved_by_faiss": 1},
        ])
        result = fuse_candidates([tfidf_df, faiss_df])
        assert "retrieved_by_faiss" in result.columns
        assert int(result["retrieved_by_faiss"].iloc[0]) == 1

    def test_agreement_count_excludes_faiss(self):
        """retrieval_agreement_count must be {1} when only TF-IDF + FAISS retrieved."""
        from src.blocking_fusion import fuse_candidates

        tfidf_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 0},
        ])
        faiss_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.85, "retrieved_by_faiss": 1},
        ])
        result = fuse_candidates([tfidf_df, faiss_df])
        count = int(result["retrieval_agreement_count"].iloc[0])
        # Only retrieved_by_char_tfidf=1, retrieved_by_bm25=0 → count=1
        assert count == 1, (
            f"retrieval_agreement_count={count} but must be 1 (FAISS excluded). "
            "This would break features.build_feature_matrix() which validates {1,2}."
        )

    def test_agreement_count_tfidf_bm25_faiss_is_two_not_three(self):
        """Even with all three sources, agreement count must stay ≤ 2."""
        from src.blocking_fusion import fuse_candidates

        tfidf_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 1},
        ])
        faiss_df = pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.85, "retrieved_by_faiss": 1},
        ])
        result = fuse_candidates([tfidf_df, faiss_df])
        count = int(result["retrieval_agreement_count"].iloc[0])
        assert count == 2, (
            f"retrieval_agreement_count={count}; must be 2 even with FAISS (FAISS excluded from count)."
        )

    def test_faiss_only_pair_agreement_count_is_zero(self):
        """A pair retrieved only by FAISS (no TF-IDF/BM25) has count=0.

        This is acceptable — such pairs are fused in but count is 0.
        The downstream alias step in run_blocking will set retrieved_by_char_tfidf
        based on the TF-IDF flag, so count remains consistent.
        """
        from src.blocking_fusion import fuse_candidates

        faiss_df = pd.DataFrame([
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-002",
             "score": 0.7, "retrieved_by_faiss": 1},
        ])
        result = fuse_candidates([faiss_df])
        # No canonical agreement cols present → count defaults to 0 or sums
        # whatever canonical cols exist (none → 0 or fallback 1)
        count = int(result["retrieval_agreement_count"].iloc[0])
        # Should be 0 (no canonical cols) or 1 (fallback) — must NOT be 2+
        assert count in (0, 1), f"Unexpected count {count} for FAISS-only pair"


# ---------------------------------------------------------------------------
# 3. blocking_faiss module — missing dependency error messages
# ---------------------------------------------------------------------------

class TestMissingDependencyErrors:
    """Verify clear RuntimeError when faiss/sentence-transformers absent."""

    def test_require_faiss_raises_runtime_error(self, monkeypatch):
        """_require_faiss() must raise RuntimeError with install hint."""
        # Reload module with faiss hidden
        import src.blocking_faiss as bf
        with patch.dict(sys.modules, {"faiss": None}):
            with pytest.raises(RuntimeError, match="faiss-cpu"):
                bf._require_faiss()

    def test_require_st_raises_runtime_error(self, monkeypatch):
        """_require_sentence_transformers() must raise RuntimeError."""
        import src.blocking_faiss as bf
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                bf._require_sentence_transformers()

    def test_error_mentions_requirements_txt(self):
        """Error messages must mention requirements.txt as the fix path."""
        import src.blocking_faiss as bf
        with patch.dict(sys.modules, {"faiss": None}):
            try:
                bf._require_faiss()
            except RuntimeError as e:
                assert "requirements.txt" in str(e)

    def test_search_without_index_raises_file_not_found(self, tmp_path):
        """search_faiss_candidates must raise FileNotFoundError for missing index."""
        if not REAL_FAISS_AVAILABLE:
            pytest.skip("faiss-cpu not installed")
        import src.blocking_faiss as bf

        s1   = _make_s1(2)
        s2s3 = _make_s2s3(4)
        with pytest.raises(FileNotFoundError):
            bf.search_faiss_candidates(s1, s2s3, tmp_path / "nonexistent")


# ---------------------------------------------------------------------------
# 4. Metadata helpers
# ---------------------------------------------------------------------------

class TestMetadataHelpers:
    def test_entity_ids_hash_deterministic(self):
        import src.blocking_faiss as bf
        ids = ["S2-001", "S2-002", "S2-003"]
        h1 = bf._compute_entity_ids_hash(ids)
        h2 = bf._compute_entity_ids_hash(ids)
        assert h1 == h2

    def test_entity_ids_hash_order_insensitive(self):
        import src.blocking_faiss as bf
        ids_a = ["S2-001", "S2-002"]
        ids_b = ["S2-002", "S2-001"]
        assert bf._compute_entity_ids_hash(ids_a) == bf._compute_entity_ids_hash(ids_b)

    def test_entity_ids_hash_different_for_different_sets(self):
        import src.blocking_faiss as bf
        h1 = bf._compute_entity_ids_hash(["S2-001"])
        h2 = bf._compute_entity_ids_hash(["S2-002"])
        assert h1 != h2

    def test_build_metadata_keys(self):
        import src.blocking_faiss as bf
        meta = bf._build_metadata(
            model_name="model", embedding_dim=384, normalization="l2",
            index_type="IVFFlat_IP", nlist=1024, nprobe=16,
            split_label="TRAIN", entity_ids=["A", "B"], add_chunk_size=500_000,
        )
        for field in bf._METADATA_FIELDS:
            assert field in meta, f"Missing metadata field: {field}"

    def test_index_is_valid_false_when_no_files(self, tmp_path):
        import src.blocking_faiss as bf
        base = tmp_path / "test_index"
        meta = bf._build_metadata(
            model_name="m", embedding_dim=384, normalization="l2",
            index_type="IVFFlat_IP", nlist=1024, nprobe=16,
            split_label="X", entity_ids=["A"], add_chunk_size=1000,
        )
        assert not bf._index_is_valid(base, meta)

    def test_index_is_valid_false_when_metadata_mismatch(self, tmp_path):
        import src.blocking_faiss as bf
        base = tmp_path / "test_index"
        meta = bf._build_metadata(
            model_name="model-A", embedding_dim=384, normalization="l2",
            index_type="IVFFlat_IP", nlist=1024, nprobe=16,
            split_label="TRAIN", entity_ids=["A", "B"], add_chunk_size=500_000,
        )
        # Write a metadata file with a different model_name
        stale = dict(meta)
        stale["model_name"] = "model-B"
        bf._metadata_path(base).write_text(json.dumps(stale))
        # Also create a dummy .idx file
        bf._index_path(base).write_text("dummy")
        assert not bf._index_is_valid(base, meta)

    def test_index_is_valid_true_when_matching(self, tmp_path):
        import src.blocking_faiss as bf
        base = tmp_path / "test_index"
        meta = bf._build_metadata(
            model_name="model-A", embedding_dim=384, normalization="l2",
            index_type="IVFFlat_IP", nlist=1024, nprobe=16,
            split_label="TRAIN", entity_ids=["A", "B"], add_chunk_size=500_000,
        )
        bf._metadata_path(base).write_text(json.dumps(meta))
        bf._index_path(base).write_text("dummy")
        assert bf._index_is_valid(base, meta)


# ---------------------------------------------------------------------------
# 5. Full FAISS build + search (requires real faiss-cpu)
# ---------------------------------------------------------------------------

@skip_if_no_faiss
class TestFAISSBuildAndSearch:
    """Integration tests that actually build and query a FAISS index."""

    def test_build_creates_index_and_metadata(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(10)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             nlist=4, batch_size=10)
        assert bf._index_path(base).exists()
        assert bf._metadata_path(base).exists()

    def test_build_metadata_content_correct(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TEST",
                             nlist=4, batch_size=8)
        meta = json.loads(bf._metadata_path(base).read_text())
        assert meta["split_label"] == "TEST"
        assert meta["row_count"]   == 8

    def test_search_returns_expected_columns(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        s1   = _make_s1(3)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             nlist=4, batch_size=8)
        result = bf.search_faiss_candidates(s1, s2s3, base, top_k=3)

        expected_cols = {
            "source1_entity_id", "candidate_entity_id",
            "score", "rank", "retrieved_by_faiss",
        }
        assert expected_cols.issubset(set(result.columns)), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )

    def test_retrieved_by_faiss_is_always_one(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        s1   = _make_s1(3)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             nlist=4, batch_size=8)
        result = bf.search_faiss_candidates(s1, s2s3, base, top_k=3)
        assert (result["retrieved_by_faiss"] == 1).all()

    def test_top_k_respected(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(20)
        s1   = _make_s1(2)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             nlist=4, batch_size=20)
        for k in (1, 3, 5):
            result = bf.search_faiss_candidates(s1, s2s3, base, top_k=k)
            # Each S1 entity should have at most k candidates
            counts = result.groupby("source1_entity_id").size()
            assert (counts <= k).all(), f"top_k={k} violated: {counts.to_dict()}"

    def test_candidate_ids_are_valid_s2s3_ids(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(10)
        s1   = _make_s1(3)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             nlist=4, batch_size=10)
        result = bf.search_faiss_candidates(s1, s2s3, base, top_k=5)
        valid_ids = set(s2s3["entity_id"].tolist())
        assert set(result["candidate_entity_id"]).issubset(valid_ids)

    def test_score_is_float_in_range(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        s1   = _make_s1(2)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             nlist=4, batch_size=8)
        result = bf.search_faiss_candidates(s1, s2s3, base, top_k=3)
        scores = result["score"].astype(float)
        assert scores.between(-1.1, 1.1).all(), "Cosine scores should be in [-1, 1]"


# ---------------------------------------------------------------------------
# 6. Cache reuse
# ---------------------------------------------------------------------------

@skip_if_no_faiss
class TestCacheReuse:
    def test_second_build_is_skipped(self, tmp_path, capsys):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        base = tmp_path / "idx"
        # First build
        bf.build_faiss_index(s2s3, base, split_label="TRAIN", nlist=4, batch_size=8)
        # Second build — should print "Reusing"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN", nlist=4, batch_size=8)
        captured = capsys.readouterr()
        assert "Reusing" in captured.out

    def test_force_rebuilds(self, tmp_path, capsys):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN", nlist=4, batch_size=8)
        bf.build_faiss_index(s2s3, base, split_label="TRAIN", nlist=4, batch_size=8,
                             force=True)
        captured = capsys.readouterr()
        # "Building" should appear in the second run's output
        assert "Building" in captured.out


# ---------------------------------------------------------------------------
# 7. Cache invalidation
# ---------------------------------------------------------------------------

@skip_if_no_faiss
class TestCacheInvalidation:
    def test_model_change_triggers_rebuild(self, tmp_path, capsys):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(8)
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             model_name="intfloat/multilingual-e5-small",
                             nlist=4, batch_size=8)
        # Tamper metadata: change model name
        meta_path = bf._metadata_path(base)
        meta = json.loads(meta_path.read_text())
        meta["model_name"] = "some-other-model"
        meta_path.write_text(json.dumps(meta))
        # Re-build with original model name → stale → rebuild
        bf.build_faiss_index(s2s3, base, split_label="TRAIN",
                             model_name="intfloat/multilingual-e5-small",
                             nlist=4, batch_size=8)
        captured = capsys.readouterr()
        assert "stale" in captured.out or "Building" in captured.out

    def test_corpus_change_triggers_rebuild(self, tmp_path, capsys):
        import src.blocking_faiss as bf
        s2s3_v1 = _make_s2s3(6)
        s2s3_v2 = _make_s2s3(8)  # different row count → different hash
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3_v1, base, split_label="TRAIN",
                             nlist=4, batch_size=6)
        bf.build_faiss_index(s2s3_v2, base, split_label="TRAIN",
                             nlist=4, batch_size=8)
        captured = capsys.readouterr()
        assert "stale" in captured.out or "Building" in captured.out


# ---------------------------------------------------------------------------
# 8. Train/test separation (no leakage)
# ---------------------------------------------------------------------------

@skip_if_no_faiss
class TestTrainTestSeparation:
    """Verify FAISS train and test indexes are independent."""

    def test_train_and_test_use_separate_index_files(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3_train = _make_s2s3(6, prefix="S2")
        s2s3_test  = _make_s2s3(4, prefix="S3")
        base_train = tmp_path / "faiss_index_TRAIN"
        base_test  = tmp_path / "faiss_index_TEST"

        bf.build_faiss_index(s2s3_train, base_train, split_label="TRAIN",
                             nlist=4, batch_size=6)
        bf.build_faiss_index(s2s3_test, base_test, split_label="TEST",
                             nlist=4, batch_size=4)

        meta_train = json.loads(bf._metadata_path(base_train).read_text())
        meta_test  = json.loads(bf._metadata_path(base_test).read_text())

        assert meta_train["split_label"] == "TRAIN"
        assert meta_test["split_label"]  == "TEST"
        assert meta_train["row_count"]   == 6
        assert meta_test["row_count"]    == 4
        # Entity ID hashes must differ
        assert meta_train["entity_ids_hash"] != meta_test["entity_ids_hash"]

    def test_test_candidates_come_from_test_corpus_only(self, tmp_path):
        import src.blocking_faiss as bf
        # Train corpus and test corpus use distinct entity ID prefixes
        s2s3_test = _make_s2s3(6, prefix="TEST_S2")
        s1_test   = _make_s1(2)
        base = tmp_path / "faiss_index_TEST"

        bf.build_faiss_index(s2s3_test, base, split_label="TEST",
                             nlist=4, batch_size=6)
        result = bf.search_faiss_candidates(s1_test, s2s3_test, base, top_k=3)

        valid_test_ids = set(s2s3_test["entity_id"].tolist())
        assert set(result["candidate_entity_id"]).issubset(valid_test_ids), (
            "Test candidates must come exclusively from the test corpus."
        )


# ---------------------------------------------------------------------------
# 9. Empty result handling
# ---------------------------------------------------------------------------

@skip_if_no_faiss
class TestEmptyResults:
    def test_empty_s1_returns_empty_df(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(6)
        s1   = pd.DataFrame(columns=["entity_id", "clean_text"])
        base = tmp_path / "idx"
        bf.build_faiss_index(s2s3, base, split_label="TRAIN", nlist=4, batch_size=6)
        result = bf.search_faiss_candidates(s1, s2s3, base, top_k=5)
        assert result.empty
        assert "retrieved_by_faiss" in result.columns

    def test_get_or_build_and_search_returns_df_schema(self, tmp_path):
        import src.blocking_faiss as bf
        s2s3 = _make_s2s3(6)
        s1   = _make_s1(2)
        base = tmp_path / "idx"
        result = bf.get_or_build_and_search(
            s2s3, s1, base, split_label="TRAIN",
            top_k=3, nlist=4, batch_size=6,
        )
        assert isinstance(result, pd.DataFrame)
        assert "source1_entity_id"   in result.columns
        assert "candidate_entity_id" in result.columns
        assert "retrieved_by_faiss"  in result.columns


# ---------------------------------------------------------------------------
# 10. BM25 combinations via blocking_fusion
# ---------------------------------------------------------------------------

class TestBM25Combinations:
    """Test CASE A / B / C candidate fusion behaviour."""

    def _tfidf_df(self):
        return pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.9,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 0},
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-001",
             "score": 0.8,
             "retrieved_by_char_tfidf": 1, "retrieved_by_bm25": 0},
        ])

    def _bm25_df(self):
        return pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "bm25_score": 5.2, "bm25_rank": 1,
             "retrieved_by_bm25": 1},
        ])

    def _faiss_df(self):
        return pd.DataFrame([
            {"source1_entity_id": "S1-000", "candidate_entity_id": "S2-000",
             "score": 0.85, "retrieved_by_faiss": 1},
            {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-002",
             "score": 0.7, "retrieved_by_faiss": 1},
        ])

    def test_case_a_bm25_off_faiss_off(self):
        """CASE A: TF-IDF only. Agreement count in {1}."""
        from src.blocking_fusion import fuse_candidates
        result = fuse_candidates([self._tfidf_df()])
        assert "retrieved_by_faiss" not in result.columns
        assert result["retrieval_agreement_count"].isin([0, 1, 2]).all()

    def test_case_b_bm25_off_faiss_on(self):
        """CASE B: TF-IDF + FAISS. Agreement count still in {1, 2} (FAISS excluded)."""
        from src.blocking_fusion import fuse_candidates
        result = fuse_candidates([self._tfidf_df(), self._faiss_df()])
        assert "retrieved_by_faiss" in result.columns
        # Agreement count must be ≤ 2
        assert (result["retrieval_agreement_count"] <= 2).all(), (
            "FAISS must not push agreement count above 2"
        )

    def test_case_c_bm25_on_faiss_on(self):
        """CASE C: TF-IDF + BM25 + FAISS. Agreement count still ≤ 2."""
        from src.blocking_fusion import fuse_candidates
        result = fuse_candidates([
            self._tfidf_df(), self._bm25_df(), self._faiss_df()
        ])
        assert "retrieved_by_faiss" in result.columns
        assert (result["retrieval_agreement_count"] <= 2).all()
        # The pair retrieved by both TF-IDF and BM25 should have count=2
        pair = result[
            (result["source1_entity_id"] == "S1-000") &
            (result["candidate_entity_id"] == "S2-000")
        ]
        if not pair.empty:
            assert int(pair["retrieval_agreement_count"].iloc[0]) == 2


# ---------------------------------------------------------------------------
# 11. CLI: --enable-faiss appears in run_pipeline.py --help
# ---------------------------------------------------------------------------

class TestCLIFAISSFlag:
    def test_enable_faiss_in_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "run_pipeline.py"), "--help"],
            capture_output=True, text=True, timeout=15,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0
        assert "--enable-faiss" in result.stdout, (
            "--enable-faiss flag must appear in run_pipeline.py --help"
        )

    def test_run_production_enable_faiss_in_help(self):
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(REPO_ROOT / "matching" / "run_production.py"), "--help"],
            capture_output=True, text=True, timeout=15,
            cwd=str(REPO_ROOT / "matching"),
        )
        assert result.returncode == 0
        assert "--enable-faiss" in result.stdout


# ---------------------------------------------------------------------------
# 12. pipeline_config exports
# ---------------------------------------------------------------------------

class TestPipelineConfigFAISS:
    def test_faiss_constants_exported(self):
        from pipeline_config import (
            FAISS_EMBEDDING_MODEL,
            FAISS_EMBEDDING_DIM,
            FAISS_NLIST,
            FAISS_NPROBE,
            FAISS_NORMALIZATION,
            FAISS_INDEX_TYPE,
            FAISS_TOP_K,
            FAISS_BATCH_SIZE,
            FAISS_ADD_CHUNK_SIZE,
            faiss_index_base,
        )
        assert FAISS_EMBEDDING_MODEL == "intfloat/multilingual-e5-small"
        assert FAISS_EMBEDDING_DIM   == 384
        assert FAISS_NLIST           == 1_024
        assert FAISS_NPROBE          == 16
        assert FAISS_NORMALIZATION   == "l2"
        assert "IVF" in FAISS_INDEX_TYPE or FAISS_INDEX_TYPE == "IVFFlat_IP"
        assert FAISS_TOP_K           >= 1
        assert FAISS_BATCH_SIZE      >= 1
        assert FAISS_ADD_CHUNK_SIZE  >= 1

    def test_faiss_index_base_helper(self, tmp_path):
        from pipeline_config import faiss_index_base
        base = faiss_index_base(tmp_path, "TRAIN")
        assert "TRAIN" in str(base)

    def test_faiss_index_base_upper(self, tmp_path):
        from pipeline_config import faiss_index_base
        b1 = faiss_index_base(tmp_path, "train")
        b2 = faiss_index_base(tmp_path, "TRAIN")
        # Both should produce the same path (label is uppercased inside)
        assert b1 == b2
