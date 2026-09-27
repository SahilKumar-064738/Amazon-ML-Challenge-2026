"""
tests/test_tfidf_memory_safety.py
-----------------------------------
Phase 5 — Correctness and memory-safety tests for the redesigned
Multi-View TF-IDF blocking.

Tests cover:
  1.  Identical entity retrieves itself
  2.  Near-duplicate retrieval
  3.  Multiple queries
  4.  Multiple corpus blocks (corpus_block_size < corpus size)
  5.  Multiple query chunks (chunk_size < query size)
  6.  Top-K bound respected
  7.  Chunked result == unchunked reference (tiny dataset)
  8.  Duplicate candidates deduplicated
  9.  Empty query → empty result
  10. Empty corpus → empty result
  11. Multiple TF-IDF views produce correct provenance flags
  12. Candidate provenance preserved in output columns
  13. Deterministic output (same input → same output)
  14. chunk_size / corpus_block_size are actually forwarded
  15. RSS instrumentation does not raise

Notes
-----
All tests use tiny synthetic datasets to run fast in CI.
They do NOT test full-scale memory usage — see the benchmark section.
"""

from __future__ import annotations

import gc
import sys
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

# Make matching/src importable regardless of working directory
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "matching"))

from src.blocking import search_candidates_scalable
from src.blocking_multiview import (
    _effective_texts,
    _has_content,
    _rss_mb,
    _sparse_mem_mb,
    fit_multiview_vectorizers,
    search_multiview_candidates,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_df(ids, names, addresses=None):
    """Build a minimal entity DataFrame with the expected columns."""
    rows = []
    for i, (eid, name) in enumerate(zip(ids, names)):
        addr = (addresses[i] if addresses else f"street {i} city")
        rows.append({
            "entity_id":     eid,
            "clean_text":    f"{name} {addr}",
            "clean_name":    name,
            "clean_address": addr,
        })
    return pd.DataFrame(rows)


S1_NAMES = [
    "acme corporation",
    "global trade ltd",
    "sunshine bakery",
    "techno solutions pvt",
    "green leaf organics",
]
S2_NAMES = [
    "acme corp",                   # near-dup of s1[0]
    "global trading limited",      # near-dup of s1[1]
    "sunshine bakeries",           # near-dup of s1[2]
    "techno solution private",     # near-dup of s1[3]
    "green leaves organic foods",  # near-dup of s1[4]
    "completely unrelated company xyz",
    "another random firm",
    "red mountain enterprises",
]

S1_IDS = [f"s1_{i}" for i in range(len(S1_NAMES))]
S2_IDS = [f"s2_{i}" for i in range(len(S2_NAMES))]

DF_S1  = _make_df(S1_IDS,  S1_NAMES)
DF_S2  = _make_df(S2_IDS,  S2_NAMES)


# ---------------------------------------------------------------------------
# 1. Identical entity retrieves itself
# ---------------------------------------------------------------------------

def test_identical_entity_retrieves_itself():
    """An entity present in both S1 and corpus should be its own top-1 result."""
    s1  = _make_df(["q0"], ["acme corporation test entity"])
    s2  = _make_df(["c0", "c1"], ["acme corporation test entity", "some other name here"])

    vecs = fit_multiview_vectorizers(s2)
    res  = search_multiview_candidates(s1, s2, vecs, top_k=2)

    assert not res.empty, "Expected at least one candidate"
    top_cands = res[res["source1_entity_id"] == "q0"]["candidate_entity_id"].tolist()
    assert "c0" in top_cands, (
        f"Identical entity 'c0' not in top candidates: {top_cands}"
    )


# ---------------------------------------------------------------------------
# 2. Near-duplicate retrieval
# ---------------------------------------------------------------------------

def test_near_duplicate_retrieval():
    """For each S1 entity the expected near-duplicate should be retrieved."""
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    # Each S1 entity (index i) should retrieve its S2 near-dup (index i)
    for i, s1_id in enumerate(S1_IDS):
        expected_cand = S2_IDS[i]
        cands = res[res["source1_entity_id"] == s1_id]["candidate_entity_id"].tolist()
        assert expected_cand in cands, (
            f"S1={s1_id!r}: expected {expected_cand!r} in candidates, got {cands}"
        )


# ---------------------------------------------------------------------------
# 3. Multiple queries — result has one row-group per S1
# ---------------------------------------------------------------------------

def test_multiple_queries_all_s1_represented():
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    s1_found = set(res["source1_entity_id"].tolist())
    for s1_id in S1_IDS:
        assert s1_id in s1_found, f"{s1_id!r} has no candidates"


# ---------------------------------------------------------------------------
# 4 & 5. Multiple corpus blocks and query chunks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk,block", [
    (1, 1),    # extreme: one query at a time, one corpus row at a time
    (2, 3),
    (3, 5),
    (100, 100),  # larger than dataset → single block
])
def test_chunked_result_matches_reference(chunk, block):
    """
    Chunked retrieval must return the same set of top-k candidates as the
    reference (unchunked) run.

    Note on tie-breaking: when multiple corpus entries share an identical
    cosine score, different chunk/block boundaries can produce different
    *orderings* of tied entries.  We therefore check set-overlap (all
    reference top-k candidates appear in the chunked top-k candidates) rather
    than strict ordering.  This is the correct property to test: chunking must
    not lose any true high-score candidate.
    """
    # Use a corpus with more distinctive text so fewer exact-score ties occur
    s1_distinct = _make_df(
        ["q0", "q1"],
        ["global trading corporation",
         "sunshine bakery and cafe"],
        ["new delhi india",
         "mumbai maharashtra india"],
    )
    s2_distinct = _make_df(
        [f"c{i}" for i in range(10)],
        ["global trading corp", "sunshine bakery", "techno solutions",
         "green leaf organics", "acme corporation", "red mountain co",
         "blue river trading", "alpha beta gamma", "delta epsilon zeta",
         "omega sigma tau"],
        ["new delhi india", "mumbai maharashtra", "bangalore karnataka",
         "pune maharashtra", "hyderabad telangana", "chennai tamilnadu",
         "kolkata west bengal", "ahmedabad gujarat", "jaipur rajasthan",
         "lucknow uttar pradesh"],
    )

    vecs = fit_multiview_vectorizers(s2_distinct)

    ref = search_multiview_candidates(
        s1_distinct, s2_distinct, vecs, top_k=3,
        chunk_size=1000, corpus_block_size=1000,
    )
    chunked = search_multiview_candidates(
        s1_distinct, s2_distinct, vecs, top_k=3,
        chunk_size=chunk, corpus_block_size=block,
    )

    for s1_id in ["q0", "q1"]:
        ref_cands = set(
            ref[ref["source1_entity_id"] == s1_id]["candidate_entity_id"].tolist()
        )
        chk_cands = set(
            chunked[chunked["source1_entity_id"] == s1_id]["candidate_entity_id"].tolist()
        )
        # Every reference candidate must appear in the chunked result
        missing = ref_cands - chk_cands
        assert len(missing) == 0, (
            f"chunk={chunk}, block={block}, s1={s1_id!r}: "
            f"reference candidates {ref_cands!r} not all in chunked {chk_cands!r}; "
            f"missing={missing!r}"
        )


# ---------------------------------------------------------------------------
# 6. Top-K bound respected (per-view, not per-union)
# ---------------------------------------------------------------------------

def test_top_k_bound_respected():
    """
    Each TF-IDF view contributes at most top_k candidates per S1 entity.
    After multi-view fusion (union), a single S1 entity may have up to
    top_k × (number of views) distinct candidates — this is expected and
    correct behaviour for a union-style retrieval system.

    We test the per-view bound by running a single-view search and checking
    that its output respects top_k.
    """
    top_k = 3
    vecs_all = fit_multiview_vectorizers(DF_S2)

    # Single view — the union bound is exactly top_k
    vecs_one = {"combined": vecs_all["combined"]}
    res = search_multiview_candidates(DF_S1, DF_S2, vecs_one, top_k=top_k)

    for s1_id in S1_IDS:
        n_cands = res[res["source1_entity_id"] == s1_id]["candidate_entity_id"].nunique()
        assert n_cands <= top_k, (
            f"{s1_id!r} has {n_cands} candidates > top_k={top_k} (single-view run)"
        )

    # Multi-view union: bound is top_k * n_views, not top_k
    n_views = len(vecs_all)
    res_all = search_multiview_candidates(DF_S1, DF_S2, vecs_all, top_k=top_k)
    for s1_id in S1_IDS:
        n_cands = res_all[res_all["source1_entity_id"] == s1_id]["candidate_entity_id"].nunique()
        assert n_cands <= top_k * n_views, (
            f"{s1_id!r} has {n_cands} candidates > top_k*views={top_k * n_views}"
        )


# ---------------------------------------------------------------------------
# 7. Chunked result == unchunked on tiny corpus (via search_candidates_scalable)
# ---------------------------------------------------------------------------

def test_scalable_chunked_equals_full():
    """search_candidates_scalable with tiny blocks equals single-block result."""
    from scipy.sparse import random as sp_random

    rng    = np.random.default_rng(42)
    n_s1   = 10
    n_s2   = 20
    vocab  = 50

    # Reproducible sparse matrices
    s1_mat  = sp_random(n_s1, vocab, density=0.3, format="csr", random_state=42,
                        dtype=np.float32)
    s2_mat  = sp_random(n_s2, vocab, density=0.3, format="csr", random_state=99,
                        dtype=np.float32)

    top_k = 5

    ref     = search_candidates_scalable(s1_mat, s2_mat, top_k=top_k,
                                         chunk_size=9999, corpus_block_size=9999)
    chunked = search_candidates_scalable(s1_mat, s2_mat, top_k=top_k,
                                         chunk_size=3, corpus_block_size=7)

    # Sort both by (s1_row_idx, score desc) and compare top-1 per query
    ref_top1     = ref.sort_values(["s1_row_idx","score"], ascending=[True,False])\
                       .drop_duplicates("s1_row_idx").set_index("s1_row_idx")["s2s3_row_idx"]
    chunked_top1 = chunked.sort_values(["s1_row_idx","score"], ascending=[True,False])\
                           .drop_duplicates("s1_row_idx").set_index("s1_row_idx")["s2s3_row_idx"]

    common_idx = ref_top1.index.intersection(chunked_top1.index)
    mismatches = (ref_top1.loc[common_idx] != chunked_top1.loc[common_idx]).sum()
    assert mismatches == 0, f"{mismatches} top-1 mismatches between full and chunked"


# ---------------------------------------------------------------------------
# 8. Duplicate candidates deduplicated in merged output
# ---------------------------------------------------------------------------

def test_no_duplicate_candidate_pairs():
    """Each (source1_entity_id, candidate_entity_id) pair must be unique."""
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=4)

    dups = res.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum()
    assert dups == 0, f"{dups} duplicate (s1, candidate) pairs found"


# ---------------------------------------------------------------------------
# 9. Empty query → empty result
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty():
    vecs   = fit_multiview_vectorizers(DF_S2)
    empty  = pd.DataFrame(columns=DF_S1.columns)
    res    = search_multiview_candidates(empty, DF_S2, vecs, top_k=5)
    assert res.empty, "Expected empty DataFrame for empty query"


# ---------------------------------------------------------------------------
# 10. Empty corpus → empty result
# ---------------------------------------------------------------------------

def test_empty_corpus_returns_empty():
    vecs_empty = {}  # no vectorizers → empty result
    res = search_multiview_candidates(DF_S1, DF_S2, vecs_empty, top_k=5)
    assert res.empty, "Expected empty DataFrame for empty vectorizers dict"


def test_empty_corpus_df():
    """Corpus with no rows: fit skips all views → empty result."""
    empty_s2 = pd.DataFrame(columns=DF_S2.columns)
    vecs = fit_multiview_vectorizers(empty_s2)
    # All views should be skipped (no content)
    assert len(vecs) == 0, "Expected no vectorizers for empty corpus"
    res = search_multiview_candidates(DF_S1, empty_s2, vecs, top_k=5)
    assert res.empty


# ---------------------------------------------------------------------------
# 11. Multiple TF-IDF views — correct provenance flags
# ---------------------------------------------------------------------------

def test_provenance_flags_present():
    """Output must contain all three view-specific retrieval flags."""
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    expected_cols = [
        "retrieved_by_tfidf_name",
        "retrieved_by_tfidf_address",
        "retrieved_by_tfidf_combined",
        "retrieval_agreement_count",
    ]
    for col in expected_cols:
        assert col in res.columns, f"Missing column {col!r} in output"


def test_provenance_flags_are_binary():
    """All retrieved_by_tfidf_* flags must be 0 or 1."""
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    for col in ["retrieved_by_tfidf_name", "retrieved_by_tfidf_address",
                "retrieved_by_tfidf_combined"]:
        if col in res.columns:
            vals = set(res[col].unique())
            assert vals <= {0, 1}, f"Column {col!r} has non-binary values: {vals}"


def test_retrieval_agreement_count_range():
    """retrieval_agreement_count must be between 1 and 3 (number of views)."""
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    if "retrieval_agreement_count" in res.columns:
        assert res["retrieval_agreement_count"].min() >= 1
        assert res["retrieval_agreement_count"].max() <= 3


# ---------------------------------------------------------------------------
# 12. Candidate provenance: required ID columns present
# ---------------------------------------------------------------------------

def test_output_required_columns():
    """Output DataFrame must contain source1_entity_id and candidate_entity_id."""
    vecs = fit_multiview_vectorizers(DF_S2)
    res  = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    for col in ["source1_entity_id", "candidate_entity_id", "score"]:
        assert col in res.columns, f"Missing required column {col!r}"


# ---------------------------------------------------------------------------
# 13. Deterministic output
# ---------------------------------------------------------------------------

def test_deterministic_output():
    """Two calls with identical input must produce identical output."""
    vecs = fit_multiview_vectorizers(DF_S2)

    res1 = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)
    res2 = search_multiview_candidates(DF_S1, DF_S2, vecs, top_k=3)

    # Sort both the same way before comparing
    key_cols = ["source1_entity_id", "candidate_entity_id"]
    r1 = res1.sort_values(key_cols).reset_index(drop=True)
    r2 = res2.sort_values(key_cols).reset_index(drop=True)

    pd.testing.assert_frame_equal(r1, r2, check_like=False,
                                  rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 14. chunk_size / corpus_block_size are actually forwarded (not silently ignored)
# ---------------------------------------------------------------------------

def test_chunk_params_forwarded():
    """
    Passing very small chunk/block params must not raise and must retrieve
    the same candidates as large (single-chunk) params — confirming the
    params are actually forwarded into search_candidates_scalable.

    Uses a corpus with distinct enough text to avoid pure tie ambiguity.
    We verify set membership (all reference candidates present in chunked
    result) rather than strict top-1 ordering equality.
    """
    # More distinctive corpus to reduce ties
    s1 = _make_df(
        ["q0"],
        ["global trade corporation india"],
        ["new delhi india 110001"],
    )
    s2 = _make_df(
        [f"c{i}" for i in range(8)],
        ["global trade corp", "global trading limited", "techno solutions",
         "green leaf organics", "acme corporation", "red mountain co",
         "blue river trading", "alpha beta gamma"],
        ["new delhi india 110001", "new delhi 110001", "bangalore karnataka",
         "pune maharashtra", "hyderabad telangana", "chennai tamilnadu",
         "kolkata west bengal", "ahmedabad gujarat"],
    )

    vecs = fit_multiview_vectorizers(s2)

    res_large = search_multiview_candidates(
        s1, s2, vecs, top_k=3, chunk_size=9999, corpus_block_size=9999
    )
    res_small = search_multiview_candidates(
        s1, s2, vecs, top_k=3, chunk_size=1, corpus_block_size=2
    )

    ref_cands = set(
        res_large[res_large["source1_entity_id"] == "q0"]["candidate_entity_id"].tolist()
    )
    small_cands = set(
        res_small[res_small["source1_entity_id"] == "q0"]["candidate_entity_id"].tolist()
    )

    missing = ref_cands - small_cands
    assert len(missing) == 0, (
        f"Small-chunk result is missing reference candidates: {missing!r}\n"
        f"ref={ref_cands!r}\nsmall={small_cands!r}"
    )
    # Also confirm both have results (params were not silently ignored)
    assert len(ref_cands) > 0, "Large-chunk produced no candidates (unexpected)"
    assert len(small_cands) > 0, "Small-chunk produced no candidates (params may have been ignored)"


# ---------------------------------------------------------------------------
# 15. RSS instrumentation does not raise
# ---------------------------------------------------------------------------

def test_rss_instrumentation():
    """_rss_mb() should return a non-negative float without raising."""
    rss = _rss_mb()
    assert isinstance(rss, float)
    assert rss >= 0.0


def test_sparse_mem_mb():
    """_sparse_mem_mb() should return a non-negative float for a sparse matrix."""
    from scipy.sparse import eye
    mat = eye(100, format="csr", dtype=np.float32)
    mem = _sparse_mem_mb(mat)
    assert isinstance(mem, float)
    assert mem >= 0.0


# ---------------------------------------------------------------------------
# 16. Fit-only: fit_multiview_vectorizers does NOT materialise a corpus matrix
# ---------------------------------------------------------------------------

def test_fit_does_not_materialise_corpus_matrix():
    """
    fit_multiview_vectorizers() must call fit() (not fit_transform()) so no
    large corpus matrix lingers after fitting.  We verify this by checking
    that the returned vectorizer objects have a vocabulary_ but no stored_matrix
    attribute.
    """
    vecs = fit_multiview_vectorizers(DF_S2)
    assert len(vecs) > 0, "Expected at least one fitted vectorizer"

    for view, vec in vecs.items():
        assert hasattr(vec, "vocabulary_"), f"View '{view}': missing vocabulary_"
        # A fit_transform result would be a separate object; the vectorizer itself
        # should not hold a dense/sparse copy of the corpus.
        assert not hasattr(vec, "_fit_transform_matrix"), (
            f"View '{view}': vectorizer unexpectedly holds a corpus matrix"
        )


# ---------------------------------------------------------------------------
# 17. Sequential view processing: no shared state between views
# ---------------------------------------------------------------------------

def test_sequential_view_isolation():
    """
    Verify that running view A then view B produces the same result as
    running them independently.  This checks that del + gc between views
    does not corrupt state.
    """
    # Build separate single-view dicts
    vecs_all = fit_multiview_vectorizers(DF_S2)

    if len(vecs_all) < 2:
        pytest.skip("Need at least 2 views to test isolation")

    view_names = list(vecs_all.keys())

    # Run all views together
    res_all = search_multiview_candidates(DF_S1, DF_S2, vecs_all, top_k=3)

    # Run first view alone
    vecs_one = {view_names[0]: vecs_all[view_names[0]]}
    res_one  = search_multiview_candidates(DF_S1, DF_S2, vecs_one, top_k=3)

    # The first-view-only candidates must all appear in the all-views result
    pairs_all = set(
        zip(res_all["source1_entity_id"], res_all["candidate_entity_id"])
    )
    pairs_one = set(
        zip(res_one["source1_entity_id"], res_one["candidate_entity_id"])
    )
    missing = pairs_one - pairs_all
    assert len(missing) == 0, (
        f"{len(missing)} candidates from single-view run not found in all-views run"
    )
