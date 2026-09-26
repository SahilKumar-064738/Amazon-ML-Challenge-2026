"""blocking_countsketch.py
-------------------------
Phase 5 — CountSketch Compressed Sparse Retrieval & Exact Re-scoring.

Adapted and generalized from Sid's retrieval architecture (Sid-techweb/AmazonML-New)
with strict memory bounding, 2D chunking, and CPU compatibility.

Architecture:
  1. Multi-view TF-IDF:
     - Name view: character n-grams (3-grams, space-free core representation).
     - Address view: word tokens.
  2. CountSketch Random Projection:
     - Projects high-dimensional sparse TF-IDF spaces (100k+ features) down to
       a compact sketch dimension (default: 1024) using a signed random hash matrix.
     - L2-normalized composite sketch:
       S = sqrt(w_name) * (Xn @ Pn) + sqrt(1 - w_name) * (Xa @ Pa)
  3. Bounded 2D Chunked Approximate Top-K Retrieval:
     - Iterates over query chunks (S1) and corpus blocks (S2/S3).
     - Never materializes full query x corpus dense matrices.
     - Maintains running Top-K approximate neighbors per query (default: sketch_top_k=40).
  4. Exact Sparse Re-scoring:
     - Computes exact row-wise cosine similarity on the original uncompressed sparse vectors
       only for the shortlist of candidate pairs.
     - Combined exact score:
       score = w_name * cos_name + (1 - w_name) * cos_addr
     - Re-ranks and retains top rerank_top_k candidates (default: 10) per S1 query.
  5. Provenance & Fusion Compatibility:
     - Returns DataFrame with [source1_entity_id, candidate_entity_id, score, rank, retrieved_by_countsketch].
     - Fully compatible with fuse_candidates() in src.blocking_fusion.
"""

from __future__ import annotations

import gc
import math
import os
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------

DEFAULT_SKETCH_DIM: int = 1024
DEFAULT_W_NAME: float = 0.5
DEFAULT_SKETCH_TOP_K: int = 40
DEFAULT_RERANK_TOP_K: int = 10
DEFAULT_QUERY_CHUNK_SIZE: int = 5_000
DEFAULT_CORPUS_BLOCK_SIZE: int = 50_000
DEFAULT_RANDOM_SEED: int = 42

_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _strip_spaces(text: Any) -> str:
    """Coerce value to string and remove all whitespace."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = str(text).strip()
    return _RE_WHITESPACE.sub("", s)


def _safe_str(text: Any) -> str:
    """Coerce value to string and strip leading/trailing whitespace."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    return str(text).strip()


def build_sketch_matrix(n_features: int, dim: int, seed: int = DEFAULT_RANDOM_SEED) -> sp.csr_matrix:
    """Generate a CountSketch sparse projection matrix of shape (n_features, dim).

    Each row has exactly 1 non-zero entry at a uniformly random column in [0, dim - 1]
    with random sign +1.0 or -1.0.

    Parameters
    ----------
    n_features : int
        Number of original sparse input features.
    dim : int
        Target sketch projection dimension.
    seed : int
        RNG seed for determinism.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse projection matrix of shape (n_features, dim) and dtype float32.
    """
    if n_features <= 0 or dim <= 0:
        return sp.csr_matrix((max(0, n_features), max(0, dim)), dtype=np.float32)

    rng = np.random.default_rng(seed)
    cols = rng.integers(0, dim, size=n_features, dtype=np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_features)
    rows = np.arange(n_features, dtype=np.int32)
    return sp.csr_matrix((signs, (rows, cols)), shape=(n_features, dim), dtype=np.float32)


def rowwise_cosine(
    A: sp.csr_matrix,
    B: sp.csr_matrix,
    ia: np.ndarray,
    ib: np.ndarray,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Compute exact cosine similarity for aligned row pairs (A[ia[i]], B[ib[i]]).

    Assumes rows of A and B are L2-normalized (standard TfidfVectorizer output).
    Memory bounded via chunking.

    Parameters
    ----------
    A : sp.csr_matrix
        First sparse matrix.
    B : sp.csr_matrix
        Second sparse matrix.
    ia : np.ndarray
        Array of row indices into A.
    ib : np.ndarray
        Array of row indices into B.
    chunk_size : int
        Batch size for computing pairwise inner products.

    Returns
    -------
    np.ndarray
        1D float32 array of cosine similarities bounded in [0.0, 1.0].
    """
    n = len(ia)
    if n == 0:
        return np.empty(0, dtype=np.float32)

    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        sub_ia = ia[s:e]
        sub_ib = ib[s:e]
        x = A[sub_ia]
        y = B[sub_ib]
        prod = x.multiply(y).sum(axis=1)
        res = np.asarray(prod).ravel().astype(np.float32)
        # Numerical guard: clamp [0.0, 1.0]
        np.clip(res, 0.0, 1.0, out=res)
        out[s:e] = res
    return out


# ---------------------------------------------------------------------------
# CountSketch Vectorizer & Index
# ---------------------------------------------------------------------------

class CountSketchVectorizer:
    """Fitted TF-IDF and CountSketch projection matrices for name and address views."""

    def __init__(
        self,
        dim: int = DEFAULT_SKETCH_DIM,
        w_name: float = DEFAULT_W_NAME,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> None:
        self.dim = dim
        self.w_name = w_name
        self.seed = seed
        self.name_vec: TfidfVectorizer | None = None
        self.addr_vec: TfidfVectorizer | None = None
        self.Pn: sp.csr_matrix | None = None
        self.Pa: sp.csr_matrix | None = None
        self.is_fitted: bool = False

    def fit(self, corpus_df: pd.DataFrame) -> "CountSketchVectorizer":
        """Fit name and address TF-IDF vectorizers on searchable corpus (S2+S3).

        Parameters
        ----------
        corpus_df : pd.DataFrame
            Corpus containing 'clean_name' and 'clean_address' (or fallback to 'clean_text').

        Returns
        -------
        CountSketchVectorizer
            Self.
        """
        # Determine columns
        has_name = "clean_name" in corpus_df.columns
        has_addr = "clean_address" in corpus_df.columns

        name_texts = (
            [_strip_spaces(x) for x in corpus_df["clean_name"]]
            if has_name
            else [_strip_spaces(x) for x in corpus_df.get("clean_text", [""] * len(corpus_df))]
        )
        addr_texts = (
            [_safe_str(x) for x in corpus_df["clean_address"]]
            if has_addr
            else [_safe_str(x) for x in corpus_df.get("clean_text", [""] * len(corpus_df))]
        )

        # Pad very short names to ensure character n-grams can be generated
        name_texts = [f"<{t}>" if t else "" for t in name_texts]

        # Fit TF-IDF
        max_df_val = 0.20 if len(corpus_df) >= 20 else 1.0
        self.name_vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 3),
            min_df=1,
            max_df=max_df_val,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.addr_vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            min_df=1,
            max_df=max_df_val,
            sublinear_tf=True,
            dtype=np.float32,
        )

        Xn = self.name_vec.fit_transform(name_texts)
        Xa = self.addr_vec.fit_transform(addr_texts)

        # Build random projection matrices
        self.Pn = build_sketch_matrix(Xn.shape[1], self.dim, seed=self.seed)
        self.Pa = build_sketch_matrix(Xa.shape[1], self.dim, seed=self.seed + 1)
        self.is_fitted = True
        return self

    def transform_sparse(self, df: pd.DataFrame) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Transform text to raw uncompressed TF-IDF sparse matrices (Xn, Xa)."""
        if not self.is_fitted or self.name_vec is None or self.addr_vec is None:
            raise RuntimeError("CountSketchVectorizer must be fitted before transform.")

        has_name = "clean_name" in df.columns
        has_addr = "clean_address" in df.columns

        name_texts = (
            [_strip_spaces(x) for x in df["clean_name"]]
            if has_name
            else [_strip_spaces(x) for x in df.get("clean_text", [""] * len(df))]
        )
        addr_texts = (
            [_safe_str(x) for x in df["clean_address"]]
            if has_addr
            else [_safe_str(x) for x in df.get("clean_text", [""] * len(df))]
        )
        name_texts = [f"<{t}>" if t else "" for t in name_texts]

        Xn = self.name_vec.transform(name_texts)
        Xa = self.addr_vec.transform(addr_texts)
        return Xn, Xa

    def sketch_from_sparse(self, Xn: sp.csr_matrix, Xa: sp.csr_matrix) -> np.ndarray:
        """Project sparse TF-IDF matrices to a dense normalized CountSketch matrix.

        Returns float32 dense numpy array of shape (N, dim).
        """
        if self.Pn is None or self.Pa is None:
            raise RuntimeError("Projection matrices not initialized.")

        a = np.float32(np.sqrt(self.w_name))
        b = np.float32(np.sqrt(1.0 - self.w_name))

        # Sparse matrix multiplication into target sketch dimension
        Sn = (Xn @ self.Pn).toarray() * a
        Sa = (Xa @ self.Pa).toarray() * b

        combined = Sn + Sa  # dense float32 (N, dim)

        # L2-normalize sketches so dot product approximates cosine similarity
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        combined /= norms
        return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Core Retrieval Function (2D Chunked, Memory-Bounded)
# ---------------------------------------------------------------------------

def search_countsketch_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    vectorizer: CountSketchVectorizer | None = None,
    top_k: int = DEFAULT_RERANK_TOP_K,
    sketch_top_k: int = DEFAULT_SKETCH_TOP_K,
    dim: int = DEFAULT_SKETCH_DIM,
    w_name: float = DEFAULT_W_NAME,
    query_chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    corpus_block_size: int = DEFAULT_CORPUS_BLOCK_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Retrieve candidate matches using CountSketch random projection and exact re-scoring.

    Guarantees:
    - Bounded memory via 2D chunking (queries x corpus).
    - Top-k candidate bound per S1 query.
    - Zero NaNs, zero Infs.
    - Deterministic output.
    - Compatible with candidate fusion (retrieved_by_countsketch flag).

    Parameters
    ----------
    s1_df : pd.DataFrame
        Query DataFrame containing 'entity_id', and ('clean_name', 'clean_address') or 'clean_text'.
    s2s3_df : pd.DataFrame
        Searchable corpus DataFrame containing 'entity_id', and ('clean_name', 'clean_address') or 'clean_text'.
    vectorizer : CountSketchVectorizer, optional
        Pre-fitted vectorizer. If None, fits a new one on s2s3_df.
    top_k : int
        Final number of re-ranked candidates to retain per S1 query (default: 10).
    sketch_top_k : int
        Number of approximate candidates retrieved by CountSketch for exact re-scoring (default: 40).
    dim : int
        CountSketch dimension (default: 1024).
    w_name : float
        Weight for name similarity in [0.0, 1.0] (default: 0.5).
    query_chunk_size : int
        Number of S1 queries per chunk (default: 5,000).
    corpus_block_size : int
        Number of S2S3 corpus records per block (default: 50,000).
    seed : int
        RNG seed for sketch projections.

    Returns
    -------
    pd.DataFrame
        Candidate pair table with columns:
        ['source1_entity_id', 'candidate_entity_id', 'score', 'rank', 'retrieved_by_countsketch']
    """
    required_cols = {"entity_id"}
    if not required_cols.issubset(s1_df.columns):
        raise ValueError(f"s1_df missing required column(s): {required_cols - set(s1_df.columns)}")
    if not required_cols.issubset(s2s3_df.columns):
        raise ValueError(f"s2s3_df missing required column(s): {required_cols - set(s2s3_df.columns)}")

    result_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "score",
        "rank",
        "retrieved_by_countsketch",
    ]

    n_s1 = len(s1_df)
    n_s2s3 = len(s2s3_df)

    if n_s1 == 0 or n_s2s3 == 0:
        return pd.DataFrame(columns=result_cols)

    # Effective K limits
    eff_sketch_k = min(sketch_top_k, n_s2s3)
    eff_rerank_k = min(top_k, eff_sketch_k)

    if eff_rerank_k < 1:
        return pd.DataFrame(columns=result_cols)

    # 1. Fit or use vectorizer
    if vectorizer is None:
        vectorizer = CountSketchVectorizer(dim=dim, w_name=w_name, seed=seed)
        vectorizer.fit(s2s3_df)

    # 2. Extract full sparse representations
    s1_Xn, s1_Xa = vectorizer.transform_sparse(s1_df)
    s2s3_Xn, s2s3_Xa = vectorizer.transform_sparse(s2s3_df)

    s1_ids = [str(x) for x in s1_df["entity_id"]]
    s2s3_ids = [str(x) for x in s2s3_df["entity_id"]]

    out_s1_ids: list[str] = []
    out_cand_ids: list[str] = []
    out_scores: list[float] = []
    out_ranks: list[int] = []

    # 3. 2D Chunked Approximate Search
    for q_start in range(0, n_s1, query_chunk_size):
        q_end = min(q_start + query_chunk_size, n_s1)
        q_count = q_end - q_start

        # Sub-slice sparse queries and build sketch
        q_sub_Xn = s1_Xn[q_start:q_end]
        q_sub_Xa = s1_Xa[q_start:q_end]
        q_sketch = vectorizer.sketch_from_sparse(q_sub_Xn, q_sub_Xa)  # (q_count, dim)

        # Running approximate top-K buffer for this query chunk
        running_approx_scores = np.full((q_count, eff_sketch_k), -1.0, dtype=np.float32)
        running_approx_indices = np.full((q_count, eff_sketch_k), -1, dtype=np.int32)

        for c_start in range(0, n_s2s3, corpus_block_size):
            c_end = min(c_start + corpus_block_size, n_s2s3)
            c_count = c_end - c_start

            # Sub-slice sparse corpus block and build sketch
            c_sub_Xn = s2s3_Xn[c_start:c_end]
            c_sub_Xa = s2s3_Xa[c_start:c_end]
            c_sketch = vectorizer.sketch_from_sparse(c_sub_Xn, c_sub_Xa)  # (c_count, dim)

            # Dense approximate similarity product: (q_count, dim) @ (dim, c_count) -> (q_count, c_count)
            approx_sim = q_sketch @ c_sketch.T

            # Update running top-K for each query in chunk
            for local_q in range(q_count):
                row_sim = approx_sim[local_q]
                block_indices = np.arange(c_start, c_end, dtype=np.int32)

                # Merge with current running state
                merged_scores = np.concatenate((running_approx_scores[local_q], row_sim))
                merged_indices = np.concatenate((running_approx_indices[local_q], block_indices))

                if len(merged_scores) <= eff_sketch_k:
                    order = np.argsort(merged_scores)[::-1]
                else:
                    part = np.argpartition(merged_scores, -eff_sketch_k)[-eff_sketch_k:]
                    order = part[np.argsort(merged_scores[part])[::-1]]

                running_approx_scores[local_q] = merged_scores[order][:eff_sketch_k]
                running_approx_indices[local_q] = merged_indices[order][:eff_sketch_k]

            del approx_sim, c_sketch

        # 4. Exact Re-scoring for this query chunk
        for local_q in range(q_count):
            global_q_idx = q_start + local_q
            candidate_corpus_indices = running_approx_indices[local_q]

            # Filter valid indices
            valid_mask = candidate_corpus_indices >= 0
            if not np.any(valid_mask):
                continue

            valid_cand_indices = candidate_corpus_indices[valid_mask]
            # Deduplicate if duplicate corpus indices were accumulated
            valid_cand_indices = np.unique(valid_cand_indices)
            n_cands = len(valid_cand_indices)

            q_idx_array = np.full(n_cands, global_q_idx, dtype=np.int32)

            # Compute exact sparse cosines
            cn = rowwise_cosine(s1_Xn, s2s3_Xn, q_idx_array, valid_cand_indices)
            ca = rowwise_cosine(s1_Xa, s2s3_Xa, q_idx_array, valid_cand_indices)

            exact_scores = (vectorizer.w_name * cn) + ((1.0 - vectorizer.w_name) * ca)

            # Filter zero-score pairs if possible, but keep top matches
            positive_mask = exact_scores > 0.0
            if np.any(positive_mask):
                exact_scores = exact_scores[positive_mask]
                valid_cand_indices = valid_cand_indices[positive_mask]
                n_cands = len(valid_cand_indices)
            else:
                # If all exact similarities are 0, keep candidates from approx search with score 0.0
                pass

            # Select top rerank_top_k
            if n_cands <= eff_rerank_k:
                rerank_order = np.argsort(exact_scores)[::-1]
            else:
                part = np.argpartition(exact_scores, -eff_rerank_k)[-eff_rerank_k:]
                rerank_order = part[np.argsort(exact_scores[part])[::-1]]

            final_cand_indices = valid_cand_indices[rerank_order][:eff_rerank_k]
            final_scores = exact_scores[rerank_order][:eff_rerank_k]

            s1_id_val = s1_ids[global_q_idx]
            for rank_idx, (c_idx, sc) in enumerate(zip(final_cand_indices, final_scores), start=1):
                out_s1_ids.append(s1_id_val)
                out_cand_ids.append(s2s3_ids[c_idx])
                out_scores.append(float(sc))
                out_ranks.append(rank_idx)

        del q_sketch
        gc.collect()

    return pd.DataFrame({
        "source1_entity_id": out_s1_ids,
        "candidate_entity_id": out_cand_ids,
        "score": out_scores,
        "rank": out_ranks,
        "retrieved_by_countsketch": 1,
    })
