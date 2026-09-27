"""
src/blocking_multiview.py
--------------------------
Multi-view TF-IDF blocking: fits separate vectorizers for name, address, and
combined text views, then fuses the per-view top-k candidate lists.

Phase 5 — Memory-Safety Redesign
---------------------------------
Root-cause of previous OOM (~130 GB on 128 GB machine):
  search_multiview_candidates() iterated over three views in a for-loop
  without explicitly deleting the previous view's corpus/query matrices.
  Python's GC is lazy; the old matrices could still be resident while the
  next view's matrices were being allocated, causing up to 3× corpus memory
  peak simultaneously.  Additionally, chunk_size and corpus_block_size were
  never forwarded from the CLI, so the hardcoded defaults were always used
  regardless of what the user specified.

Fix:
  1. Process views SEQUENTIALLY with an explicit ``del s1_mat, s2s3_mat``
     and ``gc.collect()`` after each view's search completes.
  2. Accept and forward ``chunk_size`` and ``corpus_block_size`` parameters
     all the way down to ``search_candidates_scalable()``.
  3. Emit RSS memory measurements at key lifecycle points via
     ``_log_rss()`` so memory scaling can be observed without external tools.
  4. Explicitly release vectorizer vocabulary references when no longer
     needed (optional, bounded upside but free).

Robustness rules (unchanged from Phase ≤4)
-------------------------------------------
* If a view column is absent or all-empty the view falls back to clean_text.
* If even clean_text is all-empty for a view the view is silently skipped.
* Each vectorizer is fitted with min_df=1 so single-document corpora work.
"""

from __future__ import annotations

import gc
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking import search_candidates_scalable

# ---------------------------------------------------------------------------
# Memory instrumentation — optional, zero-dep fallback
# ---------------------------------------------------------------------------

def _rss_mb() -> float:
    """Return current process RSS in MB.  Returns 0.0 if psutil is unavailable."""
    try:
        import psutil, os
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return 0.0


def _log_rss(tag: str, view: str = "") -> None:
    """Print a structured RSS checkpoint line."""
    rss = _rss_mb()
    view_str = f" view={view!r}" if view else ""
    ts = time.strftime("%H:%M:%S")
    print(f"  [MEM {ts}]{view_str} {tag}: RSS={rss:.0f} MB")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_texts(df: pd.DataFrame, col: str) -> pd.Series:
    """Return non-null, stripped texts from *col* if it exists, else ''."""
    if col not in df.columns:
        return pd.Series([""] * len(df), dtype=str)
    return df[col].fillna("").astype(str).str.strip()


def _effective_texts(df: pd.DataFrame, view: str) -> pd.Series:
    """
    Return the text series to use for *view*, falling back to clean_text
    if the primary column is missing or all-blank.
    """
    primary_col = {"name": "clean_name", "address": "clean_address"}.get(view, "clean_text")
    texts = _col_texts(df, primary_col)

    # If primary column is all-blank fall back to clean_text
    if texts.str.len().sum() == 0:
        texts = _col_texts(df, "clean_text")

    return texts


def _has_content(texts: pd.Series) -> bool:
    """True if at least one non-empty string exists in *texts*."""
    return texts.str.len().sum() > 0


def _sparse_mem_mb(mat) -> float:
    """Estimate memory of a scipy sparse matrix in MB (data + indices + indptr)."""
    try:
        return (mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes) / (1024 ** 2)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# View configuration
# ---------------------------------------------------------------------------

# Each view: (analyzer, ngram_range, max_features)
# Using float32 dtype explicitly to halve IDF/data storage vs default float64.
_VIEW_CONFIGS: dict[str, dict] = {
    "name": dict(
        analyzer="char_wb",
        ngram_range=(3, 4),
        max_features=150_000,
        sublinear_tf=True,
        min_df=1,
        dtype=__import__("numpy").float32,
    ),
    "address": dict(
        analyzer="word",
        ngram_range=(1, 1),
        max_features=100_000,
        sublinear_tf=True,
        min_df=1,
        dtype=__import__("numpy").float32,
    ),
    "combined": dict(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=250_000,
        sublinear_tf=True,
        min_df=1,
        dtype=__import__("numpy").float32,
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_multiview_vectorizers(s2s3_df: pd.DataFrame) -> dict:
    """Fit one TF-IDF vectorizer per view on the S2+S3 corpus.

    Returns a dict keyed by view name ("name", "address", "combined").
    Views whose corpus texts are entirely empty are omitted from the dict.

    Memory note
    -----------
    ``fit()`` (not ``fit_transform()``) is used so that NO corpus matrix is
    materialised here.  The fitted vectorizer stores only the vocabulary
    (int→str mapping) and IDF weights vector — both are small compared to
    the full sparse corpus matrix.  The corpus matrix is created and
    immediately discarded inside ``search_multiview_candidates()``.
    """
    _log_rss("BEFORE_FIT_ALL_VIEWS")
    fitted: dict = {}
    for view, cfg in _VIEW_CONFIGS.items():
        texts = _effective_texts(s2s3_df, view)
        if not _has_content(texts):
            print(f"    [multiview] '{view}' view skipped — corpus is all-empty.")
            continue
        try:
            vec = TfidfVectorizer(**cfg)
            fitted[view] = vec.fit(texts)
            vocab_size = len(vec.vocabulary_)
            print(
                f"    [multiview] Fitted '{view}' view: "
                f"vocab={vocab_size:,}  analyzer={cfg['analyzer']}  "
                f"ngram={cfg['ngram_range']}  max_feat={cfg['max_features']:,}"
            )
        except ValueError as exc:
            # e.g. empty vocabulary after stop-word filtering
            print(f"    [multiview] '{view}' view skipped — {exc}")
    _log_rss("AFTER_FIT_ALL_VIEWS")
    return fitted


def search_multiview_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    vectorizers: dict,
    top_k: int = 30,
    chunk_size: int = 20_000,
    corpus_block_size: int = 250_000,
) -> pd.DataFrame:
    """Retrieve top-k candidates per S1 entity across all fitted views.

    Phase 5 memory-safe implementation
    ------------------------------------
    Views are processed SEQUENTIALLY.  After each view's retrieval:

    1. ``del s1_mat, s2s3_mat`` — drops the two largest objects.
    2. ``gc.collect()``         — forces CPython to reclaim any cycles before
                                  the next view's ``transform()`` allocates.
    3. ``_log_rss()``           — records RSS so scaling is observable.

    This prevents the previous failure mode where all three corpus matrices
    (≈8–25 GB total) were simultaneously resident during the third view's
    allocation.

    Parameters
    ----------
    chunk_size : int
        S1 query chunk size forwarded to ``search_candidates_scalable()``.
        Governs the height of the intermediate similarity block:
        ``chunk_size × corpus_block_size``.  Smaller → less peak memory
        per block, more iterations.  Default 20,000.
    corpus_block_size : int
        S2+S3 corpus block size forwarded to ``search_candidates_scalable()``.
        Governs the width of the intermediate similarity block.
        Default 250,000.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, score,
                 retrieved_by_tfidf_name (0/1),
                 retrieved_by_tfidf_address (0/1),
                 retrieved_by_tfidf_combined (0/1),
                 retrieval_agreement_count
    """
    if not vectorizers:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "score"]
        )

    s1_ids   = s1_df["entity_id"].tolist()
    s2s3_ids = s2s3_df["entity_id"].tolist()

    view_rename = {
        "name":     "retrieved_by_tfidf_name",
        "address":  "retrieved_by_tfidf_address",
        "combined": "retrieved_by_tfidf_combined",
    }

    results: list[pd.DataFrame] = []

    _log_rss("BEFORE_MULTIVIEW_LOOP")

    # ── Sequential view processing ──────────────────────────────────────────
    # CRITICAL: process one view at a time and explicitly release matrices
    # before the next view begins.  This is the primary OOM fix.
    for view, vec in vectorizers.items():
        s1_texts   = _effective_texts(s1_df,   view)
        s2s3_texts = _effective_texts(s2s3_df, view)

        if not _has_content(s1_texts) or not _has_content(s2s3_texts):
            print(f"    [multiview] '{view}' view skipped at search time — texts empty.")
            continue

        _log_rss("BEFORE_TRANSFORM", view)

        # ── Transform: query matrix ──
        s1_mat = vec.transform(s1_texts)
        _log_rss(
            f"AFTER_QUERY_TRANSFORM  s1_mat={s1_mat.shape}  "
            f"nnz={s1_mat.nnz:,}  mem={_sparse_mem_mb(s1_mat):.0f} MB",
            view,
        )

        # ── Transform: corpus matrix ──
        s2s3_mat = vec.transform(s2s3_texts)
        _log_rss(
            f"AFTER_CORPUS_TRANSFORM s2s3_mat={s2s3_mat.shape}  "
            f"nnz={s2s3_mat.nnz:,}  mem={_sparse_mem_mb(s2s3_mat):.0f} MB",
            view,
        )

        # ── 2D-chunked Top-K search (similarity never fully materialised) ──
        _log_rss("BEFORE_SIMILARITY", view)
        res = search_candidates_scalable(
            s1_mat,
            s2s3_mat,
            top_k=top_k,
            chunk_size=chunk_size,
            corpus_block_size=corpus_block_size,
        )
        _log_rss("AFTER_TOPK", view)

        # ── CRITICAL: release view matrices BEFORE next iteration ──────────
        del s1_mat, s2s3_mat
        gc.collect()
        _log_rss("AFTER_RELEASE_AND_GC", view)

        # ── Map row indices → entity IDs ────────────────────────────────────
        if not res.empty:
            res = res.copy()
            res["source1_entity_id"]   = res["s1_row_idx"].map(lambda i: s1_ids[i])
            res["candidate_entity_id"] = res["s2s3_row_idx"].map(lambda j: s2s3_ids[j])
            res["view"] = view
            results.append(
                res[["source1_entity_id", "candidate_entity_id", "score", "rank", "view"]]
            )
            del res

    _log_rss("AFTER_MULTIVIEW_LOOP")

    if not results:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "score"]
        )

    merged_df = pd.concat(results, ignore_index=True)
    del results
    gc.collect()

    # ── Pivot to one row per (s1, candidate) with per-view retrieval flags ──
    merged_df["retrieved"] = 1
    pivoted = (
        merged_df
        .pivot_table(
            index=["source1_entity_id", "candidate_entity_id"],
            columns="view",
            values="retrieved",
            fill_value=0,
        )
        .reset_index()
    )
    pivoted.columns.name = None

    # Rename view columns to canonical flag names; add missing flags as 0
    for view, col_name in view_rename.items():
        if view in pivoted.columns:
            pivoted.rename(columns={view: col_name}, inplace=True)
        else:
            pivoted[col_name] = 0

    # Max score across views
    max_scores = (
        merged_df
        .groupby(["source1_entity_id", "candidate_entity_id"])["score"]
        .max()
        .reset_index()
    )

    final_df = pivoted.merge(max_scores, on=["source1_entity_id", "candidate_entity_id"])

    del merged_df, pivoted, max_scores

    # Retrieval agreement count
    flag_cols = list(view_rename.values())
    existing_flags = [c for c in flag_cols if c in final_df.columns]
    final_df["retrieval_agreement_count"] = final_df[existing_flags].sum(axis=1)

    _log_rss("AFTER_FUSION")
    print(
        f"    [multiview] Total fused candidates: {len(final_df):,}  "
        f"unique S1: {final_df['source1_entity_id'].nunique():,}"
    )
    return final_df
