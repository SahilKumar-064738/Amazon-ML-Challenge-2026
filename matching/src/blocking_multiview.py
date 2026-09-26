"""
src/blocking_multiview.py
--------------------------
Multi-view TF-IDF blocking: fits separate vectorizers for name, address, and
combined text views, then fuses the per-view top-k candidate lists.

Robustness rules
----------------
* If a view column is absent or all-empty the view falls back to clean_text.
* If even clean_text is all-empty for a view the view is silently skipped.
* Each vectorizer is fitted with min_df=1 so single-document corpora work.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.blocking import search_candidates_scalable


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_multiview_vectorizers(s2s3_df: pd.DataFrame) -> dict:
    """Fit one TF-IDF vectorizer per view on the S2+S3 corpus.

    Returns a dict keyed by view name ("name", "address", "combined").
    Views whose corpus texts are entirely empty are omitted from the dict.
    """
    view_configs = {
        "name":     TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4),
            max_features=150_000, sublinear_tf=True, min_df=1,
        ),
        "address":  TfidfVectorizer(
            analyzer="word", ngram_range=(1, 1),
            max_features=100_000, sublinear_tf=True, min_df=1,
        ),
        "combined": TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5),
            max_features=250_000, sublinear_tf=True, min_df=1,
        ),
    }

    fitted: dict = {}
    for view, vec in view_configs.items():
        texts = _effective_texts(s2s3_df, view)
        if not _has_content(texts):
            print(f"    [multiview] '{view}' view skipped — corpus is all-empty.")
            continue
        try:
            fitted[view] = vec.fit(texts)
        except ValueError as exc:
            # e.g. empty vocabulary after stop-word filtering
            print(f"    [multiview] '{view}' view skipped — {exc}")

    return fitted


def search_multiview_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    vectorizers: dict,
    top_k: int = 30,
) -> pd.DataFrame:
    """Retrieve top-k candidates per S1 entity across all fitted views.

    Returns a DataFrame with columns:
        source1_entity_id, candidate_entity_id, score,
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

    results = []
    view_rename = {
        "name":     "retrieved_by_tfidf_name",
        "address":  "retrieved_by_tfidf_address",
        "combined": "retrieved_by_tfidf_combined",
    }

    for view, vec in vectorizers.items():
        s1_texts   = _effective_texts(s1_df,   view)
        s2s3_texts = _effective_texts(s2s3_df, view)

        if not _has_content(s1_texts) or not _has_content(s2s3_texts):
            continue

        s1_mat   = vec.transform(s1_texts)
        s2s3_mat = vec.transform(s2s3_texts)

        res = search_candidates_scalable(s1_mat, s2s3_mat, top_k=top_k)

        res["source1_entity_id"]   = res["s1_row_idx"].map(lambda i: s1_ids[i])
        res["candidate_entity_id"] = res["s2s3_row_idx"].map(lambda j: s2s3_ids[j])
        res["view"] = view
        results.append(
            res[["source1_entity_id", "candidate_entity_id", "score", "rank", "view"]]
        )

    if not results:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "score"]
        )

    merged_df = pd.concat(results, ignore_index=True)

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

    # Retrieval agreement count
    flag_cols = list(view_rename.values())
    existing_flags = [c for c in flag_cols if c in final_df.columns]
    final_df["retrieval_agreement_count"] = final_df[existing_flags].sum(axis=1)

    return final_df
