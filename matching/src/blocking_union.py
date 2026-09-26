"""
blocking_union.py
-----------------
P1-4: Three-channel TF-IDF UNION blocking.

Activates name-only and address-only TF-IDF retrieval channels in addition
to the existing combined name+address TF-IDF channel, then unions all three.

Channels
~~~~~~~~
A. Combined TF-IDF  (existing)  -- char 3-5 gram on clean_text
B. Name-only TF-IDF (new)       -- word 1-2 gram on clean_name
C. Address-only TF-IDF (new)    -- char_wb 3-4 gram on clean_address

All vectorizers are fitted on the S2+S3 corpus only (NEVER on S1/val).
All channels run sequentially to stay within the ~6 GB RAM budget —
no full matrix copies are kept simultaneously.

Public API
~~~~~~~~~~
    union_three_channel_candidates(
        s1_clean_df, s2s3_clean_df, s1_feature_df, s2s3_feature_df,
        vectorizer=None, top_k=20
    ) -> pd.DataFrame
        Returns a union DataFrame with provenance flags:
            retrieved_by_combined  (char TF-IDF on clean_text)
            retrieved_by_name      (word TF-IDF on clean_name)
            retrieved_by_address   (char_wb TF-IDF on clean_address)
            retrieval_agreement_count
        and the primary cosine score column from the combined channel.

    build_candidates_dict_from_union(union_df, all_s1_ids) -> dict
        Converts the union DataFrame to a {s1_id: [cand_ids]} dict.
        Every S1 ID in all_s1_ids is guaranteed to appear as a key (P0-3).

Memory budget
~~~~~~~~~~~~~
The three channels are computed and immediately discarded one at a time.
We never hold all three sparse similarity matrices in memory simultaneously.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_combined_results(
    s1_clean_df: pd.DataFrame,
    s2s3_clean_df: pd.DataFrame,
    vectorizer,
    top_k: int,
) -> pd.DataFrame:
    """Run the existing combined char-TF-IDF channel and return its results."""
    from src.blocking import build_clean_text, search_candidates_mock

    results = search_candidates_mock(
        s1_df=s1_clean_df,
        s2s3_df=s2s3_clean_df,
        vectorizer=vectorizer,
        top_k=top_k,
        same_country_filter=False,
    )
    # Rename to standard schema expected by union logic
    results = results.rename(columns={"cosine_similarity": "combined_score"})
    return results


def _add_provenance_flag(
    union_df: pd.DataFrame,
    channel_df: pd.DataFrame,
    flag_col: str,
    score_col: str,
    rank_col: str,
) -> pd.DataFrame:
    """Mark existing pairs already in union, add new-only pairs from channel.

    Parameters
    ----------
    union_df       : current accumulated union frame
    channel_df     : new channel results (source1_entity_id, candidate_entity_id,
                     <score_col>, <rank_col>)
    flag_col       : name of the boolean provenance column to add
    score_col      : score column name in channel_df
    rank_col       : rank column name in channel_df
    """
    if union_df is None:
        # First channel — bootstrap
        union_df = channel_df[
            ["source1_entity_id", "candidate_entity_id", score_col, rank_col]
        ].copy()
        union_df[flag_col] = 1
        return union_df

    # Add flag column defaulting to 0
    union_df = union_df.copy()
    union_df[flag_col] = 0

    # Build lookup set for existing pairs
    existing_pairs: set[tuple[str, str]] = set(
        zip(
            union_df["source1_entity_id"].astype(str),
            union_df["candidate_entity_id"].astype(str),
        )
    )

    # Mark existing pairs that the new channel also retrieved
    new_pair_set: set[tuple[str, str]] = set(
        zip(
            channel_df["source1_entity_id"].astype(str),
            channel_df["candidate_entity_id"].astype(str),
        )
    )
    mask = union_df.apply(
        lambda r: (str(r["source1_entity_id"]), str(r["candidate_entity_id"]))
                  in new_pair_set,
        axis=1,
    )
    union_df.loc[mask, flag_col] = 1

    # Add NEW-only pairs from the channel
    new_only = [
        (s1, cid) for (s1, cid) in new_pair_set if (s1, cid) not in existing_pairs
    ]
    if new_only:
        new_rows = pd.DataFrame(
            new_only, columns=["source1_entity_id", "candidate_entity_id"]
        )
        # Bring in score/rank from channel_df
        ch_lookup = channel_df.set_index(
            ["source1_entity_id", "candidate_entity_id"]
        )[[score_col, rank_col]]
        new_rows = new_rows.join(
            ch_lookup,
            on=["source1_entity_id", "candidate_entity_id"],
            how="left",
        )
        # Fill existing-channel columns with 0/NaN
        for col in union_df.columns:
            if col not in new_rows.columns:
                if col == flag_col:
                    new_rows[col] = 1
                elif col.startswith("retrieved_by_"):
                    new_rows[col] = 0
                elif col == "retrieval_agreement_count":
                    new_rows[col] = 1
                else:
                    new_rows[col] = np.nan

        union_df = pd.concat([union_df, new_rows], ignore_index=True, sort=False)

    return union_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def union_three_channel_candidates(
    s1_clean_df: pd.DataFrame,
    s2s3_clean_df: pd.DataFrame,
    s1_feature_df: pd.DataFrame,
    s2s3_feature_df: pd.DataFrame,
    vectorizer=None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Run three TF-IDF channels sequentially and return their union.

    Channels run in order: combined → name-only → address-only.
    Each channel matrix is computed and discarded before the next starts
    to stay within the 6 GB RAM budget.

    Parameters
    ----------
    s1_clean_df : pd.DataFrame
        S1 records with entity_id and clean_text.
    s2s3_clean_df : pd.DataFrame
        S2+S3 records with entity_id and clean_text.
    s1_feature_df : pd.DataFrame
        S1 feature records with entity_id, clean_name, clean_address.
    s2s3_feature_df : pd.DataFrame
        S2+S3 feature records with entity_id, clean_name, clean_address.
    vectorizer : fitted TfidfVectorizer or None
        Pre-fitted combined vectorizer.  If None, one is fitted here on
        s2s3_clean_df.  NEVER fitted on s1_clean_df.
    top_k : int
        Maximum candidates per S1 entity per channel.

    Returns
    -------
    pd.DataFrame
        Union candidate frame with columns:
        source1_entity_id, candidate_entity_id,
        combined_score, name_tfidf_score, addr_tfidf_score,
        retrieved_by_combined, retrieved_by_name, retrieved_by_address,
        retrieval_agreement_count.

        Zero-score candidates are never included (P0-1 compliance).
        Candidate IDs are deduplicated within each S1 entity.
    """
    from src.blocking import build_clean_text, fit_vectorizer
    from src.blocking_extra import search_name_only, search_address_only

    # ------------------------------------------------------------------
    # Channel A: Combined char TF-IDF
    # ------------------------------------------------------------------
    if vectorizer is None:
        logger.info("[blocking_union] Fitting combined TF-IDF on S2+S3 corpus ...")
        s2s3_corpus = build_clean_text(s2s3_clean_df)
        vectorizer = fit_vectorizer(s2s3_corpus)

    logger.info("[blocking_union] Channel A: combined TF-IDF (top_k=%d) ...", top_k)
    combined_df = _build_combined_results(
        s1_clean_df, s2s3_clean_df, vectorizer, top_k
    )
    n_combined = len(combined_df)
    logger.info("[blocking_union]   Channel A produced %d candidate pairs.", n_combined)

    # Bootstrap union with combined results
    union_df = combined_df[["source1_entity_id", "candidate_entity_id", "combined_score"]].copy()
    union_df["retrieved_by_combined"] = 1

    del combined_df  # free memory before next channel

    # ------------------------------------------------------------------
    # Channel B: Name-only TF-IDF
    # ------------------------------------------------------------------
    logger.info("[blocking_union] Channel B: name-only TF-IDF (top_k=%d) ...", top_k)
    try:
        name_df = search_name_only(s1_feature_df, s2s3_feature_df, top_k=top_k)
        n_name = len(name_df)
        logger.info("[blocking_union]   Channel B produced %d candidate pairs.", n_name)
    except Exception as exc:
        warnings.warn(
            f"[blocking_union] Channel B (name-only) failed: {exc}. Skipping.",
            RuntimeWarning,
            stacklevel=2,
        )
        name_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id",
                     "name_tfidf_score", "name_tfidf_rank"]
        )
        n_name = 0

    # Merge channel B into union
    existing_pairs: set[tuple[str, str]] = set(
        zip(union_df["source1_entity_id"].astype(str),
            union_df["candidate_entity_id"].astype(str))
    )
    union_df["retrieved_by_name"] = 0

    if n_name > 0:
        name_pair_set: set[tuple[str, str]] = set(
            zip(name_df["source1_entity_id"].astype(str),
                name_df["candidate_entity_id"].astype(str))
        )
        # Flag existing pairs that name channel also found
        union_df["retrieved_by_name"] = union_df.apply(
            lambda r: 1 if (str(r["source1_entity_id"]),
                            str(r["candidate_entity_id"])) in name_pair_set else 0,
            axis=1,
        )
        # Add name-only pairs not already in union
        new_only_name = [
            (s1, cid, sc, rk)
            for _, row in name_df.iterrows()
            for s1, cid, sc, rk in [
                (str(row["source1_entity_id"]), str(row["candidate_entity_id"]),
                 float(row["name_tfidf_score"]), int(row["name_tfidf_rank"]))
            ]
            if (s1, cid) not in existing_pairs
        ]
        if new_only_name:
            new_rows = pd.DataFrame(
                new_only_name,
                columns=["source1_entity_id", "candidate_entity_id",
                         "name_tfidf_score", "name_tfidf_rank"],
            )
            new_rows["retrieved_by_combined"] = 0
            new_rows["retrieved_by_name"] = 1
            union_df = pd.concat([union_df, new_rows], ignore_index=True, sort=False)
            existing_pairs |= {(r[0], r[1]) for r in new_only_name}
        logger.info(
            "[blocking_union]   Channel B added %d new unique pairs.", len(new_only_name)
        )

    del name_df  # free memory before next channel

    # ------------------------------------------------------------------
    # Channel C: Address-only TF-IDF
    # ------------------------------------------------------------------
    logger.info("[blocking_union] Channel C: address-only TF-IDF (top_k=%d) ...", top_k)
    try:
        addr_df = search_address_only(s1_feature_df, s2s3_feature_df, top_k=top_k)
        n_addr = len(addr_df)
        logger.info("[blocking_union]   Channel C produced %d candidate pairs.", n_addr)
    except Exception as exc:
        warnings.warn(
            f"[blocking_union] Channel C (address-only) failed: {exc}. Skipping.",
            RuntimeWarning,
            stacklevel=2,
        )
        addr_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id",
                     "addr_tfidf_score", "addr_tfidf_rank"]
        )
        n_addr = 0

    union_df["retrieved_by_address"] = 0

    if n_addr > 0:
        addr_pair_set: set[tuple[str, str]] = set(
            zip(addr_df["source1_entity_id"].astype(str),
                addr_df["candidate_entity_id"].astype(str))
        )
        # Flag existing pairs
        union_df["retrieved_by_address"] = union_df.apply(
            lambda r: 1 if (str(r["source1_entity_id"]),
                            str(r["candidate_entity_id"])) in addr_pair_set else 0,
            axis=1,
        )
        # Add addr-only new pairs
        new_only_addr = [
            (s1, cid, sc, rk)
            for _, row in addr_df.iterrows()
            for s1, cid, sc, rk in [
                (str(row["source1_entity_id"]), str(row["candidate_entity_id"]),
                 float(row["addr_tfidf_score"]), int(row["addr_tfidf_rank"]))
            ]
            if (s1, cid) not in existing_pairs
        ]
        if new_only_addr:
            new_rows = pd.DataFrame(
                new_only_addr,
                columns=["source1_entity_id", "candidate_entity_id",
                         "addr_tfidf_score", "addr_tfidf_rank"],
            )
            new_rows["retrieved_by_combined"] = 0
            new_rows["retrieved_by_name"] = 0
            new_rows["retrieved_by_address"] = 1
            union_df = pd.concat([union_df, new_rows], ignore_index=True, sort=False)
        logger.info(
            "[blocking_union]   Channel C added %d new unique pairs.", len(new_only_addr)
        )

    del addr_df  # free memory

    # ------------------------------------------------------------------
    # Finalize: fill missing scores, compute agreement count, sort
    # ------------------------------------------------------------------
    for col in ("combined_score", "name_tfidf_score", "addr_tfidf_score"):
        if col not in union_df.columns:
            union_df[col] = np.nan
        else:
            union_df[col] = union_df[col].fillna(0.0)

    for col in ("name_tfidf_rank", "addr_tfidf_rank"):
        if col not in union_df.columns:
            union_df[col] = np.nan

    # Cast flag columns to int (NaN from concat → 0)
    # pandas 3.x: coerce to numeric first in case any column is StringDtype.
    for flag in ("retrieved_by_combined", "retrieved_by_name", "retrieved_by_address"):
        union_df[flag] = (
            pd.to_numeric(union_df[flag], errors="coerce").fillna(0).astype(int)
        )

    union_df["retrieval_agreement_count"] = (
        union_df["retrieved_by_combined"]
        + union_df["retrieved_by_name"]
        + union_df["retrieved_by_address"]
    )

    # Primary score for downstream use = combined channel score (or name/addr if only from those)
    union_df["cosine_similarity"] = union_df["combined_score"].fillna(0.0)

    # Sort deterministically
    union_df = union_df.sort_values(
        ["source1_entity_id", "retrieval_agreement_count",
         "cosine_similarity", "name_tfidf_score"],
        ascending=[True, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    total_pairs = len(union_df)
    n_s1 = union_df["source1_entity_id"].nunique()
    logger.info(
        "[blocking_union] UNION complete: %d total candidate pairs for %d S1 entities.",
        total_pairs, n_s1,
    )

    # Backward-compatibility aliases for add_retrieval_agreement_features()
    # which expects the original column names from the TF-IDF+BM25 union.
    # "retrieved_by_combined" is the three-channel equivalent of "retrieved_by_char_tfidf".
    # "retrieved_by_bm25" is set to 0 here (BM25 is a separate step; not run in this module).
    union_df["retrieved_by_char_tfidf"] = union_df["retrieved_by_combined"]
    union_df["retrieved_by_bm25"] = 0

    # Final column order
    final_cols = [
        "source1_entity_id", "candidate_entity_id",
        "cosine_similarity", "combined_score",
        "name_tfidf_score", "addr_tfidf_score",
        "retrieved_by_combined", "retrieved_by_name", "retrieved_by_address",
        "retrieved_by_char_tfidf", "retrieved_by_bm25",
        "retrieval_agreement_count",
    ]
    present_cols = [c for c in final_cols if c in union_df.columns]
    extra_cols = [c for c in union_df.columns if c not in final_cols]
    return union_df[present_cols + extra_cols]


def build_candidates_dict_from_union(
    union_df: pd.DataFrame,
    all_s1_ids: "list[str] | None" = None,
) -> dict:
    """Convert a union DataFrame to a {s1_id: [cand_ids]} dict.

    Guarantees every ID in *all_s1_ids* appears as a key (P0-3).
    Zero-candidate S1 entities map to an empty list.

    Parameters
    ----------
    union_df : pd.DataFrame
        Output of :func:`union_three_channel_candidates`.
    all_s1_ids : list[str] or None
        Full list of S1 entity IDs for the current split.  When provided,
        every ID is guaranteed to appear in the output dict.

    Returns
    -------
    dict[str, list[str]]
    """
    candidates: dict = {}
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            candidates[str(s1_id)] = []

    if union_df is not None and not union_df.empty:
        for s1_id, grp in union_df.groupby("source1_entity_id", sort=False):
            s1_id = str(s1_id)
            candidates[s1_id] = grp["candidate_entity_id"].astype(str).tolist()

    return candidates
