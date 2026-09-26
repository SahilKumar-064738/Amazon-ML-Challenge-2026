"""
blocking_extra.py
-----------------
Additional candidate retrieval channels for the Business Entity Resolution pipeline.

Implements four new retrieval strategies beyond the existing TF-IDF + BM25 union:

A. Name-only TF-IDF retrieval
   Fits TF-IDF on s2s3 clean_name only; searches with s1 clean_name.
   Useful for entities with very similar names but different full clean_text.

B. Address-only TF-IDF retrieval
   Fits TF-IDF on s2s3 clean_address only; searches with s1 clean_address.
   Useful for geographically co-located entities that TF-IDF on full text misses.

C. Exact / rare-token blocking
   Builds a posting list of tokens that appear in fewer than MAX_TOKEN_FREQ
   corpus documents (rare tokens).  Returns candidate pairs that share ≥ 1
   rare token.  Very high precision for distinctive business names.

D. Sorted-neighbourhood / token blocking
   Sorts all records by a blocking key (first 6 characters of clean_name)
   and creates candidate pairs within a sliding window.
   Catches transpositions and minor prefix variations missed by TF-IDF.

Design constraints:
  - All indexes are fitted on S2+S3 corpus only (never on S1 / validation).
  - No external databases, APIs, or geocoding.
  - All functions are purely in-process, challenge-safe.
  - Output schema matches union_candidate_results() input schema:
      source1_entity_id, candidate_entity_id, + channel-specific score + rank

Public API
~~~~~~~~~~
    search_name_only(s1_df, corpus_df, top_k=20) -> pd.DataFrame
        Returns: source1_entity_id, candidate_entity_id, name_tfidf_score, name_tfidf_rank

    search_address_only(s1_df, corpus_df, top_k=20) -> pd.DataFrame
        Returns: source1_entity_id, candidate_entity_id, addr_tfidf_score, addr_tfidf_rank

    search_rare_token_blocking(s1_df, corpus_df,
                               max_token_freq=5,
                               max_candidates_per_query=200) -> pd.DataFrame
        Returns: source1_entity_id, candidate_entity_id, rare_token_count, rare_token_rank

    search_sorted_neighborhood(s1_df, corpus_df,
                                window_size=10,
                                key_length=6) -> pd.DataFrame
        Returns: source1_entity_id, candidate_entity_id, snm_score, snm_rank

    evaluate_unique_recall(new_channel_df, existing_candidates_by_s1,
                           ground_truth_df) -> dict
        Measures: how many GT pairs does the new channel uniquely recover?
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TF-IDF config for name-only and address-only channels.
# Word-level (not char) since we're working with single fields.
_NAME_TFIDF_CONFIG: dict = dict(
    analyzer="word",
    ngram_range=(1, 2),
    sublinear_tf=True,
    max_features=100_000,
    min_df=1,
)

_ADDR_TFIDF_CONFIG: dict = dict(
    analyzer="char_wb",
    ngram_range=(3, 4),
    sublinear_tf=True,
    max_features=100_000,
    min_df=1,
)

# Rare-token blocking: tokens appearing in ≤ this many corpus documents
# are considered "rare" and used as blocking keys.
DEFAULT_MAX_TOKEN_FREQ: int = 5

# Maximum candidates returned per query by rare-token blocking.
# Controls candidate explosion: a very rare token that appears in 100
# records still generates ≤ 200 candidates per query.
DEFAULT_MAX_RARE_CANDIDATES: int = 200

# Sorted-neighbourhood window: how many adjacent records to compare.
DEFAULT_SNM_WINDOW: int = 10

# Length of the blocking key prefix used for sorted-neighbourhood.
DEFAULT_SNM_KEY_LENGTH: int = 6

# Required columns in corpus and query DataFrames.
_REQUIRED_CLEAN = {"entity_id", "clean_text"}
_REQUIRED_FEATURE = {"entity_id", "clean_name", "clean_address"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_df(df: pd.DataFrame, label: str, required: set) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{label} must be a pd.DataFrame, got {type(df).__name__}.")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required column(s): {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )


def _safe_text(series: pd.Series) -> list[str]:
    """Coerce a column to a list of non-null strings."""
    return [
        "" if (x is None or (isinstance(x, float) and x != x)) else str(x)
        for x in series
    ]


def _tfidf_search(
    s1_texts: list[str],
    corpus_texts: list[str],
    s1_ids: list[str],
    corpus_ids: list[str],
    config: dict,
    top_k: int,
    score_col: str,
    rank_col: str,
) -> pd.DataFrame:
    """Fit TF-IDF on corpus, search with s1 queries, return top-k candidates."""
    if not corpus_texts or not s1_ids:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     score_col, rank_col])

    # Filter empty corpus texts (TF-IDF silently handles them but warns)
    non_empty_corpus_mask = [bool(t.strip()) for t in corpus_texts]
    filtered_corpus_texts = [t for t, keep in zip(corpus_texts, non_empty_corpus_mask) if keep]
    filtered_corpus_ids = [i for i, keep in zip(corpus_ids, non_empty_corpus_mask) if keep]

    if not filtered_corpus_texts:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     score_col, rank_col])

    vectorizer = TfidfVectorizer(**config)
    try:
        corpus_matrix = vectorizer.fit_transform(filtered_corpus_texts)
    except ValueError:
        # Empty vocabulary can happen with very small datasets
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     score_col, rank_col])

    effective_k = min(top_k, len(filtered_corpus_ids))
    rows = []

    for s1_idx, (s1_id, query_text) in enumerate(zip(s1_ids, s1_texts)):
        if not query_text.strip():
            continue

        try:
            q_vec = vectorizer.transform([query_text])
        except Exception:
            continue

        # Sparse dot product for cosine (both matrices L2-normalised by TF-IDF default)
        scores = (q_vec @ corpus_matrix.T).toarray().ravel()

        if effective_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            part = np.argpartition(scores, -effective_k)[-effective_k:]
            top_indices = part[np.argsort(scores[part])[::-1]]

        rank = 1
        for idx in top_indices:
            sc = float(scores[idx])
            if sc <= 0.0:
                break
            rows.append({
                "source1_entity_id":   str(s1_id),
                "candidate_entity_id": filtered_corpus_ids[idx],
                score_col:             sc,
                rank_col:              rank,
            })
            rank += 1
            if rank > effective_k:
                break

    if not rows:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     score_col, rank_col])

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(
        ["source1_entity_id", rank_col], kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# A. Name-only TF-IDF retrieval
# ---------------------------------------------------------------------------

def search_name_only(
    s1_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """Retrieve top-k candidates using name-only TF-IDF (word n-gram 1-2).

    Fits a fresh TF-IDF vectorizer on ``corpus_df.clean_name`` and searches
    with ``s1_df.clean_name``.  The vectorizer is NEVER fitted on S1 data.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Query entities.  Must contain ``entity_id`` and ``clean_name``.
    corpus_df : pd.DataFrame
        S2+S3 searchable corpus.  Must contain ``entity_id`` and ``clean_name``.
        S1 records must NOT be included here.
    top_k : int
        Maximum candidates per query.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, name_tfidf_score,
        name_tfidf_rank.
        Only rows with name_tfidf_score > 0 are returned.
        Sorted by (source1_entity_id, name_tfidf_rank).
    """
    _validate_df(s1_df, "s1_df", _REQUIRED_FEATURE)
    _validate_df(corpus_df, "corpus_df", _REQUIRED_FEATURE)

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    return _tfidf_search(
        s1_texts=_safe_text(s1_df["clean_name"]),
        corpus_texts=_safe_text(corpus_df["clean_name"]),
        s1_ids=s1_df["entity_id"].tolist(),
        corpus_ids=corpus_df["entity_id"].tolist(),
        config=_NAME_TFIDF_CONFIG,
        top_k=top_k,
        score_col="name_tfidf_score",
        rank_col="name_tfidf_rank",
    )


# ---------------------------------------------------------------------------
# B. Address-only TF-IDF retrieval
# ---------------------------------------------------------------------------

def search_address_only(
    s1_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    top_k: int = 20,
) -> pd.DataFrame:
    """Retrieve top-k candidates using address-only TF-IDF (char_wb 3-4 grams).

    Fits a fresh TF-IDF vectorizer on ``corpus_df.clean_address`` and
    searches with ``s1_df.clean_address``.  The vectorizer is NEVER fitted
    on S1 data.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Query entities.  Must contain ``entity_id`` and ``clean_address``.
    corpus_df : pd.DataFrame
        S2+S3 searchable corpus.  Must contain ``entity_id`` and ``clean_address``.
    top_k : int
        Maximum candidates per query.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, addr_tfidf_score,
        addr_tfidf_rank.
        Only rows with addr_tfidf_score > 0 are returned.
        Sorted by (source1_entity_id, addr_tfidf_rank).
    """
    _validate_df(s1_df, "s1_df", _REQUIRED_FEATURE)
    _validate_df(corpus_df, "corpus_df", _REQUIRED_FEATURE)

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    return _tfidf_search(
        s1_texts=_safe_text(s1_df["clean_address"]),
        corpus_texts=_safe_text(corpus_df["clean_address"]),
        s1_ids=s1_df["entity_id"].tolist(),
        corpus_ids=corpus_df["entity_id"].tolist(),
        config=_ADDR_TFIDF_CONFIG,
        top_k=top_k,
        score_col="addr_tfidf_score",
        rank_col="addr_tfidf_rank",
    )


# ---------------------------------------------------------------------------
# C. Exact / rare-token blocking
# ---------------------------------------------------------------------------

def _tokenize_for_rare(text: str) -> list[str]:
    """Tokenise text for rare-token blocking (whitespace + lower-case)."""
    if not text:
        return []
    return text.lower().split()


def build_rare_token_index(
    corpus_df: pd.DataFrame,
    text_col: str = "clean_name",
    max_token_freq: int = DEFAULT_MAX_TOKEN_FREQ,
) -> dict[str, list[str]]:
    """Build posting list of rare tokens → list of corpus entity IDs.

    A token is "rare" if it appears in at most *max_token_freq* unique corpus
    documents.  Only rare tokens are indexed.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        S2+S3 corpus.  Must have ``entity_id`` and *text_col*.
    text_col : str
        Column to extract tokens from.  Default: ``"clean_name"``.
    max_token_freq : int
        Maximum document frequency to be considered rare.

    Returns
    -------
    dict[str, list[str]]
        Mapping from rare token → sorted list of entity IDs that contain it.
    """
    _validate_df(corpus_df, "corpus_df", {"entity_id", text_col})

    # Count how many documents each token appears in
    token_doc_freq: dict[str, int] = defaultdict(int)
    token_postings: dict[str, list[str]] = defaultdict(list)

    for _, row in corpus_df.iterrows():
        eid = str(row["entity_id"])
        tokens = set(_tokenize_for_rare(str(row[text_col])))
        for tok in tokens:
            if tok:
                token_doc_freq[tok] += 1
                token_postings[tok].append(eid)

    # Keep only rare tokens
    rare_index: dict[str, list[str]] = {
        tok: sorted(eids)
        for tok, eids in token_postings.items()
        if token_doc_freq[tok] <= max_token_freq
    }
    return rare_index


def search_rare_token_blocking(
    s1_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    text_col: str = "clean_name",
    max_token_freq: int = DEFAULT_MAX_TOKEN_FREQ,
    max_candidates_per_query: int = DEFAULT_MAX_RARE_CANDIDATES,
) -> pd.DataFrame:
    """Return candidates sharing at least one rare token with the query.

    Rare tokens are those appearing in ≤ *max_token_freq* corpus documents.
    For each S1 entity, the union of candidate lists for all rare tokens in
    its query text is collected, then ranked by shared rare-token count
    (descending), capped at *max_candidates_per_query*.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Query entities.  Must contain ``entity_id`` and *text_col*.
    corpus_df : pd.DataFrame
        S2+S3 corpus (NOT including S1).  Must contain ``entity_id``
        and *text_col*.
    text_col : str
        Column to use for token extraction.  Default: ``"clean_name"``.
    max_token_freq : int
        Document frequency threshold for "rare".
    max_candidates_per_query : int
        Maximum candidates returned per S1 entity.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, rare_token_count,
        rare_token_rank.
        rare_token_count = number of shared rare tokens (int ≥ 1).
        Sorted by (source1_entity_id, rare_token_rank).
    """
    _validate_df(s1_df, "s1_df", {"entity_id", text_col})
    _validate_df(corpus_df, "corpus_df", {"entity_id", text_col})

    # Build rare-token index from corpus (S2+S3) only
    rare_index = build_rare_token_index(corpus_df, text_col, max_token_freq)

    rows = []
    for _, s1_row in s1_df.iterrows():
        s1_id = str(s1_row["entity_id"])
        tokens = set(_tokenize_for_rare(str(s1_row[text_col])))

        # Count how many rare tokens each candidate shares
        cand_counts: dict[str, int] = defaultdict(int)
        for tok in tokens:
            if tok in rare_index:
                for cid in rare_index[tok]:
                    cand_counts[cid] += 1

        if not cand_counts:
            continue

        # Sort by shared rare-token count descending
        sorted_cands = sorted(cand_counts.items(), key=lambda x: -x[1])
        sorted_cands = sorted_cands[:max_candidates_per_query]

        for rank, (cid, count) in enumerate(sorted_cands, start=1):
            rows.append({
                "source1_entity_id":   s1_id,
                "candidate_entity_id": cid,
                "rare_token_count":    int(count),
                "rare_token_rank":     rank,
            })

    if not rows:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     "rare_token_count", "rare_token_rank"])

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(
        ["source1_entity_id", "rare_token_rank"], kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# D. Sorted-neighbourhood / token blocking
# ---------------------------------------------------------------------------

def _snm_blocking_key(text: str, key_length: int) -> str:
    """Build sorted-neighbourhood blocking key from text prefix."""
    cleaned = re.sub(r"[^a-z0-9]", "", text.lower())
    return cleaned[:key_length].ljust(key_length, "0")


def search_sorted_neighborhood(
    s1_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    key_col: str = "clean_name",
    window_size: int = DEFAULT_SNM_WINDOW,
    key_length: int = DEFAULT_SNM_KEY_LENGTH,
) -> pd.DataFrame:
    """Return candidates that fall within a sorted-neighbourhood window.

    Algorithm:
    1. For both S1 and S2+S3, extract a blocking key: first *key_length*
       alphanumeric characters (lowercased) of *key_col*.
    2. Sort all records (S1 and S2+S3 combined) by their blocking key.
    3. For each S1 entity, any S2+S3 entity within ±*window_size* positions
       in the sorted order is a candidate.

    This catches prefix typos, missing leading characters, and transpositions
    that TF-IDF may miss because the character n-gram overlap is low.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Query entities.  Must contain ``entity_id`` and *key_col*.
    corpus_df : pd.DataFrame
        S2+S3 corpus.  Must contain ``entity_id`` and *key_col*.
        S1 records must NOT be included here.
    key_col : str
        Column used to build the blocking key.
    window_size : int
        Number of positions on each side (before and after) in the sorted
        array to include as candidates.
    key_length : int
        Length of the prefix used as blocking key.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id, candidate_entity_id, snm_score, snm_rank.
        snm_score = fraction of key characters that match exactly (simple
        prefix similarity); snm_rank = rank within query (1 = best match).
        Sorted by (source1_entity_id, snm_rank).
    """
    _validate_df(s1_df, "s1_df", {"entity_id", key_col})
    _validate_df(corpus_df, "corpus_df", {"entity_id", key_col})

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")
    if key_length < 1:
        raise ValueError(f"key_length must be >= 1, got {key_length}.")

    # Build combined sorted array: [(blocking_key, source_type, entity_id)]
    s1_records = [
        (_snm_blocking_key(str(row[key_col]), key_length), "s1", str(row["entity_id"]))
        for _, row in s1_df.iterrows()
    ]
    corpus_records = [
        (_snm_blocking_key(str(row[key_col]), key_length), "s2s3", str(row["entity_id"]))
        for _, row in corpus_df.iterrows()
    ]

    all_records = sorted(s1_records + corpus_records, key=lambda x: x[0])
    n = len(all_records)

    # Build index: position → (blocking_key, source, entity_id)
    # and fast lookup: S1 entity_id → list of positions
    s1_positions: dict[str, list[int]] = defaultdict(list)
    for pos, (bk, src, eid) in enumerate(all_records):
        if src == "s1":
            s1_positions[eid].append(pos)

    rows = []
    for s1_id, positions in s1_positions.items():
        # Collect all S2+S3 candidates in window
        cand_scores: dict[str, float] = {}
        s1_key = _snm_blocking_key(
            str(s1_df.loc[s1_df["entity_id"] == s1_id, key_col].iloc[0]),
            key_length,
        )

        for pos in positions:
            lo = max(0, pos - window_size)
            hi = min(n, pos + window_size + 1)
            for nb_pos in range(lo, hi):
                nb_key, nb_src, nb_id = all_records[nb_pos]
                if nb_src != "s2s3":
                    continue
                # Score: fraction of key characters that match
                score = sum(
                    1 for a, b in zip(s1_key, nb_key) if a == b
                ) / max(key_length, 1)
                # Keep the best score if seen multiple times
                if nb_id not in cand_scores or score > cand_scores[nb_id]:
                    cand_scores[nb_id] = score

        if not cand_scores:
            continue

        sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
        for rank, (cid, score) in enumerate(sorted_cands, start=1):
            rows.append({
                "source1_entity_id":   s1_id,
                "candidate_entity_id": cid,
                "snm_score":           score,
                "snm_rank":            rank,
            })

    if not rows:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                     "snm_score", "snm_rank"])

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(
        ["source1_entity_id", "snm_rank"], kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation helper: unique recall measurement
# ---------------------------------------------------------------------------

def evaluate_unique_recall(
    new_channel_df: pd.DataFrame,
    existing_candidates_by_s1: dict[str, list[str]],
    ground_truth_df: Optional[pd.DataFrame],
) -> dict:
    """Measure how many GT pairs the new channel uniquely recovers.

    A "unique recovery" is a ground-truth positive pair where:
    1. The pair is retrieved by the new channel (s1_id, cand_id appears in new_channel_df).
    2. The pair is NOT already in existing_candidates_by_s1.

    This tells us whether adding the new channel is worth the added
    candidate volume.

    Parameters
    ----------
    new_channel_df : pd.DataFrame
        Must contain ``source1_entity_id`` and ``candidate_entity_id``.
    existing_candidates_by_s1 : dict[str, list[str]]
        Current candidate set (before adding new channel).
    ground_truth_df : pd.DataFrame or None
        Must contain ``source1_entity_id`` and one of:
        ``matching_entity_ids`` or ``match_entity_ids``.

    Returns
    -------
    dict with keys:
        n_new_channel_pairs     -- total candidate pairs in new channel
        n_unique_new_candidates -- candidates NOT already in existing set
        n_gt_total              -- total ground-truth positive pairs
        n_gt_recovered_by_new_only -- GT pairs recovered only by new channel
        unique_recall_lift      -- n_gt_recovered_by_new_only / n_gt_total
        candidate_overhead_ratio -- n_unique_new_candidates / n_gt_total
        recommendation          -- "KEEP" or "DISCARD" based on lift/overhead
    """
    if ground_truth_df is None:
        return {
            "n_new_channel_pairs":        len(new_channel_df),
            "n_unique_new_candidates":    0,
            "n_gt_total":                 0,
            "n_gt_recovered_by_new_only": 0,
            "unique_recall_lift":         None,
            "candidate_overhead_ratio":   None,
            "recommendation":             "CANNOT EVALUATE (no ground truth)",
        }

    # Parse ground truth
    gt_lookup: dict[str, set[str]] = {}
    match_col = None
    for col in ("matching_entity_ids", "match_entity_ids"):
        if col in ground_truth_df.columns:
            match_col = col
            break
    if match_col is None:
        raise ValueError(
            "evaluate_unique_recall(): ground_truth_df must have "
            "'matching_entity_ids' or 'match_entity_ids'."
        )
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw = str(row[match_col]).strip()
        if raw and raw.lower() not in ("nan", "none", ""):
            gt_lookup[s1_id] = {m.strip() for m in raw.split(",") if m.strip()}
        else:
            gt_lookup[s1_id] = set()

    # Build new channel set
    new_channel_set: set[tuple[str, str]] = set(
        zip(
            new_channel_df["source1_entity_id"].astype(str),
            new_channel_df["candidate_entity_id"].astype(str),
        )
    )

    # Build existing candidate set
    existing_set: set[tuple[str, str]] = set()
    for s1_id, cands in existing_candidates_by_s1.items():
        for cid in cands:
            existing_set.add((str(s1_id), str(cid)))

    n_unique_new = len(new_channel_set - existing_set)
    n_total_gt = sum(len(v) for v in gt_lookup.values())

    n_unique_gt_recovered = 0
    for s1_id, true_cands in gt_lookup.items():
        for cid in true_cands:
            pair = (s1_id, cid)
            if pair in new_channel_set and pair not in existing_set:
                n_unique_gt_recovered += 1

    lift = n_unique_gt_recovered / n_total_gt if n_total_gt > 0 else 0.0
    overhead = n_unique_new / n_total_gt if n_total_gt > 0 else float("inf")

    # Recommendation: keep if at least 1 unique GT recovery
    # AND overhead ratio ≤ 50× lift (i.e. not wildly bloating candidate set)
    if n_unique_gt_recovered > 0 and (overhead <= 0 or lift / max(overhead, 1e-9) >= 0.01):
        recommendation = "KEEP"
    else:
        recommendation = "DISCARD"

    return {
        "n_new_channel_pairs":        len(new_channel_df),
        "n_unique_new_candidates":    n_unique_new,
        "n_gt_total":                 n_total_gt,
        "n_gt_recovered_by_new_only": n_unique_gt_recovered,
        "unique_recall_lift":         round(lift, 6),
        "candidate_overhead_ratio":   round(overhead, 4),
        "recommendation":             recommendation,
    }


# ---------------------------------------------------------------------------
# Union helper: merge any extra channel into existing union format
# ---------------------------------------------------------------------------

def union_with_extra_channel(
    existing_union_df: pd.DataFrame,
    new_channel_df: pd.DataFrame,
    channel_flag_col: str,
) -> pd.DataFrame:
    """Add a new retrieval channel to the existing union DataFrame.

    Extends the union result from ``blocking.union_candidate_results()``
    with candidates from a new channel.  New candidates are added with
    channel flag = 1; existing candidates that are also in the new channel
    get their flag updated to 1.

    Parameters
    ----------
    existing_union_df : pd.DataFrame
        Output of union_candidate_results() or a prior call to this function.
        Must have ``source1_entity_id``, ``candidate_entity_id``,
        ``retrieval_agreement_count``.
    new_channel_df : pd.DataFrame
        Output of any search_* function above.  Must have
        ``source1_entity_id`` and ``candidate_entity_id``.
    channel_flag_col : str
        Name of the new boolean flag column to add (e.g. "retrieved_by_name_tfidf").

    Returns
    -------
    pd.DataFrame
        Extended union DataFrame with *channel_flag_col* added and
        ``retrieval_agreement_count`` updated.
    """
    if not isinstance(existing_union_df, pd.DataFrame):
        raise TypeError("existing_union_df must be a pd.DataFrame.")
    if not isinstance(new_channel_df, pd.DataFrame):
        raise TypeError("new_channel_df must be a pd.DataFrame.")

    result = existing_union_df.copy()

    # Add flag column, default 0
    result[channel_flag_col] = 0

    # Build set of new channel pairs
    new_set: set[tuple[str, str]] = set(
        zip(
            new_channel_df["source1_entity_id"].astype(str),
            new_channel_df["candidate_entity_id"].astype(str),
        )
    )

    # Mark existing pairs that are also in new channel
    mask_existing = result.apply(
        lambda r: (str(r["source1_entity_id"]), str(r["candidate_entity_id"])) in new_set,
        axis=1,
    )
    result.loc[mask_existing, channel_flag_col] = 1

    # Build new-only pairs (not already in existing union)
    existing_pairs: set[tuple[str, str]] = set(
        zip(
            result["source1_entity_id"].astype(str),
            result["candidate_entity_id"].astype(str),
        )
    )
    new_only_pairs = [
        (s1, cid) for s1, cid in new_set if (s1, cid) not in existing_pairs
    ]

    if new_only_pairs:
        new_rows = pd.DataFrame(new_only_pairs, columns=["source1_entity_id", "candidate_entity_id"])
        # Fill missing columns with defaults
        for col in result.columns:
            if col not in new_rows.columns:
                if col == channel_flag_col:
                    new_rows[col] = 1
                elif col in ("retrieved_by_char_tfidf", "retrieved_by_bm25",
                             "retrieved_by_name_tfidf", "retrieved_by_addr_tfidf",
                             "retrieved_by_rare_token", "retrieved_by_snm"):
                    new_rows[col] = 0
                elif col == "retrieval_agreement_count":
                    new_rows[col] = 1
                elif col in ("cosine_similarity", "bm25_score", "name_tfidf_score",
                             "addr_tfidf_score", "snm_score"):
                    new_rows[col] = 0.0
                elif col in ("rank", "bm25_rank", "name_tfidf_rank",
                             "addr_tfidf_rank", "snm_rank", "rare_token_rank"):
                    new_rows[col] = np.nan
                else:
                    new_rows[col] = np.nan

        result = pd.concat([result, new_rows], ignore_index=True, sort=False)

    # Update agreement count
    # pandas 3.x: flag columns may be StringDtype if the DataFrame passed through
    # a TSV checkpoint; coerce to numeric before sum.
    flag_cols = [c for c in result.columns if c.startswith("retrieved_by_")]
    if flag_cols:
        for _fc in flag_cols:
            result[_fc] = pd.to_numeric(result[_fc], errors="coerce").fillna(0).astype(int)
        result["retrieval_agreement_count"] = result[flag_cols].sum(axis=1)

    # Cast flag cols to int (already done above, but keep for clarity)
    for col in flag_cols:
        result[col] = result[col].astype(int)

    return result.reset_index(drop=True)
