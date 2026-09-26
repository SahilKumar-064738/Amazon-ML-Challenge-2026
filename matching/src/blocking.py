"""
blocking.py
-----------
Phase 1 — TF-IDF character n-gram blocking.

Public API (Phase 1.1):
    build_clean_text(df)         -> pd.Series
    fit_vectorizer(corpus_texts) -> TfidfVectorizer

Public API (Phase 1.2):
    search_candidates_mock(s1_df, s2s3_df, vectorizer, top_k=20)
                                 -> pd.DataFrame

Public API (Phase 1.3):
    write_candidate_pairs(candidates_by_s1, output_path)
                                 -> None
    write_blocking_scores(results_df, output_path)
                                 -> None
    log_candidate_stats(candidates_by_s1, results_df=None)
                                 -> dict

Public API (Phase 1.4):
    compute_candidate_recall(candidates_by_s1, ground_truth_df)
                                 -> dict

Public API (Phase 1.5):
    search_candidates_scalable(s1_matrix, s2s3_matrix, top_k=20,
                               chunk_size=20000)
                                 -> pd.DataFrame

Public API (Phase 1.6):
    normalize_country(country)   -> str
    is_same_country(country_a, country_b)
                                 -> bool
    search_candidates_scalable(..., same_country_filter=False,
                               s1_countries=None, s2s3_countries=None)
                                 -> pd.DataFrame

NOT implemented here (future phases):
    - ML scoring (LightGBM / CatBoost)

Design notes
~~~~~~~~~~~~
* The vectorizer is ALWAYS fitted on the combined S2+S3 searchable corpus
  (mock_clean_s2.tsv in dev).  S1 (the source corpus) is never passed to
  fit_vectorizer(); S1 records are later transformed against the already-
  fitted vectorizer when searching.
* build_clean_text() reads the ``clean_text`` column directly.  The mock
  contract (entity_id, clean_text) does NOT include clean_name or
  clean_address, so those columns are not referenced.
* search_candidates_mock() uses dense cosine similarity — acceptable only
  for the tiny mock dataset.  Do not use it on production-scale corpora.
* write_candidate_pairs() serialises the exact candidate set to TSV.  The
  written file is the definitive input for the downstream matching stage.
  Later phases MUST consume this file as-is and MUST NOT regenerate, expand,
  or shrink the candidate set.
* search_candidates_scalable() performs the same cosine retrieval as
  search_candidates_mock() but processes S1 in row-chunks so the full
  dense S1 × S2+S3 matrix is never materialised in memory.  It is safe
  to use on production-scale corpora."""

import csv
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact TF-IDF configuration — do not change without updating the spec.
_TFIDF_CONFIG: dict = dict(
    analyzer="char",
    ngram_range=(3, 5),
    sublinear_tf=True,
    max_features=250_000,
)


# ---------------------------------------------------------------------------
# Phase 1.1 — corpus preparation
# ---------------------------------------------------------------------------

def build_clean_text(df: pd.DataFrame) -> pd.Series:
    """Return the ``clean_text`` column as a Series suitable for vectorisation.

    The mock data contract guarantees that ``clean_text`` already contains
    normalised text produced upstream.  This function does NOT construct text
    from sub-fields (clean_name, clean_address, etc.) because those columns
    do not exist in the current contract.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that MUST contain a ``clean_text`` column.

    Returns
    -------
    pd.Series
        A copy of ``clean_text`` with any ``NaN`` or blank values replaced
        by an empty string ``""``.  The input DataFrame is never mutated.

    Raises
    ------
    ValueError
        If ``clean_text`` is absent from *df*.
    """
    if "clean_text" not in df.columns:
        raise ValueError(
            "build_clean_text() requires a 'clean_text' column; "
            f"found: {list(df.columns)}"
        )

    series = df["clean_text"].copy()

    # Replace NaN (should not occur given keep_default_na=False in ingestion,
    # but be defensive) and blank strings with empty string.
    n_null = int(series.isna().sum())
    if n_null:
        warnings.warn(
            f"build_clean_text(): {n_null} NaN value(s) in 'clean_text' "
            "replaced with empty string.",
            UserWarning,
            stacklevel=2,
        )
        series = series.fillna("")

    return series


# ---------------------------------------------------------------------------
# Phase 1.1 — vectorizer fitting
# ---------------------------------------------------------------------------

def fit_vectorizer(corpus_texts: pd.Series) -> TfidfVectorizer:
    """Fit and return a TF-IDF character n-gram vectorizer on *corpus_texts*.

    IMPORTANT: this function must only ever be called with the combined
    S2+S3 searchable corpus.  Passing S1 records here would contaminate the
    IDF weights with source-side data and is explicitly forbidden.

    Parameters
    ----------
    corpus_texts : pd.Series
        Pre-processed text strings from the searchable (S2+S3) corpus.
        Produced by :func:`build_clean_text`.

    Returns
    -------
    TfidfVectorizer
        A fitted vectorizer with the exact configuration::

            analyzer    = "char"
            ngram_range = (3, 5)
            sublinear_tf = True
            max_features = 250_000

    Raises
    ------
    ValueError
        If *corpus_texts* is empty (fitting on zero documents is meaningless).
    """
    if len(corpus_texts) == 0:
        raise ValueError(
            "fit_vectorizer() received an empty corpus; "
            "cannot fit TF-IDF on zero documents."
        )

    vectorizer = TfidfVectorizer(**_TFIDF_CONFIG)
    vectorizer.fit(corpus_texts)
    return vectorizer


# ---------------------------------------------------------------------------
# Phase 1.6 — generic country normalization and comparison
# ---------------------------------------------------------------------------

def normalize_country(country) -> str:
    """Normalize a country string for generic exact equality comparison.

    Leading and trailing whitespace is stripped, and the string is converted
    to lowercase (case-folded). Empty, None, or NaN values return empty string.
    No country whitelist, lookup table, or hardcoded country list is used.
    Arbitrary country strings are supported.

    Parameters
    ----------
    country : any
        Country name, code, or identifier.

    Returns
    -------
    str
        Normalized country string.
    """
    if country is None or pd.isna(country):
        return ""
    return str(country).strip().casefold()


def is_same_country(country_a, country_b) -> bool:
    """Check generic equality between two country representations.

    Compares normalized/clean country strings using exact equality.
    Returns True only if both country strings normalize to non-empty,
    identical strings. Arbitrary country strings are supported without
    any hardcoded country list or whitelist.

    Parameters
    ----------
    country_a : any
        First country representation.
    country_b : any
        Second country representation.

    Returns
    -------
    bool
        True if exact normalized equality holds and neither is empty,
        False otherwise.
    """
    norm_a = normalize_country(country_a)
    norm_b = normalize_country(country_b)
    if not norm_a or not norm_b:
        return False
    return norm_a == norm_b


# ---------------------------------------------------------------------------
# Phase 1.2 — mock-scale cosine candidate search
# ---------------------------------------------------------------------------

def search_candidates_mock(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    top_k: int = 20,
    same_country_filter: bool = False,
    s1_countries=None,
    s2s3_countries=None,
    country_col: str = "clean_country",
) -> pd.DataFrame:
    """Return top-k TF-IDF cosine candidates from S2+S3 for every S1 record.

    This function uses **dense** cosine similarity via
    ``sklearn.metrics.pairwise.cosine_similarity``.  That is only acceptable
    for the tiny mock dataset.  Do NOT use it on production-scale corpora.

    The vectorizer must already be fitted (via :func:`fit_vectorizer`) on the
    S2+S3 corpus before calling this function.  It is NEVER refitted here.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source corpus.  Must contain ``entity_id`` and ``clean_text``.
    s2s3_df : pd.DataFrame
        Combined searchable corpus (S2 + S3).  Must contain ``entity_id``
        and ``clean_text``.
    vectorizer : TfidfVectorizer
        A **fitted** vectorizer returned by :func:`fit_vectorizer`.
    top_k : int, default 20
        Maximum number of candidates to return per S1 record.
    same_country_filter : bool, default False
        When True, restricts candidates for each S1 record to records in S2+S3
        having the exact same normalized country. Default is False.
    s1_countries : iterable or None, default None
        Country strings for S1 rows. If None, searched in s1_df[country_col]
        or s1_df["country"].
    s2s3_countries : iterable or None, default None
        Country strings for S2+S3 rows. If None, searched in s2s3_df[country_col]
        or s2s3_df["country"].
    country_col : str, default "clean_country"
        Column name to check in DataFrames if s1_countries/s2s3_countries is None.

    Returns
    -------
    pd.DataFrame
        One row per (S1 record, candidate) pair with columns:

        * ``source1_entity_id``   – entity_id from S1
        * ``candidate_entity_id`` – entity_id from S2+S3
        * ``cosine_similarity``   – float in [0, 1]
        * ``rank``                – 1-based rank within each S1 record
                                    (1 = highest similarity)

        Rows are sorted by (source1_entity_id, rank).

    Raises
    ------
    ValueError
        If ``entity_id`` or ``clean_text`` are missing from either DataFrame,
        or if top_k < 1.
    RuntimeError
        If the vectorizer has not been fitted yet.
    """
    # ------------------------------------------------------------------
    # Guard clauses
    # ------------------------------------------------------------------
    for label, df in (("s1_df", s1_df), ("s2s3_df", s2s3_df)):
        for col in ("entity_id", "clean_text"):
            if col not in df.columns:
                raise ValueError(
                    f"search_candidates_mock(): '{col}' column missing "
                    f"from {label}. Found: {list(df.columns)}"
                )

    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")

    if not hasattr(vectorizer, "vocabulary_"):
        raise RuntimeError(
            "search_candidates_mock(): the vectorizer has not been fitted. "
            "Call fit_vectorizer() on the S2+S3 corpus first."
        )

    # ------------------------------------------------------------------
    # Transform both corpora — vectorizer is NOT refitted
    # ------------------------------------------------------------------
    s1_texts   = build_clean_text(s1_df).tolist()
    s2s3_texts = build_clean_text(s2s3_df).tolist()

    s1_matrix   = vectorizer.transform(s1_texts)    # shape (|S1|, vocab)
    s2s3_matrix = vectorizer.transform(s2s3_texts)  # shape (|S2+S3|, vocab)

    # ------------------------------------------------------------------
    # Dense cosine similarity  (|S1| × |S2+S3|)
    # ------------------------------------------------------------------
    sim_matrix = cosine_similarity(s1_matrix, s2s3_matrix)
    # sim_matrix[i, j] = cosine similarity between S1[i] and S2+S3[j]

    s1_ids   = s1_df["entity_id"].tolist()
    s2s3_ids = s2s3_df["entity_id"].tolist()

    # ------------------------------------------------------------------
    # Build results frame
    # ------------------------------------------------------------------
    rows = []

    if same_country_filter:
        # Resolve s1_countries
        if s1_countries is not None:
            s1_c_list = [normalize_country(c) for c in (s1_countries.tolist() if hasattr(s1_countries, "tolist") else list(s1_countries))]
        elif country_col in s1_df.columns:
            s1_c_list = [normalize_country(c) for c in s1_df[country_col]]
        elif "country" in s1_df.columns:
            s1_c_list = [normalize_country(c) for c in s1_df["country"]]
        else:
            raise ValueError(
                "search_candidates_mock(): same_country_filter=True requires "
                f"s1_countries or a '{country_col}' / 'country' column in s1_df."
            )

        # Resolve s2s3_countries
        if s2s3_countries is not None:
            s2s3_c_list = [normalize_country(c) for c in (s2s3_countries.tolist() if hasattr(s2s3_countries, "tolist") else list(s2s3_countries))]
        elif country_col in s2s3_df.columns:
            s2s3_c_list = [normalize_country(c) for c in s2s3_df[country_col]]
        elif "country" in s2s3_df.columns:
            s2s3_c_list = [normalize_country(c) for c in s2s3_df["country"]]
        else:
            raise ValueError(
                "search_candidates_mock(): same_country_filter=True requires "
                f"s2s3_countries or a '{country_col}' / 'country' column in s2s3_df."
            )

        for i, s1_id in enumerate(s1_ids):
            c1 = s1_c_list[i]
            if not c1:
                continue
            matching_j = [j for j, c2 in enumerate(s2s3_c_list) if c1 == c2]
            if not matching_j:
                continue

            scores = sim_matrix[i, matching_j]
            k_match = min(top_k, len(matching_j))
            top_sub = np.argsort(scores)[::-1][:k_match]
            for rank, sub_idx in enumerate(top_sub, start=1):
                j = matching_j[sub_idx]
                sc = float(scores[sub_idx])
                # P0-1 FIX: skip zero-score candidates
                if sc <= 0.0:
                    break   # argsort descending — all remaining are also <= 0
                rows.append(
                    {
                        "source1_entity_id":   s1_id,
                        "candidate_entity_id": s2s3_ids[j],
                        "cosine_similarity":   sc,
                        "rank":                rank,
                    }
                )
    else:
        effective_k = min(top_k, len(s2s3_ids))
        for i, s1_id in enumerate(s1_ids):
            scores = sim_matrix[i]                              # 1-D array length |S2+S3|
            # argsort descending — take top effective_k
            top_indices = np.argsort(scores)[::-1][:effective_k]

            for rank, j in enumerate(top_indices, start=1):
                sc = float(scores[j])
                # P0-1 FIX: skip zero-score candidates — they are not genuine matches
                if sc <= 0.0:
                    break   # argsort descending, so all remaining are also <= 0
                rows.append(
                    {
                        "source1_entity_id":   s1_id,
                        "candidate_entity_id": s2s3_ids[j],
                        "cosine_similarity":   sc,
                        "rank":                rank,
                    }
                )

    result_df = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "cosine_similarity",
            "rank",
        ],
    )

    # Sort for deterministic output
    result_df = result_df.sort_values(
        ["source1_entity_id", "rank"]
    ).reset_index(drop=True)

    return result_df


# ---------------------------------------------------------------------------
# Phase 1.3 — candidate_pairs.tsv writer
# ---------------------------------------------------------------------------

def write_candidate_pairs(
    candidates_by_s1: dict,
    output_path: str,
) -> None:
    """Serialise the blocking candidate set to a TSV file.

    This file is the **exact and complete** candidate set that will be
    consumed by the downstream matching stage (Phase 2+).  Later phases
    MUST NOT regenerate, expand, or shrink it — they read it as-is.

    Output format
    ~~~~~~~~~~~~~
    Tab-separated, two columns, with a header row::

        source1_entity_id\\tcandidate_entity_ids
        S1-001\\tS2-099,S2-100,S2-101,S3-999
        S1-002\\tS2-101,S3-999,S2-100,S2-099

    * ``candidate_entity_ids`` is a comma-separated list of IDs in the
      order they were supplied (highest-similarity first when the caller
      passes them in ranked order from :func:`search_candidates_mock`).
    * Duplicate IDs within a candidate list are removed while preserving
      the first occurrence order.
    * An S1 entity with zero candidates produces an empty
      ``candidate_entity_ids`` field (the cell is the empty string ``""``).
    * Exactly one row is written per S1 entity, in the iteration order of
      *candidates_by_s1*.

    Parameters
    ----------
    candidates_by_s1 : dict
        Mapping of ``source1_entity_id`` (str) to an ordered list of
        ``candidate_entity_id`` strings.  Produced by converting the
        DataFrame returned by :func:`search_candidates_mock` into this
        structure (see the test and pipeline scripts for the canonical
        conversion).
    output_path : str
        Destination file path.  Parent directories are created automatically
        if they do not already exist.

    Returns
    -------
    None
        The function writes the file and returns nothing.

    Raises
    ------
    TypeError
        If *candidates_by_s1* is not a dict.
    ValueError
        If any key or candidate value is not a string.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(candidates_by_s1, dict):
        raise TypeError(
            f"candidates_by_s1 must be a dict, got {type(candidates_by_s1).__name__}"
        )

    for s1_id, cand_list in candidates_by_s1.items():
        if not isinstance(s1_id, str):
            raise ValueError(
                f"All source1_entity_id keys must be strings; "
                f"found {type(s1_id).__name__}: {s1_id!r}"
            )
        for cid in cand_list:
            if not isinstance(cid, str):
                raise ValueError(
                    f"All candidate IDs must be strings; "
                    f"found {type(cid).__name__}: {cid!r} "
                    f"(under source1_entity_id={s1_id!r})"
                )

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Write TSV
    # ------------------------------------------------------------------
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for s1_id, cand_list in candidates_by_s1.items():
            # Deduplicate while preserving order (first occurrence wins)
            seen: set = set()
            deduped = []
            for cid in cand_list:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)

            candidate_cell = ",".join(deduped)  # empty string when deduped=[]
            writer.writerow([s1_id, candidate_cell])


def write_blocking_scores(
    results_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Serialise the already-computed Phase 1 blocking cosine similarity scores to a TSV file.

    Minimal plumbing for Phase 2.4 to persist the in-memory scores produced by
    search_candidates_mock() or search_candidates_scalable() alongside
    candidate_pairs_mock.tsv.

    This function does NOT alter candidate generation, ranking, top_k,
    TF-IDF configuration, or candidate membership.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame produced by Phase 1 candidate search containing at minimum:
        ``source1_entity_id``, ``candidate_entity_id``, and a cosine score column
        (``cosine_similarity``, ``blocking_cosine_sim``, or ``score``).
    output_path : str
        Destination TSV file path.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *results_df* is not a pandas DataFrame.
    ValueError
        If required columns are missing.
    """
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError(
            f"results_df must be a pd.DataFrame, got {type(results_df).__name__}"
        )

    score_col = None
    for cand in ("cosine_similarity", "blocking_cosine_sim", "score"):
        if cand in results_df.columns:
            score_col = cand
            break

    if "source1_entity_id" not in results_df.columns or "candidate_entity_id" not in results_df.columns or score_col is None:
        raise ValueError(
            "results_df must contain 'source1_entity_id', 'candidate_entity_id', "
            f"and a cosine score column ('cosine_similarity' / 'score'). Found: {list(results_df.columns)}"
        )

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    out_df = results_df[["source1_entity_id", "candidate_entity_id", score_col]].copy()
    if score_col != "cosine_similarity":
        out_df = out_df.rename(columns={score_col: "cosine_similarity"})

    out_df.to_csv(output_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# P0-1 Logging helper — candidate quality diagnostics
# ---------------------------------------------------------------------------

def log_candidate_stats(
    candidates_by_s1: dict,
    results_df: "pd.DataFrame | None" = None,
) -> dict:
    """Log and return candidate-count and zero-score diagnostics.

    Call this after building *candidates_by_s1* to surface:
      - number of S1 entities with zero valid candidates
      - candidate count distribution (min / mean / median / max)
      - percentage of candidate pairs with zero retrieval score

    Parameters
    ----------
    candidates_by_s1 : dict
        Mapping source1_entity_id -> list[candidate_entity_id].
    results_df : pd.DataFrame or None
        Optional long-form results frame (with a score column) to compute
        zero-score percentage.  Accepts "cosine_similarity", "score", or
        "blocking_cosine_sim".

    Returns
    -------
    dict with keys:
        n_s1_total              -- total S1 entities
        n_s1_zero_candidates    -- S1 entities with empty candidate list
        pct_s1_zero_candidates  -- % of S1 entities with no candidates
        min_candidates          -- minimum candidate count (over non-zero S1s)
        mean_candidates         -- mean candidate count (over all S1s)
        median_candidates       -- median candidate count
        max_candidates          -- maximum candidate count
        pct_zero_score_pairs    -- % of candidate pairs with score == 0.0
                                   (None if results_df not supplied)
    """
    n_total = len(candidates_by_s1)
    counts = [len(v) for v in candidates_by_s1.values()]
    n_zero_cand = sum(1 for c in counts if c == 0)
    pct_zero_cand = round(100.0 * n_zero_cand / n_total, 2) if n_total > 0 else 0.0

    non_zero_counts = [c for c in counts if c > 0]
    min_c = min(non_zero_counts) if non_zero_counts else 0
    mean_c = round(sum(counts) / n_total, 2) if n_total > 0 else 0.0
    med_c = float(np.median(counts)) if counts else 0.0
    max_c = max(counts) if counts else 0

    pct_zero_score = None
    if results_df is not None and isinstance(results_df, pd.DataFrame):
        score_col = None
        for cand in ("cosine_similarity", "score", "blocking_cosine_sim"):
            if cand in results_df.columns:
                score_col = cand
                break
        if score_col is not None and len(results_df) > 0:
            # pandas 3.x: score column may be StringDtype if loaded from TSV;
            # coerce to numeric before comparison to avoid TypeError.
            _score_num = pd.to_numeric(results_df[score_col], errors="coerce").fillna(0.0)
            n_zero = int((_score_num <= 0.0).sum())
            pct_zero_score = round(100.0 * n_zero / len(results_df), 2)

    stats = {
        "n_s1_total":             n_total,
        "n_s1_zero_candidates":   n_zero_cand,
        "pct_s1_zero_candidates": pct_zero_cand,
        "min_candidates":         min_c,
        "mean_candidates":        mean_c,
        "median_candidates":      med_c,
        "max_candidates":         max_c,
        "pct_zero_score_pairs":   pct_zero_score,
    }

    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        "[blocking] Candidate stats: total_s1=%d  zero_cand_s1=%d (%.1f%%)  "
        "min=%d mean=%.1f median=%.1f max=%d  zero_score_pct=%s",
        n_total, n_zero_cand, pct_zero_cand,
        min_c, mean_c, med_c, max_c,
        f"{pct_zero_score:.1f}%" if pct_zero_score is not None else "n/a",
    )
    if n_zero_cand > 0:
        logger.warning(
            "[blocking] %d S1 entities have zero valid candidates. "
            "They will receive an empty candidate list in the output.",
            n_zero_cand,
        )

    return stats

def compute_candidate_recall(
    candidates_by_s1: dict,
    ground_truth_df,
) -> dict:
    """Measure how many ground-truth matches were retrieved by blocking.

    Candidate recall answers the question: *of all true positive pairs that
    exist, what fraction did the blocking step preserve?*  A recall of 1.0
    means every true match is present in at least one candidate list; 0.0
    means none were retrieved.

    This function does NOT modify candidate lists.  It only measures.

    Ground-truth format
    ~~~~~~~~~~~~~~~~~~~
    *ground_truth_df* must be a :class:`pandas.DataFrame` with at minimum:

    * ``source1_entity_id``   – the S1 entity ID (string)
    * ``matching_entity_ids`` – comma-separated S2/S3 IDs that are true
                                matches for that S1 entity (string).
                                **This is the canonical column name.**

    Backward-compatible alias
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    The legacy column name ``match_entity_ids`` is also accepted when
    ``matching_entity_ids`` is absent.  If **both** columns are present
    simultaneously, a :class:`ValueError` is raised unless their values
    agree row-by-row (conflict detection prevents silent data loss).

    Rows where the match-ID column is empty or NaN are ignored (that S1
    entity has no known matches and does not contribute to recall).

    No ground truth
    ~~~~~~~~~~~~~~~
    Pass ``None`` for *ground_truth_df* to signal that ground truth is
    unavailable.  The returned dict will contain
    ``"recall": None`` and ``"unavailable": True``.  Candidate-count
    statistics are still computed from *candidates_by_s1* alone.

    Parameters
    ----------
    candidates_by_s1 : dict
        Mapping of ``source1_entity_id`` -> list of ``candidate_entity_id``
        strings.  Produced by :func:`write_candidate_pairs` input or
        directly from the Phase 1.2 search results.
    ground_truth_df : pd.DataFrame or None
        Ground-truth matches as described above, or ``None`` if unavailable.

    Returns
    -------
    dict with keys:

    * ``"recall"``              – float in [0, 1], or ``None`` if no GT
    * ``"unavailable"``         – True when ground truth was not provided
    * ``"n_s1_evaluated"``      – int, S1 entities that had GT matches
    * ``"n_gt_matches"``        – int, total ground-truth match pairs
    * ``"n_gt_retrieved"``      – int, GT matches found in candidates
    * ``"per_s1"``              – list of per-entity dicts (empty if no GT)
    * ``"min_candidates"``      – int, minimum candidate count across S1
    * ``"max_candidates"``      – int, maximum candidate count across S1
    * ``"avg_candidates"``      – float, average candidate count across S1
    * ``"n_s1_total"``          – int, total S1 entities in candidate map

    Raises
    ------
    ValueError
        If *ground_truth_df* is provided but missing required columns.
    ValueError
        If both ``matching_entity_ids`` and ``match_entity_ids`` are present
        and contain conflicting values for any row.
    """
    # ------------------------------------------------------------------
    # Candidate-count statistics (always computed)
    # ------------------------------------------------------------------
    n_s1_total = len(candidates_by_s1)
    candidate_counts = [len(v) for v in candidates_by_s1.values()]

    if candidate_counts:
        min_cands = min(candidate_counts)
        max_cands = max(candidate_counts)
        avg_cands = sum(candidate_counts) / len(candidate_counts)
    else:
        min_cands = max_cands = 0
        avg_cands = 0.0

    base_stats = {
        "n_s1_total":      n_s1_total,
        "min_candidates":  min_cands,
        "max_candidates":  max_cands,
        "avg_candidates":  round(avg_cands, 4),
    }

    # ------------------------------------------------------------------
    # Ground truth unavailable
    # ------------------------------------------------------------------
    if ground_truth_df is None:
        return {
            "recall":         None,
            "unavailable":    True,
            "n_s1_evaluated": 0,
            "n_gt_matches":   0,
            "n_gt_retrieved": 0,
            "per_s1":         [],
            **base_stats,
        }

    # ------------------------------------------------------------------
    # Validate ground-truth schema and resolve match-ID column name.
    #
    # Canonical name : "matching_entity_ids"  (Phase 2 convention)
    # Legacy alias   : "match_entity_ids"     (old Phase 1 convention)
    #
    # Rules:
    #   1. If only "matching_entity_ids" is present  -> use it (canonical).
    #   2. If only "match_entity_ids" is present     -> use it (backward compat).
    #   3. If both are present                       -> values must agree;
    #      raise ValueError if any row conflicts.
    #   4. If neither is present                     -> raise ValueError.
    # ------------------------------------------------------------------
    if "source1_entity_id" not in ground_truth_df.columns:
        raise ValueError(
            "compute_candidate_recall(): ground_truth_df is missing required "
            "column 'source1_entity_id'.  "
            f"Found: {list(ground_truth_df.columns)}"
        )

    _CANONICAL = "matching_entity_ids"
    _LEGACY    = "match_entity_ids"

    has_canonical = _CANONICAL in ground_truth_df.columns
    has_legacy    = _LEGACY    in ground_truth_df.columns

    if has_canonical and has_legacy:
        # Both present: verify they agree row-by-row (same string values).
        # Treat NaN as empty string for comparison purposes.
        canonical_vals = ground_truth_df[_CANONICAL].fillna("").astype(str)
        legacy_vals    = ground_truth_df[_LEGACY].fillna("").astype(str)
        conflicts = (canonical_vals != legacy_vals)
        if conflicts.any():
            bad_rows = ground_truth_df.index[conflicts].tolist()[:5]
            raise ValueError(
                f"compute_candidate_recall(): ground_truth_df contains both "
                f"'{_CANONICAL}' and '{_LEGACY}' columns with conflicting "
                f"values at row index(es) {bad_rows}.  "
                f"Remove the duplicate column or ensure both columns agree."
            )
        # Values agree — use canonical column.
        match_col = _CANONICAL
    elif has_canonical:
        match_col = _CANONICAL
    elif has_legacy:
        match_col = _LEGACY
    else:
        raise ValueError(
            f"compute_candidate_recall(): ground_truth_df is missing the "
            f"match-ID column.  Expected '{_CANONICAL}' (canonical) or "
            f"'{_LEGACY}' (legacy alias).  "
            f"Found: {list(ground_truth_df.columns)}"
        )

    # ------------------------------------------------------------------
    # Compute recall row-by-row
    # ------------------------------------------------------------------
    n_gt_matches   = 0
    n_gt_retrieved = 0
    n_s1_evaluated = 0
    per_s1         = []

    for _, gt_row in ground_truth_df.iterrows():
        s1_id        = str(gt_row["source1_entity_id"]).strip()
        raw_matches  = str(gt_row[match_col]).strip()

        # Skip rows with no actual matches
        if not raw_matches or raw_matches.lower() in ("nan", "none", ""):
            continue

        true_matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
        if not true_matches:
            continue

        candidate_set = set(candidates_by_s1.get(s1_id, []))
        retrieved     = true_matches & candidate_set

        n_gt_matches   += len(true_matches)
        n_gt_retrieved += len(retrieved)
        n_s1_evaluated += 1

        per_s1.append(
            {
                "source1_entity_id":  s1_id,
                "n_true_matches":     len(true_matches),
                "n_retrieved":        len(retrieved),
                "entity_recall":      round(len(retrieved) / len(true_matches), 4),
                "missing":            sorted(true_matches - candidate_set),
            }
        )

    recall = (
        round(n_gt_retrieved / n_gt_matches, 6)
        if n_gt_matches > 0
        else None
    )

    return {
        "recall":         recall,
        "unavailable":    False,
        "n_s1_evaluated": n_s1_evaluated,
        "n_gt_matches":   n_gt_matches,
        "n_gt_retrieved": n_gt_retrieved,
        "per_s1":         per_s1,
        **base_stats,
    }


# ---------------------------------------------------------------------------
# Phase 1.5 — scalable chunked sparse candidate search
# ---------------------------------------------------------------------------

def search_candidates_scalable(
    s1_matrix,
    s2s3_matrix,
    top_k: int = 20,
    chunk_size: int = 20_000,
    same_country_filter: bool = False,
    s1_countries=None,
    s2s3_countries=None,
    corpus_block_size: int = 250_000,
) -> pd.DataFrame:
    """Return top-k cosine candidates for every S1 row using memory-bounded 2D chunked sparse math.

    This function produces **identical results** to the dense implementation but never
    constructs the full ``|S1| × |S2+S3|`` dense matrix in memory.

    Algorithm (OOM-proof 2D chunked top-K extraction)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    To prevent large intermediate sparse matrices (which caused OOMs), we chunk
    both S1 and S2S3. For each S1 query chunk, we maintain a running Top-K.
    We iterate over S2S3 in blocks, compute the smaller bounded sparse product,
    and update the running Top-K for the queries.
    """
    import scipy.sparse as sp

    # ------------------------------------------------------------------
    # Guard clauses
    # ------------------------------------------------------------------
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}.")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    n_s1, vocab_s1     = s1_matrix.shape
    n_s2s3, vocab_s2s3 = s2s3_matrix.shape

    if vocab_s1 != vocab_s2s3:
        raise ValueError(
            f"Matrix vocabulary dimension mismatch: "
            f"s1_matrix has {vocab_s1} features, "
            f"s2s3_matrix has {vocab_s2s3} features.  "
            "Both must come from the same fitted vectorizer."
        )

    if n_s1 == 0 or n_s2s3 == 0:
        return pd.DataFrame(
            columns=["s1_row_idx", "s2s3_row_idx", "score", "rank"]
        )

    # ------------------------------------------------------------------
    # Phase 1.6 — Optional Same-Country Candidate Optimization
    # ------------------------------------------------------------------
    if same_country_filter:
        if s1_countries is None or s2s3_countries is None:
            raise ValueError(
                "same_country_filter=True requires both 's1_countries' and "
                "'s2s3_countries' to be provided."
            )

        if hasattr(s1_countries, "tolist"):
            s1_c_raw = s1_countries.tolist()
        elif isinstance(s1_countries, (list, tuple)):
            s1_c_raw = list(s1_countries)
        elif isinstance(s1_countries, np.ndarray):
            s1_c_raw = s1_countries.tolist()
        else:
            s1_c_raw = list(s1_countries)

        if hasattr(s2s3_countries, "tolist"):
            s2_c_raw = s2s3_countries.tolist()
        elif isinstance(s2s3_countries, (list, tuple)):
            s2_c_raw = list(s2s3_countries)
        elif isinstance(s2s3_countries, np.ndarray):
            s2_c_raw = s2s3_countries.tolist()
        else:
            s2_c_raw = list(s2s3_countries)

        if len(s1_c_raw) != n_s1:
            raise ValueError(
                f"Length of s1_countries ({len(s1_c_raw)}) does not match "
                f"s1_matrix rows ({n_s1})."
            )
        if len(s2_c_raw) != n_s2s3:
            raise ValueError(
                f"Length of s2s3_countries ({len(s2_c_raw)}) does not match "
                f"s2s3_matrix rows ({n_s2s3})."
            )

        s1_clean = [normalize_country(c) for c in s1_c_raw]
        s2_clean = [normalize_country(c) for c in s2_c_raw]

        s2_by_country = {}
        for j, c in enumerate(s2_clean):
            if c:
                s2_by_country.setdefault(c, []).append(j)

        s1_by_country = {}
        for i, c in enumerate(s1_clean):
            if c:
                s1_by_country.setdefault(c, []).append(i)

        rows_out = []
        for country, s1_idxs in s1_by_country.items():
            s2_idxs = s2_by_country.get(country, [])
            if not s2_idxs:
                continue

            # Sparse slicing of submatrices for this country
            s1_sub = s1_matrix.tocsr()[s1_idxs]
            s2_sub = s2s3_matrix.tocsr()[s2_idxs]

            # Restricted candidate search
            sub_res = search_candidates_scalable(
                s1_sub,
                s2_sub,
                top_k=top_k,
                chunk_size=chunk_size,
                same_country_filter=False,
                corpus_block_size=corpus_block_size,
            )

            for _, r in sub_res.iterrows():
                rows_out.append(
                    {
                        "s1_row_idx":   int(s1_idxs[int(r["s1_row_idx"])]),
                        "s2s3_row_idx": int(s2_idxs[int(r["s2s3_row_idx"])]),
                        "score":        float(r["score"]),
                        "rank":         int(r["rank"]),
                    }
                )

        result_df = pd.DataFrame(
            rows_out,
            columns=["s1_row_idx", "s2s3_row_idx", "score", "rank"],
        )
        if not result_df.empty:
            result_df = result_df.sort_values(
                ["s1_row_idx", "rank"]
            ).reset_index(drop=True)
        return result_df

    # ------------------------------------------------------------------
    # Prepare sparse formats
    # s1_csr  — CSR for O(1) row slicing
    # ------------------------------------------------------------------
    s1_csr  = s1_matrix.tocsr()
    s2s3_csr = s2s3_matrix.tocsr() # We will chunk S2S3 and transpose chunks

    effective_k = min(top_k, n_s2s3)

    chunk_frames = []

    # ------------------------------------------------------------------
    # 2D Chunk loop (Queries x Corpus)
    # ------------------------------------------------------------------
    for q_start in range(0, n_s1, chunk_size):
        q_end = min(q_start + chunk_size, n_s1)
        s1_chunk = s1_csr[q_start:q_end]
        
        chunk_n = q_end - q_start
        # Running top-K state for this query chunk
        running_scores = np.full((chunk_n, effective_k), -1.0, dtype=np.float32)
        running_cols   = np.full((chunk_n, effective_k), -1, dtype=np.int32)

        for c_start in range(0, n_s2s3, corpus_block_size):
            c_end = min(c_start + corpus_block_size, n_s2s3)
            # Take a block of the corpus, and transpose it to keep CSR product
            s2s3_block_T = s2s3_csr[c_start:c_end].tocsc().T

            # Bounded memory: s1_chunk @ s2s3_block_T
            sim_csr = (s1_chunk @ s2s3_block_T).tocsr()

            indptr  = sim_csr.indptr
            indices = sim_csr.indices
            data    = sim_csr.data

            for local_i in range(sim_csr.shape[0]):
                row_start = int(indptr[local_i])
                row_end   = int(indptr[local_i + 1])

                if row_end == row_start:
                    continue
                    
                nz_cols   = indices[row_start:row_end]
                nz_scores = data[row_start:row_end]

                global_cols = nz_cols + c_start

                # Merge with existing running state
                combined_scores = np.concatenate((running_scores[local_i], nz_scores))
                combined_cols   = np.concatenate((running_cols[local_i], global_cols))

                if len(combined_scores) <= effective_k:
                    order = np.argsort(combined_scores)[::-1]
                else:
                    part  = np.argpartition(combined_scores, -effective_k)[-effective_k:]
                    order = part[np.argsort(combined_scores[part])[::-1]]

                running_scores[local_i] = combined_scores[order][:effective_k]
                running_cols[local_i]   = combined_cols[order][:effective_k]

            del sim_csr, s2s3_block_T

        # Accumulate this query chunk into flat numpy arrays
        chunk_s1_idx   = []
        chunk_s2s3_idx = []
        chunk_scores   = []
        chunk_ranks    = []

        for local_i in range(chunk_n):
            global_i = q_start + local_i
            # Filter valid elements (score > 0)
            valid = running_scores[local_i] > 0
            if not np.any(valid):
                continue
            
            valid_scores = running_scores[local_i][valid]
            valid_cols   = running_cols[local_i][valid]
            n_out = len(valid_scores)

            chunk_s1_idx.extend([global_i] * n_out)
            chunk_s2s3_idx.extend(valid_cols.tolist())
            chunk_scores.extend(valid_scores.tolist())
            chunk_ranks.extend(range(1, n_out + 1))

        if chunk_s1_idx:
            chunk_frames.append(pd.DataFrame({
                "s1_row_idx":   np.array(chunk_s1_idx,   dtype=np.int32),
                "s2s3_row_idx": np.array(chunk_s2s3_idx, dtype=np.int32),
                "score":        np.array(chunk_scores,   dtype=np.float32),
                "rank":         np.array(chunk_ranks,    dtype=np.int16),
            }))

        del s1_chunk

    if not chunk_frames:
        return pd.DataFrame(
            columns=["s1_row_idx", "s2s3_row_idx", "score", "rank"]
        )

    result_df = pd.concat(chunk_frames, ignore_index=True)
    del chunk_frames

    result_df = result_df.sort_values(
        ["s1_row_idx", "rank"]
    ).reset_index(drop=True)

    return result_df


# ---------------------------------------------------------------------------
# Phase 1.7 — Multi-view candidate UNION (Char-TFIDF + BM25)
# ---------------------------------------------------------------------------

def union_candidate_results(
    tfidf_results_df: pd.DataFrame,
    bm25_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Char-TFIDF and BM25 candidate DataFrames into a single UNION set.

    For each S1 entity, the union contains every candidate ID that was
    retrieved by *either* retrieval channel.  Duplicates (same
    ``source1_entity_id`` / ``candidate_entity_id`` pair) are removed while
    preserving Char-TFIDF candidates first (Char-TFIDF order priority).

    Retrieval-agreement metadata is attached to every row:

    * ``retrieved_by_char_tfidf`` – 1 if Char-TFIDF retrieved this candidate
    * ``retrieved_by_bm25``       – 1 if BM25 retrieved this candidate
    * ``retrieval_agreement_count`` – sum of the two flags (1 or 2)

    The existing ``cosine_similarity`` / ``rank`` columns from the
    Char-TFIDF frame are preserved unchanged.  BM25-only candidates receive
    ``cosine_similarity = 0.0`` and ``rank = None`` (integer column becomes
    float/NaN, which callers should handle).

    The ``bm25_score`` and ``bm25_rank`` columns from the BM25 frame are
    included for both BM25-retrieved and combined candidates.
    Char-TFIDF-only candidates receive ``bm25_score = 0.0``,
    ``bm25_rank = None``.

    Parameters
    ----------
    tfidf_results_df : pd.DataFrame
        Output of :func:`search_candidates_mock` or equivalent.
        Expected columns: ``source1_entity_id``, ``candidate_entity_id``,
        ``cosine_similarity``, ``rank``.
    bm25_results_df : pd.DataFrame
        Output of :func:`~src.blocking_bm25.BM25Index.search` or
        :func:`~src.blocking_bm25.search_candidates_bm25`.
        Expected columns: ``source1_entity_id``, ``candidate_entity_id``,
        ``bm25_score``, ``bm25_rank``.

    Returns
    -------
    pd.DataFrame
        One row per unique (``source1_entity_id``, ``candidate_entity_id``)
        pair with columns:

        * ``source1_entity_id``
        * ``candidate_entity_id``
        * ``cosine_similarity``        – from Char-TFIDF (0.0 for BM25-only)
        * ``rank``                     – Char-TFIDF rank (NaN for BM25-only)
        * ``bm25_score``               – from BM25 (0.0 for Char-TFIDF-only)
        * ``bm25_rank``                – BM25 rank (NaN for Char-TFIDF-only)
        * ``retrieved_by_char_tfidf``  – int {0, 1}
        * ``retrieved_by_bm25``        – int {0, 1}
        * ``retrieval_agreement_count``– int {1, 2}

        Rows are sorted by (source1_entity_id, retrieved_by_char_tfidf DESC,
        rank ASC, bm25_rank ASC) for a deterministic, Char-TFIDF-priority
        ordering.

    Raises
    ------
    TypeError
        If either argument is not a pandas DataFrame.
    ValueError
        If required columns are missing.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(tfidf_results_df, pd.DataFrame):
        raise TypeError(
            f"tfidf_results_df must be a pd.DataFrame, "
            f"got {type(tfidf_results_df).__name__}."
        )
    if not isinstance(bm25_results_df, pd.DataFrame):
        raise TypeError(
            f"bm25_results_df must be a pd.DataFrame, "
            f"got {type(bm25_results_df).__name__}."
        )

    tfidf_required = {
        "source1_entity_id", "candidate_entity_id",
        "cosine_similarity", "rank",
    }
    bm25_required = {
        "source1_entity_id", "candidate_entity_id",
        "bm25_score", "bm25_rank",
    }

    missing_tfidf = tfidf_required - set(tfidf_results_df.columns)
    if missing_tfidf:
        raise ValueError(
            f"union_candidate_results(): tfidf_results_df is missing "
            f"column(s): {sorted(missing_tfidf)}.  "
            f"Found: {list(tfidf_results_df.columns)}"
        )
    missing_bm25 = bm25_required - set(bm25_results_df.columns)
    if missing_bm25:
        raise ValueError(
            f"union_candidate_results(): bm25_results_df is missing "
            f"column(s): {sorted(missing_bm25)}.  "
            f"Found: {list(bm25_results_df.columns)}"
        )

    # ------------------------------------------------------------------
    # Normalise / flag each source frame
    # ------------------------------------------------------------------
    tfidf_df = tfidf_results_df[
        ["source1_entity_id", "candidate_entity_id", "cosine_similarity", "rank"]
    ].copy()
    tfidf_df["retrieved_by_char_tfidf"] = 1
    tfidf_df["retrieved_by_bm25"]       = 0
    tfidf_df["bm25_score"]              = 0.0
    tfidf_df["bm25_rank"]               = np.nan

    bm25_df = bm25_results_df[
        ["source1_entity_id", "candidate_entity_id", "bm25_score", "bm25_rank"]
    ].copy()
    bm25_df["retrieved_by_char_tfidf"] = 0
    bm25_df["retrieved_by_bm25"]       = 1
    bm25_df["cosine_similarity"]        = 0.0
    bm25_df["rank"]                     = np.nan

    # ------------------------------------------------------------------
    # Outer-merge on the pair key
    # ------------------------------------------------------------------
    _PAIR_KEY = ["source1_entity_id", "candidate_entity_id"]

    merged = tfidf_df.merge(
        bm25_df[["source1_entity_id", "candidate_entity_id",
                 "bm25_score", "bm25_rank"]].rename(
            columns={
                "bm25_score": "_bm25_score_r",
                "bm25_rank":  "_bm25_rank_r",
            }
        ),
        on=_PAIR_KEY,
        how="outer",
        indicator=True,
    )

    # For rows present in both frames, fill in BM25 metadata from the right side
    both_mask = merged["_merge"] == "both"
    merged.loc[both_mask, "retrieved_by_bm25"] = 1
    merged.loc[both_mask, "bm25_score"] = merged.loc[both_mask, "_bm25_score_r"]
    merged.loc[both_mask, "bm25_rank"]  = merged.loc[both_mask, "_bm25_rank_r"]

    # For rows that came only from BM25 (right_only), fill from bm25_df
    right_mask = merged["_merge"] == "right_only"
    merged.loc[right_mask, "retrieved_by_char_tfidf"] = 0
    merged.loc[right_mask, "retrieved_by_bm25"]       = 1
    merged.loc[right_mask, "cosine_similarity"]        = 0.0
    merged.loc[right_mask, "rank"]                     = np.nan
    merged.loc[right_mask, "bm25_score"] = merged.loc[right_mask, "_bm25_score_r"]
    merged.loc[right_mask, "bm25_rank"]  = merged.loc[right_mask, "_bm25_rank_r"]

    # Drop merge helper columns
    merged = merged.drop(columns=["_bm25_score_r", "_bm25_rank_r", "_merge"])

    # ------------------------------------------------------------------
    # Agreement count
    # ------------------------------------------------------------------
    merged["retrieval_agreement_count"] = (
        merged["retrieved_by_char_tfidf"].astype(int)
        + merged["retrieved_by_bm25"].astype(int)
    )

    # ------------------------------------------------------------------
    # Deterministic sort:
    #   1. source1_entity_id ascending (string sort for stable order)
    #   2. retrieved_by_char_tfidf descending (Char-TFIDF candidates first)
    #   3. rank ascending (NaN → end, i.e. BM25-only go to back)
    #   4. bm25_rank ascending (NaN → end, tie-break for equal TFIDF rank)
    # ------------------------------------------------------------------
    merged = merged.sort_values(
        ["source1_entity_id",
         "retrieved_by_char_tfidf",  # ascending=False below
         "rank",                     # NaN → last
         "bm25_rank"],               # NaN → last
        ascending=[True, False, True, True],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Cast retrieval flags to int (outer merge can introduce float NaN).
    # pandas 3.x: coerce to numeric first if any column is StringDtype.
    # ------------------------------------------------------------------
    for col in ("retrieved_by_char_tfidf", "retrieved_by_bm25",
                "retrieval_agreement_count"):
        merged[col] = (
            pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        )

    # Final column order
    final_cols = [
        "source1_entity_id",
        "candidate_entity_id",
        "cosine_similarity",
        "rank",
        "bm25_score",
        "bm25_rank",
        "retrieved_by_char_tfidf",
        "retrieved_by_bm25",
        "retrieval_agreement_count",
    ]
    return merged[final_cols]


# ---------------------------------------------------------------------------
# Phase 1.7 — compare_retrieval_recall
# ---------------------------------------------------------------------------

def compare_retrieval_recall(
    tfidf_candidates_by_s1: dict,
    bm25_results_df: pd.DataFrame,
    ground_truth_df,
    label_tfidf: str = "char_tfidf",
    label_union: str = "union",
) -> dict:
    """Compare candidate recall between Char-TFIDF-only and the UNION approach.

    Computes recall and candidate-count statistics for two scenarios:

    * **Baseline**: Char-TFIDF candidates only (passed as *tfidf_candidates_by_s1*)
    * **Experiment**: UNION of Char-TFIDF and BM25 candidates

    Only training ground truth should be used here.  Do NOT pass validation
    or test ground truth.

    Parameters
    ----------
    tfidf_candidates_by_s1 : dict
        ``source1_entity_id`` → list of ``candidate_entity_id`` (Char-TFIDF only).
        Produced by :func:`write_candidate_pairs` input or Phase 1.2/1.5
        directly.
    bm25_results_df : pd.DataFrame
        Output of :class:`~src.blocking_bm25.BM25Index.search` with columns
        ``source1_entity_id``, ``candidate_entity_id``, (and optionally
        ``bm25_score``, ``bm25_rank``).
    ground_truth_df : pd.DataFrame or None
        Training ground-truth with columns ``source1_entity_id`` and
        ``match_entity_ids`` (comma-separated).
        Pass ``None`` if ground truth is unavailable.
    label_tfidf : str, default ``"char_tfidf"``
        Label used in the returned dict for the baseline scenario.
    label_union : str, default ``"union"``
        Label used in the returned dict for the experiment scenario.

    Returns
    -------
    dict with keys:

    * ``label_tfidf``  → dict (same structure as :func:`compute_candidate_recall`)
    * ``label_union``  → dict (same structure as :func:`compute_candidate_recall`)
    * ``"n_bm25_only_candidates"`` → int, new candidates added exclusively by BM25
    * ``"n_overlap_candidates"``   → int, candidates retrieved by BOTH methods
    * ``"n_tfidf_total_pairs"``    → int, total Char-TFIDF candidate pairs
    * ``"n_bm25_total_pairs"``     → int, total BM25 candidate pairs
    * ``"n_union_total_pairs"``    → int, total union candidate pairs
    """
    if not isinstance(tfidf_candidates_by_s1, dict):
        raise TypeError(
            "tfidf_candidates_by_s1 must be a dict, "
            f"got {type(tfidf_candidates_by_s1).__name__}."
        )
    if not isinstance(bm25_results_df, pd.DataFrame):
        raise TypeError(
            "bm25_results_df must be a pd.DataFrame, "
            f"got {type(bm25_results_df).__name__}."
        )

    # ------------------------------------------------------------------
    # Build the union candidates_by_s1 dict
    # ------------------------------------------------------------------
    # Start with a copy of the TFIDF dict (preserve order)
    union_candidates: dict = {
        s1_id: list(cands)
        for s1_id, cands in tfidf_candidates_by_s1.items()
    }

    # Track overlap statistics
    n_bm25_only = 0
    n_overlap   = 0

    if "source1_entity_id" in bm25_results_df.columns and \
       "candidate_entity_id" in bm25_results_df.columns:

        for s1_id, grp in bm25_results_df.groupby("source1_entity_id", sort=False):
            s1_id = str(s1_id)
            bm25_cands = grp["candidate_entity_id"].tolist()

            existing_set = set(union_candidates.get(s1_id, []))
            if s1_id not in union_candidates:
                union_candidates[s1_id] = []

            for cid in bm25_cands:
                if cid in existing_set:
                    n_overlap += 1
                else:
                    n_bm25_only += 1
                    union_candidates[s1_id].append(cid)
                    existing_set.add(cid)

    # ------------------------------------------------------------------
    # Compute recall for both scenarios
    # ------------------------------------------------------------------
    tfidf_recall = compute_candidate_recall(
        tfidf_candidates_by_s1, ground_truth_df
    )
    union_recall = compute_candidate_recall(
        union_candidates, ground_truth_df
    )

    n_tfidf_total = sum(len(v) for v in tfidf_candidates_by_s1.values())
    n_bm25_total  = (
        len(bm25_results_df)
        if isinstance(bm25_results_df, pd.DataFrame) else 0
    )
    n_union_total = sum(len(v) for v in union_candidates.values())

    return {
        label_tfidf:                tfidf_recall,
        label_union:                union_recall,
        "n_bm25_only_candidates":   n_bm25_only,
        "n_overlap_candidates":     n_overlap,
        "n_tfidf_total_pairs":      n_tfidf_total,
        "n_bm25_total_pairs":       n_bm25_total,
        "n_union_total_pairs":      n_union_total,
    }
