"""
features.py
-----------
Phase 2 — Feature Engineering.

Public API:
    load_feature_tsv(path)                          -> pd.DataFrame  (Phase 2.1)
    build_pair_table(candidate_pairs_df,
                     s1_df, s2s3_df,
                     blocking_scores_df=None)       -> pd.DataFrame  (Phase 2.1)
    add_name_similarity_features(pair_df)           -> pd.DataFrame  (Phase 2.2)
    add_address_similarity_features(pair_df)        -> pd.DataFrame  (Phase 2.3)
    load_blocking_scores(path)                      -> pd.DataFrame  (Phase 2.4)
    add_metadata_features(pair_df,
                          blocking_scores=None)     -> pd.DataFrame  (Phase 2.4)
    extract_pair_ids(pair_df)                       -> pd.DataFrame  (Phase 2.5)
    build_feature_matrix(pair_df,
                         return_ids=False)          -> pd.DataFrame  (Phase 2.5)
    write_feature_matrix(feature_df, output_path,
                         id_df=None,
                         include_ids=True)          -> None          (Phase 2.5)
    load_ground_truth(path)                         -> pd.DataFrame  (Phase 2.6)
    add_ground_truth_labels(pair_df,
                            ground_truth_df)        -> pd.DataFrame  (Phase 2.6)
    compute_candidate_positive_recall(
                            pair_df,
                            ground_truth_df)        -> dict           (Phase 2.6)
    write_labeled_pairs(labeled_df, output_path)    -> None          (Phase 2.6)

NOT implemented here (future phases):
    - Negative downsampling / 1:5 sampling (Phase 3+)
    - ML model training or inference (Phase 3+)

Design notes
~~~~~~~~~~~~
* build_pair_table() consumes the candidate_pairs TSV produced by Phase 1
  write_candidate_pairs() exactly as written — it does NOT re-run blocking,
  re-rank candidates, or alter the candidate set in any way.
* The comma-separated candidate_entity_ids cell is exploded into one row
  per (source1_entity_id, candidate_entity_id) pair, preserving the Phase 1
  ordering (first listed = highest cosine similarity).
* S1 feature columns are prefixed ``s1_``; S2/S3 candidate columns are
  prefixed ``candidate_``.
* If blocking_scores_df is supplied it must contain exactly the columns
  [source1_entity_id, candidate_entity_id, cosine_similarity].  The score
  for each pair is joined as ``blocking_cosine_similarity``.  Scores are
  NEVER recomputed here.
* An S1 row whose candidate_entity_ids cell is empty or NaN contributes
  zero pair rows — the row is silently dropped (per spec requirement 8).
* add_name_similarity_features() adds exactly four name similarity metrics
  computed between s1_clean_name and candidate_clean_name:
  1. name_jaro_winkler
  2. name_levenshtein
  3. name_token_sort_ratio
  4. name_token_set_ratio
  Values are bounded in [0.0, 1.0] and empty string cases are handled
  deterministically.
* add_metadata_features() adds exactly three metadata/pipeline features:
  1. exact_country_match (1.0 if identical non-empty, 0.0 otherwise; open-set)
  2. blocking_cosine_sim (Phase 1 blocking cosine similarity carried forward, NEVER recomputed)
  3. source_type (1 for S2 candidate, 0 for S3 candidate)
* build_feature_matrix() extracts and validates the approved 10 model features
  into a clean, numeric matrix in canonical order:
  [name_jaro_winkler, name_levenshtein, name_token_sort_ratio, name_token_set_ratio,
   address_jaccard, address_numeric_overlap, address_lcs_ratio,
   exact_country_match, blocking_cosine_sim, source_type]
  Pair identifiers (source1_entity_id, candidate_entity_id) are kept strictly
  separate from model features and can be extracted via extract_pair_ids() or
  return_ids=True.
* add_ground_truth_labels() (Phase 2.6) appends exactly one new column ``label``
  (integer 0 or 1) to the Phase 2.5 pair table.
  - label = 1 if candidate_entity_id ∈ ground-truth match set for that source1_entity_id.
  - label = 0 otherwise.
  Labels are assigned ONLY to existing Phase 1 candidate pairs.  No new pairs are
  created, no pairs are removed, no feature values are modified.
  Ground-truth positives absent from the candidate set are NOT added; they are
  counted as missed candidates / false negatives of Phase 1 blocking.
  Negative downsampling is NOT performed here (belongs to Phase 3+).
* compute_candidate_positive_recall() (Phase 2.6) reports blocking recall:
  for every ground-truth positive pair (s1_id, match_id), checks whether
  that exact pair is present in the labeled candidate set.
  This is CANDIDATE-GENERATION RECALL, not model recall or final F0.5.
* load_ground_truth() (Phase 2.6) loads and validates the challenge ground-truth TSV.
  Duplicate source1_entity_id rows with conflicting match sets raise ValueError.
  Duplicate rows with identical match sets are safely merged (set union is idempotent).
  Empty matching_entity_ids is a valid zero-match entity, not missing data.
"""

import csv
import math
import os
import re
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns required in the Phase 2.0 feature TSV files.
FEATURE_REQUIRED_COLS = {"entity_id", "clean_name", "clean_address", "clean_country"}

# Columns required in the Phase 1 candidate-pairs TSV.
CANDIDATE_PAIRS_REQUIRED_COLS = {"source1_entity_id", "candidate_entity_ids"}

# Columns required in an optional blocking-scores DataFrame.
BLOCKING_SCORES_REQUIRED_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "cosine_similarity",
}

# Final column order of the pair table.
PAIR_TABLE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "s1_clean_name",
    "s1_clean_address",
    "s1_clean_country",
    "candidate_clean_name",
    "candidate_clean_address",
    "candidate_clean_country",
]

# Column appended only when blocking scores are supplied.
BLOCKING_SCORE_COL = "blocking_cosine_similarity"

# Columns required in a candidate pair table for name similarity feature extraction.
NAME_SIMILARITY_REQUIRED_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "s1_clean_name",
    "candidate_clean_name",
}

# The four name similarity features produced in Phase 2.2.
NAME_SIMILARITY_FEATURE_COLS = [
    "name_jaro_winkler",
    "name_levenshtein",
    "name_token_sort_ratio",
    "name_token_set_ratio",
]

# Columns required in a candidate pair table for address similarity feature extraction.
ADDRESS_SIMILARITY_REQUIRED_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "s1_clean_address",
    "candidate_clean_address",
}

# The three address similarity features produced in Phase 2.3.
ADDRESS_SIMILARITY_FEATURE_COLS = [
    "address_jaccard",
    "address_numeric_overlap",
    "address_lcs_ratio",
]

# Columns required in a candidate pair table for metadata feature extraction.
METADATA_REQUIRED_COLS = {
    "source1_entity_id",
    "candidate_entity_id",
    "s1_clean_country",
    "candidate_clean_country",
}

# The three metadata features produced in Phase 2.4.
METADATA_FEATURE_COLS = [
    "exact_country_match",
    "blocking_cosine_sim",
    "source_type",
]

# The three retrieval-agreement features added in Phase 2.7 (BM25 merge).
# These flags record which retrieval channel(s) produced each candidate pair.
# Values are always int {0, 1} or {1, 2}; never NaN; never infinite.
#   retrieved_by_char_tfidf : 1 if Char-TFIDF retrieved the candidate, else 0
#   retrieved_by_bm25       : 1 if BM25 retrieved the candidate, else 0
#   retrieval_agreement_count: sum of the two flags (always 1 or 2 for union pairs)
RETRIEVAL_AGREEMENT_FEATURE_COLS = [
    "retrieved_by_char_tfidf",
    "retrieved_by_bm25",
    "retrieval_agreement_count",
]

# Default path to persisted mock blocking scores artifact.
DEFAULT_BLOCKING_SCORES_PATH = os.path.join("output", "blocking_scores_mock.tsv")

# Pair identifier columns kept separate from the model feature matrix (Phase 2.5).
PAIR_ID_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
]

# The approved 13 model feature columns in canonical order (Phase 2.5 / 2.7).
# Original 10 canonical features (Phase 2.1–2.4) are UNCHANGED.
# Three retrieval-agreement features appended (Phase 2.7 BM25 merge):
#   11. retrieved_by_char_tfidf
#   12. retrieved_by_bm25
#   13. retrieval_agreement_count
#
# IMPORTANT: build_feature_matrix() validates against this list.
# If retrieval-agreement columns are absent from the pair table (e.g. when
# running a baseline Char-TFIDF-only pipeline), callers must add them
# before calling build_feature_matrix().  The helper
# add_retrieval_agreement_defaults() in this module fills them with the
# Char-TFIDF-only defaults (1, 0, 1) for backward-compatibility.
FEATURE_COLUMNS = [
    # --- Phase 2.2: name similarity ---
    "name_jaro_winkler",
    "name_levenshtein",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    # --- Phase 2.3: address similarity ---
    "address_jaccard",
    "address_numeric_overlap",
    "address_lcs_ratio",
    # --- Phase 2.4: metadata ---
    "exact_country_match",
    "blocking_cosine_sim",
    "source_type",
    # --- Phase 2.7: retrieval-agreement (BM25 merge) ---
    "retrieved_by_char_tfidf",
    "retrieved_by_bm25",
    "retrieval_agreement_count",
]

# The original 10 features from Phase 2.1–2.4 (kept for explicit reference).
FEATURE_COLUMNS_BASELINE = [
    "name_jaro_winkler",
    "name_levenshtein",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "address_jaccard",
    "address_numeric_overlap",
    "address_lcs_ratio",
    "exact_country_match",
    "blocking_cosine_sim",
    "source_type",
]

# Default path to persisted mock feature matrix artifact.
DEFAULT_FEATURE_MATRIX_PATH = os.path.join("output", "feature_matrix_mock.tsv")

# ---------------------------------------------------------------------------
# Phase 2.6 constants
# ---------------------------------------------------------------------------

# Columns required in a ground-truth DataFrame.
GROUND_TRUTH_REQUIRED_COLS = {"source1_entity_id", "matching_entity_ids"}

# The single label column added by Phase 2.6.
LABEL_COL = "label"

# Default path to the persisted mock labeled candidate-pairs artifact.
DEFAULT_LABELED_PAIRS_PATH = os.path.join("output", "labeled_candidate_pairs_mock.tsv")


# ---------------------------------------------------------------------------
# Phase 2.1 — feature TSV loader
# ---------------------------------------------------------------------------

def load_feature_tsv(path: str) -> pd.DataFrame:
    """Load a Phase 2.0 feature TSV and return a validated DataFrame.

    Expected schema:
        entity_id | clean_name | clean_address | clean_country

    Parameters
    ----------
    path : str
        File-system path to the TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns entity_id, clean_name, clean_address,
        clean_country.  All values are preserved as strings.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing, any entity_id is blank, or
        duplicate entity_id values are found.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )

    # Required columns
    missing_cols = FEATURE_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Feature file '{path}' is missing required column(s): "
            f"{sorted(missing_cols)}.  Found: {list(df.columns)}"
        )

    # Blank entity_id
    blank_ids = df["entity_id"].str.strip() == ""
    if blank_ids.any():
        raise ValueError(
            f"Feature file '{path}' contains {int(blank_ids.sum())} row(s) "
            "with a missing or empty 'entity_id'."
        )

    # Duplicate entity_id
    dupes = df["entity_id"][df["entity_id"].duplicated(keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"Feature file '{path}' contains duplicate 'entity_id' values: "
            f"{sorted(dupes.unique().tolist())}"
        )

    return df


# ---------------------------------------------------------------------------
# Phase 2.1 — pair table builder
# ---------------------------------------------------------------------------

def build_pair_table(
    candidate_pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    blocking_scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Explode Phase 1 candidates into one row per pair and join feature fields.

    This function does NOT re-run blocking, does NOT recompute cosine scores,
    does NOT compute similarity features, and does NOT create labels.

    Parameters
    ----------
    candidate_pairs_df : pd.DataFrame
        The Phase 1 output.  Must contain:
        - ``source1_entity_id``   : str
        - ``candidate_entity_ids``: str, comma-separated candidate IDs
          (may be empty string for zero-candidate S1 rows).

    s1_df : pd.DataFrame
        Phase 2.0 feature file for the S1 corpus.  Must contain:
        entity_id, clean_name, clean_address, clean_country.

    s2s3_df : pd.DataFrame
        Phase 2.0 feature file for the S2/S3 corpus.  Must contain:
        entity_id, clean_name, clean_address, clean_country.

    blocking_scores_df : pd.DataFrame or None, default None
        Optional blocking scores from Phase 1
        ``search_candidates_mock()`` / ``search_candidates_scalable()``.
        When supplied, must contain:
        source1_entity_id, candidate_entity_id, cosine_similarity.
        The score is joined as ``blocking_cosine_similarity`` and is
        NEVER recomputed.  Pairs absent from this DataFrame receive NaN.

    Returns
    -------
    pd.DataFrame
        One row per (source1_entity_id, candidate_entity_id) pair with
        columns (in order):

        source1_entity_id
        candidate_entity_id
        s1_clean_name
        s1_clean_address
        s1_clean_country
        candidate_clean_name
        candidate_clean_address
        candidate_clean_country
        [blocking_cosine_similarity]   ← only present when scores supplied

        Row order mirrors the Phase 1 candidate list order
        (first-listed candidate = highest similarity).

    Raises
    ------
    ValueError
        If required columns are missing from any input DataFrame.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    _check_cols(candidate_pairs_df, CANDIDATE_PAIRS_REQUIRED_COLS, "candidate_pairs_df")
    _check_cols(s1_df,   FEATURE_REQUIRED_COLS, "s1_df")
    _check_cols(s2s3_df, FEATURE_REQUIRED_COLS, "s2s3_df")

    if blocking_scores_df is not None:
        _check_cols(blocking_scores_df, BLOCKING_SCORES_REQUIRED_COLS, "blocking_scores_df")

    # ------------------------------------------------------------------
    # 2. Explode comma-separated candidate_entity_ids → one row per pair
    #    Empty / NaN candidate_entity_ids cells produce zero rows.
    # ------------------------------------------------------------------
    pairs_rows = []
    for _, row in candidate_pairs_df.iterrows():
        s1_id   = str(row["source1_entity_id"]).strip()
        raw_ids = str(row["candidate_entity_ids"]).strip()

        # Empty cell or literal "nan" → zero pair rows (requirement 8)
        if not raw_ids or raw_ids.lower() == "nan":
            continue

        for cand_id in raw_ids.split(","):
            cand_id = cand_id.strip()
            if cand_id:
                pairs_rows.append(
                    {
                        "source1_entity_id":   s1_id,
                        "candidate_entity_id": cand_id,
                    }
                )

    if not pairs_rows:
        # Return empty DataFrame with correct schema
        cols = PAIR_TABLE_COLUMNS.copy()
        if blocking_scores_df is not None:
            cols.append(BLOCKING_SCORE_COL)
        return pd.DataFrame(columns=cols)

    pairs_df = pd.DataFrame(pairs_rows)

    # ------------------------------------------------------------------
    # 3. Join S1 feature fields
    # ------------------------------------------------------------------
    s1_lookup = s1_df[["entity_id", "clean_name", "clean_address", "clean_country"]].copy()
    s1_lookup = s1_lookup.rename(columns={
        "entity_id":     "source1_entity_id",
        "clean_name":    "s1_clean_name",
        "clean_address": "s1_clean_address",
        "clean_country": "s1_clean_country",
    })

    pairs_df = pairs_df.merge(s1_lookup, on="source1_entity_id", how="left")

    # ------------------------------------------------------------------
    # 4. Join candidate (S2/S3) feature fields
    # ------------------------------------------------------------------
    s2s3_lookup = s2s3_df[["entity_id", "clean_name", "clean_address", "clean_country"]].copy()
    s2s3_lookup = s2s3_lookup.rename(columns={
        "entity_id":     "candidate_entity_id",
        "clean_name":    "candidate_clean_name",
        "clean_address": "candidate_clean_address",
        "clean_country": "candidate_clean_country",
    })

    pairs_df = pairs_df.merge(s2s3_lookup, on="candidate_entity_id", how="left")

    # ------------------------------------------------------------------
    # 5. Optionally join blocking cosine similarity (do NOT recompute)
    # ------------------------------------------------------------------
    if blocking_scores_df is not None:
        score_lookup = blocking_scores_df[
            ["source1_entity_id", "candidate_entity_id", "cosine_similarity"]
        ].copy()
        score_lookup = score_lookup.rename(
            columns={"cosine_similarity": BLOCKING_SCORE_COL}
        )
        # Drop duplicate (s1_id, cand_id) score rows before joining to
        # avoid row fan-out; keep the first occurrence (highest rank).
        score_lookup = score_lookup.drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"], keep="first"
        )
        pairs_df = pairs_df.merge(
            score_lookup,
            on=["source1_entity_id", "candidate_entity_id"],
            how="left",
        )

    # ------------------------------------------------------------------
    # 6. Final column ordering and index reset
    # ------------------------------------------------------------------
    final_cols = PAIR_TABLE_COLUMNS.copy()
    if blocking_scores_df is not None:
        final_cols.append(BLOCKING_SCORE_COL)

    pairs_df = pairs_df[final_cols].reset_index(drop=True)

    return pairs_df


# ---------------------------------------------------------------------------
# Phase 2.2 — Name similarity features
# ---------------------------------------------------------------------------

def add_name_similarity_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exactly four name similarity features between s1_clean_name and candidate_clean_name.

    Features computed (Phase 2.2):
    1. name_jaro_winkler:
       Normalized Jaro-Winkler similarity in [0, 1].
    2. name_levenshtein:
       Normalized Levenshtein edit similarity (1 - normalized_Levenshtein_distance) in [0, 1].
    3. name_token_sort_ratio:
       RapidFuzz token_sort_ratio normalized to [0, 1] (score / 100.0).
    4. name_token_set_ratio:
       RapidFuzz token_set_ratio normalized to [0, 1] (score / 100.0).

    Edge Cases & Deterministic Behavior:
    - Identical non-empty strings: All four features evaluate to 1.0.
    - Both empty strings ("" vs ""):
        * name_jaro_winkler = 1.0 (empty strings are identical; character distance = 0)
        * name_levenshtein = 1.0 (0 edits required; normalized distance = 0, similarity = 1.0)
        * name_token_sort_ratio = 1.0 (empty token sequences match identically)
        * name_token_set_ratio = 0.0 (no tokens exist in empty strings; RapidFuzz evaluates
          token_set_ratio of empty token sets as 0.0)
    - One empty / one non-empty string: All four features evaluate to 0.0.
    - Missing / null values (None, NaN): Safely coerced to empty strings.
    - Empty pair table (0 rows): Returns DataFrame with all original columns plus the four
      feature columns with float dtype.
    - Output guarantees: Every feature column contains only finite numeric floats in [0.0, 1.0].
    - Non-destructive: Existing columns and their row ordering are preserved unchanged.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table from Phase 2.1 (build_pair_table).
        Must contain at least:
        - source1_entity_id
        - candidate_entity_id
        - s1_clean_name
        - candidate_clean_name

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing all columns from pair_df with the four new
        name similarity columns appended in order:
        [name_jaro_winkler, name_levenshtein, name_token_sort_ratio, name_token_set_ratio].

    Raises
    ------
    ValueError
        If any of the required columns are missing from pair_df.
    """
    _check_cols(pair_df, NAME_SIMILARITY_REQUIRED_COLS, "pair_df", caller="add_name_similarity_features")

    out_df = pair_df.copy()
    if out_df.empty:
        for col in NAME_SIMILARITY_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=float)
        return out_df

    s1_names = ["" if pd.isna(x) else str(x) for x in out_df["s1_clean_name"]]
    cand_names = ["" if pd.isna(x) else str(x) for x in out_df["candidate_clean_name"]]

    jw_scores = [
        min(1.0, max(0.0, float(JaroWinkler.normalized_similarity(s1, c))))
        for s1, c in zip(s1_names, cand_names)
    ]
    lev_scores = [
        min(1.0, max(0.0, float(Levenshtein.normalized_similarity(s1, c))))
        for s1, c in zip(s1_names, cand_names)
    ]
    sort_scores = [
        min(1.0, max(0.0, float(fuzz.token_sort_ratio(s1, c) / 100.0)))
        for s1, c in zip(s1_names, cand_names)
    ]
    set_scores = [
        min(1.0, max(0.0, float(fuzz.token_set_ratio(s1, c) / 100.0)))
        for s1, c in zip(s1_names, cand_names)
    ]

    out_df["name_jaro_winkler"] = jw_scores
    out_df["name_levenshtein"] = lev_scores
    out_df["name_token_sort_ratio"] = sort_scores
    out_df["name_token_set_ratio"] = set_scores

    return out_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_cols(
    df: pd.DataFrame, required: set, label: str, caller: str = "build_pair_table"
) -> None:
    """Raise ValueError if *df* is missing any column in *required*."""
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{caller}(): '{label}' is missing required column(s): "
            f"{sorted(missing)}.  Found: {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Phase 2.3 — Address similarity helpers (private)
# ---------------------------------------------------------------------------

def _address_token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two address strings.

    Tokenises each address by whitespace and computes set-based Jaccard:
        J(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|

    Edge cases (documented):
    - Both empty:           returns 1.0  (identical — empty set ≡ empty set)
    - One empty, one not:   returns 0.0  (disjoint)
    - Identical token sets: returns 1.0
    - Fully disjoint sets:  returns 0.0
    """
    tokens_a = set(a.split()) if a else set()
    tokens_b = set(b.split()) if b else set()

    if not tokens_a and not tokens_b:
        return 1.0                                  # both empty → identical

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return float(len(intersection)) / float(len(union))


def _address_numeric_overlap(a: str, b: str) -> float:
    """Binary overlap of digit sequences extracted from two address strings.

    Extracts all contiguous digit sequences (e.g. "12", "4") from each
    address using a regex, then returns:
        1.0  if the two sets share at least one identical digit sequence
        0.0  otherwise

    Documented edge cases:
    - No numbers in either address:            returns 0.0
    - One address has numbers, other does not: returns 0.0
    - Conflicting numbers (no common sequence): returns 0.0
    - At least one digit sequence in common:   returns 1.0
    """
    nums_a = set(re.findall(r"\d+", a))
    nums_b = set(re.findall(r"\d+", b))

    if nums_a & nums_b:
        return 1.0
    return 0.0


def _lcs_length(a: str, b: str) -> int:
    """Compute the length of the Longest Common Subsequence of strings *a* and *b*.

    Uses a space-optimised O(min(n, m)) DP algorithm (two-row rolling array).
    Characters need not be contiguous — only their relative order is preserved.

    This is Longest Common *Subsequence*, NOT Longest Common Substring.
    """
    # Ensure *a* is the shorter string to minimise memory.
    if len(a) > len(b):
        a, b = b, a

    n, m = len(a), len(b)
    # prev[j] = LCS length of a[:i] and b[:j] (previous row)
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)

    return prev[m]


def _address_lcs_ratio(a: str, b: str) -> float:
    """Longest Common Subsequence ratio between two address strings.

    Computes:
        lcs_ratio = LCS_length(a, b) / max(len(a), len(b))

    where LCS is Longest Common *Subsequence* (characters need not be
    contiguous; only their relative order must be preserved).

    Edge cases (documented):
    - Both empty:           returns 1.0  (0 / max(0, 0) → defined as 1.0)
    - One empty, one not:   returns 0.0  (LCS = 0, denominator > 0)
    - Identical strings:    returns 1.0
    - Unrelated strings:    returns a low value in [0, 1)
    """
    if not a and not b:
        return 1.0                              # both empty → identical

    denom = max(len(a), len(b))
    if denom == 0:
        return 1.0

    lcs = _lcs_length(a, b)
    return float(lcs) / float(denom)


# ---------------------------------------------------------------------------
# Phase 2.3 — Address similarity features
# ---------------------------------------------------------------------------

def add_address_similarity_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute exactly three address similarity features between
    s1_clean_address and candidate_clean_address.

    Features computed (Phase 2.3):
    1. address_jaccard
       Token-level set-based Jaccard similarity.  Each address is tokenised
       by whitespace and Jaccard(A, B) = |A∩B| / |A∪B|.  Result in [0, 1].

    2. address_numeric_overlap
       Binary indicator: 1.0 if the two addresses share at least one
       identical digit sequence (e.g. both contain "12"), 0.0 otherwise.
       Result in {0.0, 1.0}.

    3. address_lcs_ratio
       Longest Common *Subsequence* ratio:
           L / max(len(a), len(b))
       where L is the LCS length computed over the raw character sequences.
       Characters need not be contiguous — only relative order is preserved.
       This is NOT Longest Common Substring.  Result in [0, 1].

    Edge Cases & Deterministic Behavior:
    - Both addresses empty ("", ""):
        * address_jaccard          = 1.0  (empty token sets are identical)
        * address_numeric_overlap  = 0.0  (no digit sequences exist)
        * address_lcs_ratio        = 1.0  (empty strings are identical)
    - One empty / one non-empty:
        * address_jaccard          = 0.0
        * address_numeric_overlap  = 0.0
        * address_lcs_ratio        = 0.0
    - Missing / null values (None, NaN): safely coerced to empty strings.
    - Empty pair table (0 rows): returns DataFrame with all original columns
      plus the three feature columns with float dtype.
    - Output guarantees: every feature column contains only finite numeric
      floats in [0.0, 1.0]; no NaN, no Inf.
    - Non-destructive: existing columns and row ordering are unchanged.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table (from Phase 2.1/2.2).  Must contain at least:
        - source1_entity_id
        - candidate_entity_id
        - s1_clean_address
        - candidate_clean_address

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing all columns from pair_df with the three
        new address similarity columns appended in order:
        [address_jaccard, address_numeric_overlap, address_lcs_ratio].

    Raises
    ------
    ValueError
        If any of the required columns are missing from pair_df.
    """
    _check_cols(
        pair_df,
        ADDRESS_SIMILARITY_REQUIRED_COLS,
        "pair_df",
        caller="add_address_similarity_features",
    )

    out_df = pair_df.copy()

    if out_df.empty:
        for col in ADDRESS_SIMILARITY_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=float)
        return out_df

    # Coerce missing / null values to empty strings (same pattern as Phase 2.2)
    s1_addrs = ["" if pd.isna(x) else str(x) for x in out_df["s1_clean_address"]]
    cand_addrs = ["" if pd.isna(x) else str(x) for x in out_df["candidate_clean_address"]]

    jaccard_scores = [
        _address_token_jaccard(a, b) for a, b in zip(s1_addrs, cand_addrs)
    ]
    numeric_scores = [
        _address_numeric_overlap(a, b) for a, b in zip(s1_addrs, cand_addrs)
    ]
    lcs_scores = [
        _address_lcs_ratio(a, b) for a, b in zip(s1_addrs, cand_addrs)
    ]

    # Clamp to [0, 1] as a final safety net (all implementations are already
    # guaranteed to stay in range, but this guards against any future drift).
    out_df["address_jaccard"] = [
        min(1.0, max(0.0, float(v))) for v in jaccard_scores
    ]
    out_df["address_numeric_overlap"] = [
        min(1.0, max(0.0, float(v))) for v in numeric_scores
    ]
    out_df["address_lcs_ratio"] = [
        min(1.0, max(0.0, float(v))) for v in lcs_scores
    ]

    return out_df


# ---------------------------------------------------------------------------
# Phase 2.4 — Blocking scores loader
# ---------------------------------------------------------------------------

def load_blocking_scores(path: str) -> pd.DataFrame:
    """Load a Phase 1 blocking scores TSV file.

    Expected schema:
        source1_entity_id | candidate_entity_id | cosine_similarity
        (also accepts 'blocking_cosine_sim' or 'score' as the score column)

    Parameters
    ----------
    path : str
        File-system path to the TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns source1_entity_id, candidate_entity_id,
        cosine_similarity (as float).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing, or non-numeric/out-of-bounds scores are found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Blocking scores file not found: '{path}'")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )

    if "source1_entity_id" not in df.columns or "candidate_entity_id" not in df.columns:
        raise ValueError(
            f"Blocking scores file '{path}' is missing required columns. "
            f"Expected 'source1_entity_id' and 'candidate_entity_id'. Found: {list(df.columns)}"
        )

    score_col = None
    for cand in ("cosine_similarity", "blocking_cosine_sim", "score"):
        if cand in df.columns:
            score_col = cand
            break

    if score_col is None:
        raise ValueError(
            f"Blocking scores file '{path}' has no valid score column. "
            f"Expected one of ['cosine_similarity', 'blocking_cosine_sim', 'score']. Found: {list(df.columns)}"
        )

    # Convert scores to float and validate range [0, 1]
    scores = []
    for idx, val in enumerate(df[score_col]):
        try:
            f_val = float(val)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Row {idx} in '{path}' has non-numeric score: {val!r}"
            ) from e
        if not math.isfinite(f_val):
            raise ValueError(
                f"Row {idx} in '{path}' has non-finite score: {f_val}"
            )
        if not (0.0 <= f_val <= 1.0):
            raise ValueError(
                f"Row {idx} in '{path}' has score out of bounds [0.0, 1.0]: {f_val}"
            )
        scores.append(f_val)

    out = df[["source1_entity_id", "candidate_entity_id"]].copy()
    out["cosine_similarity"] = scores
    return out


# ---------------------------------------------------------------------------
# Phase 2.4 — Metadata helpers (private)
# ---------------------------------------------------------------------------

def _exact_country_match(c1, c2) -> float:
    """Exact country match comparison.

    Definition:
        1.0 if the two country values are non-empty and exactly equal
        0.0 otherwise

    Deterministic behavior for empty / missing values:
        If either c1 or c2 is empty, blank, or null (None, NaN), returns 0.0.
        Absence of country information does not assert an affirmative match.
        Never returns NaN or non-finite float.

    Open-set and generic:
        Does not use any whitelist, database, hardcoded list, or re-normalization.
        Arbitrary country strings are supported.
    """
    s1_str = "" if pd.isna(c1) else str(c1)
    cand_str = "" if pd.isna(c2) else str(c2)

    # Empty or whitespace-only strings evaluate deterministically to 0.0
    if not s1_str.strip() or not cand_str.strip():
        return 0.0

    return 1.0 if s1_str == cand_str else 0.0


def _extract_source_type(candidate_entity_id) -> int:
    """Determine candidate source type from candidate_entity_id.

    Definition:
        S2 candidate -> 1
        S3 candidate -> 0

    Validation:
        Strictly determined from candidate_entity_id prefix:
        - Starts with 'S2-' -> 1
        - Starts with 'S3-' -> 0
        - Any unexpected format raises ValueError.
        Does NOT infer from business content.
        Does NOT create one-hot columns.
    """
    if candidate_entity_id is None or pd.isna(candidate_entity_id):
        raise ValueError(
            "candidate_entity_id cannot be null/NaN when determining source_type."
        )

    cid = str(candidate_entity_id).strip()
    if not cid:
        raise ValueError(
            "candidate_entity_id cannot be empty when determining source_type."
        )

    if cid.startswith("S2-"):
        return 1
    elif cid.startswith("S3-"):
        return 0
    else:
        raise ValueError(
            f"Unexpected candidate_entity_id format: '{cid}'. "
            "Candidate IDs must start with 'S2-' (source S2) or 'S3-' (source S3)."
        )


def _validate_score_list(scores: list) -> list[float]:
    """Validate a list of scores to ensure numeric, finite, and in [0.0, 1.0]."""
    validated = []
    for idx, sc in enumerate(scores):
        if sc is None or pd.isna(sc):
            raise ValueError(
                f"Missing / NaN blocking cosine similarity score at index {idx}."
            )
        try:
            f_sc = float(sc)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Non-numeric blocking cosine similarity score at index {idx}: {sc!r}"
            ) from e
        if not math.isfinite(f_sc):
            raise ValueError(
                f"Non-finite blocking cosine similarity score at index {idx}: {f_sc}"
            )
        if not (0.0 <= f_sc <= 1.0):
            raise ValueError(
                f"Blocking cosine similarity score at index {idx} out of range [0.0, 1.0]: {f_sc}"
            )
        validated.append(f_sc)
    return validated


def _resolve_blocking_scores(
    pair_df: pd.DataFrame,
    blocking_scores: pd.DataFrame | dict | str | None = None,
) -> list[float]:
    """Resolve and validate Phase 1 blocking cosine similarity scores for pair_df.

    CRITICAL PIPELINE INVARIANT:
    This function NEVER fits a vectorizer, NEVER computes TF-IDF, NEVER computes
    cosine similarity, and NEVER fabricates/invents values.
    Scores must originate from the Phase 1 blocking stage.

    Provenance & Resolution order:
    1. If blocking_scores is a string: loaded via load_blocking_scores(path).
    2. If blocking_scores is a DataFrame: extracted by (source1_entity_id, candidate_entity_id).
    3. If blocking_scores is a dict: mapped by (source1_entity_id, candidate_entity_id)
       or {s1: {cand: score}}.
    4. If blocking_scores is None:
       a. If 'blocking_cosine_sim' is already in pair_df: values carried over.
       b. Else if 'blocking_cosine_similarity' is in pair_df: values carried over.
       c. Else if DEFAULT_BLOCKING_SCORES_PATH exists on disk: loaded from artifact.
       d. Otherwise: raises ValueError (scores unavailable).

    Validation:
        All returned scores must be finite numeric floats in [0.0, 1.0].
    """
    if pair_df.empty:
        return []

    score_lookup: dict[tuple[str, str], float] = {}

    if isinstance(blocking_scores, str):
        scores_df = load_blocking_scores(blocking_scores)
        for _, row in scores_df.iterrows():
            key = (str(row["source1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
            score_lookup[key] = float(row["cosine_similarity"])

    elif isinstance(blocking_scores, pd.DataFrame):
        _check_cols(blocking_scores, {"source1_entity_id", "candidate_entity_id"}, "blocking_scores", caller="add_metadata_features")
        score_col = None
        for cand in ("blocking_cosine_sim", "cosine_similarity", "score"):
            if cand in blocking_scores.columns:
                score_col = cand
                break
        if score_col is None:
            raise ValueError(
                f"blocking_scores DataFrame missing cosine score column. Found: {list(blocking_scores.columns)}"
            )
        for _, row in blocking_scores.iterrows():
            key = (str(row["source1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
            score_lookup[key] = float(row[score_col])

    elif isinstance(blocking_scores, dict):
        for k, v in blocking_scores.items():
            if isinstance(k, tuple) and len(k) == 2:
                score_lookup[(str(k[0]).strip(), str(k[1]).strip())] = float(v)
            elif isinstance(v, dict):
                s1_id = str(k).strip()
                for cand_id, sc in v.items():
                    score_lookup[(s1_id, str(cand_id).strip())] = float(sc)
            else:
                raise ValueError(
                    f"Invalid key/value format in blocking_scores dict: {k!r} -> {v!r}"
                )

    elif blocking_scores is None:
        if "blocking_cosine_sim" in pair_df.columns:
            raw_scores = pair_df["blocking_cosine_sim"].tolist()
            return _validate_score_list(raw_scores)
        elif "blocking_cosine_similarity" in pair_df.columns:
            raw_scores = pair_df["blocking_cosine_similarity"].tolist()
            return _validate_score_list(raw_scores)
        elif os.path.exists(DEFAULT_BLOCKING_SCORES_PATH):
            scores_df = load_blocking_scores(DEFAULT_BLOCKING_SCORES_PATH)
            for _, row in scores_df.iterrows():
                key = (str(row["source1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
                score_lookup[key] = float(row["cosine_similarity"])
        else:
            raise ValueError(
                "Phase 1 blocking cosine similarity scores are unavailable. "
                "Supply blocking_scores (as DataFrame, dict, or TSV path), or pre-join "
                "scores into pair_df. Phase 2.4 strictly forbids recomputing or inventing scores."
            )
    else:
        raise TypeError(
            f"Unsupported type for blocking_scores: {type(blocking_scores).__name__}. "
            "Expected pd.DataFrame, dict, str (path), or None."
        )

    # Lookup scores for all pairs in pair_df
    resolved_scores: list[float] = []
    missing_pairs = []
    for _, row in pair_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        cand = str(row["candidate_entity_id"]).strip()
        key = (s1, cand)
        if key in score_lookup:
            val = score_lookup[key]
            resolved_scores.append(val)
        else:
            missing_pairs.append(key)

    if missing_pairs:
        raise ValueError(
            f"Phase 1 blocking cosine similarity scores missing for {len(missing_pairs)} pair(s): "
            f"{missing_pairs[:5]}. Legitimate Phase 1 scores must be provided for every pair; "
            "scores cannot be fabricated or recomputed in Phase 2.4."
        )

    return _validate_score_list(resolved_scores)


# ---------------------------------------------------------------------------
# Phase 2.4 — Metadata features
# ---------------------------------------------------------------------------

def add_metadata_features(
    pair_df: pd.DataFrame,
    blocking_scores: pd.DataFrame | dict | str | None = None,
) -> pd.DataFrame:
    """Add exactly three approved metadata features to candidate pair table (Phase 2.4).

    Features computed / carried:
    1. exact_country_match:
       1.0 if s1_clean_country == candidate_clean_country (non-empty strings);
       0.0 otherwise.
       Open-set and generic — no country whitelist or hardcoded rules.
       Empty / missing country values deterministically evaluate to 0.0 without producing NaN.

    2. blocking_cosine_sim:
       Cosine similarity score carried over directly from Phase 1 blocking stage.
       CRITICAL: Scores are NEVER recomputed in Phase 2.4 (no TF-IDF, no refitting, no approximation).
       Must be finite numeric floats in [0.0, 1.0].
       Missing scores raise ValueError.

    3. source_type:
       Integer candidate source identity:
       1 if candidate_entity_id starts with 'S2-' (S2 candidate);
       0 if candidate_entity_id starts with 'S3-' (S3 candidate).
       Unexpected entity ID format raises ValueError.

    Output column ordering:
        Preserves all existing Phase 2.1–2.3 columns unchanged, and appends
        the three metadata features at the end in this exact order:
        [exact_country_match, blocking_cosine_sim, source_type]

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table from Phase 2.1–2.3. Must contain at least:
        source1_entity_id, candidate_entity_id, s1_clean_country, candidate_clean_country.
    blocking_scores : pd.DataFrame, dict, str, or None, default None
        Phase 1 blocking scores. Can be a DataFrame with [source1_entity_id, candidate_entity_id, cosine_similarity],
        a mapping dict, a file path to a scores TSV, or None (if already present in pair_df or loaded
        from default persisted artifact).

    Returns
    -------
    pd.DataFrame
        New DataFrame with existing columns preserved and the three metadata feature
        columns appended in order.

    Raises
    ------
    ValueError
        If required columns are missing, source IDs are invalid, or blocking scores are missing/invalid.
    """
    _check_cols(
        pair_df,
        METADATA_REQUIRED_COLS,
        "pair_df",
        caller="add_metadata_features",
    )

    out_df = pair_df.copy()

    if out_df.empty:
        for col in METADATA_FEATURE_COLS:
            if col in out_df.columns:
                out_df.drop(columns=[col], inplace=True)
        out_df["exact_country_match"] = pd.Series(dtype=float)
        out_df["blocking_cosine_sim"] = pd.Series(dtype=float)
        out_df["source_type"] = pd.Series(dtype=int)
        return out_df

    # 1. Exact country match (open-set, generic, deterministic empty handling)
    country_matches = [
        _exact_country_match(c1, c2)
        for c1, c2 in zip(out_df["s1_clean_country"], out_df["candidate_clean_country"])
    ]

    # 2. Blocking cosine similarity (carried over from Phase 1, never recomputed)
    cosine_scores = _resolve_blocking_scores(out_df, blocking_scores=blocking_scores)

    # 3. Source type (S2 -> 1, S3 -> 0, strict validation)
    source_types = [
        _extract_source_type(cid)
        for cid in out_df["candidate_entity_id"]
    ]

    # Append columns in exact order, preserving existing columns
    for col in METADATA_FEATURE_COLS:
        if col in out_df.columns:
            out_df.drop(columns=[col], inplace=True)

    out_df["exact_country_match"] = country_matches
    out_df["blocking_cosine_sim"] = cosine_scores
    out_df["source_type"] = source_types

    return out_df


# ---------------------------------------------------------------------------
# Phase 2.5 — Pair Identifiers Extraction
# ---------------------------------------------------------------------------

def extract_pair_ids(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate pair identifiers from a candidate pair table.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table containing at least ``source1_entity_id``
        and ``candidate_entity_id``.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly the columns [source1_entity_id, candidate_entity_id],
        preserving the exact row ordering and index of *pair_df*.

    Raises
    ------
    ValueError
        If required identifier columns are missing, any identifier is empty/blank,
        or duplicate candidate pairs are detected.
    """
    _check_cols(pair_df, set(PAIR_ID_COLUMNS), "pair_df", caller="extract_pair_ids")

    # Guard against blank or missing entity IDs
    s1_blank = pair_df["source1_entity_id"].isna() | (pair_df["source1_entity_id"].astype(str).str.strip() == "")
    cand_blank = pair_df["candidate_entity_id"].isna() | (pair_df["candidate_entity_id"].astype(str).str.strip() == "")
    if s1_blank.any() or cand_blank.any():
        raise ValueError(
            "extract_pair_ids(): candidate pair identifiers cannot be empty, blank, or null."
        )

    # Invariant: (source1_entity_id, candidate_entity_id) must uniquely identify each row
    dupes = pair_df.duplicated(subset=PAIR_ID_COLUMNS, keep=False)
    if dupes.any():
        sample_dupes = pair_df.loc[dupes, PAIR_ID_COLUMNS].head(2).to_dict("records")
        raise ValueError(
            f"extract_pair_ids(): duplicate candidate pair(s) detected: {sample_dupes}. "
            "(source1_entity_id, candidate_entity_id) must uniquely identify each row."
        )

    return pair_df[PAIR_ID_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Phase 2.5 — Full Feature Matrix Assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    pair_df: pd.DataFrame,
    return_ids: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble and validate the final model-ready feature matrix from the Phase 2.4/2.7 pair table.

    Extracts exactly the 13 approved predictive features in canonical order:
    1.  name_jaro_winkler          [0, 1]
    2.  name_levenshtein           [0, 1]
    3.  name_token_sort_ratio      [0, 1]
    4.  name_token_set_ratio       [0, 1]
    5.  address_jaccard            [0, 1]
    6.  address_numeric_overlap    {0, 1}
    7.  address_lcs_ratio          [0, 1]
    8.  exact_country_match        [0, 1]
    9.  blocking_cosine_sim        [0, 1]
    10. source_type                {0, 1}
    11. retrieved_by_char_tfidf    {0, 1}
    12. retrieved_by_bm25          {0, 1}
    13. retrieval_agreement_count  {1, 2}  (always 1 or 2 for union pairs)

    Validation rules:
    - Exactly the 13 canonical feature columns are included; no extra or missing features.
    - Canonical feature order is strictly enforced via FEATURE_COLUMNS.
    - Pair identifiers (source1_entity_id, candidate_entity_id) are validated and kept
      separate from the model features (never treated as numeric features).
    - Row order and row count match pair_df exactly (no sorting, deduplicating, or dropping).
    - Every feature has a numeric dtype (float or int).
    - No NaN, null, or infinite values.
    - All feature values fall strictly within their approved bounds. Out-of-range values raise ValueError.
    - Duplicate pairs raise ValueError explicitly.

    If the pair table was produced by a Char-TFIDF-only pipeline (no BM25 union),
    call :func:`add_retrieval_agreement_defaults` first to populate the three
    new columns with their Char-TFIDF-only defaults (1, 0, 1).

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table after Phase 2.4/2.7 containing all 13 feature columns
        plus pair identifiers.
    return_ids : bool, default False
        When False (default), returns the 13-column feature matrix DataFrame.
        When True, returns a 2-tuple: (feature_matrix_df, pair_ids_df).

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]
        The 13-column feature matrix, or (feature_matrix, pair_ids) if return_ids=True.

    Raises
    ------
    TypeError
        If pair_df is not a pandas DataFrame.
    ValueError
        If required columns are missing, duplicate pairs exist, non-numeric dtypes appear,
        NaN/infinite values exist, or any value is out-of-range.
    """
    if not isinstance(pair_df, pd.DataFrame):
        raise TypeError(
            f"build_feature_matrix(): expected pd.DataFrame, got {type(pair_df).__name__}."
        )

    # 1. Validate pair identifiers
    ids_df = extract_pair_ids(pair_df)

    # 2. Validate all required feature columns exist
    missing_feats = [col for col in FEATURE_COLUMNS if col not in pair_df.columns]
    if missing_feats:
        raise ValueError(
            f"build_feature_matrix(): missing required feature column(s): {missing_feats}. "
            "Input must contain all 13 approved features from Phase 2.1–2.7. "
            "For Char-TFIDF-only pipelines, call add_retrieval_agreement_defaults(pair_df) first."
        )

    # 3. Handle empty DataFrame edge case
    if pair_df.empty:
        empty_matrix = pd.DataFrame(columns=FEATURE_COLUMNS, dtype=float)
        empty_matrix["source_type"] = pd.Series(dtype=int)
        if return_ids:
            return empty_matrix, ids_df
        return empty_matrix

    # 4. Validate dtypes, NaNs, Infs, and value ranges for all 10 features
    for col in FEATURE_COLUMNS:
        series = pair_df[col]

        # Numeric dtype check
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(
                f"build_feature_matrix(): column '{col}' has non-numeric dtype '{series.dtype}'. "
                "Every feature must have a numeric dtype."
            )

        # NaN / null check
        if series.isna().any():
            n_nan = int(series.isna().sum())
            raise ValueError(
                f"build_feature_matrix(): column '{col}' contains {n_nan} NaN value(s). "
                "Feature matrix must not contain NaN."
            )

        # Infinite value check
        vals = series.to_numpy()
        if np.isinf(vals).any():
            raise ValueError(
                f"build_feature_matrix(): column '{col}' contains infinite values. "
                "Feature matrix must be strictly finite."
            )

        # Binary set validation for binary features
        # address_numeric_overlap and source_type: {0, 1}
        # retrieved_by_char_tfidf and retrieved_by_bm25: {0, 1}
        # retrieval_agreement_count: {1, 2}
        if col in ("address_numeric_overlap", "source_type",
                   "retrieved_by_char_tfidf", "retrieved_by_bm25"):
            invalid_binary = ~series.isin([0, 1, 0.0, 1.0])
            if invalid_binary.any():
                bad_val = series[invalid_binary].iloc[0]
                raise ValueError(
                    f"build_feature_matrix(): binary feature '{col}' contains invalid value {bad_val}. "
                    "Must be in {0, 1}."
                )

        if col == "retrieval_agreement_count":
            invalid_agreement = ~series.isin([1, 2, 1.0, 2.0])
            if invalid_agreement.any():
                bad_val = series[invalid_agreement].iloc[0]
                raise ValueError(
                    f"build_feature_matrix(): feature 'retrieval_agreement_count' contains "
                    f"invalid value {bad_val}. Must be in {{1, 2}} for union candidates."
                )

        # General [0, 1] range validation — retrieval_agreement_count is {1,2} so skip it
        if col != "retrieval_agreement_count":
            out_of_bounds = (series < 0.0) | (series > 1.0)
            if out_of_bounds.any():
                bad_val = series[out_of_bounds].iloc[0]
                raise ValueError(
                    f"build_feature_matrix(): feature '{col}' contains value out of approved range [0, 1]: {bad_val}. "
                    "Values must not be clipped silently."
                )

    # 5. Extract feature matrix in exact canonical order, preserving exact row order and index
    feature_matrix = pair_df[FEATURE_COLUMNS].copy()

    if return_ids:
        return feature_matrix, ids_df
    return feature_matrix


# ---------------------------------------------------------------------------
# Phase 2.5 — Feature Matrix Writer (Optional Persistence)
# ---------------------------------------------------------------------------

def write_feature_matrix(
    feature_df: pd.DataFrame,
    output_path: str,
    id_df: pd.DataFrame | None = None,
    include_ids: bool = True,
) -> None:
    """Persist the Phase 2.5/2.7 feature matrix to a TSV file.

    Parameters
    ----------
    feature_df : pd.DataFrame
        The 13-column model feature matrix in canonical order.
    output_path : str
        Destination file path.
    id_df : pd.DataFrame or None, default None
        Pair identifiers [source1_entity_id, candidate_entity_id]. When provided
        and include_ids=True, prepends pair ID columns for traceability.
    include_ids : bool, default True
        Whether to include pair identifier columns in the persisted artifact.

    Raises
    ------
    ValueError
        If feature columns do not match FEATURE_COLUMNS or row counts differ.
    """
    if list(feature_df.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"write_feature_matrix(): feature_df columns must match FEATURE_COLUMNS in order. "
            f"Expected: {FEATURE_COLUMNS}, Got: {list(feature_df.columns)}"
        )

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if include_ids and id_df is not None:
        if len(feature_df) != len(id_df):
            raise ValueError(
                f"write_feature_matrix(): row count mismatch between feature_df ({len(feature_df)}) "
                f"and id_df ({len(id_df)})."
            )
        out = pd.concat(
            [
                id_df[PAIR_ID_COLUMNS].reset_index(drop=True),
                feature_df[FEATURE_COLUMNS].reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        out = feature_df[FEATURE_COLUMNS].copy()

    out.to_csv(output_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Phase 2.6 — Ground-truth loader
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> pd.DataFrame:
    """Load and validate a challenge ground-truth TSV file (Phase 2.6).

    Expected schema (tab-separated, header on row 0):
        source1_entity_id   | matching_entity_ids
        S1-001              | S2-099,S3-010
        S1-002              | S2-101
        S1-003              |                   ← zero matches; valid

    The ``matching_entity_ids`` field is stored **as-is** (raw string).
    Parsing into a Python set is the responsibility of
    ``add_ground_truth_labels()`` and ``compute_candidate_positive_recall()``.

    Validation
    ----------
    1. Required columns ``source1_entity_id`` and ``matching_entity_ids`` exist.
    2. No ``source1_entity_id`` value is empty / blank.
    3. Duplicate ``source1_entity_id`` rows:
       * If the two rows have **identical** match sets → safely merged (kept once).
       * If the two rows have **conflicting** match sets → ``ValueError`` raised.
         Prefer failing clearly rather than silently guessing.

    Parameters
    ----------
    path : str
        File-system path to the ground-truth TSV.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with columns [source1_entity_id, matching_entity_ids].
        ``matching_entity_ids`` values are raw strings (may be empty strings for
        zero-match entities; never NaN).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        On schema violations, blank IDs, or conflicting duplicate rows.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground-truth file not found: '{path}'")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )

    # 1. Required columns
    missing_cols = GROUND_TRUTH_REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Ground-truth file '{path}' is missing required column(s): "
            f"{sorted(missing_cols)}.  Found: {list(df.columns)}"
        )

    # 2. Blank source1_entity_id
    blank_ids = df["source1_entity_id"].str.strip() == ""
    if blank_ids.any():
        raise ValueError(
            f"Ground-truth file '{path}' contains {int(blank_ids.sum())} row(s) "
            "with a missing or empty 'source1_entity_id'."
        )

    # 3. Duplicate source1_entity_id handling
    dup_mask = df["source1_entity_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = df.loc[dup_mask, "source1_entity_id"].unique().tolist()
        conflicts = []
        for s1_id in dup_ids:
            rows = df.loc[df["source1_entity_id"] == s1_id, "matching_entity_ids"].tolist()
            # Parse each row's match list into a frozenset for comparison
            parsed = [
                frozenset(
                    mid.strip()
                    for mid in raw.split(",")
                    if mid.strip()
                )
                for raw in rows
            ]
            # All parsed sets must be identical; otherwise it's a conflict
            if len(set(parsed)) > 1:
                conflicts.append(
                    f"  '{s1_id}': {[str(set(p)) for p in parsed]}"
                )

        if conflicts:
            raise ValueError(
                "Ground-truth file '{}' contains duplicate 'source1_entity_id' rows "
                "with CONFLICTING match sets (cannot safely merge):\n{}".format(
                    path, "\n".join(conflicts)
                )
            )

        # Safe merge: identical sets → keep first occurrence
        df = df.drop_duplicates(subset=["source1_entity_id"], keep="first")

    # 4. Ensure no NaN leaked into matching_entity_ids (keep_default_na=False guards
    #    against this, but belt-and-braces check)
    nan_mask = df["matching_entity_ids"].isna()
    if nan_mask.any():
        raise ValueError(
            f"Ground-truth file '{path}' contains {int(nan_mask.sum())} NaN value(s) "
            "in 'matching_entity_ids'. Empty match lists should be represented as "
            "an empty string, not NaN."
        )

    return df[["source1_entity_id", "matching_entity_ids"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 2.6 — Ground-truth label assignment
# ---------------------------------------------------------------------------

def add_ground_truth_labels(
    pair_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append a binary ``label`` column to the Phase 2.5 candidate-pair table.

    Label definition
    ----------------
    For each existing candidate pair (source1_entity_id, candidate_entity_id):

        label = 1   if candidate_entity_id ∈ ground_truth_ids for that source1_entity_id
        label = 0   otherwise

    Ground-truth positives absent from the Phase 1 candidate set are **NOT** added.
    They are counted as missed candidates / false negatives of blocking; see
    ``compute_candidate_positive_recall()`` for reporting.

    Invariants preserved
    --------------------
    * Every existing row in ``pair_df`` is retained exactly once (no rows added or removed).
    * Exact row order of ``pair_df`` is preserved.
    * All 10 Phase 2.5 feature columns are preserved unchanged.
    * Candidate IDs are not modified.
    * No negative downsampling is performed (that belongs to Phase 3+).

    Parameters
    ----------
    pair_df : pd.DataFrame
        Phase 2.5 candidate pair table.  Must contain at minimum:
        ``source1_entity_id`` and ``candidate_entity_id``.
        Typically also contains all 10 feature columns from Phase 2.1–2.4.

    ground_truth_df : pd.DataFrame
        Validated ground-truth table as returned by ``load_ground_truth()``.
        Must contain ``source1_entity_id`` and ``matching_entity_ids``.

    Returns
    -------
    pd.DataFrame
        A copy of ``pair_df`` with exactly one new column ``label`` (dtype int)
        appended as the last column.  All existing columns are preserved as-is.

    Raises
    ------
    ValueError
        If required columns are missing from either input DataFrame.
    """
    _check_cols(pair_df, {"source1_entity_id", "candidate_entity_id"}, "pair_df",
                caller="add_ground_truth_labels")
    _check_cols(ground_truth_df, GROUND_TRUTH_REQUIRED_COLS, "ground_truth_df",
                caller="add_ground_truth_labels")

    # Build lookup: source1_entity_id -> frozenset of ground-truth match IDs
    gt_lookup: dict[str, frozenset] = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matching_entity_ids"]).strip()

        if not raw_matches or raw_matches.lower() == "nan":
            # Valid zero-match entity: empty frozenset
            match_set: frozenset = frozenset()
        else:
            match_set = frozenset(
                mid.strip()
                for mid in raw_matches.split(",")
                if mid.strip()
            )

        gt_lookup[s1_id] = match_set

    # Assign labels row-by-row, preserving order
    labels: list[int] = []
    for _, row in pair_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()

        match_set = gt_lookup.get(s1_id)
        if match_set is None:
            # S1 not in ground truth at all → treat as zero-match (label = 0)
            labels.append(0)
        else:
            labels.append(1 if cand_id in match_set else 0)

    out_df = pair_df.copy()
    out_df[LABEL_COL] = labels
    return out_df


# ---------------------------------------------------------------------------
# Phase 2.6 — Candidate positive recall reporting
# ---------------------------------------------------------------------------

def compute_candidate_positive_recall(
    pair_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
) -> dict:
    """Compute candidate-generation recall for the Phase 1 blocking stage.

    For every ground-truth positive pair (source1_entity_id, matching_entity_id),
    checks whether that exact pair is present as a candidate in ``pair_df``.

    IMPORTANT: This is CANDIDATE-GENERATION RECALL (blocking recall), not model
    recall, not final matching recall, and not F0.5.  It measures what fraction
    of ground-truth positive pairs survived the Phase 1 blocking / retrieval step
    and therefore appear as labeled candidates for the classifier to learn from.

    Parameters
    ----------
    pair_df : pd.DataFrame
        The labeled (or unlabeled) candidate pair table from Phase 1/2.
        Must contain ``source1_entity_id`` and ``candidate_entity_id``.

    ground_truth_df : pd.DataFrame
        Validated ground-truth table as returned by ``load_ground_truth()``.
        Must contain ``source1_entity_id`` and ``matching_entity_ids``.

    Returns
    -------
    dict with keys:
        total_gt_positives  : int   – total ground-truth positive pairs
        retrieved_positives : int   – GT positives present in the candidate set
        missed_positives    : int   – GT positives absent from the candidate set
        missed_pairs        : list  – list of (source1_entity_id, matching_entity_id)
                                      tuples for missed positives
        candidate_positive_recall : float  – retrieved / total (0.0 if total == 0)

    Raises
    ------
    ValueError
        If required columns are missing from either input DataFrame.
    """
    _check_cols(pair_df, {"source1_entity_id", "candidate_entity_id"}, "pair_df",
                caller="compute_candidate_positive_recall")
    _check_cols(ground_truth_df, GROUND_TRUTH_REQUIRED_COLS, "ground_truth_df",
                caller="compute_candidate_positive_recall")

    # Build candidate set: frozenset of (s1_id, cand_id) tuples for O(1) lookup
    candidate_set: frozenset = frozenset(
        (str(row["source1_entity_id"]).strip(), str(row["candidate_entity_id"]).strip())
        for _, row in pair_df.iterrows()
    )

    total_gt_positives = 0
    retrieved_positives = 0
    missed_pairs: list[tuple[str, str]] = []

    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw_matches = str(row["matching_entity_ids"]).strip()

        if not raw_matches or raw_matches.lower() == "nan":
            # Zero-match entity: no positive pairs to evaluate
            continue

        match_ids = [
            mid.strip()
            for mid in raw_matches.split(",")
            if mid.strip()
        ]
        # Deduplicate within this row (handles S2-099,S2-099,S3-010 → {S2-099,S3-010})
        seen: set[str] = set()
        for match_id in match_ids:
            if match_id in seen:
                continue
            seen.add(match_id)

            total_gt_positives += 1
            if (s1_id, match_id) in candidate_set:
                retrieved_positives += 1
            else:
                missed_pairs.append((s1_id, match_id))

    missed_positives = total_gt_positives - retrieved_positives
    recall = (
        float(retrieved_positives) / float(total_gt_positives)
        if total_gt_positives > 0
        else 0.0
    )

    return {
        "total_gt_positives": total_gt_positives,
        "retrieved_positives": retrieved_positives,
        "missed_positives": missed_positives,
        "missed_pairs": missed_pairs,
        "candidate_positive_recall": recall,
    }


# ---------------------------------------------------------------------------
# Phase 2.6 — Labeled pairs writer (optional persistence)
# ---------------------------------------------------------------------------

def write_labeled_pairs(
    labeled_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Persist the Phase 2.6 labeled candidate-pair table to a TSV file.

    The output retains all columns from ``labeled_df`` exactly as-is,
    which should be:
        source1_entity_id, candidate_entity_id, [10 feature columns], label

    This does NOT overwrite the Phase 2.5 feature matrix
    (``feature_matrix_mock.tsv``).

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled candidate-pair table from ``add_ground_truth_labels()``.
        Must contain ``source1_entity_id``, ``candidate_entity_id``, and ``label``.

    output_path : str
        Destination file path (e.g. ``output/labeled_candidate_pairs_mock.tsv``).

    Raises
    ------
    ValueError
        If required columns are missing from ``labeled_df``.
    """
    _check_cols(
        labeled_df,
        {"source1_entity_id", "candidate_entity_id", LABEL_COL},
        "labeled_df",
        caller="write_labeled_pairs",
    )

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    labeled_df.to_csv(output_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Phase 2.7 — Retrieval-agreement feature helpers (BM25 merge)
# ---------------------------------------------------------------------------

def add_retrieval_agreement_defaults(
    pair_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add retrieval-agreement columns with Char-TFIDF-only defaults.

    Used when the candidate pairs were produced by the Char-TFIDF-only
    baseline pipeline (no BM25 union).  All pairs are assumed to have been
    retrieved by Char-TFIDF and not by BM25.

    Columns added (if not already present):

    * ``retrieved_by_char_tfidf``   = 1  (int)
    * ``retrieved_by_bm25``         = 0  (int)
    * ``retrieval_agreement_count`` = 1  (int)

    If any of these columns already exist in *pair_df* they are left
    unchanged (no overwrite).

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table.  Must contain pair identifier columns.

    Returns
    -------
    pd.DataFrame
        A copy of *pair_df* with the three retrieval-agreement columns
        added (or preserved).

    Raises
    ------
    TypeError
        If *pair_df* is not a pandas DataFrame.
    """
    if not isinstance(pair_df, pd.DataFrame):
        raise TypeError(
            f"add_retrieval_agreement_defaults(): expected pd.DataFrame, "
            f"got {type(pair_df).__name__}."
        )

    result = pair_df.copy()

    if "retrieved_by_char_tfidf" not in result.columns:
        result["retrieved_by_char_tfidf"] = 1
    if "retrieved_by_bm25" not in result.columns:
        result["retrieved_by_bm25"] = 0
    if "retrieval_agreement_count" not in result.columns:
        result["retrieval_agreement_count"] = (
            result["retrieved_by_char_tfidf"].astype(int)
            + result["retrieved_by_bm25"].astype(int)
        )

    # Ensure integer dtype
    for col in RETRIEVAL_AGREEMENT_FEATURE_COLS:
        result[col] = result[col].astype(int)

    return result


def add_retrieval_agreement_features(
    pair_df: pd.DataFrame,
    union_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join retrieval-agreement flags from the UNION result onto *pair_df*.

    Used when the candidate pairs were produced by the multi-view UNION
    pipeline (Char-TFIDF + BM25).  The three agreement flags from
    *union_df* are joined onto *pair_df* by (source1_entity_id,
    candidate_entity_id).

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair feature table (output of Phase 2.4).  Must contain
        ``source1_entity_id`` and ``candidate_entity_id``.
    union_df : pd.DataFrame
        Output of :func:`~src.blocking.union_candidate_results`.  Must
        contain ``source1_entity_id``, ``candidate_entity_id``,
        ``retrieved_by_char_tfidf``, ``retrieved_by_bm25``,
        ``retrieval_agreement_count``.

    Returns
    -------
    pd.DataFrame
        *pair_df* with the three retrieval-agreement columns added.
        Rows in *pair_df* that have no matching entry in *union_df* receive
        defaults: retrieved_by_char_tfidf=1, retrieved_by_bm25=0,
        retrieval_agreement_count=1 (Char-TFIDF-only assumption).

    Raises
    ------
    TypeError
        If either argument is not a pandas DataFrame.
    ValueError
        If required columns are missing from either DataFrame.
    """
    if not isinstance(pair_df, pd.DataFrame):
        raise TypeError(
            f"add_retrieval_agreement_features(): pair_df must be "
            f"pd.DataFrame, got {type(pair_df).__name__}."
        )
    if not isinstance(union_df, pd.DataFrame):
        raise TypeError(
            f"add_retrieval_agreement_features(): union_df must be "
            f"pd.DataFrame, got {type(union_df).__name__}."
        )

    _check_cols(
        pair_df,
        {"source1_entity_id", "candidate_entity_id"},
        "pair_df",
        caller="add_retrieval_agreement_features",
    )

    required_union_cols = {
        "source1_entity_id", "candidate_entity_id",
        "retrieved_by_char_tfidf", "retrieved_by_bm25",
        "retrieval_agreement_count",
    }
    missing_union = required_union_cols - set(union_df.columns)
    if missing_union:
        raise ValueError(
            f"add_retrieval_agreement_features(): union_df is missing "
            f"column(s): {sorted(missing_union)}.  "
            f"Found: {list(union_df.columns)}"
        )

    _PAIR_KEY = ["source1_entity_id", "candidate_entity_id"]
    agreement_cols = _PAIR_KEY + RETRIEVAL_AGREEMENT_FEATURE_COLS

    result = pair_df.merge(
        union_df[agreement_cols].drop_duplicates(subset=_PAIR_KEY),
        on=_PAIR_KEY,
        how="left",
    )

    # Fill defaults for any unmatched rows (should not occur in a correct union,
    # but be defensive)
    for col, default in [
        ("retrieved_by_char_tfidf", 1),
        ("retrieved_by_bm25", 0),
        ("retrieval_agreement_count", 1),
    ]:
        if result[col].isna().any():
            result[col] = result[col].fillna(default)

    # Ensure integer dtype
    for col in RETRIEVAL_AGREEMENT_FEATURE_COLS:
        result[col] = result[col].astype(int)

    return result


# ---------------------------------------------------------------------------
# Stage D — Length-ratio features
# ---------------------------------------------------------------------------
#
# Source inspiration: ChandrimaNandi/Amazon-ML-Hackathon-2026 src/similarity.py
#   functions: length_diff(), length_ratio()
#
# We adapt the concept but do NOT copy the full 35-feature system.
# Only two name-length features are added here because:
#   - Entity name length asymmetry is a meaningful signal (a very long name
#     paired with a very short name is unlikely to be a match).
#   - Both features are deterministic, NaN-free, and handle edge cases safely.
#   - They complement the existing 4 name-similarity features without overlap.
#
# DEFINITIONS:
#   name_length_diff  = |len(s1_clean_name) - len(candidate_clean_name)|
#                       normalised to [0, 1] by max(len_a, len_b).
#                       0.0 when lengths are equal; 1.0 when one is empty and
#                       the other is not.
#                       Both empty → 0.0 (identical empty strings).
#
#   name_length_ratio = min(len_a, len_b) / max(len_a, len_b)
#                       0.0 when one string is empty and the other is not.
#                       Both empty → 1.0 (identical empty strings).
#
# Both features are in [0, 1], finite, never NaN.
# ---------------------------------------------------------------------------

#: The two length features added in Stage D.
LENGTH_FEATURE_COLS: list[str] = [
    "name_length_diff",
    "name_length_ratio",
]

# Update the canonical feature list to 15 columns (13 + 2 new length features).
# IMPORTANT: FEATURE_COLUMNS_BASELINE (10) and FEATURE_COLUMNS (13) are kept
# for backward compatibility.  FEATURE_COLUMNS_V2 is the full 15-feature set.
FEATURE_COLUMNS_V2: list[str] = FEATURE_COLUMNS + LENGTH_FEATURE_COLS

# ---------------------------------------------------------------------------
# Feature set versioning — new improvements (additive, backward-compatible)
# ---------------------------------------------------------------------------
#
# Each V-N set is a strict superset of the previous one.  LightGBM training
# must use exactly ONE of these lists as feature_cols.
#
# V3 adds 9 address-component features (src/address_components.py).
# V4 adds 8 frequency-aware features (src/frequency_features.py).
# V5 adds 8 RRF / rank-aware features (src/rrf_features.py).
# V_FULL = V5 (all 40 features).
#
# IMPORTANT: build_feature_matrix() validates against FEATURE_COLUMNS (13).
# For V3+ feature sets, callers must skip build_feature_matrix() and instead
# feed the full pair table directly to train_lightgbm() with the desired
# FEATURE_COLUMNS_V* list.
#
# All new feature columns are:
#   - Finite floats in [0, 1]
#   - Never NaN
#   - Never Inf
#   - Deterministic
# ---------------------------------------------------------------------------

#: 9 address-component features added in V3.
ADDRESS_COMPONENT_FEATURE_COLS: list[str] = [
    "postal_code_exact_match",
    "postal_code_mismatch",
    "house_number_match",
    "house_number_conflict",
    "unit_match",
    "floor_match",
    "locality_token_overlap",
    "street_number_match",
    "component_agreement_count",
]

#: 8 frequency-aware features added in V4.
FREQUENCY_FEATURE_COLS: list[str] = [
    "name_freq_s1",
    "name_freq_cand",
    "token_freq_max",
    "token_freq_min",
    "rare_token_overlap",
    "rare_shared_token_count",
    "rarity_weighted_name_sim",
    "rarity_weighted_addr_sim",
]

#: 8 RRF / rank-aware features added in V5.
RRF_FEATURE_COLS: list[str] = [
    "tfidf_rank_norm",
    "bm25_rank_norm",
    "name_tfidf_rank_norm",
    "addr_tfidf_rank_norm",
    "rare_token_rank_norm",
    "snm_rank_norm",
    "rrf_score",
    "n_channels_agreeing",
]

#: V3 = V2 + 9 address-component features (24 total).
FEATURE_COLUMNS_V3: list[str] = FEATURE_COLUMNS_V2 + ADDRESS_COMPONENT_FEATURE_COLS

#: V4 = V3 + 8 frequency-aware features (32 total).
FEATURE_COLUMNS_V4: list[str] = FEATURE_COLUMNS_V3 + FREQUENCY_FEATURE_COLS

#: V5 = V4 + 8 RRF/rank-aware features (40 total). Full feature set.
FEATURE_COLUMNS_V5: list[str] = FEATURE_COLUMNS_V4 + RRF_FEATURE_COLS

#: 15 difference features added in V6 (token diff + numeric diff).
DIFF_FEATURE_COLS: list[str] = [
    # Token difference — name
    "name_common_token_count",
    "name_s1_only_token_count",
    "name_cand_only_token_count",
    "name_token_difference_ratio",
    "name_diff_token_sim",
    # Token difference — address
    "addr_common_token_count",
    "addr_s1_only_token_count",
    "addr_cand_only_token_count",
    # Numeric difference
    "numeric_common_count",
    "numeric_s1_only_count",
    "numeric_cand_only_count",
    "numeric_overlap_ratio",
    "numeric_conflict",
    "numeric_set_equal",
    "numeric_rel_diff",
]

#: V6 = V5 + 15 difference features (55 total).
FEATURE_COLUMNS_V6: list[str] = FEATURE_COLUMNS_V5 + DIFF_FEATURE_COLS

#: 16 runner-up margin and ambiguity/competition features added in V7.
MARGIN_FEATURE_COLS: list[str] = [
    "candidate_rank",
    "top_candidate_score",
    "second_candidate_score",
    "top1_top2_margin",
    "score_margin_vs_other",
    "score_gap_to_next",
    "score_gap_to_top",
    "relative_margin",
    "name_sim_margin",
    "addr_sim_margin",
    "candidate_count",
    "n_near_ties",
    "n_strong_candidates",
    "candidates_within_01_of_top",
    "competition_density",
    "is_ambiguous_candidate_set",
]

#: V7 = V6 + 16 margin/ambiguity features (71 total).
FEATURE_COLUMNS_V7: list[str] = FEATURE_COLUMNS_V6 + MARGIN_FEATURE_COLS

#: 10 RapidFuzz similarity features added in V8.
RAPIDFUZZ_FEATURE_COLS: list[str] = [
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_key_ratio",
    "name_key_partial_ratio",
    "name_key_jaro_winkler",
    "addr_fuzz_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_diff_token_sim",
]

#: V8 = V7 + 10 RapidFuzz features (81 total).
FEATURE_COLUMNS_V8: list[str] = FEATURE_COLUMNS_V7 + RAPIDFUZZ_FEATURE_COLS

#: 8 Indic transliteration features added in V9.
TRANSLITERATION_FEATURE_COLS: list[str] = [
    "translit_token_mapped_count",
    "translit_token_mapped_ratio",
    "translit_name_fuzz_ratio",
    "translit_name_token_sort_ratio",
    "translit_name_token_set_ratio",
    "translit_similarity_gain",
    "translit_addr_mapped_count",
    "translit_addr_fuzz_ratio",
]

#: V9 = V8 + 8 transliteration features (89 total).
FEATURE_COLUMNS_V9: list[str] = FEATURE_COLUMNS_V8 + TRANSLITERATION_FEATURE_COLS

#: Alias for the current recommended full feature set.
FEATURE_COLUMNS_FULL: list[str] = FEATURE_COLUMNS_V9


def _name_length_diff(a: str, b: str) -> float:
    """Normalised absolute length difference between two name strings.

    Returns |len(a) - len(b)| / max(len(a), len(b)).
    Both empty → 0.0.  One empty, one non-empty → 1.0.
    Result in [0, 1], always finite.
    """
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    denom = max(la, lb)
    return float(abs(la - lb)) / float(denom)


def _name_length_ratio(a: str, b: str) -> float:
    """Length ratio min/max between two name strings.

    Returns min(len(a), len(b)) / max(len(a), len(b)).
    Both empty → 1.0.  One empty, one non-empty → 0.0.
    Result in [0, 1], always finite.
    """
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    denom = max(la, lb)
    if denom == 0:
        return 1.0
    return float(min(la, lb)) / float(denom)


def add_length_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute two name-length features for every candidate pair.

    Features computed (Stage D):
    1. name_length_diff
       Normalised absolute length difference of s1_clean_name and
       candidate_clean_name: |len(a) - len(b)| / max(len(a), len(b)).
       Result in [0, 1].  0.0 = identical lengths.

    2. name_length_ratio
       Length ratio of the shorter name to the longer name:
       min(len(a), len(b)) / max(len(a), len(b)).
       Result in [0, 1].  1.0 = identical lengths.

    Edge cases (all handled deterministically):
    - Both empty ("", ""):
        name_length_diff  = 0.0
        name_length_ratio = 1.0
    - One empty, one non-empty:
        name_length_diff  = 1.0
        name_length_ratio = 0.0
    - Both identical, non-empty:
        name_length_diff  = 0.0
        name_length_ratio = 1.0
    - Missing / null values (None, NaN): coerced to empty string.
    - Empty pair table (0 rows): columns added with float dtype.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table.  Must contain at least:
        - ``source1_entity_id``
        - ``candidate_entity_id``
        - ``s1_clean_name``
        - ``candidate_clean_name``

    Returns
    -------
    pd.DataFrame
        A copy of *pair_df* with two new columns appended:
        [name_length_diff, name_length_ratio].

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    _check_cols(
        pair_df,
        NAME_SIMILARITY_REQUIRED_COLS,   # same required cols as name similarity
        "pair_df",
        caller="add_length_features",
    )

    out_df = pair_df.copy()

    if out_df.empty:
        for col in LENGTH_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=float)
        return out_df

    s1_names   = ["" if pd.isna(x) else str(x) for x in out_df["s1_clean_name"]]
    cand_names = ["" if pd.isna(x) else str(x) for x in out_df["candidate_clean_name"]]

    diff_scores  = [_name_length_diff(a, b)  for a, b in zip(s1_names, cand_names)]
    ratio_scores = [_name_length_ratio(a, b) for a, b in zip(s1_names, cand_names)]

    # Clamp as safety net (implementations already guarantee [0,1])
    out_df["name_length_diff"]  = [min(1.0, max(0.0, v)) for v in diff_scores]
    out_df["name_length_ratio"] = [min(1.0, max(0.0, v)) for v in ratio_scores]

    return out_df
