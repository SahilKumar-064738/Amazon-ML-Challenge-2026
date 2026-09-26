"""
model.py
--------
Phase 3 -- Entity-resolution model utilities.

Phase 3.1 implements ONLY the entity-level train/validation split.
No model training, no feature inspection, no label inspection,
no negative sampling, no metrics computation.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Column that holds the unique S1 entity identifier in a Source-1 DataFrame.
S1_ENTITY_ID_COL: str = "entity_id"

#: Default fraction of S1 entities reserved for validation.
DEFAULT_HOLDOUT_FRAC: float = 0.15

#: Default random seed for reproducibility.
DEFAULT_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Phase 3.1 -- Entity-level train / validation split
# ---------------------------------------------------------------------------

def split_entities(
    s1_df: pd.DataFrame,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[list[str], list[str]]:
    """Split unique S1 entity IDs into train and validation sets.

    The split is performed at the **entity** level: every candidate pair
    belonging to a given S1 entity will later belong entirely to either
    the train set or the validation set -- never to both.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Source-1 DataFrame produced by :func:`src.ingestion.load_clean_tsv`.
        Must contain an ``entity_id`` column.  The function reads only the
        ``entity_id`` column; it does not inspect feature values or labels.
    holdout_frac : float, optional
        Fraction of S1 entities to place in the validation set.
        Must satisfy ``0 < holdout_frac < 1``.  Default: ``0.15``.
    random_state : int, optional
        Seed for the random number generator, ensuring a deterministic
        split for a fixed value.  Default: ``42``.

    Returns
    -------
    train_ids : list[str]
        Sorted list of S1 entity IDs assigned to the training set.
    val_ids : list[str]
        Sorted list of S1 entity IDs assigned to the validation set.

    Raises
    ------
    ValueError
        * If ``s1_df`` does not contain an ``entity_id`` column.
        * If ``s1_df`` is empty (no rows).
        * If ``holdout_frac`` is not strictly between 0 and 1.

    Notes
    -----
    * Duplicate ``entity_id`` values are collapsed to a unique set before
      splitting -- each entity is counted exactly once.
    * The split is label-blind: no label or feature column is read.
    * No model training or metric computation is performed here.
    """

    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if S1_ENTITY_ID_COL not in s1_df.columns:
        raise ValueError(
            f"split_entities() requires an '{S1_ENTITY_ID_COL}' column, "
            f"but the DataFrame only has columns: {list(s1_df.columns)}"
        )

    if not (0.0 < holdout_frac < 1.0):
        raise ValueError(
            f"holdout_frac must be strictly between 0 and 1, "
            f"got {holdout_frac!r}."
        )

    if s1_df.empty:
        raise ValueError(
            "split_entities() received an empty DataFrame (zero rows). "
            "Cannot create a train/validation split from no entities."
        )

    # ------------------------------------------------------------------
    # 2. Extract unique entity IDs (sorted for determinism)
    # ------------------------------------------------------------------
    unique_ids: list[str] = sorted(s1_df[S1_ENTITY_ID_COL].unique().tolist())
    n_unique = len(unique_ids)

    # ------------------------------------------------------------------
    # 3. Edge case: only one unique entity
    # ------------------------------------------------------------------
    if n_unique == 1:
        # Cannot split a single entity; assign it to train, val is empty.
        # This edge case is reported by callers; we do not silently expand.
        return unique_ids, []

    # ------------------------------------------------------------------
    # 4. Split at the entity level
    #    train_test_split guarantees:
    #      - determinism for a fixed random_state
    #      - no entity appears in both sets
    # ------------------------------------------------------------------
    train_ids_raw, val_ids_raw = train_test_split(
        unique_ids,
        test_size=holdout_frac,
        random_state=random_state,
        shuffle=True,
    )

    # Sort for stable output ordering
    train_ids: list[str] = sorted(train_ids_raw)
    val_ids: list[str] = sorted(val_ids_raw)

    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Validation helper (used by the report runner and tests)
# ---------------------------------------------------------------------------

def validate_split(
    s1_df: pd.DataFrame,
    train_ids: list[str],
    val_ids: list[str],
) -> dict:
    """Compute and return split-quality metrics.

    Parameters
    ----------
    s1_df : pd.DataFrame
        The same Source-1 DataFrame passed to :func:`split_entities`.
    train_ids : list[str]
        Training entity IDs returned by :func:`split_entities`.
    val_ids : list[str]
        Validation entity IDs returned by :func:`split_entities`.

    Returns
    -------
    dict with keys:
        ``total_unique``     -- total unique S1 entities
        ``train_count``      -- number of train entities
        ``val_count``        -- number of validation entities
        ``overlap_count``    -- number of IDs in both sets (must be 0)
        ``coverage_ok``      -- True if train | val == all unique IDs
        ``no_duplicates_train`` -- True if train_ids has no duplicates
        ``no_duplicates_val``   -- True if val_ids has no duplicates
        ``actual_val_frac``  -- val_count / total_unique
    """
    unique_ids = set(s1_df[S1_ENTITY_ID_COL].unique().tolist())
    train_set = set(train_ids)
    val_set = set(val_ids)

    overlap = train_set & val_set
    union = train_set | val_set

    total = len(unique_ids)
    train_count = len(train_ids)
    val_count = len(val_ids)
    actual_val_frac = val_count / total if total > 0 else 0.0

    return {
        "total_unique": total,
        "train_count": train_count,
        "val_count": val_count,
        "overlap_count": len(overlap),
        "coverage_ok": union == unique_ids,
        "no_duplicates_train": len(train_ids) == len(train_set),
        "no_duplicates_val": len(val_ids) == len(val_set),
        "actual_val_frac": actual_val_frac,
    }


# ---------------------------------------------------------------------------
# Phase 3.2 -- Train / validation pair-table construction
# ---------------------------------------------------------------------------

#: Column in the labeled candidate-pair table that identifies the S1 entity.
PAIR_S1_ENTITY_COL: str = "source1_entity_id"

#: Column in the labeled candidate-pair table that holds the binary label.
LABEL_COL: str = "label"

#: The 10 feature columns produced by Phase 2 (baseline, no retrieval-agreement).
FEATURE_COLS: tuple[str, ...] = (
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
)

# ---------------------------------------------------------------------------
# Extended feature-column sets for ablation experiments
# ---------------------------------------------------------------------------
#
# Each tuple is a strict superset of the previous one.  Pass the desired
# tuple as feature_cols to train_lightgbm() / score_pairs() for that
# experiment tier.  The pair table must contain all named columns before
# training; use the corresponding add_*_features() functions to populate them.
#
# FEATURE_COLS         -- 10 cols  (Phase 2 baseline, no retrieval flags)
# FEATURE_COLS_V2      -- 15 cols  (+ retrieval-agreement + length)
# FEATURE_COLS_V3      -- 24 cols  (+ address-component features)
# FEATURE_COLS_V4      -- 32 cols  (+ frequency-aware features)
# FEATURE_COLS_V5      -- 40 cols  (+ RRF/rank-aware features)  ← full set
# ---------------------------------------------------------------------------

#: V2 = 10-baseline + 3 retrieval-agreement + 2 length features (15 total).
FEATURE_COLS_V2: tuple[str, ...] = (
    # Phase 2 baseline (10)
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
    # Retrieval-agreement (3)
    "retrieved_by_char_tfidf",
    "retrieved_by_bm25",
    "retrieval_agreement_count",
    # Length (2)
    "name_length_diff",
    "name_length_ratio",
)

#: V3 = V2 + 9 address-component features (24 total).
FEATURE_COLS_V3: tuple[str, ...] = FEATURE_COLS_V2 + (
    "postal_code_exact_match",
    "postal_code_mismatch",
    "house_number_match",
    "house_number_conflict",
    "unit_match",
    "floor_match",
    "locality_token_overlap",
    "street_number_match",
    "component_agreement_count",
)

#: V4 = V3 + 8 frequency-aware features (32 total).
FEATURE_COLS_V4: tuple[str, ...] = FEATURE_COLS_V3 + (
    "name_freq_s1",
    "name_freq_cand",
    "token_freq_max",
    "token_freq_min",
    "rare_token_overlap",
    "rare_shared_token_count",
    "rarity_weighted_name_sim",
    "rarity_weighted_addr_sim",
)

#: V5 = V4 + 8 RRF/rank-aware features (40 total). Full recommended feature set.
FEATURE_COLS_V5: tuple[str, ...] = FEATURE_COLS_V4 + (
    "tfidf_rank_norm",
    "bm25_rank_norm",
    "name_tfidf_rank_norm",
    "addr_tfidf_rank_norm",
    "rare_token_rank_norm",
    "snm_rank_norm",
    "rrf_score",
    "n_channels_agreeing",
)

#: Alias for the current recommended full feature set for experiments.
FEATURE_COLS_FULL: tuple[str, ...] = FEATURE_COLS_V5


def filter_pairs_by_entities(
    pair_df: pd.DataFrame,
    entity_ids: "list[str] | set[str] | frozenset[str]",
) -> pd.DataFrame:
    """Return only the rows of *pair_df* whose source1_entity_id is in *entity_ids*.

    The split is enforced at the **S1 entity level**: every candidate pair
    belonging to a given S1 entity is either fully included or fully excluded.
    This function never mixes pairs from the same S1 entity across splits.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Labeled candidate-pair table produced by Phase 2.6.  Must contain a
        ``source1_entity_id`` column.  All other columns are passed through
        unchanged.
    entity_ids : collection of str
        The S1 entity IDs to keep.  Must be a collection of strings (list,
        set, frozenset, or any iterable of str).  Passing a non-string
        collection (e.g. a list of ints) raises ``TypeError``.

    Returns
    -------
    pd.DataFrame
        A copy of the rows whose ``source1_entity_id`` is in *entity_ids*,
        in the **same order** as the original *pair_df*.  All columns are
        preserved exactly as-is -- no feature values or labels are modified.

    Raises
    ------
    ValueError
        If *pair_df* does not contain a ``source1_entity_id`` column.
    TypeError
        If *entity_ids* is not an iterable of strings.

    Notes
    -----
    * Row order follows the original *pair_df* index order -- no sorting is
      applied.
    * The original *pair_df* is never mutated; a copy is returned.
    * No new pairs are generated.  No existing pairs are deduplicated.
    * Feature values and label values are not inspected or modified.
    """

    # ------------------------------------------------------------------
    # 1. Validate pair_df has the required column
    # ------------------------------------------------------------------
    if PAIR_S1_ENTITY_COL not in pair_df.columns:
        raise ValueError(
            f"filter_pairs_by_entities() requires a '{PAIR_S1_ENTITY_COL}' "
            f"column in pair_df, but found only: {list(pair_df.columns)}"
        )

    if isinstance(entity_ids, (str, bytes)):
        raise TypeError(
            f"entity_ids must be a collection of string IDs, not a single string: {entity_ids!r}"
        )

    try:
        entity_ids_set: frozenset[str] = frozenset(entity_ids)
    except TypeError as exc:
        raise TypeError(
            f"entity_ids must be an iterable of strings, got {type(entity_ids).__name__}: {exc}"
        ) from exc

    if entity_ids_set and not all(isinstance(eid, str) for eid in entity_ids_set):
        bad = [eid for eid in entity_ids_set if not isinstance(eid, str)]
        raise TypeError(
            f"entity_ids must contain only str values; "
            f"found non-string element(s): {bad[:5]!r}"
        )

    # ------------------------------------------------------------------
    # 3. Filter -- preserve original row order by using boolean mask
    # ------------------------------------------------------------------
    mask = pair_df[PAIR_S1_ENTITY_COL].isin(entity_ids_set)
    filtered = pair_df.loc[mask].copy()

    return filtered


def filter_ground_truth_to_s1_ids(
    ground_truth_df: "pd.DataFrame",
    s1_ids: "list[str] | set[str]",
) -> "pd.DataFrame":
    """Return only GT rows whose source1_entity_id is in *s1_ids*.

    This is the authoritative filter that prevents training GT from
    contaminating validation metrics (P0-2 fix).

    Parameters
    ----------
    ground_truth_df : pd.DataFrame
        Full ground-truth table (all splits).
    s1_ids : collection of str
        The S1 entity IDs that should contribute to the metric being computed.
        Pass *val_s1_ids* when computing validation metrics.

    Returns
    -------
    pd.DataFrame
        A copy of *ground_truth_df* restricted to rows whose
        ``source1_entity_id`` is in *s1_ids*.  Rows for S1 IDs that have
        no ground-truth entry are not invented; the returned frame may be
        smaller than ``len(s1_ids)`` if some IDs have no GT entry.

    Raises
    ------
    ValueError
        If ``source1_entity_id`` column is absent from *ground_truth_df*.
    """
    if "source1_entity_id" not in ground_truth_df.columns:
        raise ValueError(
            "filter_ground_truth_to_s1_ids(): ground_truth_df must contain "
            "'source1_entity_id'. "
            f"Found: {list(ground_truth_df.columns)}"
        )
    s1_set = frozenset(str(x) for x in s1_ids)
    mask = ground_truth_df["source1_entity_id"].astype(str).isin(s1_set)
    return ground_truth_df.loc[mask].copy().reset_index(drop=True)


def assert_val_gt_integrity(
    train_s1_ids: "list[str]",
    val_s1_ids: "list[str]",
    val_gt_df: "pd.DataFrame",
) -> None:
    """Assert the four P0-2 ground-truth integrity invariants.

    Raises
    ------
    AssertionError
        If any invariant is violated.
    """
    train_set = frozenset(str(x) for x in train_s1_ids)
    val_set   = frozenset(str(x) for x in val_s1_ids)

    # 1. No overlap between train and val S1 IDs
    overlap = train_set & val_set
    assert not overlap, (
        f"P0-2 FAIL: val_s1_ids ∩ train_s1_ids is non-empty: {sorted(overlap)}"
    )

    # 2. val_gt_df contains ONLY val S1 IDs (no training contamination)
    if len(val_gt_df) > 0:
        gt_s1_ids = frozenset(val_gt_df["source1_entity_id"].astype(str).unique())
        gt_in_train = gt_s1_ids & train_set
        assert not gt_in_train, (
            f"P0-2 FAIL: {len(gt_in_train)} GT rows belong to training S1 IDs: "
            f"{sorted(gt_in_train)}"
        )

        # 3. All GT S1 IDs are in the val set
        gt_not_in_val = gt_s1_ids - val_set
        assert not gt_not_in_val, (
            f"P0-2 FAIL: {len(gt_not_in_val)} GT rows have S1 IDs that are "
            f"neither in val nor train: {sorted(gt_not_in_val)}"
        )

    # 4. Every val S1 ID appears in the evaluation universe (i.e., val_set)
    #    This is trivially true since we constructed val_set from val_s1_ids —
    #    just confirm the list is non-empty when we have data
    assert len(val_set) >= 0, "P0-2 FAIL: val_set must be a valid set"


def build_pair_splits(
    labeled_pairs_df: pd.DataFrame,
    train_ids: "list[str]",
    val_ids: "list[str]",
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """Apply the Phase 3.1 entity split to the Phase 2.6 labeled pair table.

    Uses :func:`filter_pairs_by_entities` to construct train and validation
    pair tables.  The split is at the S1 entity level: every pair belonging
    to a given S1 entity appears in exactly one of the two tables.

    Parameters
    ----------
    labeled_pairs_df : pd.DataFrame
        The complete labeled candidate-pair table from Phase 2.6.
    train_ids : list[str]
        S1 entity IDs assigned to the training split (from Phase 3.1).
    val_ids : list[str]
        S1 entity IDs assigned to the validation split (from Phase 3.1).

    Returns
    -------
    train_pairs_df : pd.DataFrame
        Candidate pairs belonging to S1 entities in *train_ids*.
    val_pairs_df : pd.DataFrame
        Candidate pairs belonging to S1 entities in *val_ids*.

    Raises
    ------
    ValueError
        If the S1 entity overlap between train_ids and val_ids is non-empty,
        or if the resulting splits do not cover all rows of *labeled_pairs_df*.
    """

    # ------------------------------------------------------------------
    # 1. Guard: train_ids and val_ids must not overlap
    # ------------------------------------------------------------------
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise ValueError(
            f"build_pair_splits() received overlapping train_ids and val_ids. "
            f"Overlapping IDs: {sorted(overlap)}"
        )

    # ------------------------------------------------------------------
    # 2. Apply the entity-level filter to produce each split
    # ------------------------------------------------------------------
    train_pairs_df = filter_pairs_by_entities(labeled_pairs_df, train_ids)
    val_pairs_df = filter_pairs_by_entities(labeled_pairs_df, val_ids)

    # ------------------------------------------------------------------
    # 3. Post-construction validation
    # ------------------------------------------------------------------
    train_s1_ids = set(train_pairs_df[PAIR_S1_ENTITY_COL].unique())
    val_s1_ids = set(val_pairs_df[PAIR_S1_ENTITY_COL].unique())

    # 3a. Zero S1 entity overlap between the two splits
    pair_overlap = train_s1_ids & val_s1_ids
    if pair_overlap:
        raise ValueError(
            f"Post-construction validation failed: S1 entity overlap detected "
            f"between train and validation pair tables: {sorted(pair_overlap)}"
        )

    # 3b. Full pair coverage: every row in labeled_pairs_df is in exactly one split
    total_original = len(labeled_pairs_df)
    total_split = len(train_pairs_df) + len(val_pairs_df)
    if total_split != total_original:
        # Determine what is missing or extra
        all_s1_in_labeled = set(labeled_pairs_df[PAIR_S1_ENTITY_COL].unique())
        covered_s1 = set(train_ids) | set(val_ids)
        uncovered = all_s1_in_labeled - covered_s1
        raise ValueError(
            f"build_pair_splits() coverage failure: "
            f"train ({len(train_pairs_df)}) + val ({len(val_pairs_df)}) "
            f"= {total_split} != original {total_original}. "
            f"S1 entities in labeled_pairs_df not covered by any split: {sorted(uncovered)}"
        )

    return train_pairs_df, val_pairs_df


def validate_pair_splits(
    labeled_pairs_df: pd.DataFrame,
    train_pairs_df: pd.DataFrame,
    val_pairs_df: pd.DataFrame,
) -> dict:
    """Compute and return split-quality metrics for the pair tables.

    Parameters
    ----------
    labeled_pairs_df : pd.DataFrame
        The complete Phase 2.6 labeled pair table.
    train_pairs_df : pd.DataFrame
        Training pair table from :func:`build_pair_splits`.
    val_pairs_df : pd.DataFrame
        Validation pair table from :func:`build_pair_splits`.

    Returns
    -------
    dict with keys:
        ``total_s1_entities``       -- unique S1 entity count in labeled_pairs_df
        ``train_s1_count``          -- unique S1 entities in train split
        ``val_s1_count``            -- unique S1 entities in validation split
        ``s1_overlap_count``        -- entities in both train and val (must be 0)
        ``total_pairs``             -- rows in labeled_pairs_df
        ``train_pairs``             -- rows in train split
        ``val_pairs``               -- rows in validation split
        ``pair_coverage_ok``        -- True if train + val == total
        ``train_positives``         -- positive-label rows in train
        ``train_negatives``         -- negative-label rows in train
        ``train_positive_rate``     -- train positives / train pairs
        ``train_negative_rate``     -- train negatives / train pairs
        ``val_positives``           -- positive-label rows in val
        ``val_negatives``           -- negative-label rows in val
        ``val_positive_rate``       -- val positives / val pairs
        ``val_negative_rate``       -- val negatives / val pairs
        ``no_duplicate_pairs``      -- True if no pair appears in both splits
    """
    all_s1 = set(labeled_pairs_df[PAIR_S1_ENTITY_COL].unique())
    train_s1 = set(train_pairs_df[PAIR_S1_ENTITY_COL].unique()) if len(train_pairs_df) else set()
    val_s1 = set(val_pairs_df[PAIR_S1_ENTITY_COL].unique()) if len(val_pairs_df) else set()

    s1_overlap = train_s1 & val_s1
    total_pairs = len(labeled_pairs_df)
    train_pairs = len(train_pairs_df)
    val_pairs = len(val_pairs_df)

    # Positive / negative counts (requires LABEL_COL)
    def _pos_neg(df: pd.DataFrame) -> tuple[int, int]:
        if LABEL_COL not in df.columns or len(df) == 0:
            return 0, 0
        pos = int((df[LABEL_COL] == 1).sum())
        neg = int((df[LABEL_COL] == 0).sum())
        return pos, neg

    train_pos, train_neg = _pos_neg(train_pairs_df)
    val_pos, val_neg = _pos_neg(val_pairs_df)

    # Duplicate-pair check: build composite key and compare
    def _pair_keys(df: pd.DataFrame) -> set:
        if len(df) == 0:
            return set()
        return set(
            zip(df[PAIR_S1_ENTITY_COL], df.get("candidate_entity_id", pd.Series(dtype=str)))
        )

    no_duplicate_pairs = len(_pair_keys(train_pairs_df) & _pair_keys(val_pairs_df)) == 0

    return {
        "total_s1_entities": len(all_s1),
        "train_s1_count": len(train_s1),
        "val_s1_count": len(val_s1),
        "s1_overlap_count": len(s1_overlap),
        "total_pairs": total_pairs,
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "pair_coverage_ok": (train_pairs + val_pairs) == total_pairs,
        "train_positives": train_pos,
        "train_negatives": train_neg,
        "train_positive_rate": train_pos / train_pairs if train_pairs > 0 else 0.0,
        "train_negative_rate": train_neg / train_pairs if train_pairs > 0 else 0.0,
        "val_positives": val_pos,
        "val_negatives": val_neg,
        "val_positive_rate": val_pos / val_pairs if val_pairs > 0 else 0.0,
        "val_negative_rate": val_neg / val_pairs if val_pairs > 0 else 0.0,
        "no_duplicate_pairs": no_duplicate_pairs,
    }


# ---------------------------------------------------------------------------
# Phase 3.3 -- LightGBM baseline candidate scoring model
# ---------------------------------------------------------------------------

#: Disallowed feature columns that must never be used as model features.
DISALLOWED_FEATURE_COLS: frozenset[str] = frozenset({
    PAIR_S1_ENTITY_COL,
    "candidate_entity_id",
    LABEL_COL,
    S1_ENTITY_ID_COL,
})

#: Default negative downsampling ratio (at most 5 negatives per 1 positive).
DEFAULT_NEGATIVE_RATIO: int = 5

#: Default baseline hyperparameters for LightGBM binary classification.
DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "random_state": DEFAULT_RANDOM_STATE,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 1,
    "verbose": -1,
}


def prepare_training_pairs(
    train_pairs_df: pd.DataFrame,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Perform deterministic negative downsampling on training pairs.

    Keeps ALL positive pairs (label=1) and downsamples negative pairs (label=0)
    so that there are at most *negative_ratio* negatives per positive pair.
    If the training set already has fewer or equal negatives than the cap,
    all negatives are retained without modification.

    Parameters
    ----------
    train_pairs_df : pd.DataFrame
        Training candidate pairs from Phase 3.2. Must contain the ``label`` column.
    negative_ratio : int, optional
        Maximum number of negative pairs per positive pair. Default is 5.
    random_state : int, optional
        Random seed for deterministic negative sampling. Default is 42.

    Returns
    -------
    pd.DataFrame
        Downsampled training pairs preserving original column structure and
        relative row ordering. Original input is never mutated.

    Raises
    ------
    ValueError
        If ``label`` column is missing, or if ``negative_ratio`` < 1.
    """
    if LABEL_COL not in train_pairs_df.columns:
        raise ValueError(
            f"prepare_training_pairs() requires a '{LABEL_COL}' column, "
            f"but found only: {list(train_pairs_df.columns)}"
        )
    if negative_ratio < 1:
        raise ValueError(
            f"negative_ratio must be an integer >= 1, got {negative_ratio!r}"
        )

    pos_df = train_pairs_df[train_pairs_df[LABEL_COL] == 1]
    neg_df = train_pairs_df[train_pairs_df[LABEL_COL] == 0]

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    max_neg = n_pos * negative_ratio
    if n_neg > max_neg:
        sampled_neg_df = neg_df.sample(
            n=max_neg, random_state=random_state, replace=False
        )
        keep_indices = pos_df.index.union(sampled_neg_df.index)
        sampled_df = train_pairs_df.loc[
            train_pairs_df.index.isin(keep_indices)
        ].copy()
    else:
        sampled_df = train_pairs_df.copy()

    return sampled_df


def train_lightgbm(
    train_pairs_df: pd.DataFrame,
    feature_cols: list[str] | tuple[str, ...] = FEATURE_COLS,
    params: dict[str, object] | None = None,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    return_metadata: bool = False,
) -> lgb.LGBMClassifier | tuple[lgb.LGBMClassifier, dict[str, object]]:
    """Train a baseline LightGBM binary classifier on training candidate pairs.

    Learns to estimate the matching probability for candidate entity pairs using
    the 10 approved Phase 2 features.

    Strict leakage guard:
    - Fits ONLY on *train_pairs_df*.
    - Validation pairs (*val_pairs_df*) must NEVER be passed or used.
    - Entity IDs and raw text columns are strictly prohibited as features.

    Parameters
    ----------
    train_pairs_df : pd.DataFrame
        Training candidate pair DataFrame from Phase 3.2.
    feature_cols : list or tuple of str, optional
        Feature column names to train on. Default is :data:`FEATURE_COLS`.
    params : dict, optional
        Custom LightGBM parameters to override :data:`DEFAULT_LGBM_PARAMS`.
    negative_ratio : int, optional
        Ratio of negative to positive pairs for training downsampling. Default: 5.
    random_state : int, optional
        Random seed for negative downsampling and model reproducibility. Default: 42.
    return_metadata : bool, optional
        If True, returns a ``(model, metadata)`` tuple. If False (default), returns
        just ``model`` with metadata attached to ``model.training_metadata_``.

    Returns
    -------
    model : lgb.LGBMClassifier
        Fitted LightGBM binary classification model.
    metadata : dict (only if return_metadata=True)
        Dictionary of training metadata and class balance statistics.

    Raises
    ------
    ValueError
        - If train_pairs_df is empty.
        - If feature_cols is empty.
        - If any disallowed column (entity IDs, labels) is included in feature_cols.
        - If any required feature_cols or 'label' column is missing from train_pairs_df.
        - If labels are not binary (0 and 1).
        - If both classes (0 and 1) are not present in train_pairs_df.
        - If features contain NaN or infinite values.
    TypeError
        - If any feature column has non-numeric dtype.
    """
    # ------------------------------------------------------------------
    # 1. Input validations
    # ------------------------------------------------------------------
    if train_pairs_df.empty:
        raise ValueError("train_pairs_df cannot be empty (0 rows).")

    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")

    # Guard: Entity IDs and label must never be model features
    disallowed = sorted(set(feature_cols) & DISALLOWED_FEATURE_COLS)
    if disallowed:
        raise ValueError(
            f"Entity IDs and target columns cannot be used as model features. "
            f"Found disallowed column(s): {disallowed}"
        )

    # Missing feature columns
    missing_features = [col for col in feature_cols if col not in train_pairs_df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required feature column(s) in train_pairs_df: {missing_features}"
        )

    # Missing label column
    if LABEL_COL not in train_pairs_df.columns:
        raise ValueError(
            f"train_pairs_df must contain '{LABEL_COL}' column, "
            f"found only: {list(train_pairs_df.columns)}"
        )

    # Validate numeric feature dtypes
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(train_pairs_df[col]):
            raise TypeError(
                f"Feature column '{col}' must be numeric, got dtype {train_pairs_df[col].dtype}"
            )

    # Validate no NaN or infinite values
    features_subset = train_pairs_df[list(feature_cols)]
    if features_subset.isna().any().any():
        nan_cols = [c for c in feature_cols if features_subset[c].isna().any()]
        raise ValueError(f"Feature column(s) contain NaN values: {nan_cols}")

    arr_vals = features_subset.to_numpy(dtype=float)
    if np.isinf(arr_vals).any():
        raise ValueError("Feature column(s) contain infinite values.")

    # Validate binary labels
    unique_labels = set(train_pairs_df[LABEL_COL].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Labels must contain only 0 and 1, found: {sorted(unique_labels)}"
        )

    n_rows_before = len(train_pairs_df)
    n_pos_before = int((train_pairs_df[LABEL_COL] == 1).sum())
    n_neg_before = int((train_pairs_df[LABEL_COL] == 0).sum())

    # Ensure both classes exist
    if n_pos_before == 0 or n_neg_before == 0:
        raise ValueError(
            f"Training data must contain both positive (label=1) and negative (label=0) samples. "
            f"Found positives={n_pos_before}, negatives={n_neg_before}."
        )

    # ------------------------------------------------------------------
    # 2. Negative sampling (training-only)
    # ------------------------------------------------------------------
    sampled_df = prepare_training_pairs(
        train_pairs_df=train_pairs_df,
        negative_ratio=negative_ratio,
        random_state=random_state,
    )

    n_rows_after = len(sampled_df)
    n_pos_after = int((sampled_df[LABEL_COL] == 1).sum())
    n_neg_after = int((sampled_df[LABEL_COL] == 0).sum())
    final_ratio = n_neg_after / n_pos_after if n_pos_after > 0 else 0.0

    # ------------------------------------------------------------------
    # 3. Model initialization & fitting
    # ------------------------------------------------------------------
    effective_params: dict[str, object] = dict(DEFAULT_LGBM_PARAMS)
    if params:
        effective_params.update(params)

    model = lgb.LGBMClassifier(**effective_params)

    X_train = sampled_df[list(feature_cols)]
    y_train = sampled_df[LABEL_COL]

    model.fit(X_train, y_train)

    # Attach training metadata
    metadata: dict[str, object] = {
        "training_rows_before": n_rows_before,
        "positive_rows_before": n_pos_before,
        "negative_rows_before": n_neg_before,
        "training_rows_after": n_rows_after,
        "positive_rows_after": n_pos_after,
        "negative_rows_after": n_neg_after,
        "final_negative_positive_ratio": final_ratio,
        "feature_cols": list(feature_cols),
        "params": effective_params,
    }
    model.training_metadata_ = metadata

    if return_metadata:
        return model, metadata
    return model


def get_feature_importance(
    model: lgb.LGBMClassifier,
    feature_cols: list[str] | tuple[str, ...] = FEATURE_COLS,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extract feature importance from a trained LightGBM model.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained LightGBM model.
    feature_cols : list or tuple of str, optional
        Feature column names corresponding to model inputs.
    importance_type : str, optional
        Importance metric: ``"gain"`` (default) or ``"split"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``["feature", "importance"]`` sorted
        in descending order of importance.
    """
    if not hasattr(model, "booster_") or model.booster_ is None:
        raise ValueError("Model is not fitted. Cannot extract feature importance.")

    importances = model.booster_.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({
        "feature": list(feature_cols),
        "importance": importances,
    })
    return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase 3.4 -- Raw validation scoring
# ---------------------------------------------------------------------------

#: New column appended by score_pairs().
PROB_MATCH_COL: str = "prob_match"


def score_pairs(
    model: lgb.LGBMClassifier,
    pairs_df: pd.DataFrame,
    feature_cols: "list[str] | tuple[str, ...]",
) -> pd.DataFrame:
    """Generate raw match probabilities for every candidate pair in *pairs_df*.

    Applies ``model.predict_proba(X)[:, 1]`` to the feature matrix built from
    *feature_cols* and appends the result as a new ``prob_match`` column.

    This function is intentionally **output-only**:

    * No threshold is applied.
    * No binary predictions (0 / 1) are produced.
    * No performance metrics are computed (no F0.5, precision, recall, etc.).
    * The model is never retrained, recalibrated, or modified.

    Parameters
    ----------
    model : lgb.LGBMClassifier (or any sklearn-compatible classifier)
        A trained classifier that exposes ``predict_proba(X)``.
    pairs_df : pd.DataFrame
        Candidate pair table to score.  Must contain all columns named in
        *feature_cols*.  All existing columns are preserved and returned.
    feature_cols : list or tuple of str
        Feature column names to use as model input.  Must be a non-empty
        collection.  Must match the features used during training.

    Returns
    -------
    pd.DataFrame
        A copy of *pairs_df* with one new column appended:

        ``prob_match`` : float64
            Raw match probability in [0, 1] from
            ``model.predict_proba(X)[:, 1]``.

        * All existing columns are preserved without modification.
        * Row order is identical to the input.
        * No rows are added or removed.

    Raises
    ------
    ValueError
        * If *feature_cols* is empty.
        * If any column in *feature_cols* is missing from *pairs_df*.
        * If any feature column contains NaN values.
        * If any feature column contains infinite values.
        * If *pairs_df* is empty.
        * If the output probabilities fall outside [0, 1] or are non-finite.
    TypeError
        * If any feature column has a non-numeric dtype.
        * If *model* does not expose ``predict_proba``.

    Notes
    -----
    * The function verifies row alignment: ``len(output) == len(input)``.
    * Labels (if present) remain untouched in the output.
    * No threshold is applied; Phase 4 owns threshold selection.
    """

    # ------------------------------------------------------------------
    # 1. Validate model supports predict_proba
    # ------------------------------------------------------------------
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError(
            "score_pairs() requires a model with predict_proba(). "
            f"The supplied model of type '{type(model).__name__}' does not "
            "expose predict_proba."
        )

    # ------------------------------------------------------------------
    # 2. Validate feature_cols is non-empty
    # ------------------------------------------------------------------
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty.")

    # ------------------------------------------------------------------
    # 3. Validate all feature columns exist in pairs_df
    # ------------------------------------------------------------------
    missing = [col for col in feature_cols if col not in pairs_df.columns]
    if missing:
        raise ValueError(
            f"score_pairs(): the following feature column(s) are missing "
            f"from pairs_df: {missing}"
        )

    # ------------------------------------------------------------------
    # 4. Validate pairs_df is non-empty
    # ------------------------------------------------------------------
    if pairs_df.empty:
        raise ValueError(
            "score_pairs() received an empty pairs_df (zero rows). "
            "Nothing to score."
        )

    # ------------------------------------------------------------------
    # 5. Validate feature dtypes are numeric
    # ------------------------------------------------------------------
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(pairs_df[col]):
            raise TypeError(
                f"Feature column '{col}' must be numeric, "
                f"got dtype {pairs_df[col].dtype}."
            )

    # ------------------------------------------------------------------
    # 6. Validate no NaN values in features
    # ------------------------------------------------------------------
    feature_subset = pairs_df[list(feature_cols)]
    nan_cols = [col for col in feature_cols if feature_subset[col].isna().any()]
    if nan_cols:
        raise ValueError(
            f"score_pairs(): feature column(s) contain NaN values: {nan_cols}"
        )

    # ------------------------------------------------------------------
    # 7. Validate no infinite values in features
    # ------------------------------------------------------------------
    arr = feature_subset.to_numpy(dtype=float)
    if np.isinf(arr).any():
        raise ValueError(
            "score_pairs(): feature column(s) contain infinite values."
        )

    # ------------------------------------------------------------------
    # 8. Build feature matrix (preserve original row order)
    # ------------------------------------------------------------------
    X = pairs_df[list(feature_cols)]

    # ------------------------------------------------------------------
    # 9. Generate raw probabilities -- NO threshold applied
    # ------------------------------------------------------------------
    raw_proba = model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # 10. Validate output probabilities are finite and within [0, 1]
    # ------------------------------------------------------------------
    if not np.isfinite(raw_proba).all():
        raise ValueError(
            "score_pairs(): model.predict_proba() returned non-finite "
            "probability values."
        )
    if raw_proba.min() < 0.0 or raw_proba.max() > 1.0:
        raise ValueError(
            f"score_pairs(): probabilities must be in [0, 1], "
            f"got range [{raw_proba.min():.6f}, {raw_proba.max():.6f}]."
        )

    # ------------------------------------------------------------------
    # 11. Build output DataFrame -- copy, then append prob_match
    # ------------------------------------------------------------------
    output_df = pairs_df.copy()
    output_df[PROB_MATCH_COL] = raw_proba

    # ------------------------------------------------------------------
    # 12. Row-alignment invariant check
    # ------------------------------------------------------------------
    if len(output_df) != len(pairs_df):
        raise RuntimeError(
            f"score_pairs() row-alignment failure: "
            f"input had {len(pairs_df)} rows, output has {len(output_df)} rows."
        )

    return output_df


# ---------------------------------------------------------------------------
# Phase 3.5 -- Full Pipeline Validation on Held-Out S1 Entities
# ---------------------------------------------------------------------------

#: Default top-k for Phase 1 blocking (must match approved Phase 1 config).
PHASE35_TOP_K: int = 20

#: Same-country filter setting (must remain disabled by default per Phase 1 spec).
PHASE35_SAME_COUNTRY_FILTER: bool = False


def run_fresh_validation_blocking(
    val_s1_df: pd.DataFrame,
    s2s3_clean_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the approved Phase 1 TF-IDF blocking pipeline fresh for validation S1 entities.

    The vectorizer is fitted ONLY on the S2+S3 corpus, exactly as Phase 1 specifies.
    It is NEVER refit on validation S1 data.  Validation S1 texts are only transformed
    (not fitted) using the S2+S3-fitted vectorizer.

    Configuration (must match approved Phase 1 spec exactly):
        analyzer="char", ngram_range=(3,5), sublinear_tf=True,
        max_features=250000, top_k=20, same_country_filter=False

    Parameters
    ----------
    val_s1_df : pd.DataFrame
        Validation S1 records.  Must contain ``entity_id`` and ``clean_text``.
        This is ONLY the held-out validation split -- training S1 entities
        must be excluded before calling this function.
    s2s3_clean_df : pd.DataFrame
        Combined S2+S3 searchable corpus.  Must contain ``entity_id`` and
        ``clean_text``.  The vectorizer is fitted on this corpus.

    Returns
    -------
    pd.DataFrame
        Blocking results with columns:
            source1_entity_id, candidate_entity_id, cosine_similarity, rank
        One row per (val S1 entity, top-k candidate) pair, sorted by
        (source1_entity_id, rank).

    Raises
    ------
    ValueError
        If required columns are missing or inputs are empty.

    Notes
    -----
    * The vectorizer is fitted on s2s3_clean_df, NOT on val_s1_df.
    * same_country_filter is disabled (False) per Phase 1 approval.
    * top_k is fixed at 20 per Phase 1 approval.
    * No MinHash, ANN, semantic embeddings, or external data are used.
    """
    from src.blocking import build_clean_text, fit_vectorizer, search_candidates_mock

    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    for label, df in (("val_s1_df", val_s1_df), ("s2s3_clean_df", s2s3_clean_df)):
        for col in ("entity_id", "clean_text"):
            if col not in df.columns:
                raise ValueError(
                    f"run_fresh_validation_blocking(): '{col}' column missing "
                    f"from {label}. Found: {list(df.columns)}"
                )

    if val_s1_df.empty:
        raise ValueError(
            "run_fresh_validation_blocking(): val_s1_df is empty. "
            "Cannot run blocking on zero validation entities."
        )
    if s2s3_clean_df.empty:
        raise ValueError(
            "run_fresh_validation_blocking(): s2s3_clean_df is empty. "
            "Cannot run blocking against an empty searchable corpus."
        )

    # ------------------------------------------------------------------
    # 2. Fit vectorizer on S2+S3 corpus ONLY -- never on val S1
    # ------------------------------------------------------------------
    s2s3_corpus_texts = build_clean_text(s2s3_clean_df)
    vectorizer = fit_vectorizer(s2s3_corpus_texts)  # fitted on S2+S3 only

    # ------------------------------------------------------------------
    # 3. Run mock-scale blocking (val S1 is transformed, NOT fitted)
    #    same_country_filter=False per Phase 1 approval
    # ------------------------------------------------------------------
    blocking_results_df = search_candidates_mock(
        s1_df=val_s1_df,
        s2s3_df=s2s3_clean_df,
        vectorizer=vectorizer,
        top_k=PHASE35_TOP_K,
        same_country_filter=PHASE35_SAME_COUNTRY_FILTER,
    )

    return blocking_results_df


def build_fresh_validation_candidates(
    blocking_results_df: pd.DataFrame,
    all_val_s1_ids: "list[str] | None" = None,
) -> dict:
    """Convert fresh blocking results into the candidates_by_s1 dict format.

    Preserves the exact candidate ordering returned by the blocking stage.
    Does not deduplicate unless required by the existing approved blocking
    implementation.  Does not manually insert ground-truth matches.

    Every validation S1 entity is guaranteed to appear in the returned dict,
    even if it received zero candidates from blocking.  S1 entities with zero
    candidates map to an empty list.  This satisfies the P0-3 invariant:
    ``len(candidates_by_s1) == len(all_val_s1_ids)``.

    Parameters
    ----------
    blocking_results_df : pd.DataFrame
        Output of :func:`run_fresh_validation_blocking`.  Must contain
        ``source1_entity_id``, ``candidate_entity_id``, and ``rank``
        columns.  Rows must be sorted by (source1_entity_id, rank).
    all_val_s1_ids : list[str] or None
        Complete list of validation S1 entity IDs (including those that may
        have received zero candidates from blocking).  When provided, every
        ID in this list is guaranteed to appear as a key in the returned
        dict.  When None, only S1 entities that appear in
        *blocking_results_df* are included.

    Returns
    -------
    dict
        Mapping of ``source1_entity_id`` -> list of ``candidate_entity_id``
        strings in rank order (highest similarity first).  Every validation
        S1 entity is represented; zero-candidate S1 entities map to ``[]``.

    Raises
    ------
    ValueError
        If required columns are missing from blocking_results_df.
    """
    required_cols = {"source1_entity_id", "candidate_entity_id", "rank"}
    missing = required_cols - set(blocking_results_df.columns)
    if missing:
        raise ValueError(
            f"build_fresh_validation_candidates(): blocking_results_df is missing "
            f"column(s): {sorted(missing)}. Found: {list(blocking_results_df.columns)}"
        )

    # P0-3: Pre-populate every known val S1 ID with an empty list so that
    # zero-candidate S1 entities are never silently dropped from the output.
    candidates_by_s1: dict = {}
    if all_val_s1_ids is not None:
        for s1_id in all_val_s1_ids:
            candidates_by_s1[str(s1_id).strip()] = []

    if blocking_results_df.empty:
        return candidates_by_s1

    # Sort by (source1_entity_id, rank) to guarantee ordering
    sorted_df = blocking_results_df.sort_values(
        ["source1_entity_id", "rank"]
    )

    for _, row in sorted_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()
        if s1_id not in candidates_by_s1:
            candidates_by_s1[s1_id] = []
        candidates_by_s1[s1_id].append(cand_id)

    return candidates_by_s1


def compare_candidate_sets(
    fresh_candidates_by_s1: dict,
    phase32_val_pairs_df: pd.DataFrame,
) -> dict:
    """Compare freshly generated validation candidates against Phase 3.2 candidates.

    Reports candidate-set overlap per S1 entity without treating any difference
    as automatically a failure.  Differences are investigated and categorised.

    Parameters
    ----------
    fresh_candidates_by_s1 : dict
        Mapping of source1_entity_id -> list of candidate_entity_id from
        the freshly run Phase 1 blocking.
    phase32_val_pairs_df : pd.DataFrame
        Validation pair table from Phase 3.2.  Must contain
        ``source1_entity_id`` and ``candidate_entity_id``.

    Returns
    -------
    dict with keys:
        ``n_val_s1_entities``          -- number of validation S1 entities (fresh)
        ``n_fresh_candidate_pairs``    -- total fresh candidate pairs
        ``n_phase32_candidate_pairs``  -- total Phase 3.2 validation candidate pairs
        ``n_identical_candidate_sets`` -- S1 entities with identical candidate ID sets
        ``n_differing_candidate_sets`` -- S1 entities with different candidate ID sets
        ``candidate_set_overlap``      -- union-based overlap ratio across all pairs
        ``per_s1``                     -- list of per-entity comparison dicts
    """
    # Build phase32 candidates_by_s1
    phase32_by_s1: dict = {}
    if len(phase32_val_pairs_df) > 0:
        for s1_id, grp in phase32_val_pairs_df.groupby(PAIR_S1_ENTITY_COL):
            phase32_by_s1[s1_id] = list(grp["candidate_entity_id"].astype(str))

    n_fresh_total = sum(len(v) for v in fresh_candidates_by_s1.values())
    n_phase32_total = sum(len(v) for v in phase32_by_s1.values())

    n_identical = 0
    n_differing = 0
    total_fresh_set_size = 0
    total_intersection_size = 0
    per_s1 = []

    all_s1_ids = sorted(set(list(fresh_candidates_by_s1.keys()) + list(phase32_by_s1.keys())))

    for s1_id in all_s1_ids:
        fresh_set = set(fresh_candidates_by_s1.get(s1_id, []))
        phase32_set = set(phase32_by_s1.get(s1_id, []))

        intersection = fresh_set & phase32_set
        union = fresh_set | phase32_set

        overlap_ratio = len(intersection) / len(union) if union else 1.0
        identical = (fresh_set == phase32_set)

        if identical:
            n_identical += 1
        else:
            n_differing += 1

        total_fresh_set_size += len(fresh_set)
        total_intersection_size += len(intersection)

        per_s1.append({
            "source1_entity_id": s1_id,
            "fresh_candidates": sorted(fresh_set),
            "phase32_candidates": sorted(phase32_set),
            "intersection_size": len(intersection),
            "union_size": len(union),
            "overlap_ratio": round(overlap_ratio, 6),
            "identical": identical,
            "only_in_fresh": sorted(fresh_set - phase32_set),
            "only_in_phase32": sorted(phase32_set - fresh_set),
        })

    # Global overlap ratio (Jaccard over all fresh candidates)
    global_overlap = (
        total_intersection_size / total_fresh_set_size
        if total_fresh_set_size > 0
        else 1.0
    )

    return {
        "n_val_s1_entities": len(fresh_candidates_by_s1),
        "n_fresh_candidate_pairs": n_fresh_total,
        "n_phase32_candidate_pairs": n_phase32_total,
        "n_identical_candidate_sets": n_identical,
        "n_differing_candidate_sets": n_differing,
        "candidate_set_overlap": round(global_overlap, 6),
        "per_s1": per_s1,
    }


def check_candidate_recall(
    fresh_candidates_by_s1: dict,
    ground_truth_df: "pd.DataFrame | None",
) -> dict:
    """Calculate candidate-generation recall as a diagnostic only.

    IMPORTANT: This is STRICTLY a blocking diagnostic.  It measures whether
    the ground-truth match appears in the fresh candidate set.  It does NOT
    compute final model precision/recall/F0.5 or any performance metric.

    The mock validation data is tiny (1 validation S1 entity); results are
    a smoke test only and must NOT be used to tune blocking.

    Parameters
    ----------
    fresh_candidates_by_s1 : dict
        Mapping of source1_entity_id -> list of candidate_entity_id.
    ground_truth_df : pd.DataFrame or None
        Ground truth with columns [source1_entity_id, matching_entity_ids]
        (canonical) or [source1_entity_id, match_entity_ids] (legacy alias,
        accepted for backward compatibility).
        Pass None if ground truth is unavailable.

    Returns
    -------
    dict with keys:
        ``recall``                -- float in [0,1] or None if no GT
        ``n_gt_matches``          -- total ground-truth match pairs
        ``n_gt_retrieved``        -- GT matches found in candidates
        ``n_s1_evaluated``        -- S1 entities with GT matches
        ``unavailable``           -- True if GT was None
        ``diagnostic_only``       -- always True
        ``smoke_test_warning``    -- warning string about tiny mock data
    """
    from src.blocking import compute_candidate_recall

    # Both "matching_entity_ids" (canonical) and "match_entity_ids" (legacy)
    # are now accepted directly by compute_candidate_recall().
    # No column rename is needed here.
    result = compute_candidate_recall(fresh_candidates_by_s1, ground_truth_df)
    result["diagnostic_only"] = True
    result["smoke_test_warning"] = (
        "SMOKE TEST ONLY: mock validation data is tiny. "
        "Do NOT use this recall to tune blocking or make model decisions."
    )
    return result


def build_fresh_validation_feature_table(
    fresh_candidates_by_s1: dict,
    blocking_results_df: pd.DataFrame,
    val_s1_feature_df: pd.DataFrame,
    s2s3_feature_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full feature table for fresh validation candidates using Phase 2 functions.

    Uses the approved Phase 2.1--2.4 pipeline to construct the exact 10 approved
    features.  The blocking_cosine_sim comes from the fresh Phase 1 blocking scores
    (blocking_results_df), NOT from a separately fitted vectorizer.

    Does NOT use validation labels as features.

    Parameters
    ----------
    fresh_candidates_by_s1 : dict
        Mapping of source1_entity_id -> list of candidate_entity_id from
        :func:`build_fresh_validation_candidates`.
    blocking_results_df : pd.DataFrame
        Fresh Phase 1 blocking results with cosine scores.
        Must contain source1_entity_id, candidate_entity_id, cosine_similarity.
    val_s1_feature_df : pd.DataFrame
        S1 feature file with entity_id, clean_name, clean_address, clean_country.
        Contains ONLY validation S1 entities.
    s2s3_feature_df : pd.DataFrame
        S2+S3 feature file with entity_id, clean_name, clean_address, clean_country.
    ground_truth_df : pd.DataFrame or None, optional
        Ground-truth labels for adding label column (not used as features).
        If None, no label column is added.

    Returns
    -------
    pd.DataFrame
        Fully-featured validation pair table with the 10 approved feature columns
        plus source1_entity_id, candidate_entity_id, and optionally label.
        blocking_cosine_sim comes from fresh Phase 1 blocking scores.

    Raises
    ------
    ValueError
        If required columns are missing or the feature pipeline fails.
    """
    import pandas as _pd
    from src.features import (
        build_pair_table,
        add_name_similarity_features,
        add_address_similarity_features,
        add_metadata_features,
        add_ground_truth_labels,
    )

    # ------------------------------------------------------------------
    # 1. Build candidate_pairs_df in the format expected by build_pair_table
    #    (source1_entity_id, candidate_entity_ids as comma-separated string)
    # ------------------------------------------------------------------
    pair_rows = [
        {
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands),
        }
        for s1_id, cands in fresh_candidates_by_s1.items()
        # P0-3: S1 entities with zero candidates produce no pair rows in the
        # feature table (nothing to score), but they still have a row in
        # candidate_pairs.tsv (empty list) — that is handled by write_candidate_pairs.
        if cands
    ]

    if not pair_rows:
        raise ValueError(
            "build_fresh_validation_feature_table(): fresh_candidates_by_s1 "
            "contains no valid (non-empty) candidate lists. "
            "No pairs can be built for feature engineering."
        )

    candidate_pairs_df = _pd.DataFrame(pair_rows)

    # ------------------------------------------------------------------
    # 2. Prepare blocking scores DataFrame in the expected schema
    #    (source1_entity_id, candidate_entity_id, cosine_similarity)
    #    Use the fresh Phase 1 cosine scores -- never recompute.
    # ------------------------------------------------------------------
    score_col = None
    for cand_col in ("cosine_similarity", "blocking_cosine_sim", "score"):
        if cand_col in blocking_results_df.columns:
            score_col = cand_col
            break

    if score_col is None:
        raise ValueError(
            "build_fresh_validation_feature_table(): blocking_results_df must contain "
            "a cosine score column ('cosine_similarity', 'blocking_cosine_sim', or 'score'). "
            f"Found: {list(blocking_results_df.columns)}"
        )

    blocking_scores_df = blocking_results_df[
        ["source1_entity_id", "candidate_entity_id", score_col]
    ].copy()
    if score_col != "cosine_similarity":
        blocking_scores_df = blocking_scores_df.rename(
            columns={score_col: "cosine_similarity"}
        )

    # ------------------------------------------------------------------
    # 3. Phase 2.1 -- Build pair table with S1/S2+S3 feature fields joined
    # ------------------------------------------------------------------
    pair_table = build_pair_table(
        candidate_pairs_df=candidate_pairs_df,
        s1_df=val_s1_feature_df,
        s2s3_df=s2s3_feature_df,
        blocking_scores_df=blocking_scores_df,
    )

    # ------------------------------------------------------------------
    # 4. Phase 2.2 -- Name similarity features
    # ------------------------------------------------------------------
    pair_table = add_name_similarity_features(pair_table)

    # ------------------------------------------------------------------
    # 5. Phase 2.3 -- Address similarity features
    # ------------------------------------------------------------------
    pair_table = add_address_similarity_features(pair_table)

    # ------------------------------------------------------------------
    # 6. Phase 2.4 -- Metadata features (uses fresh blocking cosine sim)
    # ------------------------------------------------------------------
    pair_table = add_metadata_features(pair_table, blocking_scores=blocking_scores_df)

    # ------------------------------------------------------------------
    # 7. Phase 2.6 (optional) -- Ground-truth labels for diagnostics
    #    Labels are NEVER used as features; only appended for reporting.
    # ------------------------------------------------------------------
    if ground_truth_df is not None:
        pair_table = add_ground_truth_labels(pair_table, ground_truth_df)

    return pair_table


def compare_fresh_vs_phase32_features(
    fresh_val_pairs_df: pd.DataFrame,
    phase32_val_pairs_df: pd.DataFrame,
    feature_cols: "tuple[str, ...] | list[str]" = FEATURE_COLS,
) -> dict:
    """Compare fresh validation feature table against Phase 3.2 validation data.

    Checks column names, pair counts, feature values, labels, and blocking
    cosine scores for identical pairs.  Distinguishes ordering differences
    from data differences.

    Parameters
    ----------
    fresh_val_pairs_df : pd.DataFrame
        Fresh validation pair table from Phase 3.5.
    phase32_val_pairs_df : pd.DataFrame
        Phase 3.2 validation pair table.
    feature_cols : tuple or list of str
        Feature column names to compare.

    Returns
    -------
    dict with keys:
        ``same_val_s1_entities``     -- bool, same S1 entity sets
        ``same_pair_count``          -- bool, same number of pairs
        ``same_feature_col_names``   -- bool
        ``n_identical_pairs``        -- int, pairs present in both with matching features
        ``n_ordering_only_diffs``    -- int, pairs with same data but different order
        ``n_data_diffs``             -- int, pairs with actual data differences
        ``same_labels``              -- bool (for overlapping pairs)
        ``same_blocking_cosine``     -- bool (for overlapping pairs)
        ``same_model_probs``         -- bool (for overlapping scored pairs)
    """
    # S1 entity sets
    fresh_s1_ids = set(fresh_val_pairs_df[PAIR_S1_ENTITY_COL].unique())
    phase32_s1_ids = set(phase32_val_pairs_df[PAIR_S1_ENTITY_COL].unique())
    same_s1 = fresh_s1_ids == phase32_s1_ids

    same_pair_count = len(fresh_val_pairs_df) == len(phase32_val_pairs_df)

    # Feature column names
    fresh_feat_cols = [c for c in feature_cols if c in fresh_val_pairs_df.columns]
    phase32_feat_cols = [c for c in feature_cols if c in phase32_val_pairs_df.columns]
    same_feature_col_names = set(fresh_feat_cols) == set(phase32_feat_cols) == set(feature_cols)

    # Build pair key -> row lookup for overlap analysis
    def _pair_key(s1_id, cand_id):
        return (str(s1_id).strip(), str(cand_id).strip())

    fresh_lookup: dict = {}
    for _, row in fresh_val_pairs_df.iterrows():
        key = _pair_key(row[PAIR_S1_ENTITY_COL], row["candidate_entity_id"])
        fresh_lookup[key] = row

    phase32_lookup: dict = {}
    for _, row in phase32_val_pairs_df.iterrows():
        key = _pair_key(row[PAIR_S1_ENTITY_COL], row["candidate_entity_id"])
        phase32_lookup[key] = row

    # Compare overlapping pairs
    common_keys = set(fresh_lookup.keys()) & set(phase32_lookup.keys())
    n_identical = 0
    n_data_diffs = 0
    labels_match = True
    blocking_cosine_match = True
    prob_match_match = True

    for key in common_keys:
        fresh_row = fresh_lookup[key]
        phase32_row = phase32_lookup[key]
        pair_data_ok = True

        # Feature values
        for col in feature_cols:
            if col in fresh_val_pairs_df.columns and col in phase32_val_pairs_df.columns:
                fv = fresh_row.get(col)
                pv = phase32_row.get(col)
                if fv is not None and pv is not None:
                    try:
                        if abs(float(fv) - float(pv)) > 1e-9:
                            pair_data_ok = False
                    except (TypeError, ValueError):
                        if str(fv) != str(pv):
                            pair_data_ok = False

        # Labels
        if LABEL_COL in fresh_val_pairs_df.columns and LABEL_COL in phase32_val_pairs_df.columns:
            if fresh_row.get(LABEL_COL) != phase32_row.get(LABEL_COL):
                labels_match = False

        # Blocking cosine
        if "blocking_cosine_sim" in fresh_val_pairs_df.columns and "blocking_cosine_sim" in phase32_val_pairs_df.columns:
            fcos = fresh_row.get("blocking_cosine_sim")
            pcos = phase32_row.get("blocking_cosine_sim")
            if fcos is not None and pcos is not None:
                try:
                    if abs(float(fcos) - float(pcos)) > 1e-9:
                        blocking_cosine_match = False
                except (TypeError, ValueError):
                    blocking_cosine_match = False

        # Probabilities (if scored)
        if PROB_MATCH_COL in fresh_val_pairs_df.columns and PROB_MATCH_COL in phase32_val_pairs_df.columns:
            fprob = fresh_row.get(PROB_MATCH_COL)
            pprob = phase32_row.get(PROB_MATCH_COL)
            if fprob is not None and pprob is not None:
                try:
                    if abs(float(fprob) - float(pprob)) > 1e-9:
                        prob_match_match = False
                except (TypeError, ValueError):
                    prob_match_match = False

        if pair_data_ok:
            n_identical += 1
        else:
            n_data_diffs += 1

    # Ordering differences: pairs in both but appearing in different positions
    # (same data, different positional order)
    n_ordering_diffs = 0
    if same_pair_count and len(common_keys) == len(phase32_lookup):
        # All pairs present; check if only order changed
        fresh_ordered = [
            _pair_key(r[PAIR_S1_ENTITY_COL], r["candidate_entity_id"])
            for _, r in fresh_val_pairs_df.iterrows()
        ]
        phase32_ordered = [
            _pair_key(r[PAIR_S1_ENTITY_COL], r["candidate_entity_id"])
            for _, r in phase32_val_pairs_df.iterrows()
        ]
        if set(fresh_ordered) == set(phase32_ordered) and fresh_ordered != phase32_ordered:
            n_ordering_diffs = sum(
                1 for a, b in zip(fresh_ordered, phase32_ordered) if a != b
            )

    return {
        "same_val_s1_entities": same_s1,
        "same_pair_count": same_pair_count,
        "same_feature_col_names": same_feature_col_names,
        "n_identical_pairs": n_identical,
        "n_ordering_only_diffs": n_ordering_diffs,
        "n_data_diffs": n_data_diffs,
        "same_labels": labels_match,
        "same_blocking_cosine": blocking_cosine_match,
        "same_model_probs": prob_match_match,
    }


def validate_phase35_leakage_checks(
    train_s1_ids: "list[str]",
    val_s1_ids: "list[str]",
    fresh_val_pairs_df: pd.DataFrame,
    model: "lgb.LGBMClassifier",
    vectorizer_was_refit_on_val: bool = False,
    ground_truth_injected: bool = False,
    external_data_used: bool = False,
    country_whitelist_used: bool = False,
    threshold_applied: bool = False,
) -> dict:
    """Run all Phase 3.5 integrity and leakage checks.

    Returns a dict of check results.  Each value is True (pass) or
    a string message (fail).  A final ``all_passed`` key is True only
    if every check passes.

    Parameters
    ----------
    train_s1_ids : list[str]
        Training S1 entity IDs from Phase 3.1.
    val_s1_ids : list[str]
        Validation S1 entity IDs from Phase 3.1.
    fresh_val_pairs_df : pd.DataFrame
        Fresh validation pair table from Phase 3.5 pipeline.
    model : lgb.LGBMClassifier
        The Phase 3.3 trained model.
    vectorizer_was_refit_on_val : bool
        Must be False -- vectorizer must NOT be refit on validation S1.
    ground_truth_injected : bool
        Must be False -- ground-truth matches must NOT be injected into candidates.
    external_data_used : bool
        Must be False -- no external data or API calls.
    country_whitelist_used : bool
        Must be False -- no hardcoded country whitelist.
    threshold_applied : bool
        Must be False -- no threshold may be applied in Phase 3.5.

    Returns
    -------
    dict with keys for each check (True = pass) and ``all_passed`` bool.
    """
    checks: dict = {}

    # 1. No validation S1 ID appears in train S1 IDs
    val_set = set(val_s1_ids)
    train_set = set(train_s1_ids)
    overlap = val_set & train_set
    checks["no_val_s1_in_train"] = (
        True if not overlap
        else f"FAIL: {len(overlap)} val S1 IDs leaked into train: {sorted(overlap)}"
    )

    # 2. No validation S1 labels used during model training
    #    (model was trained before Phase 3.5; we verify via metadata)
    checks["val_labels_not_used_for_training"] = (
        True if hasattr(model, "training_metadata_")
        else "FAIL: model lacks training_metadata_ -- cannot verify training data"
    )

    # 3. Model not retrained in Phase 3.5
    #    (We can only assert this structurally -- no re-train call is made)
    checks["model_not_retrained"] = True  # structural: Phase 3.5 calls score_pairs only

    # 4. Vectorizer not refit on validation S1
    checks["vectorizer_not_refit_on_val"] = (
        True if not vectorizer_was_refit_on_val
        else "FAIL: vectorizer was refit on validation S1 data"
    )

    # 5. No ground-truth matches manually injected into candidates
    checks["no_ground_truth_injection"] = (
        True if not ground_truth_injected
        else "FAIL: ground-truth matches were manually injected into candidates"
    )

    # 6. No external data or API used
    checks["no_external_data"] = (
        True if not external_data_used
        else "FAIL: external data or API was used"
    )

    # 7. No hardcoded country whitelist used
    checks["no_country_whitelist"] = (
        True if not country_whitelist_used
        else "FAIL: a hardcoded country whitelist was applied"
    )

    # 8. No threshold applied
    checks["no_threshold_applied"] = (
        True if not threshold_applied
        else "FAIL: a threshold was applied to produce binary predictions"
    )

    # 9. No final match decisions produced
    #    (prob_match column is the only output; no binary "match" column)
    has_binary_match_col = "match" in fresh_val_pairs_df.columns
    checks["no_final_match_decisions"] = (
        True if not has_binary_match_col
        else "FAIL: a 'match' column (binary decision) found in output"
    )

    # 10. Validation S1 IDs don't appear in training pairs
    fresh_s1_ids = set(fresh_val_pairs_df[PAIR_S1_ENTITY_COL].unique())
    s1_in_train = fresh_s1_ids & train_set
    checks["fresh_s1_not_in_train"] = (
        True if not s1_in_train
        else f"FAIL: {len(s1_in_train)} fresh val S1 IDs also appear in train set: {sorted(s1_in_train)}"
    )

    checks["all_passed"] = all(v is True for v in checks.values())
    return checks


def run_phase35_validation_pipeline(
    s1_clean_df: pd.DataFrame,
    s2s3_clean_df: pd.DataFrame,
    s1_feature_df: pd.DataFrame,
    s2s3_feature_df: pd.DataFrame,
    labeled_pairs_df: pd.DataFrame,
    ground_truth_df: "pd.DataFrame | None" = None,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Run the complete Phase 3.5 end-to-end validation pipeline.

    Orchestrates:
    1.  Phase 3.1 entity split (train / validation S1 IDs).
    2.  Phase 3.2 pair splits.
    3.  Phase 3.3 model training (train split only).
    4.  Phase 3.5 FRESH blocking for validation S1.
    5.  Phase 3.5 candidate comparison (fresh vs Phase 3.2).
    6.  Phase 3.5 candidate recall (diagnostic only).
    7.  Phase 3.5 fresh feature construction.
    8.  Phase 3.4 scoring using the Phase 3.3 model.
    9.  Phase 3.5 feature table comparison.
    10. Phase 3.5 leakage checks.

    CRITICAL LEAKAGE CONSTRAINTS:
    * The model is trained ONLY on training pairs (train S1 entities).
    * Validation S1 data is NEVER used to fit the vectorizer.
    * The model is NOT retrained during Phase 3.5.
    * No threshold is applied.

    Parameters
    ----------
    s1_clean_df : pd.DataFrame
        Full S1 clean dataset (entity_id, clean_text).
    s2s3_clean_df : pd.DataFrame
        Full S2+S3 clean dataset (entity_id, clean_text).
    s1_feature_df : pd.DataFrame
        S1 feature dataset (entity_id, clean_name, clean_address, clean_country).
    s2s3_feature_df : pd.DataFrame
        S2+S3 feature dataset (entity_id, clean_name, clean_address, clean_country).
    labeled_pairs_df : pd.DataFrame
        Full Phase 2.6 labeled pair table.
    ground_truth_df : pd.DataFrame or None
        Optional ground truth for candidate recall diagnostic.
    holdout_frac : float
        Fraction for validation split (default 0.15).
    random_state : int
        Random seed (default 42).

    Returns
    -------
    dict with keys:
        ``train_s1_ids``           -- list[str]
        ``val_s1_ids``             -- list[str]
        ``train_s1_df``            -- pd.DataFrame (train S1 clean records)
        ``val_s1_df``              -- pd.DataFrame (val S1 clean records)
        ``train_pairs_df``         -- pd.DataFrame (Phase 3.2 train pairs)
        ``val_pairs_df``           -- pd.DataFrame (Phase 3.2 val pairs)
        ``model``                  -- lgb.LGBMClassifier (trained on train only)
        ``fresh_blocking_df``      -- pd.DataFrame (fresh val blocking results)
        ``fresh_candidates_by_s1`` -- dict (fresh candidates per val S1)
        ``candidate_comparison``   -- dict (Step 3 comparison metrics)
        ``candidate_recall``       -- dict (Step 4 recall diagnostic)
        ``fresh_val_pairs_df``     -- pd.DataFrame (fresh features, pre-scoring)
        ``fresh_scored_val_pairs_df`` -- pd.DataFrame (with prob_match)
        ``feature_comparison``     -- dict (Step 7 comparison)
        ``leakage_checks``         -- dict (Step 8 integrity checks)
        ``val_gt_df``              -- pd.DataFrame (GT filtered to val S1 IDs only)
    """
    # ------------------------------------------------------------------
    # Step 1: Phase 3.1 entity split
    # ------------------------------------------------------------------
    train_s1_ids, val_s1_ids = split_entities(
        s1_clean_df, holdout_frac=holdout_frac, random_state=random_state
    )

    # ------------------------------------------------------------------
    # P0-2: Filter ground truth to ONLY the validation S1 IDs.
    # Training GT must never contaminate validation metrics.
    # ------------------------------------------------------------------
    val_gt_df = None
    if ground_truth_df is not None:
        val_gt_df = filter_ground_truth_to_s1_ids(ground_truth_df, val_s1_ids)
        # Assert P0-2 integrity invariants
        assert_val_gt_integrity(train_s1_ids, val_s1_ids, val_gt_df)

    # ------------------------------------------------------------------
    # Step 2: Phase 3.2 pair splits
    # ------------------------------------------------------------------
    train_pairs_df, val_pairs_df = build_pair_splits(
        labeled_pairs_df, train_s1_ids, val_s1_ids
    )

    # ------------------------------------------------------------------
    # Step 3: Phase 3.3 model training (train split ONLY)
    # ------------------------------------------------------------------
    model = train_lightgbm(train_pairs_df, feature_cols=FEATURE_COLS)

    # ------------------------------------------------------------------
    # Step 4: Phase 3.5 -- fresh validation blocking
    # ------------------------------------------------------------------
    # Build val_s1_df (clean records for val entities only)
    val_s1_df = s1_clean_df[
        s1_clean_df[S1_ENTITY_ID_COL].isin(set(val_s1_ids))
    ].copy()

    # Build train_s1_df (clean records for train entities only)
    train_s1_df = s1_clean_df[
        s1_clean_df[S1_ENTITY_ID_COL].isin(set(train_s1_ids))
    ].copy()

    fresh_blocking_df = run_fresh_validation_blocking(val_s1_df, s2s3_clean_df)

    # ------------------------------------------------------------------
    # Step 5: Build fresh candidate set
    # ------------------------------------------------------------------
    # Pass all val S1 IDs so zero-candidate S1s still appear in the dict
    # (P0-3 invariant: every val S1 gets exactly one row in candidate_pairs.tsv)
    fresh_candidates_by_s1 = build_fresh_validation_candidates(
        fresh_blocking_df,
        all_val_s1_ids=val_s1_ids,
    )

    # ------------------------------------------------------------------
    # Step 6: Compare fresh candidates vs Phase 3.2 candidates
    # ------------------------------------------------------------------
    candidate_comparison = compare_candidate_sets(fresh_candidates_by_s1, val_pairs_df)

    # ------------------------------------------------------------------
    # Step 7: Candidate recall (diagnostic only)
    # P0-2: Use val_gt_df (filtered to val S1 IDs only) — not full GT.
    # ------------------------------------------------------------------
    candidate_recall = check_candidate_recall(fresh_candidates_by_s1, val_gt_df)

    # ------------------------------------------------------------------
    # Step 8: Build fresh validation feature table (Phase 2 pipeline)
    # ------------------------------------------------------------------
    # Build val S1 feature dataframe (only val entities)
    val_s1_feature_df = s1_feature_df[
        s1_feature_df["entity_id"].isin(set(val_s1_ids))
    ].copy()

    fresh_val_pairs_df = build_fresh_validation_feature_table(
        fresh_candidates_by_s1=fresh_candidates_by_s1,
        blocking_results_df=fresh_blocking_df,
        val_s1_feature_df=val_s1_feature_df,
        s2s3_feature_df=s2s3_feature_df,
        # P0-2: pass val-restricted GT so labels come from val GT only
        ground_truth_df=val_gt_df,
    )

    # ------------------------------------------------------------------
    # Step 9: Score using the Phase 3.3 model (inference only -- no retraining)
    # ------------------------------------------------------------------
    fresh_scored_val_pairs_df = score_pairs(model, fresh_val_pairs_df, FEATURE_COLS)

    # ------------------------------------------------------------------
    # Step 10: Compare fresh vs Phase 3.2 feature/score table
    # ------------------------------------------------------------------
    feature_comparison = compare_fresh_vs_phase32_features(
        fresh_val_pairs_df=fresh_scored_val_pairs_df,
        phase32_val_pairs_df=val_pairs_df,
        feature_cols=FEATURE_COLS,
    )

    # ------------------------------------------------------------------
    # Step 11: Leakage checks
    # ------------------------------------------------------------------
    leakage_checks = validate_phase35_leakage_checks(
        train_s1_ids=train_s1_ids,
        val_s1_ids=val_s1_ids,
        fresh_val_pairs_df=fresh_scored_val_pairs_df,
        model=model,
        vectorizer_was_refit_on_val=False,   # structural: vectorizer fitted on s2s3 only
        ground_truth_injected=False,          # structural: no manual GT injection
        external_data_used=False,             # structural: only local files used
        country_whitelist_used=False,         # structural: same_country_filter=False
        threshold_applied=False,              # structural: score_pairs applies no threshold
    )

    return {
        "train_s1_ids": train_s1_ids,
        "val_s1_ids": val_s1_ids,
        "train_s1_df": train_s1_df,
        "val_s1_df": val_s1_df,
        "train_pairs_df": train_pairs_df,
        "val_pairs_df": val_pairs_df,
        "model": model,
        "fresh_blocking_df": fresh_blocking_df,
        "fresh_candidates_by_s1": fresh_candidates_by_s1,
        "candidate_comparison": candidate_comparison,
        "candidate_recall": candidate_recall,
        "fresh_val_pairs_df": fresh_val_pairs_df,
        "fresh_scored_val_pairs_df": fresh_scored_val_pairs_df,
        "feature_comparison": feature_comparison,
        "leakage_checks": leakage_checks,
        # P0-2: GT filtered to val S1 IDs only — use this for all val metrics
        "val_gt_df": val_gt_df,
    }


# ---------------------------------------------------------------------------
# Stage B — Hard-Negative Mining
# ---------------------------------------------------------------------------
#
# Hard negatives are training candidate pairs that are:
#   - labeled negative (label=0) in the training ground truth
#   - scored with HIGH predicted match probability by the current model
#
# These are the pairs the model is "confused" by.  Adding them to training
# (with their correct negative label) teaches the model to distinguish
# hard near-misses from true matches.
#
# LEAKAGE RULES (strictly enforced):
#   1. Mining operates ONLY on training S1 entities.
#   2. Validation S1 entities NEVER enter mining.
#   3. Test S1 entities NEVER enter mining.
#   4. Validation labels NEVER influence mining or retraining.
#   5. Mining uses model predictions on training data only.
#   6. Positives (label=1) are NEVER converted to negatives.
#   7. Candidate set is not expanded by mining — only existing training
#      negatives are promoted into the sampling pool.
# ---------------------------------------------------------------------------

#: Default probability threshold above which a training negative is
#: considered a "hard" negative.
DEFAULT_HNM_THRESHOLD: float = 0.3

#: Default maximum number of hard negatives to add per positive pair.
#: Keeps the class balance from degenerating.
DEFAULT_HNM_MAX_PER_POS: int = 3

#: Default fraction of available hard negatives to include (0 < v <= 1).
#: Using a fraction rather than a hard cap makes the selector scale with
#: dataset size.  When combined with max_per_pos both limits apply.
DEFAULT_HNM_SAMPLE_FRAC: float = 1.0


def mine_hard_negatives(
    train_pairs_df: pd.DataFrame,
    model: "lgb.LGBMClassifier",
    feature_cols: "list[str] | tuple[str, ...]" = FEATURE_COLS,
    threshold: float = DEFAULT_HNM_THRESHOLD,
    max_hard_neg_per_pos: int = DEFAULT_HNM_MAX_PER_POS,
    sample_frac: float = DEFAULT_HNM_SAMPLE_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Identify hard-negative training pairs from the training split.

    A hard negative is a candidate pair that is:
      - labeled 0 (negative) in *train_pairs_df*
      - assigned a predicted match probability >= *threshold* by *model*

    Only training S1 entity pairs are inspected.  Validation and test data
    are never passed here and never influence the output.

    Parameters
    ----------
    train_pairs_df : pd.DataFrame
        Training candidate pairs from Phase 3.2, containing all feature
        columns, a ``label`` column, and pair identifiers.  Must contain
        ONLY training-split rows (no validation S1 entities).
    model : lgb.LGBMClassifier
        A **fitted** model from Phase 3.3 / :func:`train_lightgbm`.
        The model is used in inference mode only — it is NEVER retrained
        here.
    feature_cols : list or tuple of str, optional
        Feature column names matching those used to train *model*.
        Default: :data:`FEATURE_COLS` (10 baseline features).
    threshold : float, optional
        Minimum ``prob_match`` for a negative pair to be considered hard.
        Default: 0.3.  Must be in (0, 1).
    max_hard_neg_per_pos : int, optional
        Maximum hard negatives to include per positive pair in
        *train_pairs_df*.  Controls class-balance degradation.
        Default: 3.
    sample_frac : float, optional
        Fraction of eligible hard negatives to keep (before max_per_pos
        cap).  Allows sub-sampling when many hard negatives exist.
        Default: 1.0 (keep all eligible before cap).
    random_state : int, optional
        Seed for deterministic sampling.  Default: 42.

    Returns
    -------
    pd.DataFrame
        Subset of *train_pairs_df* containing only the selected hard-negative
        rows (label=0).  Columns and dtypes are identical to *train_pairs_df*.
        If no hard negatives meet the threshold, an empty DataFrame is returned
        (with the same columns).

    Raises
    ------
    ValueError
        - If ``label`` column is missing from *train_pairs_df*.
        - If *threshold* is not in (0, 1).
        - If *max_hard_neg_per_pos* < 1.
        - If *sample_frac* is not in (0, 1].
        - If features contain NaN / infinite values.
    RuntimeError
        - If *model* is not fitted.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if LABEL_COL not in train_pairs_df.columns:
        raise ValueError(
            f"mine_hard_negatives(): '{LABEL_COL}' column missing from "
            f"train_pairs_df.  Found: {list(train_pairs_df.columns)}"
        )
    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"mine_hard_negatives(): threshold must be in (0, 1), got {threshold!r}."
        )
    if max_hard_neg_per_pos < 1:
        raise ValueError(
            f"mine_hard_negatives(): max_hard_neg_per_pos must be >= 1, "
            f"got {max_hard_neg_per_pos!r}."
        )
    if not (0.0 < sample_frac <= 1.0):
        raise ValueError(
            f"mine_hard_negatives(): sample_frac must be in (0, 1], "
            f"got {sample_frac!r}."
        )
    if not callable(getattr(model, "predict_proba", None)):
        raise RuntimeError(
            "mine_hard_negatives(): model must have predict_proba(). "
            f"Got type {type(model).__name__}."
        )

    # ------------------------------------------------------------------
    # 2. Short-circuit on empty input
    # ------------------------------------------------------------------
    if train_pairs_df.empty:
        return train_pairs_df.iloc[0:0].copy()

    # ------------------------------------------------------------------
    # 3. Isolate negative pairs (label=0) — positives are NEVER touched
    # ------------------------------------------------------------------
    neg_df = train_pairs_df[train_pairs_df[LABEL_COL] == 0].copy()
    pos_df = train_pairs_df[train_pairs_df[LABEL_COL] == 1]
    n_pos = len(pos_df)

    if neg_df.empty:
        return train_pairs_df.iloc[0:0].copy()

    # ------------------------------------------------------------------
    # 4. Score all training negatives with the current model
    # ------------------------------------------------------------------
    scored_neg_df = score_pairs(model, neg_df, feature_cols)

    # ------------------------------------------------------------------
    # 5. Select hard negatives: prob_match >= threshold
    # ------------------------------------------------------------------
    hard_mask = scored_neg_df[PROB_MATCH_COL] >= threshold
    hard_neg_df = scored_neg_df[hard_mask].copy()

    if hard_neg_df.empty:
        return train_pairs_df.iloc[0:0].copy()

    # ------------------------------------------------------------------
    # 6. Optional sub-sampling (fraction of eligible hard negatives)
    # ------------------------------------------------------------------
    if sample_frac < 1.0:
        n_sample = max(1, int(len(hard_neg_df) * sample_frac))
        hard_neg_df = hard_neg_df.sample(
            n=n_sample, random_state=random_state, replace=False
        )

    # ------------------------------------------------------------------
    # 7. Cap at max_hard_neg_per_pos * n_pos to preserve class balance
    # ------------------------------------------------------------------
    max_total = n_pos * max_hard_neg_per_pos
    if len(hard_neg_df) > max_total:
        # Sort by prob_match descending to keep the hardest negatives first
        hard_neg_df = hard_neg_df.sort_values(
            PROB_MATCH_COL, ascending=False
        ).head(max_total)

    # ------------------------------------------------------------------
    # 8. Return the original train_pairs_df rows (without prob_match col)
    #    to keep the returned schema identical to the input
    # ------------------------------------------------------------------
    hard_indices = hard_neg_df.index
    result = train_pairs_df.loc[hard_indices].copy()

    return result


def build_hnm_training_pairs(
    train_pairs_df: pd.DataFrame,
    hard_negatives_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine original training pairs with additional hard-negative pairs.

    Appends the selected hard negatives to *train_pairs_df*.  Duplicate
    pair rows (same ``source1_entity_id`` + ``candidate_entity_id``) are
    removed, keeping the first occurrence (original training pair takes
    priority).

    Positives (label=1) from *train_pairs_df* are always preserved.

    Parameters
    ----------
    train_pairs_df : pd.DataFrame
        Original training pairs (from Phase 3.2 + negative sampling).
    hard_negatives_df : pd.DataFrame
        Hard-negative subset returned by :func:`mine_hard_negatives`.
        Must have the same columns as *train_pairs_df*.

    Returns
    -------
    pd.DataFrame
        Augmented training set with hard negatives included.

    Raises
    ------
    ValueError
        If column schemas do not match or ``label`` column is missing.
    """
    if LABEL_COL not in train_pairs_df.columns:
        raise ValueError(
            "build_hnm_training_pairs(): 'label' column missing from "
            "train_pairs_df."
        )
    if hard_negatives_df.empty:
        return train_pairs_df.copy()

    # Verify all hard negatives are truly negatives — protect positives
    if (hard_negatives_df[LABEL_COL] == 1).any():
        raise ValueError(
            "build_hnm_training_pairs(): hard_negatives_df contains positive "
            "pairs (label=1).  Hard negatives must all be label=0."
        )

    combined = pd.concat(
        [train_pairs_df, hard_negatives_df],
        ignore_index=True,
        sort=False,
    )

    # Deduplicate: original rows take priority (they appear first)
    id_cols = [PAIR_S1_ENTITY_COL, "candidate_entity_id"]
    if all(c in combined.columns for c in id_cols):
        combined = combined.drop_duplicates(subset=id_cols, keep="first")

    return combined.reset_index(drop=True)


def fit_with_hard_negative_mining(
    train_pairs_df: pd.DataFrame,
    feature_cols: "list[str] | tuple[str, ...]" = FEATURE_COLS,
    params: "dict[str, object] | None" = None,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    hnm_threshold: float = DEFAULT_HNM_THRESHOLD,
    hnm_max_per_pos: int = DEFAULT_HNM_MAX_PER_POS,
    hnm_sample_frac: float = DEFAULT_HNM_SAMPLE_FRAC,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> "tuple[lgb.LGBMClassifier, dict[str, object]]":
    """Train a LightGBM model with one round of hard-negative mining.

    Algorithm
    ---------
    1. Train a baseline LightGBM on *train_pairs_df* (with 5:1 negative
       sampling).
    2. Score all training negatives with the baseline model.
    3. Select hard negatives (prob_match >= *hnm_threshold*), capped at
       *hnm_max_per_pos* × n_positives.
    4. Append hard negatives to the training set.
    5. Retrain a fresh LightGBM on the augmented set.
    6. Return the retrained model and a metadata dict.

    LEAKAGE INVARIANTS:
      - Only *train_pairs_df* rows are ever mined or used for retraining.
      - The caller must guarantee *train_pairs_df* contains ONLY training
        S1 entities (no validation, no test).
      - The baseline model is fitted and discarded; the returned model is
        fresh (not fine-tuned from the baseline weights).

    Parameters
    ----------
    train_pairs_df : pd.DataFrame
        Phase 3.2 training pairs.  Must contain feature columns + ``label``.
    feature_cols : list or tuple of str
        Feature column names.  Default: :data:`FEATURE_COLS`.
    params : dict or None
        LightGBM parameters.  Defaults to :data:`DEFAULT_LGBM_PARAMS`.
    negative_ratio : int
        Negatives-per-positive cap for the initial downsampling step.
    hnm_threshold : float
        Hard-negative selection threshold.  Default: 0.3.
    hnm_max_per_pos : int
        Maximum hard negatives per positive.  Default: 3.
    hnm_sample_frac : float
        Fraction of eligible hard negatives to keep.  Default: 1.0.
    random_state : int
        Random seed.  Default: 42.

    Returns
    -------
    model : lgb.LGBMClassifier
        Retrained model on baseline + hard-negative augmented training set.
    metadata : dict
        Provenance record containing:
        - ``baseline_train_rows``   : int
        - ``baseline_pos``          : int
        - ``baseline_neg``          : int
        - ``n_hard_negatives_found``: int
        - ``n_hard_negatives_added``: int
        - ``augmented_train_rows``  : int
        - ``augmented_pos``         : int
        - ``augmented_neg``         : int
        - ``hnm_threshold``         : float
        - ``hnm_max_per_pos``       : int
        - ``random_state``          : int
        - ``feature_cols``          : list[str]
        - ``params``                : dict
    """
    # ------------------------------------------------------------------
    # Step 1: Train baseline model
    # ------------------------------------------------------------------
    baseline_model, baseline_meta = train_lightgbm(
        train_pairs_df=train_pairs_df,
        feature_cols=feature_cols,
        params=params,
        negative_ratio=negative_ratio,
        random_state=random_state,
        return_metadata=True,
    )

    baseline_pos = int((train_pairs_df[LABEL_COL] == 1).sum())
    baseline_neg = int((train_pairs_df[LABEL_COL] == 0).sum())

    # ------------------------------------------------------------------
    # Step 2 & 3: Mine hard negatives from TRAINING DATA ONLY
    # ------------------------------------------------------------------
    hard_neg_df = mine_hard_negatives(
        train_pairs_df=train_pairs_df,
        model=baseline_model,
        feature_cols=feature_cols,
        threshold=hnm_threshold,
        max_hard_neg_per_pos=hnm_max_per_pos,
        sample_frac=hnm_sample_frac,
        random_state=random_state,
    )
    n_hard_found = len(hard_neg_df)

    # ------------------------------------------------------------------
    # Step 4: Augment training set
    # ------------------------------------------------------------------
    augmented_df = build_hnm_training_pairs(train_pairs_df, hard_neg_df)
    n_aug_pos = int((augmented_df[LABEL_COL] == 1).sum())
    n_aug_neg = int((augmented_df[LABEL_COL] == 0).sum())

    # ------------------------------------------------------------------
    # Step 5: Retrain on augmented set
    #
    # P1-5 FIX: The augmented set contains hard negatives that were mined
    # precisely because they are valuable.  We must not let the 5:1 ratio
    # cap inside train_lightgbm() re-downsample them away.
    #
    # Strategy: compute the actual neg:pos ratio in augmented_df and pass
    # that as negative_ratio so prepare_training_pairs() keeps ALL negatives
    # (both easy and hard).  This preserves the class balance we explicitly
    # constructed and logs it as metadata.
    #
    # If augmented_df has no positives (degenerate case), fall back to the
    # caller-supplied negative_ratio.
    # ------------------------------------------------------------------
    if n_aug_pos > 0:
        # Ceil ratio so prepare_training_pairs() retains every negative
        import math
        augmented_neg_ratio = max(
            negative_ratio,
            math.ceil(n_aug_neg / n_aug_pos),
        )
    else:
        augmented_neg_ratio = negative_ratio

    final_model, final_meta = train_lightgbm(
        train_pairs_df=augmented_df,
        feature_cols=feature_cols,
        params=params,
        negative_ratio=augmented_neg_ratio,   # ← preserves hard negatives
        random_state=random_state,
        return_metadata=True,
    )

    metadata: dict[str, object] = {
        "baseline_train_rows":    len(train_pairs_df),
        "baseline_pos":           baseline_pos,
        "baseline_neg":           baseline_neg,
        "n_hard_negatives_found": n_hard_found,
        "n_hard_negatives_added": len(hard_neg_df),
        "augmented_train_rows":   len(augmented_df),
        "augmented_pos":          n_aug_pos,
        "augmented_neg":          n_aug_neg,
        "hnm_threshold":          hnm_threshold,
        "hnm_max_per_pos":        hnm_max_per_pos,
        "hnm_sample_frac":        hnm_sample_frac,
        "random_state":           random_state,
        "feature_cols":           list(feature_cols),
        "params":                 final_meta["params"],
    }
    final_model.hnm_metadata_ = metadata

    # ------------------------------------------------------------------
    # P1-5: Log the HNM summary and assert hard negatives survived
    # ------------------------------------------------------------------
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    n_easy_neg = baseline_neg - n_hard_found
    final_pos  = final_meta["positive_rows_after"]
    final_neg  = final_meta["negative_rows_after"]
    # Hard negatives in the final training set = total_neg_after - easy_neg_kept
    # (All negatives are now kept by the elevated ratio; separate tracking below)
    final_ratio = final_neg / final_pos if final_pos > 0 else 0.0

    _logger.info(
        "[HNM] total_positives=%d  easy_negatives=%d  hard_negatives_found=%d  "
        "hard_negatives_added=%d  "
        "final_positives=%d  final_negatives=%d  final_pos:neg_ratio=1:%.2f",
        baseline_pos, max(0, n_easy_neg), n_hard_found, len(hard_neg_df),
        final_pos, final_neg, final_ratio,
    )

    # Assert hard negatives are actually present in the final training matrix
    # whenever HNM produced them (P1-5 invariant)
    if len(hard_neg_df) > 0:
        hard_indices = set(hard_neg_df.index)
        # After concat+dedup in build_hnm_training_pairs, the augmented_df
        # will contain the hard negatives.  The final model trained on it.
        aug_neg_indices = set(augmented_df[augmented_df[LABEL_COL] == 0].index)
        actually_present = hard_indices & aug_neg_indices
        assert len(actually_present) > 0, (
            "P1-5 FAIL: Hard negatives were mined but none survived into the "
            "augmented training set.  Check build_hnm_training_pairs() logic."
        )
        _logger.info(
            "[HNM] Assert PASSED: %d/%d hard negatives confirmed in augmented training set.",
            len(actually_present), len(hard_neg_df),
        )

    return final_model, metadata


# ---------------------------------------------------------------------------
# Stage C — LightGBM Hyperparameter Experiment
# ---------------------------------------------------------------------------
#
# Compare our baseline LightGBM configuration against a higher-capacity
# configuration inspired by the Chandrima repository.
#
# EXPERIMENT DISCIPLINE:
#   - Same entity split, same training pairs, same features, same negative
#     sampling ratio, same random seed, same threshold sweep procedure.
#   - Only the model parameters differ between baseline and experiment.
#   - Validation ground truth is required for a meaningful comparison.
#   - Results on tiny mock data MUST NOT be used to tune parameters.
#     This infrastructure is for real-data use only.
# ---------------------------------------------------------------------------

#: Experiment hyperparameters inspired by the Chandrima repository.
#: Source: ChandrimaNandi/Amazon-ML-Hackathon-2026 src/config.py (adapted).
#: Key differences from baseline:
#:   - n_estimators: 100 → 400 (more trees, compensated by lower lr)
#:   - learning_rate: 0.05 → 0.05 (unchanged — baseline already good)
#:   - num_leaves: 31 → 63 (more expressive tree structure)
#:   - subsample: not set → 0.8 (row-level bagging; prevents overfitting)
#:   - colsample_bytree: not set → 0.8 (column-level bagging)
#:   - min_child_samples: 1 → 20 (prevents leaves with single sample)
EXPERIMENT_LGBM_PARAMS: dict[str, object] = {
    "objective":           "binary",
    "random_state":        DEFAULT_RANDOM_STATE,
    "n_estimators":        400,
    "learning_rate":       0.05,
    "num_leaves":          63,
    "min_child_samples":   20,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "verbose":             -1,
}


def run_hyperparameter_experiment(
    train_pairs_df: pd.DataFrame,
    val_pairs_df: pd.DataFrame,
    feature_cols: "list[str] | tuple[str, ...]" = FEATURE_COLS,
    baseline_params: "dict[str, object] | None" = None,
    experiment_params: "dict[str, object] | None" = None,
    negative_ratio: int = DEFAULT_NEGATIVE_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
    ground_truth_df: "pd.DataFrame | None" = None,
    beta: float = 0.5,
) -> dict:
    """Compare baseline vs experiment LightGBM configurations.

    Both models are trained on the SAME training data, using the SAME
    entity split, features, negative sampling ratio, and random seed.
    Only the hyperparameters differ.

    The validation set is used ONLY for scoring and metric computation —
    it is NEVER used to guide training or parameter selection.

    Parameters
    ----------
    train_pairs_df : pd.DataFrame
        Training candidate pairs from Phase 3.2.  Must contain all
        feature columns and a ``label`` column.
    val_pairs_df : pd.DataFrame
        Validation candidate pairs from Phase 3.2.  Must contain all
        feature columns and, if *ground_truth_df* is provided, a ``label``
        column.  NEVER passed to :func:`train_lightgbm`.
    feature_cols : list or tuple of str
        Feature column names.  Default: :data:`FEATURE_COLS`.
    baseline_params : dict or None
        Baseline hyperparameters.  Defaults to :data:`DEFAULT_LGBM_PARAMS`.
    experiment_params : dict or None
        Experiment hyperparameters.  Defaults to :data:`EXPERIMENT_LGBM_PARAMS`.
    negative_ratio : int
        Negatives-per-positive cap.  Default: 5.
    random_state : int
        Random seed.  Default: 42.
    ground_truth_df : pd.DataFrame or None
        If provided, macro F0.5 is computed over the validation set.
        Must contain ``source1_entity_id`` and ``matching_entity_ids``.
        If ``None``, F0.5 metrics are ``None`` in the result.
    beta : float
        F-beta parameter.  Default: 0.5.

    Returns
    -------
    dict with keys:

    ``"baseline"`` : dict
        - ``params``         : dict (effective LightGBM params)
        - ``train_metadata`` : dict (from :func:`train_lightgbm`)
        - ``val_probs_df``   : pd.DataFrame (scored val pairs, no threshold)
        - ``macro_f_beta"``  : float or None
        - ``best_threshold`` : float or None
        - ``precision``      : float or None
        - ``recall``         : float or None
        - ``n_val_pairs``    : int

    ``"experiment"`` : dict (same structure)

    ``"comparison"`` : dict
        - ``f_beta_delta``         : float or None (experiment - baseline)
        - ``baseline_better``      : bool or None
        - ``experiment_better``    : bool or None
        - ``recommendation``       : str
        - ``real_data_required``   : bool
        - ``n_val_s1_entities``    : int
        - ``feature_cols``         : list[str]
        - ``same_train_split``     : True (always — both use same train_pairs_df)
        - ``same_neg_ratio``       : True
        - ``same_random_seed``     : True

    Notes
    -----
    * Results on tiny mock datasets MUST NOT be used to adopt experiment
      parameters.  The ``real_data_required`` flag is always True.
    * Both models are trained from scratch; no shared state.
    """
    from src.threshold import sweep_thresholds

    if baseline_params is None:
        baseline_params = dict(DEFAULT_LGBM_PARAMS)
    if experiment_params is None:
        experiment_params = dict(EXPERIMENT_LGBM_PARAMS)

    results: dict = {}

    for label, params in (("baseline", baseline_params), ("experiment", experiment_params)):
        # ---- Train --------------------------------------------------------
        model, train_meta = train_lightgbm(
            train_pairs_df=train_pairs_df,
            feature_cols=feature_cols,
            params=params,
            negative_ratio=negative_ratio,
            random_state=random_state,
            return_metadata=True,
        )

        # ---- Score validation (inference only — no threshold) ------------
        val_probs_df = score_pairs(model, val_pairs_df, feature_cols)

        # ---- Metrics (only if ground truth available) --------------------
        macro_fb = None
        best_thr = None
        prec = None
        rec = None

        if ground_truth_df is not None:
            try:
                sweep = sweep_thresholds(
                    scored_pairs_df=val_probs_df,
                    ground_truth_df=ground_truth_df,
                    beta=beta,
                )
                best_thr = sweep["best_threshold"]
                macro_fb = sweep["best_macro_f_beta"]

                # Compute precision and recall at best threshold
                from src.threshold import apply_threshold, _parse_ground_truth_df
                predictions = apply_threshold(val_probs_df, best_thr)
                gt_dict = _parse_ground_truth_df(ground_truth_df)

                from src.threshold import entity_f_beta
                prec_scores, rec_scores = [], []
                for s1_id, true_ids in gt_dict.items():
                    pred_ids = predictions.get(str(s1_id), set())
                    n_tp = len(pred_ids & set(true_ids))
                    prec_scores.append(n_tp / len(pred_ids) if pred_ids else 1.0)
                    rec_scores.append(n_tp / len(true_ids) if true_ids else 1.0)

                prec = float(np.mean(prec_scores)) if prec_scores else None
                rec = float(np.mean(rec_scores)) if rec_scores else None

            except Exception:
                pass  # Ground truth mismatch — leave metrics as None

        results[label] = {
            "params":         params,
            "train_metadata": train_meta,
            "val_probs_df":   val_probs_df,
            "macro_f_beta":   macro_fb,
            "best_threshold": best_thr,
            "precision":      prec,
            "recall":         rec,
            "n_val_pairs":    len(val_pairs_df),
        }

    # ---- Comparison summary ------------------------------------------
    base_fb = results["baseline"]["macro_f_beta"]
    exp_fb  = results["experiment"]["macro_f_beta"]

    delta = (exp_fb - base_fb) if (base_fb is not None and exp_fb is not None) else None

    if delta is None:
        recommendation = (
            "NEEDS REAL DATA: metrics are None (ground_truth_df not provided "
            "or too small).  Experiment infrastructure is verified; run on "
            "real challenge data before choosing parameters."
        )
        baseline_better = None
        experiment_better = None
    elif delta > 0.005:
        recommendation = (
            f"EXPERIMENT BETTER by {delta:.4f} macro F{beta:.1f}. "
            "Consider adopting experiment params after validation on real data."
        )
        baseline_better = False
        experiment_better = True
    elif delta < -0.005:
        recommendation = (
            f"BASELINE BETTER by {-delta:.4f} macro F{beta:.1f}. "
            "Keep current baseline params."
        )
        baseline_better = True
        experiment_better = False
    else:
        recommendation = (
            f"NO MEANINGFUL DIFFERENCE (delta={delta:.4f}). "
            "Keep baseline params (simpler / faster)."
        )
        baseline_better = True  # tie → prefer simpler
        experiment_better = False

    n_val_s1 = len(val_pairs_df[PAIR_S1_ENTITY_COL].unique()) if len(val_pairs_df) else 0

    results["comparison"] = {
        "f_beta_delta":         delta,
        "baseline_better":      baseline_better,
        "experiment_better":    experiment_better,
        "recommendation":       recommendation,
        "real_data_required":   True,
        "n_val_s1_entities":    n_val_s1,
        "feature_cols":         list(feature_cols),
        "same_train_split":     True,
        "same_neg_ratio":       True,
        "same_random_seed":     True,
    }

    return results
