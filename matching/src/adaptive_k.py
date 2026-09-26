"""
adaptive_k.py
-------------
Adaptive candidate count (adaptive K) infrastructure for the Business Entity
Resolution pipeline.

Instead of a fixed top_k=20 for all S1 entities, adaptive K adjusts the
candidate count per entity based on signals that indicate whether more or
fewer candidates are needed:

Signals used to increase K (→ more candidates):
  - High name frequency (common name → need more candidates to find the right one)
  - Low retrieval score gap (top score close to subsequent scores → ambiguous)
  - Low retrieval agreement (only one channel retrieved the top candidate)
  - Low rarity (common entity → many similar candidates exist)

Signals used to decrease K (→ fewer candidates):
  - Very high top retrieval score (clear winner → don't need extras)
  - High retrieval agreement (both channels agree on top → high confidence)
  - High rarity (very unique name → few false positives expected)

Design constraints:
  - K is bounded: K_MIN ≤ K ≤ K_MAX to prevent candidate explosion
  - The candidate VOLUME metric is tracked and logged
  - This is an EXPERIMENT infrastructure: fixed K=20 remains the default
    until ablation confirms adaptive K improves F0.5

Public API
~~~~~~~~~~
    AdaptiveKConfig              -- dataclass holding K bounds and thresholds
    compute_adaptive_k(...)      -- compute per-entity K values
    apply_adaptive_k(...)        -- filter an existing candidate table to adaptive K
    evaluate_k_settings(...)     -- measure recall + volume at different K values
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------

#: Default minimum K (never return fewer candidates than this per entity).
DEFAULT_K_MIN: int = 10

#: Default fixed K (current pipeline default).
DEFAULT_K_FIXED: int = 20

#: Default maximum K (never return more candidates than this per entity).
DEFAULT_K_MAX: int = 50

#: K values to test in the ablation sweep.
K_SWEEP_VALUES: list[int] = [10, 20, 30, 50]


@dataclass
class AdaptiveKConfig:
    """Configuration for adaptive-K computation.

    Attributes
    ----------
    k_min : int
        Minimum candidates returned per entity.
    k_max : int
        Maximum candidates returned per entity.
    k_fixed : int
        Fallback fixed K when no signals are available.

    high_score_threshold : float
        Top retrieval score above which K is decreased toward k_min.
    low_score_threshold : float
        Top retrieval score below which K is increased toward k_max.

    agreement_high_threshold : float
        n_channels_agreeing / max_channels above which K is decreased.
    agreement_low_threshold : float
        n_channels_agreeing / max_channels below which K is increased.

    rarity_high_threshold : float
        Name rarity above which K is decreased.
    rarity_low_threshold : float
        Name rarity below which K is increased.
    """
    k_min: int = DEFAULT_K_MIN
    k_max: int = DEFAULT_K_MAX
    k_fixed: int = DEFAULT_K_FIXED

    high_score_threshold: float = 0.85
    low_score_threshold: float = 0.40

    agreement_high_threshold: float = 0.67   # 4+ of 6 channels
    agreement_low_threshold: float = 0.17    # 1 of 6 channels

    rarity_high_threshold: float = 0.80
    rarity_low_threshold: float = 0.30


# ---------------------------------------------------------------------------
# Core adaptive-K computation
# ---------------------------------------------------------------------------

def compute_adaptive_k(
    s1_entity_ids: list[str],
    retrieval_scores: dict[str, float],
    retrieval_agreement: dict[str, float],
    name_rarity: dict[str, float],
    config: Optional[AdaptiveKConfig] = None,
) -> dict[str, int]:
    """Compute per-entity adaptive K values.

    Parameters
    ----------
    s1_entity_ids : list[str]
        All S1 entity IDs to compute K for.
    retrieval_scores : dict[str, float]
        Mapping from S1 entity ID → top retrieval score (cosine or BM25).
        Missing entries use 0.5 (neutral).
    retrieval_agreement : dict[str, float]
        Mapping from S1 entity ID → fraction of channels that retrieved the
        top candidate (n_channels_agreeing / max_channels).
        Missing entries use 0.5 (neutral).
    name_rarity : dict[str, float]
        Mapping from S1 entity ID → name rarity score [0,1].
        1.0 = maximally unique, 0.0 = very common.
        Missing entries use 0.5 (neutral).
    config : AdaptiveKConfig or None
        Configuration.  Uses defaults if None.

    Returns
    -------
    dict[str, int]
        Mapping from S1 entity ID → adaptive K value.
        All values are in [config.k_min, config.k_max].
    """
    if config is None:
        config = AdaptiveKConfig()

    result: dict[str, int] = {}

    for s1_id in s1_entity_ids:
        top_score = retrieval_scores.get(s1_id, 0.5)
        agreement = retrieval_agreement.get(s1_id, 0.5)
        rarity = name_rarity.get(s1_id, 0.5)

        # Start at fixed K
        k = float(config.k_fixed)

        # High top score → decrease K (we already have the right candidate)
        if top_score >= config.high_score_threshold:
            k = k * 0.75  # reduce by 25%

        # Low top score → increase K (need more candidates to find the match)
        elif top_score <= config.low_score_threshold:
            k = k * 1.5  # increase by 50%

        # High retrieval agreement → slight decrease (channels already agree)
        if agreement >= config.agreement_high_threshold:
            k = k * 0.85

        # Low retrieval agreement → increase (only one channel sees it)
        elif agreement <= config.agreement_low_threshold:
            k = k * 1.25

        # High rarity → decrease K (unique name, few false positives)
        if rarity >= config.rarity_high_threshold:
            k = k * 0.75

        # Low rarity (common name) → increase K
        elif rarity <= config.rarity_low_threshold:
            k = k * 1.35

        # Clamp to [k_min, k_max]
        k_int = int(round(k))
        k_int = max(config.k_min, min(config.k_max, k_int))

        result[s1_id] = k_int

    return result


def apply_adaptive_k(
    candidates_df: pd.DataFrame,
    adaptive_k_map: dict[str, int],
    rank_col: str = "rank",
) -> pd.DataFrame:
    """Filter a candidate table to per-entity adaptive K counts.

    For each S1 entity, retains only the top *k* candidates according to
    *rank_col*, where *k* comes from *adaptive_k_map*.  If an entity is not
    in the map, the fixed default K is used.

    Parameters
    ----------
    candidates_df : pd.DataFrame
        Candidate pair table.  Must contain ``source1_entity_id``
        and *rank_col*.
    adaptive_k_map : dict[str, int]
        Output of :func:`compute_adaptive_k`.
    rank_col : str
        Column holding the per-entity rank (1-based).  Default: ``"rank"``.

    Returns
    -------
    pd.DataFrame
        Filtered candidate table with ≤ k rows per S1 entity.
        Row order is preserved.
    """
    if "source1_entity_id" not in candidates_df.columns:
        raise ValueError(
            "apply_adaptive_k(): candidates_df must have 'source1_entity_id'."
        )
    if rank_col not in candidates_df.columns:
        raise ValueError(
            f"apply_adaptive_k(): rank column '{rank_col}' not found in candidates_df."
        )

    if candidates_df.empty:
        return candidates_df.copy()

    keep_mask = candidates_df.apply(
        lambda row: (
            row[rank_col] is not None
            and not (isinstance(row[rank_col], float) and not np.isfinite(row[rank_col]))
            and float(row[rank_col]) <= adaptive_k_map.get(
                str(row["source1_entity_id"]), DEFAULT_K_FIXED
            )
        ),
        axis=1,
    )

    return candidates_df.loc[keep_mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# K-sweep ablation evaluator
# ---------------------------------------------------------------------------

def evaluate_k_settings(
    scored_candidates_df: pd.DataFrame,
    ground_truth_df: Optional[pd.DataFrame],
    k_values: list[int] = K_SWEEP_VALUES,
    score_col: str = "cosine_similarity",
    rank_col: Optional[str] = None,
) -> list[dict]:
    """Measure candidate recall and volume at different K values.

    For each K in *k_values*:
    1. Retain only the top K candidates per S1 entity (by *score_col*).
    2. Measure candidate recall against *ground_truth_df*.
    3. Record candidate volume (total pairs) and avg per entity.

    Parameters
    ----------
    scored_candidates_df : pd.DataFrame
        Candidate table.  Must have ``source1_entity_id``,
        ``candidate_entity_id``, and *score_col*.
    ground_truth_df : pd.DataFrame or None
        Ground truth with ``source1_entity_id`` and one of
        ``matching_entity_ids`` / ``match_entity_ids``.
    k_values : list[int]
        K values to sweep.
    score_col : str
        Column to rank candidates by (descending).  Default: cosine_similarity.
    rank_col : str or None
        Pre-computed rank column.  If provided, used instead of recomputing.

    Returns
    -------
    list[dict]
        One dict per K value with keys:
        k, n_total_candidates, avg_candidates_per_entity, candidate_recall,
        n_gt_total, n_gt_recovered, n_s1_entities.
    """
    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(scored_candidates_df.columns)
    if missing:
        raise ValueError(
            f"evaluate_k_settings(): missing columns: {sorted(missing)}"
        )

    # Parse ground truth
    gt_lookup: dict[str, set[str]] = {}
    if ground_truth_df is not None:
        match_col = None
        for col in ("matching_entity_ids", "match_entity_ids"):
            if col in ground_truth_df.columns:
                match_col = col
                break
        if match_col:
            for _, row in ground_truth_df.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw = str(row[match_col]).strip()
                if raw and raw.lower() not in ("nan", "none", ""):
                    gt_lookup[s1_id] = {m.strip() for m in raw.split(",") if m.strip()}

    results = []

    for k in k_values:
        # Compute per-entity top-k (by score_col if rank_col not available)
        if rank_col and rank_col in scored_candidates_df.columns:
            filtered = scored_candidates_df[
                scored_candidates_df[rank_col].fillna(float("inf")) <= k
            ].copy()
        elif score_col in scored_candidates_df.columns:
            # Re-rank by score within each S1 entity
            scored_candidates_df = scored_candidates_df.copy()
            scored_candidates_df["_rank_tmp"] = scored_candidates_df.groupby(
                "source1_entity_id"
            )[score_col].rank(method="first", ascending=False)
            filtered = scored_candidates_df[
                scored_candidates_df["_rank_tmp"] <= k
            ].copy()
        else:
            warnings.warn(
                f"evaluate_k_settings(): neither '{score_col}' nor "
                f"'{rank_col}' found.  Using all candidates as-is."
            )
            filtered = scored_candidates_df.copy()

        n_total = len(filtered)
        s1_entities = filtered["source1_entity_id"].unique()
        n_s1 = len(s1_entities)
        avg_per_entity = n_total / n_s1 if n_s1 > 0 else 0.0

        # Candidate recall
        n_gt_total = sum(len(v) for v in gt_lookup.values())
        n_gt_recovered = 0

        if gt_lookup:
            candidate_set: set[tuple[str, str]] = set(
                zip(
                    filtered["source1_entity_id"].astype(str),
                    filtered["candidate_entity_id"].astype(str),
                )
            )
            for s1_id, true_cands in gt_lookup.items():
                for cid in true_cands:
                    if (s1_id, cid) in candidate_set:
                        n_gt_recovered += 1

        recall = n_gt_recovered / n_gt_total if n_gt_total > 0 else None

        results.append({
            "k":                        k,
            "n_total_candidates":       n_total,
            "avg_candidates_per_entity": round(avg_per_entity, 2),
            "candidate_recall":         round(recall, 6) if recall is not None else None,
            "n_gt_total":               n_gt_total,
            "n_gt_recovered":           n_gt_recovered,
            "n_s1_entities":            n_s1,
        })

    return results
