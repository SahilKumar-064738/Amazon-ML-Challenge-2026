"""
entity_decision.py
------------------
P1-6: Entity-level decision layer.

Operates AFTER pair scoring (prob_match column) and BEFORE final output.
Makes per-S1-entity match decisions independently of the ML classifier
configuration.

Key design rules
~~~~~~~~~~~~~~~~
1. Collects all candidate pair probabilities per S1 entity.
2. Applies the calibrated pair threshold (prob_match >= threshold).
3. Sorts surviving candidates by probability (descending).
4. Removes duplicate candidate IDs (deduplication).
5. Preserves multiple genuine matches when their scores support them.
6. Produces an empty match list for S1 entities with zero surviving candidates
   — no match is forced.
7. Does NOT hard-code "exactly one match per S1".
8. Threshold and all configuration values are explicit parameters.

This module makes NO machine-learning calls.  It reads only prob_match
and the threshold supplied by the caller.

Public API
~~~~~~~~~~
    EntityDecisionConfig          -- dataclass holding all tunable parameters
    make_entity_decisions(scored_pairs_df, config) -> dict[str, list[str]]
        Returns {s1_id: [matched_ids sorted by score desc]}.
        Every S1 ID in scored_pairs_df is in the output (P0-3 variant for final).

    write_matching_results(decisions, output_path, all_s1_ids=None) -> None
        Writes the challenge-format matching_results.tsv.
        Every S1 ID is guaranteed to appear (empty row for zero matches).

    decisions_to_dataframe(decisions) -> pd.DataFrame
        Convenience helper for downstream metric computation.
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EntityDecisionConfig:
    """Tunable parameters for the entity-level decision stage.

    Attributes
    ----------
    threshold : float
        Minimum prob_match for a candidate to be considered a match.
        Must be in [0.0, 1.0].  Default: 0.5.
        Set to the value returned by sweep_thresholds() on the validation set.
    min_score_margin : float
        Minimum difference between the top candidate score and the threshold
        before accepting a match.  0.0 means accept any pair at or above the
        threshold.  Increase to be more conservative.  Default: 0.0.
    max_matches_per_s1 : int or None
        Hard cap on predicted matches per S1 entity.  None = no cap (challenge
        allows one-to-many).  Default: None.
    deduplicate_candidates : bool
        Remove duplicate candidate IDs within each S1 entity's result list.
        Should always be True.  Default: True.
    prob_col : str
        Name of the probability column.  Default: "prob_match".
    s1_col : str
        Name of the S1 entity ID column.  Default: "source1_entity_id".
    cand_col : str
        Name of the candidate entity ID column.  Default: "candidate_entity_id".
    """
    threshold: float = 0.5
    min_score_margin: float = 0.0
    max_matches_per_s1: Optional[int] = None
    deduplicate_candidates: bool = True
    prob_col: str = "prob_match"
    s1_col: str = "source1_entity_id"
    cand_col: str = "candidate_entity_id"

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(
                f"EntityDecisionConfig.threshold must be in [0, 1], "
                f"got {self.threshold!r}."
            )
        if self.min_score_margin < 0.0:
            raise ValueError(
                f"EntityDecisionConfig.min_score_margin must be >= 0, "
                f"got {self.min_score_margin!r}."
            )
        if self.max_matches_per_s1 is not None and self.max_matches_per_s1 < 1:
            raise ValueError(
                f"EntityDecisionConfig.max_matches_per_s1 must be >= 1 or None, "
                f"got {self.max_matches_per_s1!r}."
            )


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------

def make_entity_decisions(
    scored_pairs_df: pd.DataFrame,
    config: EntityDecisionConfig,
    all_s1_ids: "list[str] | None" = None,
) -> "dict[str, list[str]]":
    """Apply entity-level decisions to a scored candidate pair table.

    For each S1 entity:
    1. Collect all (candidate_entity_id, prob_match) pairs.
    2. Apply threshold: keep only prob_match >= config.threshold.
    3. Apply score-margin filter if config.min_score_margin > 0.
    4. Sort surviving candidates by prob_match descending.
    5. Deduplicate candidate IDs (first-occurrence order kept).
    6. Apply max_matches_per_s1 cap if set.
    7. S1 entities with no survivors produce an empty list — no match forced.

    Parameters
    ----------
    scored_pairs_df : pd.DataFrame
        Must contain config.s1_col, config.cand_col, config.prob_col.
    config : EntityDecisionConfig
        Decision configuration.
    all_s1_ids : list[str] or None
        When provided, every S1 ID is guaranteed to appear in the output
        dict (P0-3 invariant for the final results file).

    Returns
    -------
    dict[str, list[str]]
        Maps every S1 entity ID to a list of matched candidate IDs
        (sorted by descending prob_match, deduplicated).
        S1 entities with no candidates above threshold map to [].

    Raises
    ------
    TypeError
        If scored_pairs_df is not a DataFrame.
    ValueError
        If required columns are missing.
    """
    if not isinstance(scored_pairs_df, pd.DataFrame):
        raise TypeError(
            f"make_entity_decisions(): scored_pairs_df must be a pd.DataFrame, "
            f"got {type(scored_pairs_df).__name__}."
        )

    required = {config.s1_col, config.cand_col, config.prob_col}
    missing = required - set(scored_pairs_df.columns)
    if missing:
        raise ValueError(
            f"make_entity_decisions(): scored_pairs_df is missing column(s): "
            f"{sorted(missing)}. Found: {list(scored_pairs_df.columns)}"
        )

    # Pre-populate with all known S1 IDs (P0-3 guarantee)
    decisions: dict[str, list[str]] = {}
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            decisions[str(s1_id)] = []

    if scored_pairs_df.empty:
        return decisions

    # Also pre-populate from the scored pairs (handles case where all_s1_ids not given)
    for s1_id in scored_pairs_df[config.s1_col].unique():
        decisions.setdefault(str(s1_id), [])

    # Process each S1 entity
    for s1_id, group in scored_pairs_df.groupby(config.s1_col, sort=False):
        s1_str = str(s1_id)
        probs = group[config.prob_col].to_numpy(dtype=float)
        cands = group[config.cand_col].astype(str).tolist()

        # Step 2: threshold filter
        above_mask = probs >= config.threshold

        # Step 3: score-margin filter (if set)
        if config.min_score_margin > 0.0 and above_mask.any():
            top_score = float(probs[above_mask].max())
            above_mask = above_mask & (probs >= config.threshold + config.min_score_margin)

        surviving_cands = [c for c, keep in zip(cands, above_mask) if keep]
        surviving_probs = probs[above_mask].tolist()

        # Step 4: sort by probability descending
        if surviving_cands:
            sorted_pairs = sorted(
                zip(surviving_probs, surviving_cands),
                key=lambda x: x[0],
                reverse=True,
            )
            surviving_cands = [cid for _, cid in sorted_pairs]

        # Step 5: deduplicate (preserve first-occurrence order = highest score first)
        if config.deduplicate_candidates and surviving_cands:
            seen: set = set()
            deduped = []
            for cid in surviving_cands:
                if cid not in seen:
                    seen.add(cid)
                    deduped.append(cid)
            surviving_cands = deduped

        # Step 6: cap at max_matches_per_s1
        if config.max_matches_per_s1 is not None:
            surviving_cands = surviving_cands[: config.max_matches_per_s1]

        decisions[s1_str] = surviving_cands

    # Log summary
    n_with_matches = sum(1 for v in decisions.values() if v)
    n_empty = sum(1 for v in decisions.values() if not v)
    total_matches = sum(len(v) for v in decisions.values())
    logger.info(
        "[entity_decision] threshold=%.3f  s1_total=%d  with_matches=%d  "
        "no_match=%d  total_predicted_pairs=%d",
        config.threshold, len(decisions), n_with_matches, n_empty, total_matches,
    )

    return decisions


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_matching_results(
    decisions: "dict[str, list[str]]",
    output_path: str,
    all_s1_ids: "list[str] | None" = None,
) -> None:
    """Write per-entity match decisions to a TSV file.

    Output format (challenge spec)::

        source1_entity_id<TAB>matched_entity_ids
        S1-001<TAB>S2-099,S2-102
        S1-002<TAB>
        S1-003<TAB>S2-044

    Every S1 ID in *decisions* (and every ID in *all_s1_ids* if provided)
    gets exactly one row.  S1 entities with no matches produce an empty
    ``matched_entity_ids`` field.

    Parameters
    ----------
    decisions : dict[str, list[str]]
        Output of :func:`make_entity_decisions`.
    output_path : str
        Destination TSV path.  Parent dirs are created automatically.
    all_s1_ids : list[str] or None
        If supplied, every ID here is guaranteed to appear (P0-3 compliance).

    Raises
    ------
    TypeError
        If *decisions* is not a dict.
    """
    if not isinstance(decisions, dict):
        raise TypeError(
            f"write_matching_results(): decisions must be a dict, "
            f"got {type(decisions).__name__}."
        )

    # Merge all_s1_ids into decisions to guarantee coverage
    final_decisions: dict[str, list[str]] = {}
    if all_s1_ids is not None:
        for s1_id in all_s1_ids:
            final_decisions[str(s1_id)] = []
    for s1_id, matches in decisions.items():
        final_decisions[str(s1_id)] = list(matches)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(
            fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in sorted(final_decisions.keys()):
            matches = final_decisions[s1_id]
            writer.writerow([s1_id, ",".join(matches)])

    # Assert: number of rows == number of S1 entities
    written = len(final_decisions)
    logger.info(
        "[entity_decision] Wrote matching_results.tsv: %d rows → %s",
        written, output_path,
    )


def decisions_to_dataframe(
    decisions: "dict[str, list[str]]",
) -> pd.DataFrame:
    """Convert a decisions dict to a two-column DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: source1_entity_id (str), matched_entity_ids (str, comma-sep).
        One row per S1 entity, sorted by source1_entity_id.
    """
    rows = [
        {
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),
        }
        for s1_id, matches in sorted(decisions.items())
    ]
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
