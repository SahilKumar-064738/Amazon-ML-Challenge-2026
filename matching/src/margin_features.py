"""
margin_features.py
------------------
Phase 2 — Runner-Up, Competitive Margin, and Ambiguity Features.

Merges runner-up margin and candidate competition features from
Sid-techweb/AmazonML-New into the existing Business Entity Resolution pipeline.

SOURCE (Sid's repo):
    business_entity_resolution/src/features.py:
        - add_query_context() (lines 42–57):
            n_cands       = count of candidates for entity
            comb_max      = max score among candidates
            comb_gap      = gap to best other candidate
            comb_gap_next = gap to immediately next-ranked candidate
            name_gap      = cos_name - max(cos_name)
            addr_gap      = cos_addr - max(cos_addr)
        - add_margins() (lines 163–173):
            m_cos_name, m_cos_addr, m_nm_tset, m_full_ratio, m_ad_tset, m_num_q_cov
            n_name_ties   = count of candidates with score >= 0.95 * max_score
    experiments/s4_crossfit.py & stack_v2.py:
        - margin12      = top1_score - top2_score

Features computed (16 features):
    --- Runner-Up & Score Margin Features ---
    1.  candidate_rank               int    1-based rank within this S1's candidates (by score descending)
    2.  top_candidate_score          float  Highest candidate score for this S1 entity (Sid: comb_max)
    3.  second_candidate_score       float  Second-highest score for this S1 (0.0 if only 1 candidate)
    4.  top1_top2_margin             float  top1_score - top2_score (Sid: margin12)
    5.  score_margin_vs_other        float  score - best_other_score (Sid: comb_gap)
                                            (positive for rank 1, negative for runners-up)
    6.  score_gap_to_next            float  score - next_score (Sid: comb_gap_next)
    7.  score_gap_to_top             float  score - top_candidate_score (<= 0.0; 0.0 for rank 1)
    8.  relative_margin              float  (top1 - top2) / (top1 + 1e-6), bounded in [0, 1]
    9.  name_sim_margin              float  name_jaro_winkler - max(other name_jaro_winkler)
    10. addr_sim_margin              float  address_jaccard - max(other address_jaccard)

    --- Ambiguity & Competition Features ---
    11. candidate_count              int    Total candidate count for this S1 entity (Sid: n_cands)
    12. n_near_ties                  int    Count of candidates with score >= 0.95 * top_score (Sid: n_name_ties)
    13. n_strong_candidates          int    Count of candidates with score >= 0.70 * top_score and score >= 0.50
    14. candidates_within_01_of_top  int    Count of candidates with score >= top_score - 0.10
    15. competition_density          float  n_near_ties / candidate_count in [0, 1]
    16. is_ambiguous_candidate_set   int    {0, 1}: 1 if n_near_ties >= 2 and top1_top2_margin < 0.05

All features are:
    - Finite numeric floats or integers
    - Strictly NaN-free (all edge cases handled deterministically)
    - Strictly Inf-free
    - Deterministic
    - Grouped strictly per S1 entity (source1_entity_id) without data leakage

Public API:
    MARGIN_FEATURE_COLS : list[str]
    add_margin_features(pair_df, score_col="blocking_cosine_sim") -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature Column Names (16 features)
# ---------------------------------------------------------------------------

MARGIN_FEATURE_COLS: list[str] = [
    # Runner-up & score margins
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
    # Ambiguity & competition
    "candidate_count",
    "n_near_ties",
    "n_strong_candidates",
    "candidates_within_01_of_top",
    "competition_density",
    "is_ambiguous_candidate_set",
]


# ---------------------------------------------------------------------------
# Core computation logic
# ---------------------------------------------------------------------------

def add_margin_features(
    pair_df: pd.DataFrame,
    score_col: str | None = None,
) -> pd.DataFrame:
    """Compute 16 runner-up, margin, and ambiguity features for each candidate pair.

    Calculates competitive signals within each S1 entity's candidate pool.
    Preserves all existing columns and row ordering of `pair_df`.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table. Must contain at minimum:
        ``source1_entity_id``, ``candidate_entity_id``.
        Optionally uses:
        - `score_col` (e.g. ``blocking_cosine_sim`` or ``cosine_similarity``)
        - ``name_jaro_winkler`` (for name_sim_margin)
        - ``address_jaccard`` (for addr_sim_margin)
    score_col : str or None, default None
        Name of the similarity / retrieval score column. If None, resolves
        automatically by checking ['blocking_cosine_sim', 'cosine_similarity',
        'score', 'blocking_cosine_similarity']. If none is present, defaults
        to 1.0 / rank proxy.

    Returns
    -------
    pd.DataFrame
        Copy of `pair_df` with the 16 margin and ambiguity columns appended.
    """
    if not isinstance(pair_df, pd.DataFrame):
        raise TypeError(f"add_margin_features: expected pd.DataFrame, got {type(pair_df).__name__}")

    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(
            f"add_margin_features: pair_df missing required column(s): {sorted(missing)}. "
            f"Found: {list(pair_df.columns)}"
        )

    out_df = pair_df.copy()

    # Empty table edge case
    if out_df.empty:
        int_cols = {
            "candidate_rank", "candidate_count", "n_near_ties",
            "n_strong_candidates", "candidates_within_01_of_top",
            "is_ambiguous_candidate_set",
        }
        for col in MARGIN_FEATURE_COLS:
            dtype = np.int32 if col in int_cols else np.float64
            out_df[col] = pd.Series(dtype=dtype)
        return out_df

    # Resolve score column
    resolved_score_col = None
    if score_col is not None and score_col in out_df.columns:
        resolved_score_col = score_col
    else:
        for candidate in ["blocking_cosine_sim", "cosine_similarity", "score", "blocking_cosine_similarity"]:
            if candidate in out_df.columns:
                resolved_score_col = candidate
                break

    # If no score column is found, synthesize a valid finite fallback in [0, 1]
    if resolved_score_col is None:
        scores = np.ones(len(out_df), dtype=np.float64)
    else:
        scores = pd.to_numeric(out_df[resolved_score_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    # Name and address similarity arrays for specific margins (if present, else zeros)
    has_name_sim = "name_jaro_winkler" in out_df.columns
    name_sims = (
        pd.to_numeric(out_df["name_jaro_winkler"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if has_name_sim
        else np.zeros(len(out_df), dtype=np.float64)
    )

    has_addr_sim = "address_jaccard" in out_df.columns
    addr_sims = (
        pd.to_numeric(out_df["address_jaccard"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if has_addr_sim
        else np.zeros(len(out_df), dtype=np.float64)
    )

    # Pre-allocate output feature arrays
    n_rows = len(out_df)
    candidate_rank = np.ones(n_rows, dtype=np.int32)
    top_candidate_score = np.zeros(n_rows, dtype=np.float64)
    second_candidate_score = np.zeros(n_rows, dtype=np.float64)
    top1_top2_margin = np.zeros(n_rows, dtype=np.float64)
    score_margin_vs_other = np.zeros(n_rows, dtype=np.float64)
    score_gap_to_next = np.zeros(n_rows, dtype=np.float64)
    score_gap_to_top = np.zeros(n_rows, dtype=np.float64)
    relative_margin = np.zeros(n_rows, dtype=np.float64)
    name_sim_margin = np.zeros(n_rows, dtype=np.float64)
    addr_sim_margin = np.zeros(n_rows, dtype=np.float64)
    candidate_count = np.ones(n_rows, dtype=np.int32)
    n_near_ties = np.ones(n_rows, dtype=np.int32)
    n_strong_candidates = np.zeros(n_rows, dtype=np.int32)
    candidates_within_01_of_top = np.ones(n_rows, dtype=np.int32)
    competition_density = np.zeros(n_rows, dtype=np.float64)
    is_ambiguous_candidate_set = np.zeros(n_rows, dtype=np.int32)

    # Group row indices by source1_entity_id, preserving original appearance order
    s1_ids = out_df["source1_entity_id"].astype(str).values
    grouped_indices: dict[str, list[int]] = {}
    for idx, s1_id in enumerate(s1_ids):
        if s1_id not in grouped_indices:
            grouped_indices[s1_id] = []
        grouped_indices[s1_id].append(idx)

    # Compute group-level competitive & margin statistics
    for s1_id, idxs in grouped_indices.items():
        k = len(idxs)
        grp_scores = scores[idxs]
        grp_name_sims = name_sims[idxs]
        grp_addr_sims = addr_sims[idxs]

        # Sort indices within group by score descending (stable sort to keep original rank on ties)
        sorted_pos = np.argsort(-grp_scores, kind="stable")
        sorted_scores = grp_scores[sorted_pos]

        top1 = float(sorted_scores[0])
        top2 = float(sorted_scores[1]) if k >= 2 else 0.0
        margin12 = float(top1 - top2) if k >= 2 else float(top1)
        if k == 1:
            rel_margin = 1.0
        else:
            rel_margin = float(margin12 / (top1 + 1e-6)) if top1 > 0 else 0.0
        rel_margin = min(1.0, max(0.0, rel_margin))

        # Ambiguity metrics
        near_ties_count = int(np.sum(grp_scores >= (0.95 * top1)))
        strong_count = int(np.sum((grp_scores >= (0.70 * top1)) & (grp_scores >= 0.50)))
        within_01_count = int(np.sum(grp_scores >= (top1 - 0.10)))
        density = float(near_ties_count / k) if k > 0 else 0.0
        is_ambiguous = int(near_ties_count >= 2 and margin12 < 0.05)

        # Name and Address top similarities
        sorted_name_sims = np.sort(grp_name_sims)[::-1]
        top1_name = float(sorted_name_sims[0])
        top2_name = float(sorted_name_sims[1]) if k >= 2 else 0.0

        sorted_addr_sims = np.sort(grp_addr_sims)[::-1]
        top1_addr = float(sorted_addr_sims[0])
        top2_addr = float(sorted_addr_sims[1]) if k >= 2 else 0.0

        # Assign per-candidate features
        # Map rank: rank 1 is highest score
        for rank_1b, orig_local_pos in enumerate(sorted_pos, start=1):
            row_idx = idxs[orig_local_pos]
            sc = float(grp_scores[orig_local_pos])

            candidate_rank[row_idx] = rank_1b
            top_candidate_score[row_idx] = top1
            second_candidate_score[row_idx] = top2
            top1_top2_margin[row_idx] = margin12
            candidate_count[row_idx] = k
            n_near_ties[row_idx] = near_ties_count
            n_strong_candidates[row_idx] = strong_count
            candidates_within_01_of_top[row_idx] = within_01_count
            competition_density[row_idx] = density
            is_ambiguous_candidate_set[row_idx] = is_ambiguous
            relative_margin[row_idx] = rel_margin

            # Score gap to top candidate (<= 0.0)
            score_gap_to_top[row_idx] = float(sc - top1)

            # Score margin vs best other candidate (Sid: comb_gap)
            if k == 1:
                score_margin_vs_other[row_idx] = sc
            elif rank_1b == 1:
                score_margin_vs_other[row_idx] = float(sc - top2)
            else:
                score_margin_vs_other[row_idx] = float(sc - top1)

            # Score gap to next candidate (Sid: comb_gap_next)
            if rank_1b < k:
                next_sc = float(sorted_scores[rank_1b])
                score_gap_to_next[row_idx] = float(sc - next_sc)
            else:
                score_gap_to_next[row_idx] = sc if k == 1 else 0.0

            # Name similarity margin
            n_sc = float(grp_name_sims[orig_local_pos])
            if k == 1:
                name_sim_margin[row_idx] = n_sc
            elif np.isclose(n_sc, top1_name):
                name_sim_margin[row_idx] = float(n_sc - top2_name)
            else:
                name_sim_margin[row_idx] = float(n_sc - top1_name)

            # Address similarity margin
            a_sc = float(grp_addr_sims[orig_local_pos])
            if k == 1:
                addr_sim_margin[row_idx] = a_sc
            elif np.isclose(a_sc, top1_addr):
                addr_sim_margin[row_idx] = float(a_sc - top2_addr)
            else:
                addr_sim_margin[row_idx] = float(a_sc - top1_addr)

    # Attach computed feature columns
    out_df["candidate_rank"] = candidate_rank
    out_df["top_candidate_score"] = top_candidate_score
    out_df["second_candidate_score"] = second_candidate_score
    out_df["top1_top2_margin"] = top1_top2_margin
    out_df["score_margin_vs_other"] = score_margin_vs_other
    out_df["score_gap_to_next"] = score_gap_to_next
    out_df["score_gap_to_top"] = score_gap_to_top
    out_df["relative_margin"] = relative_margin
    out_df["name_sim_margin"] = name_sim_margin
    out_df["addr_sim_margin"] = addr_sim_margin
    out_df["candidate_count"] = candidate_count
    out_df["n_near_ties"] = n_near_ties
    out_df["n_strong_candidates"] = n_strong_candidates
    out_df["candidates_within_01_of_top"] = candidates_within_01_of_top
    out_df["competition_density"] = competition_density
    out_df["is_ambiguous_candidate_set"] = is_ambiguous_candidate_set

    return out_df
