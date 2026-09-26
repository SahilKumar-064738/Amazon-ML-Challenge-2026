"""
rrf_features.py
---------------
Rank-aware retrieval fusion features for the Business Entity Resolution pipeline.

Implements Reciprocal Rank Fusion (RRF) and related features that capture
HOW CONFIDENTLY each retrieval channel retrieved a candidate, not just
WHETHER it was retrieved.

Background
~~~~~~~~~~
The existing union retrieval gives binary flags (retrieved_by_char_tfidf,
retrieved_by_bm25).  RRF goes further: it produces a continuous score from
the per-channel RANKS, which is a stronger signal for LightGBM than binary
flags alone.

RRF formula:
    RRF(d) = Σ_channel  1 / (k + rank_channel(d))
where k is a smoothing constant (default 60, from the original RRF paper).

Features produced
~~~~~~~~~~~~~~~~~
Per candidate pair row:
    1.  tfidf_rank_norm       -- Char-TF-IDF rank normalised to [0,1]
                                 (1/rank, so rank-1 = 1.0, high rank → 0.0)
    2.  bm25_rank_norm        -- BM25 rank normalised to [0,1]
    3.  name_tfidf_rank_norm  -- Name-only TF-IDF rank normalised (0 if absent)
    4.  addr_tfidf_rank_norm  -- Address-only TF-IDF rank normalised (0 if absent)
    5.  rare_token_rank_norm  -- Rare-token blocking rank normalised (0 if absent)
    6.  snm_rank_norm         -- Sorted-neighbourhood rank normalised (0 if absent)
    7.  rrf_score             -- RRF aggregate score over all channels
                                 Normalised to [0,1] by max possible RRF.
    8.  n_channels_agreeing   -- Count of channels that retrieved this pair {0..6}
                                 Normalised to [0,1] by 6.

All values:
    - Finite floats in [0.0, 1.0]
    - Never NaN (absent channel contribution = 0.0)
    - Deterministic

Public API
~~~~~~~~~~
    add_rrf_features(pair_df, union_df=None) -> pd.DataFrame
        Computes rank-aware features from whatever rank columns are present
        in *pair_df* or supplied via *union_df*.

        Operates in two modes:
        1. If *pair_df* already contains rank columns (from union_df join),
           compute directly.
        2. If *union_df* is provided, join the rank columns first, then compute.

    compute_rrf_score(ranks_by_channel, k=60) -> float
        Low-level RRF score for a dict of {channel_name: rank}.
        Rank=None or missing means the channel did not retrieve this pair.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: RRF smoothing constant (from Cormack et al. 2009).
RRF_K: int = 60

#: Columns produced by this module (8 features).
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

# Mapping from rank column names → feature column names
# (channel_rank_col, feature_col, default_if_absent)
_RANK_CHANNEL_MAP: list[tuple[str, str]] = [
    ("rank",             "tfidf_rank_norm"),
    ("bm25_rank",        "bm25_rank_norm"),
    ("name_tfidf_rank",  "name_tfidf_rank_norm"),
    ("addr_tfidf_rank",  "addr_tfidf_rank_norm"),
    ("rare_token_rank",  "rare_token_rank_norm"),
    ("snm_rank",         "snm_rank_norm"),
]

# Maximum number of channels we consider (used for n_channels normalisation)
_N_CHANNELS_MAX: int = len(_RANK_CHANNEL_MAP)


# ---------------------------------------------------------------------------
# Low-level RRF score
# ---------------------------------------------------------------------------

def compute_rrf_score(
    ranks_by_channel: dict[str, Optional[float]],
    k: int = RRF_K,
) -> float:
    """Compute a raw (unnormalised) RRF score for one candidate pair.

    Parameters
    ----------
    ranks_by_channel : dict[str, Optional[float]]
        Mapping of channel name to rank (1-based, positive integer).
        Pass ``None`` or omit a channel to indicate it did not retrieve this pair.
        Non-finite or ≤ 0 ranks are treated as missing (no contribution).
    k : int
        RRF smoothing constant.  Default: 60.

    Returns
    -------
    float
        RRF score ≥ 0.  Sum of 1/(k + rank) over all channels that
        retrieved this pair.  0.0 if no channel retrieved it.
    """
    score = 0.0
    for rank in ranks_by_channel.values():
        if rank is None:
            continue
        try:
            r = float(rank)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(r) or r <= 0:
            continue
        score += 1.0 / (k + r)
    return score


def _max_rrf_score(k: int = RRF_K, n_channels: int = _N_CHANNELS_MAX) -> float:
    """Maximum possible RRF score when retrieved at rank 1 by all channels."""
    return n_channels * (1.0 / (k + 1.0))


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _rank_to_norm(rank_val: float) -> float:
    """Convert a 1-based rank to a normalised score in (0, 1].

    rank 1 → 1.0; rank 2 → 0.5; rank k → 1/k.
    Missing / invalid → 0.0.
    """
    if rank_val is None or (isinstance(rank_val, float) and not np.isfinite(rank_val)):
        return 0.0
    try:
        r = float(rank_val)
    except (ValueError, TypeError):
        return 0.0
    if r <= 0:
        return 0.0
    return min(1.0, 1.0 / r)


# ---------------------------------------------------------------------------
# Public API — DataFrame-level feature adder
# ---------------------------------------------------------------------------

def add_rrf_features(
    pair_df: pd.DataFrame,
    union_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute 8 rank-aware RRF features for every candidate pair.

    The function reads rank columns from *pair_df* (if present) or from
    *union_df* (joined by source1_entity_id + candidate_entity_id).
    Channels not present receive a rank contribution of 0.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table.  Must contain at least:
        ``source1_entity_id``, ``candidate_entity_id``.
        May optionally already contain any of: rank, bm25_rank,
        name_tfidf_rank, addr_tfidf_rank, rare_token_rank, snm_rank.
    union_df : pd.DataFrame or None
        Extended union result (from union_candidate_results or
        union_with_extra_channel).  If provided, rank columns are joined
        from here onto pair_df.  If None, only columns already present
        in pair_df are used.

    Returns
    -------
    pd.DataFrame
        Copy of *pair_df* with 8 new columns appended:
        :data:`RRF_FEATURE_COLS`.

    Raises
    ------
    ValueError
        If required identifier columns are missing.
    """
    required = {"source1_entity_id", "candidate_entity_id"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(
            f"add_rrf_features(): pair_df is missing column(s): "
            f"{sorted(missing)}.  Found: {list(pair_df.columns)}"
        )

    out_df = pair_df.copy()

    # Optionally join rank columns from union_df
    if union_df is not None:
        if not isinstance(union_df, pd.DataFrame):
            raise TypeError("union_df must be a pd.DataFrame or None.")

        rank_cols_to_join = [
            col for col, _ in _RANK_CHANNEL_MAP
            if col in union_df.columns and col not in out_df.columns
        ]
        if rank_cols_to_join:
            _PAIR_KEY = ["source1_entity_id", "candidate_entity_id"]
            join_cols = _PAIR_KEY + rank_cols_to_join
            out_df = out_df.merge(
                union_df[join_cols].drop_duplicates(subset=_PAIR_KEY),
                on=_PAIR_KEY,
                how="left",
            )

    # Initialise feature columns for empty table
    if out_df.empty:
        for col in RRF_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=float)
        return out_df

    # Compute per-row features
    max_rrf = _max_rrf_score()

    tfidf_rank_norms = []
    bm25_rank_norms = []
    name_rank_norms = []
    addr_rank_norms = []
    rare_rank_norms = []
    snm_rank_norms = []
    rrf_scores = []
    n_channels_list = []

    for _, row in out_df.iterrows():
        # Extract each rank (None if column absent or NaN)
        def _get_rank(col: str) -> Optional[float]:
            if col not in out_df.columns:
                return None
            val = row[col]
            if val is None or (isinstance(val, float) and not np.isfinite(val)):
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        r_tfidf = _get_rank("rank")
        r_bm25  = _get_rank("bm25_rank")
        r_name  = _get_rank("name_tfidf_rank")
        r_addr  = _get_rank("addr_tfidf_rank")
        r_rare  = _get_rank("rare_token_rank")
        r_snm   = _get_rank("snm_rank")

        # Normalised rank features
        tfidf_rank_norms.append(_rank_to_norm(r_tfidf))
        bm25_rank_norms.append(_rank_to_norm(r_bm25))
        name_rank_norms.append(_rank_to_norm(r_name))
        addr_rank_norms.append(_rank_to_norm(r_addr))
        rare_rank_norms.append(_rank_to_norm(r_rare))
        snm_rank_norms.append(_rank_to_norm(r_snm))

        # RRF score
        ranks = {
            "tfidf": r_tfidf,
            "bm25":  r_bm25,
            "name":  r_name,
            "addr":  r_addr,
            "rare":  r_rare,
            "snm":   r_snm,
        }
        raw_rrf = compute_rrf_score(ranks)
        normalised_rrf = raw_rrf / max_rrf if max_rrf > 0 else 0.0
        rrf_scores.append(min(1.0, max(0.0, normalised_rrf)))

        # Number of channels agreeing
        n_agree = sum(1 for r in ranks.values() if r is not None and r > 0)
        n_channels_list.append(n_agree / _N_CHANNELS_MAX)

    out_df["tfidf_rank_norm"]      = tfidf_rank_norms
    out_df["bm25_rank_norm"]       = bm25_rank_norms
    out_df["name_tfidf_rank_norm"] = name_rank_norms
    out_df["addr_tfidf_rank_norm"] = addr_rank_norms
    out_df["rare_token_rank_norm"] = rare_rank_norms
    out_df["snm_rank_norm"]        = snm_rank_norms
    out_df["rrf_score"]            = rrf_scores
    out_df["n_channels_agreeing"]  = n_channels_list

    return out_df
