import pandas as pd
import numpy as np


def _to_num(series_or_scalar, default=0.0):
    """Coerce a Series (or scalar fallback) to float64, filling NaN with default.

    Handles pandas 3.x ArrowDtype / StringDtype columns that come back from
    TSV checkpoints loaded with ``dtype=str``.  On pandas <3.0 this is a
    no-op because ``pd.to_numeric`` already works on object-dtype columns.
    """
    if isinstance(series_or_scalar, pd.Series):
        return pd.to_numeric(series_or_scalar, errors="coerce").fillna(default)
    # scalar fallback (e.g. df.get('col', 0) returned 0 because col is absent)
    try:
        return float(series_or_scalar)
    except (TypeError, ValueError):
        return float(default)


def build_controlled_training_pairs(
    candidate_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    features_df: pd.DataFrame,
    neg_ratio: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Builds a training dataset with controlled negative sampling.
    Categories of negatives:
    1. Lexical near-duplicate but wrong entity
    2. Address collision but wrong entity
    3. Retrieval hard negative (high retrieval rank/score but wrong)
    4. Random negative

    Pandas 3.x compatibility
    ------------------------
    All numeric comparisons on columns read from TSV checkpoints are guarded
    with ``pd.to_numeric(..., errors='coerce')`` because pandas 3.x uses a
    PyArrow-backed StringDtype for ``dtype=str`` columns that refuses direct
    numeric comparisons.
    """
    np.random.seed(random_state)

    # 1. Label all candidates
    # Explode candidate pairs
    exploded = candidate_pairs_df.assign(
        candidate_entity_id=candidate_pairs_df["candidate_entity_ids"].str.split(",")
    ).explode("candidate_entity_id")

    exploded = exploded[exploded["candidate_entity_id"].str.strip() != ""]

    # Merge GT to find positives
    gt = ground_truth_df[["source1_entity_id", "matching_entity_ids"]].copy()
    gt["matching_entity_ids"] = gt["matching_entity_ids"].astype(str).str.split(",")
    gt_exploded = gt.explode("matching_entity_ids")
    gt_exploded.rename(columns={"matching_entity_ids": "candidate_entity_id"}, inplace=True)
    gt_exploded["is_match"] = 1

    labeled = pd.merge(
        exploded, gt_exploded,
        on=["source1_entity_id", "candidate_entity_id"],
        how="left",
    )
    # is_match may be NaN for non-matches; fillna then cast to int.
    # pandas 3.x: is_match column may be float64 after the left-merge, so
    # fillna(0) is safe here (NaN in a float column, not a str column).
    labeled["is_match"] = labeled["is_match"].fillna(0).astype(int)

    positives = labeled[labeled["is_match"] == 1].copy()
    negatives = labeled[labeled["is_match"] == 0].copy()

    if len(positives) == 0:
        return labeled  # Fallback if no positives

    n_positives = len(positives)
    target_negatives = n_positives * neg_ratio

    # We assume features_df has already been computed for all candidates to do HNM
    if not features_df.empty:
        neg_with_feats = pd.merge(
            negatives, features_df,
            on=["source1_entity_id", "candidate_entity_id"],
            how="inner",
        )

        # ── Coerce all numeric-intent columns to float64 ──────────────────
        # Columns like 'score', 'rank', similarity scores may arrive as
        # StringDtype on pandas 3.x (loaded from TSV with dtype=str).
        # We coerce them here once so all downstream comparisons are safe.
        name_jw  = _to_num(neg_with_feats.get("name_jaro_winkler",  pd.Series(dtype=float)))
        name_jac = _to_num(neg_with_feats.get("name_jaccard",       pd.Series(dtype=float)))
        addr_jw  = _to_num(neg_with_feats.get("addr_jaro_winkler",  pd.Series(dtype=float)))
        addr_jac = _to_num(neg_with_feats.get("addr_jaccard",       pd.Series(dtype=float)))
        rank_num = _to_num(neg_with_feats.get("rank",               pd.Series(dtype=float)), default=999.0)
        score_num= _to_num(neg_with_feats.get("score",              pd.Series(dtype=float)), default=0.0)

        # Ensure they are proper Series aligned to neg_with_feats index
        def _align(val, ref_df):
            if isinstance(val, pd.Series) and len(val) == len(ref_df):
                val = val.values  # strip index to avoid alignment issues
            if isinstance(val, (float, int)):
                return pd.Series([val] * len(ref_df), index=ref_df.index)
            return pd.Series(val, index=ref_df.index)

        name_jw   = _align(name_jw,   neg_with_feats)
        name_jac  = _align(name_jac,  neg_with_feats)
        addr_jw   = _align(addr_jw,   neg_with_feats)
        addr_jac  = _align(addr_jac,  neg_with_feats)
        rank_num  = _align(rank_num,  neg_with_feats)
        score_num = _align(score_num, neg_with_feats)

        # 1. Lexical near-duplicate
        lexical_mask = (name_jw >= 0.78) | (name_jac >= 0.75)
        lexical_negs = neg_with_feats[lexical_mask]

        # 2. Address collision
        addr_mask = (addr_jw >= 0.72) | (addr_jac >= 0.70)
        addr_negs = neg_with_feats[addr_mask & ~lexical_mask]

        # 3. Retrieval hard negative (high score, low rank)
        retrieval_mask = (rank_num <= 3) | (score_num >= 0.35)
        retrieval_negs = neg_with_feats[retrieval_mask & ~lexical_mask & ~addr_mask]

        # 4. Random negative
        random_negs = neg_with_feats[~lexical_mask & ~addr_mask & ~retrieval_mask]

        # Sample proportionally
        q_lexical   = int(target_negatives * 0.4)
        q_addr      = int(target_negatives * 0.3)
        q_retrieval = int(target_negatives * 0.2)
        q_random    = target_negatives - q_lexical - q_addr - q_retrieval

        def safe_sample(df, n):
            if len(df) <= n:
                return df
            return df.sample(n=n, random_state=random_state)

        sampled_negs = pd.concat([
            safe_sample(lexical_negs,   q_lexical),
            safe_sample(addr_negs,      q_addr),
            safe_sample(retrieval_negs, q_retrieval),
            safe_sample(random_negs,    target_negatives),  # truncated below
        ])

        # Ensure we don't exceed target
        if len(sampled_negs) > target_negatives:
            sampled_negs = sampled_negs.sample(n=target_negatives, random_state=random_state)

        final_df = pd.concat([positives, sampled_negs])
    else:
        # Fallback if features not passed
        if len(negatives) > target_negatives:
            negatives = negatives.sample(n=target_negatives, random_state=random_state)
        final_df = pd.concat([positives, negatives])

    return final_df
