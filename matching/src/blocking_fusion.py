import pandas as pd

# Columns that contribute to retrieval_agreement_count.
# IMPORTANT: retrieved_by_faiss is intentionally excluded here.
# features.build_feature_matrix() validates retrieval_agreement_count
# strictly as {1, 2}.  FAISS is a third retrieval source but its provenance
# flag is informational only and must not inflate the agreement count.
_AGREEMENT_COLS = {"retrieved_by_char_tfidf", "retrieved_by_bm25"}

def fuse_candidates(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Fuses multiple candidate DataFrames.
    Each dataframe must have at least ['source1_entity_id', 'candidate_entity_id'].
    We outer merge them all to preserve all provenance flags and scores.
    """
    if not dfs:
        return pd.DataFrame()
        
    merged_df = dfs[0]
    for df in dfs[1:]:
        if df.empty:
            continue
        if merged_df.empty:
            merged_df = df
            continue
            
        merged_df = pd.merge(
            merged_df, 
            df, 
            on=['source1_entity_id', 'candidate_entity_id'], 
            how='outer',
            suffixes=('', '_dup')
        )
        
        # Combine overlapping columns if any (e.g., if multiple dataframes have 'score')
        for col in list(merged_df.columns):
            if col.endswith('_dup'):
                orig_col = col[:-4]
                if orig_col.startswith('retrieved_by_'):
                    # For retrieval flags: logical OR (take max of 0/1 values)
                    merged_df[orig_col] = (
                        pd.to_numeric(merged_df[orig_col], errors="coerce").fillna(0)
                        .combine(
                            pd.to_numeric(merged_df[col], errors="coerce").fillna(0),
                            max
                        )
                    )
                else:
                    # For scores / other columns: keep original, fill NaN from dup
                    merged_df[orig_col] = merged_df[orig_col].fillna(merged_df[col])
                merged_df = merged_df.drop(columns=[col])
                
    # Fill NaN for retrieval flags with 0.
    # pandas 3.x uses a PyArrow-backed StringDtype for columns originating from
    # TSV reads with dtype=str.  We must coerce to numeric before fillna/astype
    # to avoid "Invalid value '0' for dtype 'str'" TypeError.
    for col in merged_df.columns:
        if col.startswith('retrieved_by_'):
            merged_df[col] = (
                pd.to_numeric(merged_df[col], errors="coerce")
                .fillna(0)
                .astype(int)
            )

    # Calculate retrieval agreement — counts only the two canonical sources
    # (TF-IDF and BM25) so the value stays in {1, 2} as required by
    # features.build_feature_matrix().  retrieved_by_faiss is carried through
    # as an informational provenance column but does NOT affect the count.
    agreement_cols = [
        c for c in merged_df.columns
        if c in _AGREEMENT_COLS
    ]
    if agreement_cols:
        merged_df['retrieval_agreement_count'] = merged_df[agreement_cols].sum(axis=1)
    else:
        # Fallback: if neither canonical column is present yet (e.g. called
        # before the alias step in run_blocking), default to 1.
        merged_df['retrieval_agreement_count'] = 1

    return merged_df
