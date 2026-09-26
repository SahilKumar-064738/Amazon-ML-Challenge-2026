import pandas as pd

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
        for col in merged_df.columns:
            if col.endswith('_dup'):
                orig_col = col[:-4]
                # Keep max of scores, or fillna for missing
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

    # Calculate retrieval agreement
    retrieval_cols = [c for c in merged_df.columns if c.startswith('retrieved_by_')]
    merged_df['retrieval_agreement_count'] = merged_df[retrieval_cols].sum(axis=1)
    
    return merged_df
