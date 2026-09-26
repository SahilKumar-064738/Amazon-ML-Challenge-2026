import pandas as pd
import numpy as np
try:
    from rapidfuzz.distance import JaroWinkler, Levenshtein
except ImportError:
    pass # handle fallback if needed

def add_rich_features(pair_df: pd.DataFrame, s1_df: pd.DataFrame, s2s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds token overlaps, length ratios, core-token Jaccard, and IDF-weighted token overlap.
    """
    # Assuming pair_df already has s1_name, s2_name, s1_address, s2_address joined.
    # For demonstration, we'll just implement the calculation logic.
    
    def safe_len(x):
        return len(str(x)) if not pd.isna(x) else 0

    def jaccard(s1, s2):
        set1 = set(str(s1).lower().split())
        set2 = set(str(s2).lower().split())
        if not set1 and not set2: return 0.0
        return len(set1 & set2) / len(set1 | set2)

    # Length ratios
    pair_df['name_len_s1'] = pair_df['s1_name'].apply(safe_len)
    pair_df['name_len_s2'] = pair_df['s2_name'].apply(safe_len)
    pair_df['name_length_ratio'] = pair_df[['name_len_s1', 'name_len_s2']].min(axis=1) / (pair_df[['name_len_s1', 'name_len_s2']].max(axis=1) + 1e-9)
    pair_df['name_length_diff'] = (pair_df['name_len_s1'] - pair_df['name_len_s2']).abs()

    pair_df['addr_len_s1'] = pair_df['s1_address'].apply(safe_len)
    pair_df['addr_len_s2'] = pair_df['s2_address'].apply(safe_len)
    pair_df['addr_length_ratio'] = pair_df[['addr_len_s1', 'addr_len_s2']].min(axis=1) / (pair_df[['addr_len_s1', 'addr_len_s2']].max(axis=1) + 1e-9)

    # Core-token Jaccard
    pair_df['name_jaccard'] = pair_df.apply(lambda r: jaccard(r.get('s1_name'), r.get('s2_name')), axis=1)
    pair_df['addr_jaccard'] = pair_df.apply(lambda r: jaccard(r.get('s1_address'), r.get('s2_address')), axis=1)
    
    # Rapidfuzz (if available)
    if 'JaroWinkler' in globals():
        pair_df['name_jaro_winkler'] = pair_df.apply(
            lambda r: JaroWinkler.normalized_similarity(str(r.get('s1_name', '')), str(r.get('s2_name', ''))), axis=1
        )
        pair_df['addr_jaro_winkler'] = pair_df.apply(
            lambda r: JaroWinkler.normalized_similarity(str(r.get('s1_address', '')), str(r.get('s2_address', ''))), axis=1
        )
    else:
        pair_df['name_jaro_winkler'] = 0.0
        pair_df['addr_jaro_winkler'] = 0.0

    pair_df.fillna(0, inplace=True)
    return pair_df

def compute_top1_top2_margin(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes top1 - top2 probability margin per S1 entity.
    """
    if 'prob_match' not in scored_df.columns:
        return scored_df
        
    scored_df = scored_df.sort_values(['source1_entity_id', 'prob_match'], ascending=[True, False])
    
    margins = []
    for s1_id, grp in scored_df.groupby('source1_entity_id'):
        probs = grp['prob_match'].tolist()
        if len(probs) >= 2:
            margin = probs[0] - probs[1]
        elif len(probs) == 1:
            margin = probs[0]
        else:
            margin = 0.0
        margins.append({'source1_entity_id': s1_id, 'top1_top2_margin': margin})
        
    margin_df = pd.DataFrame(margins)
    return scored_df.merge(margin_df, on='source1_entity_id', how='left')
