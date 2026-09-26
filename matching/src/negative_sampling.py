import pandas as pd
import numpy as np

def build_controlled_training_pairs(
    candidate_pairs_df: pd.DataFrame, 
    ground_truth_df: pd.DataFrame, 
    features_df: pd.DataFrame, 
    neg_ratio: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Builds a training dataset with controlled negative sampling.
    Categories of negatives:
    1. Lexical near-duplicate but wrong entity
    2. Address collision but wrong entity
    3. Retrieval hard negative (high retrieval rank/score but wrong)
    4. Random negative
    """
    np.random.seed(random_state)
    
    # 1. Label all candidates
    # Explode candidate pairs
    exploded = candidate_pairs_df.assign(
        candidate_entity_id=candidate_pairs_df['candidate_entity_ids'].str.split(',')
    ).explode('candidate_entity_id')
    
    exploded = exploded[exploded['candidate_entity_id'].str.strip() != '']
    
    # Merge GT to find positives
    gt = ground_truth_df[['source1_entity_id', 'matching_entity_ids']].copy()
    gt['matching_entity_ids'] = gt['matching_entity_ids'].astype(str).str.split(',')
    gt_exploded = gt.explode('matching_entity_ids')
    gt_exploded.rename(columns={'matching_entity_ids': 'candidate_entity_id'}, inplace=True)
    gt_exploded['is_match'] = 1
    
    labeled = pd.merge(exploded, gt_exploded, on=['source1_entity_id', 'candidate_entity_id'], how='left')
    labeled['is_match'] = labeled['is_match'].fillna(0).astype(int)
    
    positives = labeled[labeled['is_match'] == 1].copy()
    negatives = labeled[labeled['is_match'] == 0].copy()
    
    if len(positives) == 0:
        return labeled # Fallback if no positives
        
    n_positives = len(positives)
    target_negatives = n_positives * neg_ratio
    
    # We assume features_df has already been computed for all candidates to do HNM
    if not features_df.empty:
        neg_with_feats = pd.merge(negatives, features_df, on=['source1_entity_id', 'candidate_entity_id'], how='inner')
        
        # 1. Lexical near-duplicate
        lexical_mask = (neg_with_feats.get('name_jaro_winkler', 0) >= 0.78) | (neg_with_feats.get('name_jaccard', 0) >= 0.75)
        lexical_negs = neg_with_feats[lexical_mask]
        
        # 2. Address collision
        addr_mask = (neg_with_feats.get('addr_jaro_winkler', 0) >= 0.72) | (neg_with_feats.get('addr_jaccard', 0) >= 0.70)
        addr_negs = neg_with_feats[addr_mask & ~lexical_mask]
        
        # 3. Retrieval hard negative (high score, low rank)
        retrieval_mask = (neg_with_feats.get('rank', 999) <= 3) | (neg_with_feats.get('score', 0) >= 0.35)
        retrieval_negs = neg_with_feats[retrieval_mask & ~lexical_mask & ~addr_mask]
        
        # 4. Random negative
        random_negs = neg_with_feats[~lexical_mask & ~addr_mask & ~retrieval_mask]
        
        # Sample proportionally
        q_lexical = int(target_negatives * 0.4)
        q_addr = int(target_negatives * 0.3)
        q_retrieval = int(target_negatives * 0.2)
        q_random = target_negatives - q_lexical - q_addr - q_retrieval
        
        def safe_sample(df, n):
            if len(df) <= n: return df
            return df.sample(n=n, random_state=random_state)
            
        sampled_negs = pd.concat([
            safe_sample(lexical_negs, q_lexical),
            safe_sample(addr_negs, q_addr),
            safe_sample(retrieval_negs, q_retrieval),
            safe_sample(random_negs, target_negatives) # Will be truncated to whatever is needed
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
