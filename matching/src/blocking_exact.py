import pandas as pd
import numpy as np

def build_exact_indexes(df: pd.DataFrame, cols: list[str]) -> dict:
    """Build inverted indexes for exact matches.
    cols: e.g. ['clean_name', 'clean_address', 'clean_text']
    Returns a dict mapping col_name -> {value: [entity_id1, entity_id2, ...]}
    """
    indexes = {col: {} for col in cols}
    for row in df.itertuples(index=False):
        ent_id = getattr(row, "entity_id")
        for col in cols:
            val = getattr(row, col, "")
            if pd.isna(val) or val == "":
                continue
            val = str(val).strip()
            if not val:
                continue
            
            if val not in indexes[col]:
                indexes[col][val] = []
            indexes[col][val].append(ent_id)
            
    return indexes

def search_exact_candidates(s1_df: pd.DataFrame, exact_indexes: dict, cols: list[str]) -> pd.DataFrame:
    """Search candidates using the exact match inverted indexes.
    Returns DataFrame with [source1_entity_id, candidate_entity_id, retrieved_by_exact_name, etc.]
    """
    rows = []
    
    # Precompute a fast lookup map for S1
    for row in s1_df.itertuples(index=False):
        s1_id = getattr(row, "entity_id")
        matches = {} # candidate_id -> list of cols where it matched
        
        for col in cols:
            val = getattr(row, col, "")
            if pd.isna(val) or val == "":
                continue
            val = str(val).strip()
            if not val:
                continue
                
            if val in exact_indexes[col]:
                for s2s3_id in exact_indexes[col][val]:
                    if s2s3_id not in matches:
                        matches[s2s3_id] = set()
                    matches[s2s3_id].add(col)
                    
        for s2s3_id, matched_cols in matches.items():
            record = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": s2s3_id,
                "score": 1.0,
                "rank": 1
            }
            # Add provenance flags
            for c in cols:
                record[f"retrieved_by_exact_{c}"] = 1 if c in matched_cols else 0
            rows.append(record)
            
    if not rows:
        columns = ["source1_entity_id", "candidate_entity_id", "score", "rank"] + [f"retrieved_by_exact_{c}" for c in cols]
        return pd.DataFrame(columns=columns)
        
    df = pd.DataFrame(rows)
    return df
