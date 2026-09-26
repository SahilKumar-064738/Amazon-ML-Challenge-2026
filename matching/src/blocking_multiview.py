import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import search_candidates_scalable

def build_view_texts(df: pd.DataFrame, view: str) -> pd.Series:
    if view == "name":
        col = "clean_name"
    elif view == "address":
        col = "clean_address"
    else:
        col = "clean_text"
        
    if col not in df.columns:
        # Fallback if column missing
        return pd.Series([""] * len(df))
        
    return df[col].fillna("").astype(str)

def fit_multiview_vectorizers(s2s3_df: pd.DataFrame) -> dict:
    views = {
        "name": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=150_000, sublinear_tf=True),
        "address": TfidfVectorizer(analyzer="word", ngram_range=(1, 1), max_features=100_000, sublinear_tf=True),
        "combined": TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=250_000, sublinear_tf=True)
    }
    
    fitted = {}
    for view, vec in views.items():
        texts = build_view_texts(s2s3_df, view)
        fitted[view] = vec.fit(texts)
    return fitted

def search_multiview_candidates(
    s1_df: pd.DataFrame, 
    s2s3_df: pd.DataFrame, 
    vectorizers: dict, 
    top_k: int = 30
) -> pd.DataFrame:
    
    results = []
    
    s1_ids = s1_df["entity_id"].tolist()
    s2s3_ids = s2s3_df["entity_id"].tolist()
    
    for view, vec in vectorizers.items():
        s1_texts = build_view_texts(s1_df, view)
        s2s3_texts = build_view_texts(s2s3_df, view)
        
        s1_mat = vec.transform(s1_texts)
        s2s3_mat = vec.transform(s2s3_texts)
        
        # OOM-proof scalable chunked search
        res = search_candidates_scalable(s1_mat, s2s3_mat, top_k=top_k)
        
        # Map indices to IDs
        res["source1_entity_id"] = res["s1_row_idx"].map(lambda i: s1_ids[i])
        res["candidate_entity_id"] = res["s2s3_row_idx"].map(lambda j: s2s3_ids[j])
        res["view"] = view
        
        results.append(res[["source1_entity_id", "candidate_entity_id", "score", "rank", "view"]])
        
    merged_df = pd.concat(results, ignore_index=True)
    
    # Pivot to create retrieval provenance flags
    merged_df["retrieved"] = 1
    pivoted = merged_df.pivot_table(
        index=["source1_entity_id", "candidate_entity_id"],
        columns="view",
        values="retrieved",
        fill_value=0
    ).reset_index()
    
    pivoted.columns.name = None
    pivoted.rename(columns={
        "name": "retrieved_by_tfidf_name",
        "address": "retrieved_by_tfidf_address",
        "combined": "retrieved_by_tfidf_combined"
    }, inplace=True)
    
    # We also want the max score across views as the primary score for ranking
    max_scores = merged_df.groupby(["source1_entity_id", "candidate_entity_id"])["score"].max().reset_index()
    
    final_df = pivoted.merge(max_scores, on=["source1_entity_id", "candidate_entity_id"])
    return final_df
