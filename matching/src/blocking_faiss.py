import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def build_faiss_index(s2s3_df: pd.DataFrame, index_path: str, model_name: str = 'intfloat/multilingual-e5-small', batch_size: int = 1000):
    """Build a disk-backed FAISS index for the corpus."""
    model = SentenceTransformer(model_name)
    d = model.get_sentence_embedding_dimension()
    
    # Train index on a sample to avoid loading all into memory
    sample_size = min(100_000, len(s2s3_df))
    sample_texts = s2s3_df['clean_text'].fillna('').sample(n=sample_size, random_state=42).tolist()
    print("Training FAISS index...")
    sample_embeddings = model.encode(sample_texts, batch_size=batch_size, show_progress_bar=True)
    
    nlist = 1024
    quantizer = faiss.IndexFlatIP(d)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(sample_embeddings)
    index_ivf.train(sample_embeddings)
    
    # We use OnDiskInvertedLists to avoid keeping 10.3M vectors in RAM
    invlists = faiss.OnDiskInvertedLists(index_ivf.nlist, index_ivf.code_size, index_path + ".ivfdata")
    index_ivf.replace_invlists(invlists)
    
    # Add to index in chunks
    print("Adding vectors to FAISS index...")
    texts = s2s3_df['clean_text'].fillna('').tolist()
    ids = np.arange(len(texts))
    
    chunk_size = 500_000
    for start_idx in range(0, len(texts), chunk_size):
        end_idx = min(start_idx + chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]
        chunk_ids = ids[start_idx:end_idx]
        
        emb = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True)
        faiss.normalize_L2(emb)
        index_ivf.add_with_ids(emb, chunk_ids)
        
    # Write the index metadata (the inverted lists are already on disk)
    faiss.write_index(index_ivf, index_path)
    print("FAISS index built.")

def search_faiss_candidates(s1_df: pd.DataFrame, s2s3_df: pd.DataFrame, index_path: str, model_name: str = 'intfloat/multilingual-e5-small', top_k: int = 20, batch_size: int = 1000) -> pd.DataFrame:
    """Search candidates using the disk-backed FAISS index."""
    model = SentenceTransformer(model_name)
    
    index = faiss.read_index(index_path)
    index.nprobe = 16
    
    s1_texts = s1_df['clean_text'].fillna('').tolist()
    s1_ids = s1_df['entity_id'].tolist()
    s2s3_ids = s2s3_df['entity_id'].tolist()
    
    rows = []
    chunk_size = 50_000
    
    for start_idx in range(0, len(s1_texts), chunk_size):
        end_idx = min(start_idx + chunk_size, len(s1_texts))
        chunk_texts = s1_texts[start_idx:end_idx]
        
        emb = model.encode(chunk_texts, batch_size=batch_size, show_progress_bar=True)
        faiss.normalize_L2(emb)
        
        scores, I = index.search(emb, top_k)
        
        for i, (score_arr, idx_arr) in enumerate(zip(scores, I)):
            global_i = start_idx + i
            s1_id = s1_ids[global_i]
            
            for rank, (score, s2s3_idx) in enumerate(zip(score_arr, idx_arr), start=1):
                if s2s3_idx == -1:
                    continue
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": s2s3_ids[s2s3_idx],
                    "score": float(score),
                    "rank": rank,
                    "retrieved_by_embedding": 1
                })
                
    return pd.DataFrame(rows)
