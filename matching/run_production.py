"""
run_production.py
-----------------
Full production pipeline with CLI args, per-step checkpointing, and a
RAM-aware BM25 guard.

Usage
-----
# Full run (train + test inference):
    python run_production.py

# Train only (stops after threshold sweep, before test inference):
    python run_production.py --split train

# Test inference only (requires completed train checkpoint):
    python run_production.py --split test

# Resume from last completed checkpoint:
    python run_production.py          # automatically skips done steps

# Force re-run everything from scratch:
    python run_production.py --force

# Disable BM25 (use TF-IDF only -- saves ~8-12 GB RAM):
    python run_production.py --skip-bm25

# Change top-k and checkpoint directory:
    python run_production.py --top-k 75 --checkpoint-dir output/ckpt

Checkpoints
-----------
Each major step writes a sentinel file to --checkpoint-dir (default:
output/checkpoints/).  On restart, completed steps are skipped.  To re-run
a specific step, delete its sentinel file:

    Step 1  ckpt/s1_vectorizer_fitted
    Step 2  ckpt/s2_train_blocking_done
    Step 3  ckpt/s3_train_features_done
    Step 4  ckpt/s4_model_trained
    Step 5  ckpt/s5_threshold_selected
    Step 6  ckpt/s6_test_blocking_done
    Step 7  ckpt/s7_test_scored
    Step 8  ckpt/s8_outputs_written

State saved between steps (in checkpoint-dir):
    vectorizer.pkl          fitted TF-IDF vectorizer
    train_results.tsv       blocking scores (train)
    train_candidates.tsv    candidate pairs (train)
    train_labeled.tsv       labeled feature table
    model.pkl               trained LightGBM model
    threshold.txt           selected threshold value
    feature_cols.txt        feature column names (one per line)
    test_results.tsv        blocking scores (test)
    test_candidates.tsv     candidate pairs (test)
    test_scored.tsv         scored test pairs

RAM notes
---------
On 16 GB with ~4 GB free, BM25 on a 10M-row corpus will OOM.
- Use --skip-bm25 to run TF-IDF-only blocking (fits in ~2-3 GB).
- TF-IDF scalable path never materialises the full similarity matrix.
- BM25 is attempted after TF-IDF; if it raises MemoryError it is skipped
  automatically and a warning is printed.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Argument parsing — done before any heavy imports so --help is instant
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Business Entity Resolution — production pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which split to run. 'train' stops after threshold sweep. "
             "'test' runs inference only (requires completed train state). "
             "'all' (default) runs end-to-end.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=50,
        metavar="K",
        help="Candidates per S1 entity from blocking (default: 50). "
             "Increase to 75-100 if blocking recall is low. "
             "Decrease for a tighter candidate set.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoints and re-run all steps from scratch.",
    )
    p.add_argument(
        "--skip-bm25",
        action="store_true",
        help="Disable BM25 blocking (TF-IDF only). Saves 8-12 GB RAM on "
             "large corpora. Recommended on machines with <= 16 GB total RAM.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=os.path.join("output", "checkpoints"),
        metavar="DIR",
        help="Directory for checkpoint sentinels and intermediate state "
             "(default: output/checkpoints/).",
    )
    p.add_argument(
        "--data-dir",
        default=os.path.join("dataset", "real"),
        metavar="DIR",
        help="Directory containing prepared data files "
             "(default: dataset/real/).",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for final submission files (default: output/).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for feature extraction and LightGBM (default: 8).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Query chunk size for TF-IDF 2D blocking (default: 20000).",
    )
    p.add_argument(
        "--corpus-block-size",
        type=int,
        default=250000,
        help="Corpus block size for TF-IDF 2D blocking (default: 250000).",
    )
    p.add_argument(
        "--faiss-batch-size",
        type=int,
        default=1000,
        help="Batch size for FAISS sentence-transformers encoding (default: 1000).",
    )
    p.add_argument(
        "--enable-faiss",
        action="store_true",
        help=(
            "Enable FAISS semantic retrieval. Requires faiss-cpu and "
            "sentence-transformers to be installed (see requirements.txt). "
            "The index is built automatically on first run and reused on "
            "subsequent runs unless the corpus or configuration changes. "
            "FAISS is OFF by default; set this flag to activate it."
        ),
    )
    p.add_argument(
        "--neg-ratio",
        type=int,
        default=5,
        help="Negative to positive sampling ratio for training (default: 5).",
    )
    return p.parse_args()


ARGS = _parse_args()

# ---------------------------------------------------------------------------
# Lazy imports — only pulled in after arg parsing
# ---------------------------------------------------------------------------

from src.ingestion import load_clean_tsv
from src.blocking import (
    build_clean_text,
    fit_vectorizer,
    search_candidates_scalable,
    write_candidate_pairs,
    write_blocking_scores,
    compute_candidate_recall,
)

try:
    from src.blocking_bm25 import search_candidates_bm25
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

try:
    from src.blocking import union_candidate_results
    _UNION_AVAILABLE = True
except ImportError:
    _UNION_AVAILABLE = False

from src.features import (
    load_feature_tsv,
    load_blocking_scores,
    build_pair_table,
    add_name_similarity_features,
    add_address_similarity_features,
    add_metadata_features,
    load_ground_truth,
    add_ground_truth_labels,
    compute_candidate_positive_recall,
    write_labeled_pairs,
)
from src.model import (
    LABEL_COL,
    split_entities,
    build_pair_splits,
    score_pairs,
    filter_ground_truth_to_s1_ids,
)
from src.threshold import (
    sweep_thresholds,
    apply_threshold,
    singleton_report,
)

# V2 features + HNM with graceful fallback to V1
try:
    from src.features import (
        add_retrieval_agreement_features,
        add_length_features,
        FEATURE_COLUMNS_V2 as _FEATURE_COLS,
    )
    from src.model import fit_with_hard_negative_mining as _train_fn
    _USE_V2 = True
except ImportError:
    from src.model import FEATURE_COLS as _FEATURE_COLS, train_lightgbm as _train_fn
    _USE_V2 = False

# ---------------------------------------------------------------------------
# Paths derived from CLI args
# ---------------------------------------------------------------------------

REAL  = ARGS.data_dir
OUT   = ARGS.output_dir
CKPT  = ARGS.checkpoint_dir
TOP_K = ARGS.top_k

os.makedirs(OUT,  exist_ok=True)
os.makedirs(CKPT, exist_ok=True)

# Prepared inputs
CLEAN_S1_TRAIN   = os.path.join(REAL, "clean_s1_train.tsv")
CLEAN_S2S3_TRAIN = os.path.join(REAL, "clean_s2s3_train.tsv")
FEAT_S1_TRAIN    = os.path.join(REAL, "feature_s1_train.tsv")
FEAT_S2S3_TRAIN  = os.path.join(REAL, "feature_s2s3_train.tsv")
GT_TRAIN         = os.path.join(REAL, "ground_truth_train.tsv")
CLEAN_S1_TEST    = os.path.join(REAL, "clean_s1_test.tsv")
CLEAN_S2S3_TEST  = os.path.join(REAL, "clean_s2s3_test.tsv")
FEAT_S1_TEST     = os.path.join(REAL, "feature_s1_test.tsv")
FEAT_S2S3_TEST   = os.path.join(REAL, "feature_s2s3_test.tsv")

# Checkpoint state files (pickles / TSVs kept between steps)
CKPT_VECTORIZER       = os.path.join(CKPT, "vectorizer.pkl")
CKPT_TRAIN_RESULTS    = os.path.join(CKPT, "train_results.tsv")
CKPT_TRAIN_CAND       = os.path.join(CKPT, "train_candidates.tsv")
CKPT_TRAIN_LABELED    = os.path.join(CKPT, "train_labeled.tsv")
CKPT_MODEL            = os.path.join(CKPT, "model.pkl")
CKPT_THRESHOLD        = os.path.join(CKPT, "threshold.txt")
CKPT_FEATURE_COLS     = os.path.join(CKPT, "feature_cols.txt")
CKPT_VAL_IDS          = os.path.join(CKPT, "val_ids.txt")   # one entity_id per line
CKPT_TEST_RESULTS     = os.path.join(CKPT, "test_results.tsv")
CKPT_TEST_CAND        = os.path.join(CKPT, "test_candidates.tsv")
CKPT_TEST_SCORED      = os.path.join(CKPT, "test_scored.tsv")
_S = lambda name: os.path.join(CKPT, name)
SENT_VECTORIZER  = _S("s1_vectorizer_fitted")
SENT_TRAIN_BLOCK = _S("s2_train_blocking_done")
SENT_TRAIN_FEAT  = _S("s3_train_features_done")
SENT_MODEL       = _S("s4_model_trained")
SENT_THRESHOLD   = _S("s5_threshold_selected")
SENT_TEST_BLOCK  = _S("s6_test_blocking_done")
SENT_TEST_SCORED = _S("s7_test_scored")
SENT_OUTPUTS     = _S("s8_outputs_written")

# Final submission files
O_CANDIDATE_PAIRS  = os.path.join(OUT, "candidate_pairs.tsv")
O_MATCHING_RESULTS = os.path.join(OUT, "matching_results.tsv")

SEP = "=" * 60


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _done(sentinel: str) -> bool:
    """Return True if this step's sentinel exists and --force was not set."""
    return (not ARGS.force) and os.path.isfile(sentinel)


def _mark_done(sentinel: str) -> None:
    with open(sentinel, "w") as f:
        f.write(f"completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def _skip(step_name: str) -> None:
    print(f"  [SKIP] {step_name} (checkpoint found — delete {CKPT}/ to re-run)")


# ---------------------------------------------------------------------------
# Blocking helper: TF-IDF scalable + optional BM25 union
# ---------------------------------------------------------------------------

def _map_indices_to_ids(raw_df: pd.DataFrame,
                        s1_ids: list[str],
                        s2s3_ids: list[str]) -> pd.DataFrame:
    """Map integer row indices from search_candidates_scalable to entity IDs."""
    df = raw_df.copy()
    df["source1_entity_id"]   = df["s1_row_idx"].astype(int).map(lambda i: s1_ids[i])
    df["candidate_entity_id"] = df["s2s3_row_idx"].astype(int).map(lambda j: s2s3_ids[j])
    df = df.rename(columns={"score": "cosine_similarity"})
    return df[["source1_entity_id", "candidate_entity_id", "cosine_similarity", "rank"]]


def run_blocking(s1_df: pd.DataFrame,
                 s2s3_df: pd.DataFrame,
                 vectorizers: dict,
                 top_k: int,
                 label: str = "",
                 skip_bm25: bool = False,
                 skip_faiss: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Run the master retrieval pipeline: Exact Match + Multi-View TF-IDF + FAISS + BM25, then fuse.

    Parameters
    ----------
    skip_faiss : bool
        When True (default) FAISS is not executed, preserving the existing
        baseline behaviour.  Set to False only when --enable-faiss is passed.

    Returns (results_df, candidates_by_s1 dict).
    """
    s1_ids   = s1_df["entity_id"].tolist()
    s2s3_ids = s2s3_df["entity_id"].tolist()
    
    t_start = time.time()
    
    # ── 1. EXACT MATCH ──────────────
    print(f"  [{label}] Exact Match Indexing ...")
    try:
        from src.blocking_exact import build_exact_indexes, search_exact_candidates
        exact_idx = build_exact_indexes(s2s3_df, ['clean_name', 'clean_address'])
        exact_df = search_exact_candidates(s1_df, exact_idx, ['clean_name', 'clean_address'])
        print(f"    Exact pairs: {len(exact_df)}")
    except ImportError:
        print(f"    Exact match skipped (module missing).")
        exact_df = pd.DataFrame()
        
    # ── 2. MULTI-VIEW TF-IDF ──────────────
    print(f"  [{label}] Multi-View TF-IDF (top_k={top_k}) ...")
    try:
        from src.blocking_multiview import search_multiview_candidates
        # We assume vectorizers are now a dict of fitted vectorizers {"name": vec, "address": vec, "combined": vec}
        # In a real run we pass chunk_size and corpus_block_size from ARGS down to scalable_search, 
        # but for simplicity we rely on the function defaults or modify it to accept kwargs.
        mv_df = search_multiview_candidates(s1_df, s2s3_df, vectorizers, top_k=top_k)
        print(f"    Multi-View TF-IDF pairs: {len(mv_df)}")
    except ImportError:
        print(f"    Multi-view skipped (module missing).")
        mv_df = pd.DataFrame()

    # ── 3. FAISS SEMANTIC ──────────────
    faiss_df = pd.DataFrame()
    if skip_faiss:
        print(f"  [{label}] FAISS skipped (not enabled; use --enable-faiss to activate).")
    else:
        print(f"  [{label}] FAISS Semantic Retrieval (top_k={top_k}) ...")
        # Import here — raises RuntimeError with actionable message if
        # faiss-cpu or sentence-transformers are not installed.
        from src.blocking_faiss import get_or_build_and_search
        try:
            from pipeline_config import faiss_index_base, FAISS_BATCH_SIZE, FAISS_ADD_CHUNK_SIZE
        except ImportError:
            # Fallback: infer base from checkpoint dir
            faiss_index_base_fn = lambda ckpt, lbl: os.path.join(ckpt, f"faiss_index_{lbl}")
            FAISS_BATCH_SIZE = ARGS.faiss_batch_size
            FAISS_ADD_CHUNK_SIZE = 500_000
            idx_base = faiss_index_base_fn(CKPT, label.upper())
        else:
            idx_base = str(faiss_index_base(CKPT, label))

        faiss_df = get_or_build_and_search(
            s2s3_df      = s2s3_df,
            s1_df        = s1_df,
            index_base   = idx_base,
            split_label  = label.upper(),
            top_k        = top_k,
            batch_size   = ARGS.faiss_batch_size,
        )
        print(f"    FAISS pairs: {len(faiss_df)}")
        
    # ── 4. BM25 ──────────────
    bm25_df = pd.DataFrame()
    if skip_bm25 or not _BM25_AVAILABLE:
        print(f"  [{label}] BM25 skipped.")
    else:
        print(f"  [{label}] BM25 blocking (top_k={top_k}) ...")
        try:
            bm25_df = search_candidates_bm25(s1_df, s2s3_df, top_k=top_k)
            print(f"    BM25 pairs: {len(bm25_df)}")
        except Exception as exc:
            print(f"  [{label}] BM25 failed ({exc}).")

    # ── 5. FUSION ──────────────
    print(f"  [{label}] Candidate Fusion ...")
    try:
        from src.blocking_fusion import fuse_candidates
        results_df = fuse_candidates([exact_df, mv_df, faiss_df, bm25_df])
        print(f"    Fused pairs: {len(results_df)}  (Total time: {time.time()-t_start:.1f}s)")
    except ImportError:
        print(f"    Fusion failed, using mv_df only.")
        results_df = mv_df

    # ── 6. RETRIEVAL AGREEMENT ALIASES ──────────────
    # add_retrieval_agreement_features() expects:
    #   retrieved_by_char_tfidf   — combined multi-view TF-IDF provenance flag
    #   retrieved_by_bm25         — BM25 provenance flag
    #   retrieval_agreement_count — already computed by fuse_candidates()
    #
    # Mapping from actual fused columns:
    #   retrieved_by_tfidf_combined  -> retrieved_by_char_tfidf
    #   bm25_score / bm25_rank presence -> retrieved_by_bm25
    if not results_df.empty:
        # --- retrieved_by_char_tfidf ---
        # Use the 'combined' view flag from blocking_multiview as the canonical
        # TF-IDF signal. Fall back to OR of name/address views if combined is absent.
        if "retrieved_by_tfidf_combined" in results_df.columns:
            results_df["retrieved_by_char_tfidf"] = results_df["retrieved_by_tfidf_combined"].fillna(0).astype(int)
        elif "retrieved_by_tfidf_name" in results_df.columns or "retrieved_by_tfidf_address" in results_df.columns:
            # At least one TF-IDF view is present — a pair was TF-IDF-retrieved
            # if any view retrieved it.
            tfidf_cols = [c for c in results_df.columns if c.startswith("retrieved_by_tfidf_")]
            results_df["retrieved_by_char_tfidf"] = (
                results_df[tfidf_cols].fillna(0).max(axis=1).astype(int)
            )
        else:
            # No TF-IDF columns present at all — default to 0
            results_df["retrieved_by_char_tfidf"] = 0

        # --- retrieved_by_bm25 ---
        # blocking_bm25 does not emit a retrieved_by_bm25 flag; it emits
        # bm25_score and bm25_rank. A non-null, positive bm25_score means
        # the pair was BM25-retrieved. When BM25 is skipped the column is absent.
        if "bm25_score" in results_df.columns:
            results_df["retrieved_by_bm25"] = (
                results_df["bm25_score"].notna() &
                (pd.to_numeric(results_df["bm25_score"], errors="coerce").fillna(0) > 0)
            ).astype(int)
        elif "bm25_rank" in results_df.columns:
            # bm25_rank present but no score column: treat presence as flag
            results_df["retrieved_by_bm25"] = results_df["bm25_rank"].notna().astype(int)
        else:
            # BM25 was skipped (--skip-bm25) or unavailable — zero-fill
            results_df["retrieved_by_bm25"] = 0

        # --- retrieval_agreement_count ---
        # Always recompute AFTER the alias columns are set so the count
        # reflects the canonical (retrieved_by_char_tfidf + retrieved_by_bm25)
        # pair, not whatever fuse_candidates computed from the raw retrieved_by_*
        # flags (which may not have included the aliases yet).
        #
        # Clamp minimum to 1: build_feature_matrix() validates {1, 2}.
        # A pair retrieved only by FAISS or exact-match (both canonical flags=0)
        # would produce 0, which fails the validator.  Such pairs genuinely
        # entered the candidate set (via another retrieval mechanism) so
        # defaulting to 1 is conservative and correct.
        raw_count = (
            results_df["retrieved_by_char_tfidf"].astype(int)
            + results_df["retrieved_by_bm25"].astype(int)
        )
        results_df["retrieval_agreement_count"] = raw_count.clip(lower=1)

    candidates_by_s1: dict[str, list[str]] = {}
    if not results_df.empty:
        for s1_id, grp in results_df.groupby("source1_entity_id", sort=False):
            candidates_by_s1[s1_id] = grp["candidate_entity_id"].tolist()

    return results_df, candidates_by_s1


# ---------------------------------------------------------------------------
# Feature builder helper
# ---------------------------------------------------------------------------

def build_features(cand_pairs_df: pd.DataFrame,
                   feat_s1: pd.DataFrame,
                   feat_s2s3: pd.DataFrame,
                   scores_df: pd.DataFrame,
                   results_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature table from a candidate pair set.

    Mirrors the train-side feature construction exactly so that train and
    test feature columns are always consistent.
    """
    from src.features_rich import add_rich_features

    pair_df = build_pair_table(cand_pairs_df, feat_s1, feat_s2s3, scores_df)
    pair_df = add_name_similarity_features(pair_df)
    pair_df = add_address_similarity_features(pair_df)
    pair_df = add_metadata_features(pair_df, blocking_scores=scores_df)
    if _USE_V2:
        pair_df = add_retrieval_agreement_features(pair_df, results_df)
        pair_df = add_length_features(pair_df)

    # Add the same rich features that the train path adds via add_rich_features().
    # These are computed from s1_name / s2_name / s1_address / s2_address which
    # are derived from the feature TSV files.  We join them here the same way
    # the train path does (pair_df already has s1_clean_name etc. from
    # build_pair_table, so rename them to the expected s1_name/s2_name keys).
    pair_df = pair_df.rename(columns={
        "s1_clean_name":    "s1_name",
        "candidate_clean_name": "s2_name",
        "s1_clean_address": "s1_address",
        "candidate_clean_address": "s2_address",
    })
    # Pass empty DataFrames for s1_df/s2s3_df since add_rich_features only
    # uses pair_df columns (s1_name, s2_name, s1_address, s2_address).
    pair_df = add_rich_features(pair_df,
                                pd.DataFrame(columns=feat_s1.columns),
                                pd.DataFrame(columns=feat_s2s3.columns))
    # Restore canonical column names for downstream compatibility
    pair_df = pair_df.rename(columns={
        "s1_name":    "s1_clean_name",
        "s2_name":    "candidate_clean_name",
        "s1_address": "s1_clean_address",
        "s2_address": "candidate_clean_address",
    })
    return pair_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(SEP)
    print("Production Pipeline — Business Entity Resolution")
    print(f"  split={ARGS.split}  top_k={TOP_K}  skip_bm25={ARGS.skip_bm25}  enable_faiss={ARGS.enable_faiss}")
    print(f"  force={ARGS.force}  checkpoint_dir={CKPT}")
    print(f"  features={'V2 (15)' if _USE_V2 else 'V1 (10)'}")
    print(SEP)

    # ── Validate input files exist ────────────────────────────────────────
    need_train = ARGS.split in ("train", "all")
    need_test  = ARGS.split in ("test",  "all")

    train_files = [CLEAN_S1_TRAIN, CLEAN_S2S3_TRAIN,
                   FEAT_S1_TRAIN,  FEAT_S2S3_TRAIN, GT_TRAIN]
    test_files  = [CLEAN_S1_TEST, CLEAN_S2S3_TEST,
                   FEAT_S1_TEST,  FEAT_S2S3_TEST]

    missing = []
    if need_train:
        missing += [f for f in train_files if not os.path.isfile(f)]
    if need_test:
        missing += [f for f in test_files  if not os.path.isfile(f)]

    if missing:
        print("\nERROR: Prepared data files missing. Run first:")
        print("  python prepare_real_data.py\n")
        for f in missing:
            print(f"  MISSING: {f}")
        raise SystemExit(1)

    if need_test and not need_train:
        # --split test: require completed train checkpoints
        required_train_ckpts = [SENT_MODEL, SENT_THRESHOLD,
                                 CKPT_MODEL, CKPT_THRESHOLD,
                                 CKPT_VECTORIZER, CKPT_FEATURE_COLS]
        missing_ckpts = [f for f in required_train_ckpts
                         if not os.path.isfile(f)]
        if missing_ckpts:
            print("\nERROR: --split test requires completed train checkpoints.")
            print("Run training first:  python run_production.py --split train\n")
            for f in missing_ckpts:
                print(f"  MISSING: {f}")
            raise SystemExit(1)

    # ====================================================================
    # TRAIN SIDE
    # ====================================================================

    if need_train:

        # ── Step 1: Load training data ────────────────────────────────────
        print("\n[STEP 1] Loading training data ...")
        t0 = time.time()
        s1_train   = load_clean_tsv(CLEAN_S1_TRAIN)
        s2s3_train = load_clean_tsv(CLEAN_S2S3_TRAIN)
        feat_s1    = load_feature_tsv(FEAT_S1_TRAIN)
        feat_s2s3  = load_feature_tsv(FEAT_S2S3_TRAIN)
        gt_df      = load_ground_truth(GT_TRAIN)
        print(f"  S1 train rows    : {len(s1_train)}")
        print(f"  S2+S3 train rows : {len(s2s3_train)}")
        print(f"  Ground truth rows: {len(gt_df)}")
        print(f"  Loaded in {time.time()-t0:.1f}s")

        # ── Step 2: Fit vectorizer ────────────────────────────────────────
        if _done(SENT_VECTORIZER) and os.path.isfile(CKPT_VECTORIZER):
            _skip("Fit TF-IDF vectorizers")
            with open(CKPT_VECTORIZER, "rb") as f:
                vectorizer = pickle.load(f)
            print(f"  Loaded multi-view vectorizers.")
        else:
            print("\n[STEP 2] Fitting Multi-View TF-IDF vectorizers on S2+S3 train corpus ...")
            t0 = time.time()
            from src.blocking_multiview import fit_multiview_vectorizers
            vectorizer = fit_multiview_vectorizers(s2s3_train)
            with open(CKPT_VECTORIZER, "wb") as f:
                pickle.dump(vectorizer, f, protocol=4)
            _mark_done(SENT_VECTORIZER)
            print(f"  Fitted multi-view vectorizers in {time.time()-t0:.1f}s")

        # ── Step 3: Training blocking ─────────────────────────────────────
        if _done(SENT_TRAIN_BLOCK) and os.path.isfile(CKPT_TRAIN_RESULTS):
            _skip("Training blocking")
            train_results_df = pd.read_csv(CKPT_TRAIN_RESULTS, sep="\t",
                                           dtype=str, keep_default_na=False)
            # The checkpoint TSV stores the score under 'score' (from
            # blocking_multiview) or 'cosine_similarity' depending on which
            # blocking path ran.  Normalise to 'cosine_similarity' float here.
            # pandas 3.x: columns loaded with dtype=str need explicit numeric
            # coercion; .astype(float) on a StringDtype series also works but
            # pd.to_numeric is more robust for missing/empty values.
            if "cosine_similarity" not in train_results_df.columns:
                score_src = "score" if "score" in train_results_df.columns else None
                if score_src:
                    train_results_df["cosine_similarity"] = pd.to_numeric(
                        train_results_df[score_src], errors="coerce"
                    ).fillna(0.0)
                else:
                    train_results_df["cosine_similarity"] = 0.0
            else:
                train_results_df["cosine_similarity"] = pd.to_numeric(
                    train_results_df["cosine_similarity"], errors="coerce"
                ).fillna(0.0)
            if "rank" not in train_results_df.columns:
                # Reconstruct rank as row-order within each S1 entity group
                train_results_df["rank"] = (
                    train_results_df
                    .sort_values(["source1_entity_id", "cosine_similarity"],
                                 ascending=[True, False])
                    .groupby("source1_entity_id")
                    .cumcount()
                )
            else:
                train_results_df["rank"] = pd.to_numeric(
                    train_results_df["rank"], errors="coerce"
                ).fillna(0).astype(int)
            # Reconstruct candidates_by_s1 from saved TSV
            train_cand_raw = pd.read_csv(CKPT_TRAIN_CAND, sep="\t",
                                         dtype=str, keep_default_na=False)
            train_candidates: dict[str, list[str]] = {}
            for _, row in train_cand_raw.iterrows():
                raw = str(row["candidate_entity_ids"]).strip()
                train_candidates[row["source1_entity_id"]] = (
                    [x for x in raw.split(",") if x] if raw else []
                )
        else:
            print("\n[STEP 3] Training blocking ...")
            t0 = time.time()
            train_results_df, train_candidates = run_blocking(
                s1_train, s2s3_train, vectorizer,
                top_k=TOP_K, label="TRAIN",
                skip_bm25=ARGS.skip_bm25,
                skip_faiss=not ARGS.enable_faiss,
            )
            # Persist
            train_results_df.to_csv(CKPT_TRAIN_RESULTS, sep="\t", index=False)
            write_candidate_pairs(train_candidates, CKPT_TRAIN_CAND)
            recall = compute_candidate_recall(train_candidates, gt_df)
            print(f"  Train blocking recall : {recall['recall']}")
            if recall["recall"] is not None and recall["recall"] < 0.85:
                print(f"  WARNING: recall {recall['recall']:.3f} < 0.85 "
                      "-- consider --top-k 75 or --top-k 100")
            _mark_done(SENT_TRAIN_BLOCK)
            print(f"  Blocking done in {time.time()-t0:.1f}s")

        # ── Normalise score column from blocking ─────────────────────────
        # run_blocking() (and blocking_multiview) emits 'score'; the resume
        # path may already have 'cosine_similarity'.  Ensure both a float
        # 'cosine_similarity' and integer 'rank' column always exist in
        # train_results_df before Step 4 uses it.
        if "cosine_similarity" not in train_results_df.columns:
            _sc = "score" if "score" in train_results_df.columns else None
            train_results_df["cosine_similarity"] = (
                pd.to_numeric(train_results_df[_sc], errors="coerce").fillna(0.0)
                if _sc else 0.0
            )
        else:
            train_results_df["cosine_similarity"] = pd.to_numeric(
                train_results_df["cosine_similarity"], errors="coerce"
            ).fillna(0.0)
        if "rank" not in train_results_df.columns:
            train_results_df["rank"] = (
                train_results_df
                .sort_values(["source1_entity_id", "cosine_similarity"],
                             ascending=[True, False])
                .groupby("source1_entity_id")
                .cumcount()
            )

        # ── Step 4: Training features ─────────────────────────────────────
        if _done(SENT_TRAIN_FEAT) and os.path.isfile(CKPT_TRAIN_LABELED):
            _skip("Training feature engineering")
            labeled_df = pd.read_csv(CKPT_TRAIN_LABELED, sep="\t",
                                     dtype=str, keep_default_na=False)
            # Find feature columns — same exclusion list as the fresh-run path
            _NON_FEATURE_COLS_RESUME = frozenset({
                "source1_entity_id", "candidate_entity_id",
                "label", "is_match", "matching_entity_ids",
                "candidate_entity_ids",
                "s1_name", "s2_name", "s1_address", "s2_address",
                "score", "cosine_similarity", "rank",
                "retrieved_by_tfidf_address",
                "retrieved_by_tfidf_combined",
                "retrieved_by_tfidf_name",
                "bm25_score", "bm25_rank",
            })
            feature_cols = [
                c for c in labeled_df.columns
                if c not in _NON_FEATURE_COLS_RESUME
            ]
            for col in feature_cols:
                if col in labeled_df.columns:
                    labeled_df[col] = pd.to_numeric(labeled_df[col], errors='coerce')
            if LABEL_COL in labeled_df.columns:
                labeled_df[LABEL_COL] = labeled_df[LABEL_COL].astype(int)
            n_pos = int((labeled_df[LABEL_COL] == 1).sum())
            n_neg = int((labeled_df[LABEL_COL] == 0).sum())
            print(f"  Labeled pairs: {len(labeled_df)}  (pos={n_pos}, neg={n_neg})")
        else:
            print("\n[STEP 4] Building training features & hard negative mining ...")
            t0 = time.time()
            
            # Use raw scores/features directly from the fusion output (train_results_df)
            cand_pairs_df = pd.DataFrame([
                {"source1_entity_id": s1,
                 "candidate_entity_ids": ",".join(cands)}
                for s1, cands in train_candidates.items()
            ])
            
            # Merge S1 and S2/S3 text fields for rich features
            # NOTE: s1_train and s2s3_train only have entity_id + clean_text.
            #       The name/address columns live in feat_s1 / feat_s2s3.
            s1_feat_idx    = feat_s1.set_index('entity_id')
            s2s3_feat_idx  = feat_s2s3.set_index('entity_id')
            pair_df = train_results_df.copy()
            pair_df['s1_name']    = pair_df['source1_entity_id'].map(s1_feat_idx['clean_name'])
            pair_df['s2_name']    = pair_df['candidate_entity_id'].map(s2s3_feat_idx['clean_name'])
            pair_df['s1_address'] = pair_df['source1_entity_id'].map(s1_feat_idx['clean_address'])
            pair_df['s2_address'] = pair_df['candidate_entity_id'].map(s2s3_feat_idx['clean_address'])
            
            from src.features_rich import add_rich_features
            pair_df = add_rich_features(pair_df, s1_train, s2s3_train)
            
            from src.negative_sampling import build_controlled_training_pairs
            labeled_df = build_controlled_training_pairs(
                cand_pairs_df, gt_df, pair_df, neg_ratio=ARGS.neg_ratio
            )
            
            # Map 'is_match' to LABEL_COL
            labeled_df[LABEL_COL] = labeled_df['is_match']
            
            # Ensure no NaNs in numeric feature columns.
            # pandas 3.x uses a PyArrow-backed StringDtype for columns loaded
            # with dtype=str; fillna(0) on those raises TypeError because 0 is
            # not a valid string fill value.  We therefore restrict fillna to
            # numeric columns only — string ID columns are never NaN here.
            _num_cols = labeled_df.select_dtypes(include="number").columns
            labeled_df[_num_cols] = labeled_df[_num_cols].fillna(0)

            n_pos = int((labeled_df[LABEL_COL] == 1).sum())
            n_neg = int((labeled_df[LABEL_COL] == 0).sum())
            print(f"  Pair table rows: {len(pair_df)}")
            print(f"  Labeled pairs: {len(labeled_df)}  (pos={n_pos}, neg={n_neg})")

            write_labeled_pairs(labeled_df, CKPT_TRAIN_LABELED)
            # Save feature column list for test-side resume.
            # Exclude:
            #  (a) ID / label columns that are never model features
            #  (b) raw blocking passthrough columns that come from
            #      train_results_df but are NOT produced by build_features()
            #      (which is what generates test_pair_df).  Keeping them would
            #      cause score_pairs() to fail with "column missing" on the
            #      test side because build_features() never emits them.
            _NON_FEATURE_COLS = frozenset({
                # identity / label
                "source1_entity_id", "candidate_entity_id",
                "label", "is_match", "matching_entity_ids",
                "candidate_entity_ids",
                # rich-feature scratch columns (not model inputs)
                "s1_name", "s2_name", "s1_address", "s2_address",
                # raw blocking internals not emitted by build_features()
                "score", "cosine_similarity", "rank",
                "retrieved_by_tfidf_address",
                "retrieved_by_tfidf_combined",
                "retrieved_by_tfidf_name",
                # BM25 blocking internals (present when --skip-bm25 is off)
                "bm25_score", "bm25_rank",
            })
            feature_cols = [
                c for c in labeled_df.columns
                if c not in _NON_FEATURE_COLS
            ]
            
            with open(CKPT_FEATURE_COLS, "w") as f:
                f.write("\n".join(feature_cols))
            _mark_done(SENT_TRAIN_FEAT)
            print(f"  Features done in {time.time()-t0:.1f}s")

        # ── Step 5: Entity split + model training ─────────────────────────
        if _done(SENT_MODEL) and os.path.isfile(CKPT_MODEL):
            _skip("Model training")
            with open(CKPT_MODEL, "rb") as f:
                model = pickle.load(f)
            with open(CKPT_FEATURE_COLS) as f:
                feature_cols = [l.strip() for l in f if l.strip()]
            print(f"  Loaded model: n_estimators={model.n_estimators_}")
            print(f"  Feature cols : {len(feature_cols)}")
            # Recover val_ids from checkpoint (written when model was trained).
            # Fall back to recomputing from the current s1_train split if the
            # file is absent (e.g., upgrading from an older checkpoint).
            if os.path.isfile(CKPT_VAL_IDS):
                with open(CKPT_VAL_IDS) as f:
                    val_ids = [l.strip() for l in f if l.strip()]
            else:
                _, val_ids = split_entities(s1_train)
                with open(CKPT_VAL_IDS, "w") as f:
                    f.write("\n".join(val_ids))
        else:
            print("\n[STEP 5] Entity split + LightGBM training ...")
            t0 = time.time()
            with open(CKPT_FEATURE_COLS) as f:
                feature_cols = [l.strip() for l in f if l.strip()]

            train_ids, val_ids = split_entities(s1_train)
            print(f"  Train S1 entities: {len(train_ids)}")
            print(f"  Val   S1 entities: {len(val_ids)}")

            train_pairs_df, val_pairs_df = build_pair_splits(
                labeled_pairs_df=labeled_df,
                train_ids=train_ids,
                val_ids=val_ids,
            )
            print(f"  Train pairs: {len(train_pairs_df)}  "
                  f"Val pairs: {len(val_pairs_df)}")

            if _USE_V2:
                model, meta = _train_fn(train_pairs_df,
                                        feature_cols=feature_cols)
                print(f"  Hard negatives added: "
                      f"{meta.get('n_hard_negatives_added', 'N/A')}")
            else:
                model = _train_fn(train_pairs_df, feature_cols=feature_cols)

            print(f"  Model trained: n_estimators={model.n_estimators_}")
            # Persist model + split info for threshold step
            with open(CKPT_MODEL, "wb") as f:
                pickle.dump(model, f, protocol=4)
            with open(CKPT_FEATURE_COLS, "w") as fh:
                fh.write("\n".join(feature_cols))
            # Persist val_ids so Step 6 can filter GT without re-splitting
            with open(CKPT_VAL_IDS, "w") as fh:
                fh.write("\n".join(val_ids))
            # Also persist val_pairs for threshold step
            val_pairs_df.to_csv(
                os.path.join(CKPT, "val_pairs.tsv"), sep="\t", index=False
            )
            _mark_done(SENT_MODEL)
            print(f"  Training done in {time.time()-t0:.1f}s")

        # ── Step 6: Threshold sweep ───────────────────────────────────────
        if _done(SENT_THRESHOLD) and os.path.isfile(CKPT_THRESHOLD):
            _skip("Threshold sweep")
            with open(CKPT_THRESHOLD) as f:
                best_t = float(f.read().strip())
            with open(CKPT_FEATURE_COLS) as f:
                feature_cols = [l.strip() for l in f if l.strip()]
            print(f"  Loaded threshold: {best_t:.4f}")
        else:
            print("\n[STEP 6] Threshold sweep on validation set ...")
            t0 = time.time()
            with open(CKPT_FEATURE_COLS) as f:
                feature_cols = [l.strip() for l in f if l.strip()]

            val_pairs_path = os.path.join(CKPT, "val_pairs.tsv")
            if os.path.isfile(val_pairs_path):
                val_pairs_df = pd.read_csv(val_pairs_path, sep="\t",
                                           dtype=str, keep_default_na=False)
                for col in feature_cols:
                    if col in val_pairs_df.columns:
                        val_pairs_df[col] = val_pairs_df[col].astype(float)
                if LABEL_COL in val_pairs_df.columns:
                    val_pairs_df[LABEL_COL] = val_pairs_df[LABEL_COL].astype(int)
            else:
                # Reconstruct val split from labeled_df
                train_ids, val_ids = split_entities(s1_train)
                _, val_pairs_df = build_pair_splits(labeled_df,
                                                    train_ids, val_ids)

            if len(val_pairs_df) > 0:
                scored_val = score_pairs(model, val_pairs_df,
                                         feature_cols=feature_cols)

                # Filter GT to validation S1 entities only so training
                # entities cannot inflate or deflate the sweep metric.
                val_gt_df = filter_ground_truth_to_s1_ids(gt_df, val_ids)
                print(f"  Val GT rows : {len(val_gt_df)}  "
                      f"(full GT: {len(gt_df)})")

                sweep  = sweep_thresholds(scored_val, val_gt_df)
                best_t = sweep["best_threshold"]
                best_f = sweep["best_macro_f_beta"]
                print(f"  Best threshold : {best_t:.4f}")
                print(f"  Best macro-F0.5: {best_f:.4f}  "
                      "(validation — NOT the challenge score)")
                diag = singleton_report(
                    apply_threshold(scored_val, best_t), val_gt_df
                )
                print(f"  Singleton F0.5     : "
                      f"{diag['singleton']['macro_f_beta']:.4f}")
                print(f"  Non-singleton F0.5 : "
                      f"{diag['non_singleton']['macro_f_beta']:.4f}")
            else:
                best_t = 0.5
                print("  Val set empty — using default threshold 0.5")

            with open(CKPT_THRESHOLD, "w") as f:
                f.write(str(best_t))
            _mark_done(SENT_THRESHOLD)
            print(f"  Sweep done in {time.time()-t0:.1f}s")

        if ARGS.split == "train":
            print("\n" + SEP)
            print("TRAIN SPLIT COMPLETE")
            print(f"  Model checkpoint  : {CKPT_MODEL}")
            print(f"  Best threshold    : {best_t:.4f}")
            print(f"  To run inference  : python run_production.py --split test")
            print(SEP)
            return

    # ====================================================================
    # TEST SIDE
    # ====================================================================

    # Load persisted state if entering from --split test
    if not need_train:
        print("\n[RESUME] Loading train-side state from checkpoints ...")
        with open(CKPT_VECTORIZER, "rb") as f:
            vectorizer = pickle.load(f)
        with open(CKPT_MODEL, "rb") as f:
            model = pickle.load(f)
        with open(CKPT_THRESHOLD) as f:
            best_t = float(f.read().strip())
        with open(CKPT_FEATURE_COLS) as f:
            feature_cols = [l.strip() for l in f if l.strip()]
        # vectorizer is a dict of multi-view TfidfVectorizers
        vocab_info = {k: len(v.vocabulary_) for k, v in vectorizer.items()} if isinstance(vectorizer, dict) else len(vectorizer.vocabulary_)
        print(f"  Vectorizer vocab : {vocab_info}")
        print(f"  Model estimators : {model.n_estimators_}")
        print(f"  Threshold        : {best_t:.4f}")
        print(f"  Feature cols     : {len(feature_cols)}")

    # ── Step 7: Test blocking ─────────────────────────────────────────────
    print("\n[STEP 7] Loading test data ...")
    s1_test   = load_clean_tsv(CLEAN_S1_TEST)
    s2s3_test = load_clean_tsv(CLEAN_S2S3_TEST)
    feat_s1_t = load_feature_tsv(FEAT_S1_TEST)
    feat_s2_t = load_feature_tsv(FEAT_S2S3_TEST)
    print(f"  S1 test rows    : {len(s1_test)}")
    print(f"  S2+S3 test rows : {len(s2s3_test)}")

    if _done(SENT_TEST_BLOCK) and os.path.isfile(CKPT_TEST_RESULTS):
        _skip("Test blocking")
        test_results_df = pd.read_csv(CKPT_TEST_RESULTS, sep="\t",
                                      dtype=str, keep_default_na=False)
        # Same normalisation as the train-side resume: checkpoint may store
        # 'score' instead of 'cosine_similarity', and 'rank' may be absent.
        if "cosine_similarity" not in test_results_df.columns:
            score_src = "score" if "score" in test_results_df.columns else None
            if score_src:
                test_results_df["cosine_similarity"] = pd.to_numeric(
                    test_results_df[score_src], errors="coerce"
                ).fillna(0.0)
            else:
                test_results_df["cosine_similarity"] = 0.0
        else:
            test_results_df["cosine_similarity"] = pd.to_numeric(
                test_results_df["cosine_similarity"], errors="coerce"
            ).fillna(0.0)
        if "rank" not in test_results_df.columns:
            test_results_df["rank"] = (
                test_results_df
                .sort_values(["source1_entity_id", "cosine_similarity"],
                             ascending=[True, False])
                .groupby("source1_entity_id")
                .cumcount()
            )
        else:
            test_results_df["rank"] = pd.to_numeric(
                test_results_df["rank"], errors="coerce"
            ).fillna(0).astype(int)
        test_cand_raw = pd.read_csv(CKPT_TEST_CAND, sep="\t",
                                    dtype=str, keep_default_na=False)
        test_candidates: dict[str, list[str]] = {}
        for _, row in test_cand_raw.iterrows():
            raw = str(row["candidate_entity_ids"]).strip()
            test_candidates[row["source1_entity_id"]] = (
                [x for x in raw.split(",") if x] if raw else []
            )
    else:
        print("\n[STEP 7] Test blocking ...")
        t0 = time.time()
        test_results_df, test_candidates = run_blocking(
            s1_test, s2s3_test, vectorizer,
            top_k=TOP_K, label="TEST",
            skip_bm25=ARGS.skip_bm25,
            skip_faiss=not ARGS.enable_faiss,
        )
        test_results_df.to_csv(CKPT_TEST_RESULTS, sep="\t", index=False)
        write_candidate_pairs(test_candidates, CKPT_TEST_CAND)
        # Also write the final submission candidate_pairs.tsv now
        write_candidate_pairs(test_candidates, O_CANDIDATE_PAIRS)
        _mark_done(SENT_TEST_BLOCK)
        print(f"  Test blocking done in {time.time()-t0:.1f}s")

    # ── Normalise test blocking score column ────────────────────────────
    # run_blocking() emits 'score'; ensure a float 'cosine_similarity' and
    # integer 'rank' are always present before Step 8 uses them.
    if "cosine_similarity" not in test_results_df.columns:
        _sc = "score" if "score" in test_results_df.columns else None
        test_results_df["cosine_similarity"] = (
            pd.to_numeric(test_results_df[_sc], errors="coerce").fillna(0.0)
            if _sc else 0.0
        )
    else:
        test_results_df["cosine_similarity"] = pd.to_numeric(
            test_results_df["cosine_similarity"], errors="coerce"
        ).fillna(0.0)
    if "rank" not in test_results_df.columns:
        test_results_df["rank"] = (
            test_results_df
            .sort_values(["source1_entity_id", "cosine_similarity"],
                         ascending=[True, False])
            .groupby("source1_entity_id")
            .cumcount()
        )

    # Ensure candidate_pairs.tsv is always written (even on resume)
    if not os.path.isfile(O_CANDIDATE_PAIRS):
        write_candidate_pairs(test_candidates, O_CANDIDATE_PAIRS)

    # ── Step 8: Test features + scoring ──────────────────────────────────
    if _done(SENT_TEST_SCORED) and os.path.isfile(CKPT_TEST_SCORED):
        _skip("Test feature engineering + scoring")
        scored_test = pd.read_csv(CKPT_TEST_SCORED, sep="\t",
                                  dtype=str, keep_default_na=False)
        scored_test["prob_match"] = scored_test["prob_match"].astype(float)
    else:
        print("\n[STEP 8] Test feature engineering + scoring ...")
        t0 = time.time()
        # cosine_similarity is now guaranteed to exist as float (normalised above)
        test_scores_df = test_results_df[
            ["source1_entity_id", "candidate_entity_id", "cosine_similarity"]
        ].copy()
        test_cand_pairs_df = pd.DataFrame([
            {"source1_entity_id": s1,
             "candidate_entity_ids": ",".join(cands)}
            for s1, cands in test_candidates.items()
        ])
        test_pair_df = build_features(
            test_cand_pairs_df, feat_s1_t, feat_s2_t,
            test_scores_df, test_results_df,
        )
        print(f"  Test pairs to score: {len(test_pair_df)}")
        scored_test = score_pairs(model, test_pair_df,
                                   feature_cols=feature_cols)
        scored_test[["source1_entity_id", "candidate_entity_id",
                     "prob_match"]].to_csv(
            CKPT_TEST_SCORED, sep="\t", index=False
        )
        _mark_done(SENT_TEST_SCORED)
        print(f"  Scoring done in {time.time()-t0:.1f}s")

    # ── Step 9: Apply threshold + write outputs ───────────────────────────
    print("\n[STEP 9] Applying threshold + writing submission files ...")
    t0 = time.time()

    preds = apply_threshold(scored_test, threshold=best_t)

    n_matched   = sum(1 for v in preds.values() if len(v) > 0)
    n_singleton = sum(1 for v in preds.values() if len(v) == 0)
    print(f"  Threshold        : {best_t:.4f}")
    print(f"  Matched entities : {n_matched}")
    print(f"  Singletons       : {n_singleton}")

    from src.postprocess import (
        dedupe_matches,
        filter_valid_ids,
        ensure_full_coverage,
        check_candidate_consistency,
        write_matching_results,
        write_candidate_pairs_final,
    )

    # 5.1 Dedup
    preds = dedupe_matches(preds)

    # 5.2 Filter invalid IDs
    valid_ids = set(s2s3_test["entity_id"].tolist())
    preds = filter_valid_ids(preds, valid_ids)

    # 5.3 Full coverage
    all_s1 = s1_test["entity_id"].tolist()
    preds = ensure_full_coverage(preds, all_s1)

    # 5.4 Candidate consistency — abort on violations
    consistency = check_candidate_consistency(preds, test_candidates)
    if not consistency["passed"]:
        raise RuntimeError(
            f"Candidate consistency check failed with "
            f"{consistency['n_violations']} violations. "
            "Fix blocking/inference mismatch before writing output."
        )

    # 5.5 Write output files
    n_match = write_matching_results(preds, O_MATCHING_RESULTS)
    n_cand  = write_candidate_pairs_final(test_candidates, O_CANDIDATE_PAIRS)
    print(f"  Written -> {O_MATCHING_RESULTS}  ({n_match} rows)")
    print(f"  Written -> {O_CANDIDATE_PAIRS}  ({n_cand} rows)")

    _mark_done(SENT_OUTPUTS)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + SEP)
    print("PIPELINE COMPLETE")
    print(SEP)
    print(f"\n  Submission files:")
    print(f"    {O_MATCHING_RESULTS}   <- upload to leaderboard")
    print(f"    {O_CANDIDATE_PAIRS}    <- include in submission zip")
    print(f"\n  Validate before submitting:")
    print("    python utils/validate_submission.py \\")
    print("        --matching output/matching_results.tsv \\")
    print("        --candidate output/candidate_pairs.tsv \\")
    print("        --test-dir dataset/test")
    print(SEP)


if __name__ == "__main__":
    main()