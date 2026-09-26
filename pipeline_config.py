"""
pipeline_config.py
------------------
Centralised, minimal configuration for the combined pipeline.

All path resolution is done relative to REPO_ROOT so the pipeline works
correctly regardless of the directory from which the user runs it.

Callers should import from this module rather than hardcoding paths.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root — determined from this file's location.
# All other paths are resolved relative to this.
# ---------------------------------------------------------------------------
REPO_ROOT: Path = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Raw data (immutable inputs — NEVER overwritten)
# ---------------------------------------------------------------------------
RAW_DIR: Path       = REPO_ROOT / "dataset" / "raw"
RAW_TRAIN_DIR: Path = RAW_DIR / "train"
RAW_TEST_DIR: Path  = RAW_DIR / "test"

# Raw file names (the 7 challenge files)
RAW_TRAIN_FILES = [
    RAW_TRAIN_DIR / "train_source1.tsv",
    RAW_TRAIN_DIR / "train_source2.tsv",
    RAW_TRAIN_DIR / "train_source3.tsv",
    RAW_TRAIN_DIR / "train_ground_truth.tsv",
]
RAW_TEST_FILES = [
    RAW_TEST_DIR / "test_source1.tsv",
    RAW_TEST_DIR / "test_source2.tsv",
    RAW_TEST_DIR / "test_source3.tsv",
]
ALL_RAW_FILES = RAW_TRAIN_FILES + RAW_TEST_FILES

# ---------------------------------------------------------------------------
# Preprocessing outputs
# ---------------------------------------------------------------------------
PREPROCESSING_OUTPUT_DIR: Path  = REPO_ROOT / "output" / "preprocessing"
PREPROCESSED_DATA_DIR: Path     = PREPROCESSING_OUTPUT_DIR / "processed"
PREPROCESSING_METADATA_DIR: Path = PREPROCESSING_OUTPUT_DIR / "reports"

# Per-split preprocessed sub-directories
PREPROCESSED_TRAIN_DIR: Path = PREPROCESSED_DATA_DIR / "train"
PREPROCESSED_TEST_DIR:  Path = PREPROCESSED_DATA_DIR / "test"

# Checkpoint sentinel for preprocessing stage
CKPT_PREPROCESSING_DONE: Path = REPO_ROOT / "checkpoints" / "preprocessing_complete"
CKPT_BRIDGE_DONE:         Path = REPO_ROOT / "checkpoints" / "bridge_complete"

# ---------------------------------------------------------------------------
# Bridge / MLnAWS input (dataset/real/)
# ---------------------------------------------------------------------------
REAL_DIR: Path = REPO_ROOT / "dataset" / "real"

# Explicit file paths for contract validation
MATCHING_INPUT_FILES = {
    "train": [
        REAL_DIR / "clean_s1_train.tsv",
        REAL_DIR / "clean_s2s3_train.tsv",
        REAL_DIR / "feature_s1_train.tsv",
        REAL_DIR / "feature_s2s3_train.tsv",
        REAL_DIR / "ground_truth_train.tsv",
    ],
    "test": [
        REAL_DIR / "clean_s1_test.tsv",
        REAL_DIR / "clean_s2s3_test.tsv",
        REAL_DIR / "feature_s1_test.tsv",
        REAL_DIR / "feature_s2s3_test.tsv",
    ],
}

# ---------------------------------------------------------------------------
# Matching / ML outputs
# ---------------------------------------------------------------------------
MATCHING_OUTPUT_DIR: Path    = REPO_ROOT / "output" / "matching"
MATCHING_CHECKPOINT_DIR: Path = REPO_ROOT / "checkpoints" / "matching"

FINAL_MATCHING_RESULTS: Path  = MATCHING_OUTPUT_DIR / "matching_results.tsv"
FINAL_CANDIDATE_PAIRS:  Path  = MATCHING_OUTPUT_DIR / "candidate_pairs.tsv"

# Checkpoint sentinels for matching stages
CKPT_MATCHING_DONE: Path = REPO_ROOT / "checkpoints" / "matching_complete"

# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------
LOGS_DIR: Path = REPO_ROOT / "logs"

# ---------------------------------------------------------------------------
# Pipeline settings — safe defaults; can be overridden via CLI args
# ---------------------------------------------------------------------------
DEFAULT_WORKERS: int    = 8        # conservative for 16 vCPU (leaves headroom)
DEFAULT_CHUNK_SIZE: int = 100_000  # rows per preprocessing chunk
# NOTE: --chunk-size is accepted by run_pipeline.py CLI but preprocessing
# (preprocessing/src/preprocess.py) does not currently use chunked loading.
# The parameter is documented as "not wired through" to preprocessing.
# See run_pipeline._run_preprocessing() for details.
DEFAULT_TOP_K: int      = 50       # blocking top-k candidates per S1 entity
DEFAULT_NEG_RATIO: int  = 5        # negative:positive sampling ratio
DEFAULT_RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Parallelism / threading guard-rails
# ---------------------------------------------------------------------------
# These are environment variables that LightGBM and BLAS use for thread counts.
# Set conservatively to prevent CPU oversubscription when using joblib workers.
# Callers should apply these BEFORE importing lightgbm / numpy.
THREADING_ENV_VARS: dict[str, str] = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}

def apply_threading_env(n_workers: int = DEFAULT_WORKERS) -> None:
    """Set thread-count environment variables to prevent oversubscription.

    For n_workers parallel processes on 16 vCPUs, each process should use
    at most floor(16 / n_workers) threads for BLAS/OpenMP.  We cap at 4
    threads per process to leave headroom for the OS and I/O.

    Call this BEFORE importing lightgbm, numpy, or sklearn.
    """
    import math
    threads_per_worker = max(1, min(4, math.floor(16 / max(1, n_workers))))
    thread_str = str(threads_per_worker)
    for var in THREADING_ENV_VARS:
        os.environ.setdefault(var, thread_str)

# ---------------------------------------------------------------------------
# FAISS semantic retrieval configuration
# ---------------------------------------------------------------------------
# Embedding model — intfloat/multilingual-e5-small is a lightweight (117 MB)
# multilingual sentence-transformers model that produces 384-d embeddings.
# It runs on CPU without any GPU dependency.  All FAISS code reads this
# constant; do NOT hardcode the model name elsewhere.
FAISS_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"

# Output dimension of the above model.  Used to pre-allocate FAISS index
# structures and as part of cache-validity metadata.
FAISS_EMBEDDING_DIM: int = 384

# IVF nlist — number of Voronoi cells for the IVFFlat index.
# For ~10 M vectors a value of 1024 gives a good speed/recall tradeoff.
# For ~24 M vectors (S2+S3 combined) consider increasing to 4096 with
# FAISS_NPROBE=64 for better recall at the cost of ~4× index-build time.
# For small datasets (< nlist vectors) the code automatically falls back to
# IndexFlatIP (exact search) so this constant is safe for any dataset size.
FAISS_NLIST: int = 1_024

# Number of cells to probe at query time.  Higher → better recall, slower.
# At nlist=1024: nprobe=16 → ~1.6% cells probed, recall typically >90%.
# At nlist=4096 for 24M: use nprobe=64 for comparable recall.
FAISS_NPROBE: int = 16

# Normalization strategy: "l2" → L2-normalize embeddings before indexing/
# querying, turning inner-product into cosine similarity.
FAISS_NORMALIZATION: str = "l2"

# Index type tag written to metadata sidecar for cache-invalidation checks.
FAISS_INDEX_TYPE: str = "IVFFlat_IP"

# Default top-K for FAISS retrieval.  Overridable via --top-k CLI flag.
FAISS_TOP_K: int = DEFAULT_TOP_K  # mirrors the global blocking top-k (50)

# Sentence-transformers encoding batch size.  Keep conservative to avoid
# OOM on large corpora; overridable via --faiss-batch-size.
FAISS_BATCH_SIZE: int = 1_000

# Chunk size for adding vectors to the index in build_faiss_index().
# 500 k vectors per chunk × 384 dims × 4 bytes ≈ 768 MB peak per chunk,
# which is safe on a 128 GB machine while avoiding an all-at-once allocation.
FAISS_ADD_CHUNK_SIZE: int = 500_000

# ---------------------------------------------------------------------------
# FAISS index + metadata paths (under the matching checkpoint directory)
# ---------------------------------------------------------------------------
# The index is split into two files:
#   <base>.idx          — FAISS index metadata (written by faiss.write_index)
#   <base>.idx.ivfdata  — on-disk inverted lists (OnDiskInvertedLists)
#   <base>_metadata.json — cache-validity sidecar
#
# One index per split label (TRAIN / TEST) lives under the matching
# checkpoint directory so it participates in the normal checkpoint lifecycle.

def faiss_index_base(checkpoint_dir: "Path | str", split_label: str) -> Path:
    """Return the base path for a FAISS index (without extension).

    Files written:
        <base>.idx
        <base>.idx.ivfdata
        <base>_metadata.json
    """
    return Path(checkpoint_dir) / f"faiss_index_{split_label.upper()}"

def get_git_sha() -> str:
    """Return the current git commit SHA (short), or 'unknown' if unavailable."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Pipeline version
# ---------------------------------------------------------------------------
PIPELINE_VERSION: str = "1.0.0"
