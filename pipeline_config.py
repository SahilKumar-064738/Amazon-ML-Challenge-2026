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
# Pipeline version
# ---------------------------------------------------------------------------
PIPELINE_VERSION: str = "1.0.0"

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
