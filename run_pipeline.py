#!/usr/bin/env python3
"""
run_pipeline.py
---------------
Single entrypoint for the combined Business Entity Resolution pipeline.

Usage
-----
    python run_pipeline.py                   # full pipeline (default)
    python run_pipeline.py --help            # show all options
    python run_pipeline.py --force           # re-run from scratch
    python run_pipeline.py --skip-preprocessing  # skip to matching stage
    python run_pipeline.py --skip-matching       # run preprocessing only
    python run_pipeline.py --resume              # resume from last checkpoint
    python run_pipeline.py --skip-bm25           # disable BM25 (saves ~8-12 GB RAM)
    python run_pipeline.py --workers 8           # override CPU workers
    python run_pipeline.py --top-k 75            # increase blocking candidates

Pipeline stages
---------------
  [01/08] Validate environment
  [02/08] Validate raw dataset
  [03/08] Run preprocessing (MLnNor)
  [04/08] Run bridge (convert preprocessed -> MLnAWS format)
  [05/08] Validate MLnAWS input contract
  [06/08] Run entity-resolution pipeline (MLnAWS)
  [07/08] Validate final outputs
  [08/08] Write pipeline report

Input (place raw files here):
    dataset/raw/train/train_source{1,2,3}.tsv
    dataset/raw/train/train_ground_truth.tsv
    dataset/raw/test/test_source{1,2,3}.tsv

Outputs:
    output/matching/matching_results.tsv
    output/matching/candidate_pairs.tsv
    output/pipeline_report.json
    output/preprocessing/reports/preprocessing_report.json
    logs/pipeline_<timestamp>.log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Set threading env vars BEFORE importing numpy/lightgbm to prevent
# CPU oversubscription on 16 vCPU / 128 GB GCP machine.
# ---------------------------------------------------------------------------
_early_workers = next(
    (int(sys.argv[sys.argv.index("--workers") + 1])
     for i, a in enumerate(sys.argv)
     if a == "--workers" and i + 1 < len(sys.argv)),
    8,
)
import math as _math
_threads = str(max(1, min(4, _math.floor(16 / max(1, _early_workers)))))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _threads)

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path regardless of invocation directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Now import pipeline modules (after env setup)
# ---------------------------------------------------------------------------
from pipeline_config import (
    RAW_DIR, REAL_DIR, PREPROCESSED_DATA_DIR, PREPROCESSING_METADATA_DIR,
    MATCHING_OUTPUT_DIR, MATCHING_CHECKPOINT_DIR, LOGS_DIR,
    DEFAULT_WORKERS, DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, DEFAULT_NEG_RATIO,
    PIPELINE_VERSION, get_git_sha,
)
from pipeline_logging import setup_logging, get_logger, StepTimer
from pipeline_checkpoints import CheckpointStore
from pipeline_validation import (
    validate_raw_dataset,
    validate_preprocessed_dataset,
    validate_matching_inputs,
    validate_final_outputs,
)
from preprocessing.bridge import run_bridge, validate_contract

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Business Entity Resolution — combined pipeline.\n"
            "Runs preprocessing (MLnNor) followed by matching (MLnAWS) "
            "from a single command."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input-dir", default=None, metavar="DIR",
        help=(
            "Root directory containing raw data in train/ and test/ subdirs. "
            f"Default: dataset/raw/ (inside repo root)."
        ),
    )
    p.add_argument(
        "--output-dir", default=None, metavar="DIR",
        help="Override the output directory for final results. Default: output/matching/.",
    )
    p.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
        help=f"CPU workers for feature extraction and LightGBM (default: {DEFAULT_WORKERS}). "
             "Capped at 16 for a 16-vCPU machine.",
    )
    p.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, metavar="N",
        help=f"Rows per preprocessing chunk (default: {DEFAULT_CHUNK_SIZE}). "
             "Reduce to save RAM; increase for throughput.",
    )
    p.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, metavar="K",
        help=f"Blocking candidates per S1 entity (default: {DEFAULT_TOP_K}). "
             "Increase to 75-100 if recall is low.",
    )
    p.add_argument(
        "--neg-ratio", type=int, default=DEFAULT_NEG_RATIO, metavar="R",
        help=f"Negative:positive sampling ratio for training (default: {DEFAULT_NEG_RATIO}).",
    )
    p.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip the preprocessing stage. The cleaned dataset in dataset/real/ "
             "must already exist and will be validated before matching starts.",
    )
    p.add_argument(
        "--skip-matching", action="store_true",
        help="Stop after preprocessing + bridge. Useful for validating preprocessing "
             "output before committing to a long ML run.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Ignore ALL existing checkpoints and re-run every stage from scratch. "
             "WARNING: this deletes existing output files.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from the last successful checkpoint (default behaviour). "
             "This flag is provided for clarity; resumption is already the default.",
    )
    p.add_argument(
        "--skip-bm25", action="store_true",
        help="Disable BM25 blocking (use TF-IDF only). Saves ~8-12 GB RAM. "
             "Recommended for machines with <= 32 GB RAM.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to process (default: train test).",
    )
    p.add_argument(
        "--checkpoint-dir", default=None, metavar="DIR",
        help="Override the checkpoint directory. Default: checkpoints/ inside repo root.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate environment and raw dataset, then exit without processing.",
    )
    return p


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _run_preprocessing(
    raw_dir: Path,
    preprocessed_dir: Path,
    metadata_dir: Path,
    splits: list[str],
    chunk_size: int,
    log,
) -> dict:
    """Run MLnNor preprocessing stage.

    NOTE — ``chunk_size`` parameter
    --------------------------------
    The ``--chunk-size`` CLI option is accepted by the combined pipeline for
    forward-compatibility but is **not currently passed through** to
    ``preprocessing.src.preprocess.run()``.  That function loads each split
    entirely into RAM; chunked streaming would require a more significant
    redesign of the preprocessing layer.

    On the target 128 GB machine the full 2.5 GB dataset loads comfortably.
    If chunked preprocessing becomes necessary in the future, implement it
    inside ``preprocessing/src/preprocess.py`` and thread the parameter
    through here.
    """
    log.info("Running MLnNor preprocessing...")
    if chunk_size != DEFAULT_CHUNK_SIZE:
        log.warning(
            "--chunk-size %d specified but preprocessing does not currently "
            "use chunked loading; the parameter is ignored for this stage.",
            chunk_size,
        )

    # Import preprocessing here (after sys.path setup and env vars)
    from preprocessing.src.preprocess import run as preprocess_run
    from preprocessing.src.validation import FatalValidationError

    # MLnNor expects input as: input_dir/{split}/{split}_source{1,2,3}.tsv
    # raw_dir already has that layout (raw/train/train_source1.tsv, etc.)
    # We also need to set chunk size — MLnNor currently loads full splits.
    # For the 2.5 GB dataset on 128 GB RAM, each split loads comfortably.
    # Chunked processing is handled by the bridge layer which streams outputs.

    try:
        report = preprocess_run(
            input_dir=raw_dir,
            output_processed_dir=preprocessed_dir,
            output_metadata_dir=metadata_dir,
            splits=splits,
        )
    except FatalValidationError as e:
        raise RuntimeError(
            f"Preprocessing failed with a validation error:\n{e}"
        ) from e

    return report


def _run_bridge_stage(
    preprocessed_dir: Path,
    real_dir: Path,
    splits: list[str],
    log,
) -> dict:
    """Run the bridge layer: convert MLnNor outputs to MLnAWS inputs."""
    log.info("Running bridge (converting preprocessed -> MLnAWS format)...")
    report = run_bridge(
        preprocessed_dir=preprocessed_dir,
        real_dir=real_dir,
        splits=splits,
    )
    return report


def _run_matching(
    real_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path,
    workers: int,
    top_k: int,
    neg_ratio: int,
    skip_bm25: bool,
    splits: list[str],
    force: bool,
    log,
) -> None:
    """Run the MLnAWS entity-resolution pipeline by importing run_production logic."""
    log.info("Running MLnAWS entity-resolution pipeline...")

    # Ensure the matching src is importable
    matching_src = REPO_ROOT / "matching" / "src"
    if str(matching_src.parent) not in sys.path:
        sys.path.insert(0, str(matching_src.parent))

    # Import the matching pipeline
    # run_production.py uses ARGS = _parse_args() at module level which reads
    # sys.argv. We call its main() equivalent by injecting the arguments
    # we want and invoking the pipeline logic directly, or we subprocess it
    # with the right args. The cleanest approach: call run_production.main()
    # after adjusting sys.argv.

    # Determine which split mode to use
    need_train = "train" in splits
    need_test  = "test" in splits
    if need_train and need_test:
        split_arg = "all"
    elif need_train:
        split_arg = "train"
    else:
        split_arg = "test"

    # Rebuild sys.argv for run_production's argument parser
    matching_argv = [
        "run_production.py",
        "--split", split_arg,
        "--top-k", str(top_k),
        "--workers", str(workers),
        "--data-dir", str(real_dir),
        "--output-dir", str(output_dir),
        "--checkpoint-dir", str(checkpoint_dir),
        "--neg-ratio", str(neg_ratio),
    ]
    if skip_bm25:
        matching_argv.append("--skip-bm25")
    if force:
        matching_argv.append("--force")

    old_argv = sys.argv[:]
    old_cwd  = os.getcwd()
    try:
        sys.argv = matching_argv
        # Change working directory so relative paths in run_production.py
        # (which uses os.path.join("dataset", "real", ...)) still resolve.
        os.chdir(str(REPO_ROOT))

        # Add matching directory to path for "from src.X import Y" style imports
        matching_root = str(REPO_ROOT / "matching")
        if matching_root not in sys.path:
            sys.path.insert(0, matching_root)

        # Import and run
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_production",
            str(REPO_ROOT / "matching" / "run_production.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()

    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_pipeline_report(
    report_data: dict,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pipeline_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    pipeline_start = time.time()

    parser = _build_parser()
    args = parser.parse_args()

    # Mutually exclusive sanity check
    if args.skip_preprocessing and args.skip_matching:
        parser.error("--skip-preprocessing and --skip-matching cannot both be set.")

    # Resolve paths
    raw_dir         = Path(args.input_dir).resolve() if args.input_dir else RAW_DIR
    output_dir      = Path(args.output_dir).resolve() if args.output_dir else MATCHING_OUTPUT_DIR
    checkpoint_dir  = (
        Path(args.checkpoint_dir).resolve() if args.checkpoint_dir
        else REPO_ROOT / "checkpoints"
    )
    matching_ckpt_dir = checkpoint_dir / "matching"

    # Workers: cap at 16 for a 16-vCPU machine
    workers = min(16, max(1, args.workers))

    # Setup logging
    log_path = setup_logging(logs_dir=LOGS_DIR)
    log = get_logger("pipeline")

    log.info("=" * 60)
    log.info("Business Entity Resolution — Combined Pipeline")
    log.info("  Version      : %s", PIPELINE_VERSION)
    log.info("  Git SHA      : %s", get_git_sha())
    log.info("  Raw data     : %s", raw_dir)
    log.info("  Output       : %s", output_dir)
    log.info("  Checkpoints  : %s", checkpoint_dir)
    log.info("  Workers      : %d", workers)
    log.info("  Top-k        : %d", args.top_k)
    log.info("  BM25         : %s", "disabled" if args.skip_bm25 else "enabled")
    log.info("  Log file     : %s", log_path)
    log.info("=" * 60)

    store = CheckpointStore(checkpoint_dir)
    if args.force:
        log.info("[force] Clearing all checkpoints for full re-run.")
        store.clear_all()

    # Pipeline report accumulator
    report: dict = {
        "pipeline_version":   PIPELINE_VERSION,
        "git_sha":            get_git_sha(),
        "started_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "raw_dir":        str(raw_dir),
            "output_dir":     str(output_dir),
            "workers":        workers,
            "top_k":          args.top_k,
            "chunk_size":     args.chunk_size,
            "neg_ratio":      args.neg_ratio,
            "skip_bm25":      args.skip_bm25,
            "splits":         args.splits,
        },
        "stages": {},
        "status": "running",
    }

    n_stages = 8
    stage_num = 0

    def log_stage(n: int, name: str) -> None:
        log.info("[%02d/%02d] %s", n, n_stages, name)

    try:
        # ── [01/08] Validate environment ─────────────────────────────────
        stage_num = 1
        log_stage(stage_num, "Validating environment...")
        with StepTimer("env validation", log):
            python_ver = sys.version.split()[0]
            if tuple(int(x) for x in python_ver.split(".")[:2]) < (3, 10):
                raise RuntimeError(
                    f"Python 3.10+ required, found {python_ver}"
                )
            # Verify required packages importable
            _check_imports()
            log.info("  Python %s — environment OK", python_ver)
        report["stages"]["01_env"] = {"status": "ok", "python": python_ver}

        # ── [02/08] Validate raw dataset ─────────────────────────────────
        stage_num = 2
        log_stage(stage_num, "Validating raw dataset...")
        with StepTimer("raw validation", log):
            raw_info = validate_raw_dataset(raw_dir, splits=args.splits)
        report["stages"]["02_raw_validation"] = raw_info
        log.info("  Raw dataset validated OK")

        if args.dry_run:
            log.info("  --dry-run: stopping after raw validation.")
            report["status"] = "dry_run_complete"
            _finalize_report(report, pipeline_start, output_dir)
            return 0

        # ── [03/08] Preprocessing ─────────────────────────────────────────
        stage_num = 3
        log_stage(stage_num, "Preprocessing (MLnNor)...")

        if args.skip_preprocessing:
            log.info("  --skip-preprocessing: verifying that preprocessed data exists...")
            with StepTimer("preprocessed validation", log):
                prep_info = validate_preprocessed_dataset(
                    PREPROCESSED_DATA_DIR, splits=args.splits
                )
            report["stages"]["03_preprocessing"] = {"status": "skipped", **prep_info}
            log.info("  Preprocessed data found and validated OK")

        elif store.is_done(CheckpointStore.PREPROCESSING, force=args.force):
            log.info("  [SKIP] Preprocessing (checkpoint found)")
            report["stages"]["03_preprocessing"] = {"status": "resumed_from_checkpoint"}

        else:
            with StepTimer("preprocessing", log):
                prep_report = _run_preprocessing(
                    raw_dir=raw_dir,
                    preprocessed_dir=PREPROCESSED_DATA_DIR,
                    metadata_dir=PREPROCESSING_METADATA_DIR,
                    splits=args.splits,
                    chunk_size=args.chunk_size,
                    log=log,
                )

            log.info("  Validating preprocessed outputs...")
            with StepTimer("preprocessed validation", log):
                validate_preprocessed_dataset(
                    PREPROCESSED_DATA_DIR, splits=args.splits
                )

            store.mark_done(CheckpointStore.PREPROCESSING)
            report["stages"]["03_preprocessing"] = {
                "status": "completed",
                "report": prep_report,
            }
            log.info("  Preprocessing complete")

        # ── [04/08] Bridge ────────────────────────────────────────────────
        stage_num = 4
        log_stage(stage_num, "Bridge (preprocessed -> MLnAWS format)...")

        if args.skip_preprocessing and not store.is_done(CheckpointStore.BRIDGE, force=False):
            # When skipping preprocessing, we still need to check if the
            # bridge outputs exist. If they do, skip the bridge too.
            try:
                validate_contract(REAL_DIR, splits=args.splits)
                log.info("  Bridge outputs exist and are valid — skipping bridge.")
                report["stages"]["04_bridge"] = {"status": "skipped_outputs_exist"}
            except ValueError:
                # Bridge outputs don't exist; run the bridge from preprocessed data
                log.info("  Bridge outputs absent — running bridge from preprocessed data.")
                with StepTimer("bridge", log):
                    bridge_report = _run_bridge_stage(
                        PREPROCESSED_DATA_DIR, REAL_DIR, args.splits, log
                    )
                store.mark_done(CheckpointStore.BRIDGE)
                report["stages"]["04_bridge"] = {"status": "completed", "report": bridge_report}

        elif store.is_done(CheckpointStore.BRIDGE, force=args.force):
            log.info("  [SKIP] Bridge (checkpoint found)")
            report["stages"]["04_bridge"] = {"status": "resumed_from_checkpoint"}

        else:
            with StepTimer("bridge", log):
                bridge_report = _run_bridge_stage(
                    PREPROCESSED_DATA_DIR, REAL_DIR, args.splits, log
                )
            store.mark_done(CheckpointStore.BRIDGE)
            report["stages"]["04_bridge"] = {
                "status": "completed",
                "report": bridge_report,
            }
            log.info("  Bridge complete")

        # ── [05/08] Validate MLnAWS input contract ────────────────────────
        stage_num = 5
        log_stage(stage_num, "Validating MLnAWS input contract...")
        with StepTimer("contract validation", log):
            contract_info = validate_matching_inputs(REAL_DIR, splits=args.splits)
        store.mark_done(CheckpointStore.CONTRACT_VALIDATED)
        report["stages"]["05_contract"] = contract_info
        log.info("  MLnAWS input contract validated OK")

        if args.skip_matching:
            log.info("  --skip-matching: stopping after preprocessing.")
            report["status"] = "preprocessing_only_complete"
            path = _finalize_report(report, pipeline_start, output_dir)
            log.info("Pipeline report written to: %s", path)
            log.info("To run matching: python run_pipeline.py --skip-preprocessing")
            return 0

        # ── [06/08] MLnAWS entity-resolution ─────────────────────────────
        stage_num = 6
        log_stage(stage_num, "Running entity-resolution pipeline (MLnAWS)...")
        with StepTimer("MLnAWS matching", log):
            _run_matching(
                real_dir=REAL_DIR,
                output_dir=output_dir,
                checkpoint_dir=matching_ckpt_dir,
                workers=workers,
                top_k=args.top_k,
                neg_ratio=args.neg_ratio,
                skip_bm25=args.skip_bm25,
                splits=args.splits,
                force=args.force,
                log=log,
            )
        store.mark_done(CheckpointStore.MATCHING_COMPLETE)
        report["stages"]["06_matching"] = {"status": "completed"}
        log.info("  MLnAWS pipeline complete")

        # ── [07/08] Validate final outputs ───────────────────────────────
        stage_num = 7
        log_stage(stage_num, "Validating final outputs...")
        with StepTimer("final output validation", log):
            final_info = validate_final_outputs(output_dir)
        store.mark_done(CheckpointStore.FINAL_VALIDATED)
        report["stages"]["07_final_validation"] = final_info
        log.info("  Final outputs validated OK")

        # ── [08/08] Write pipeline report ────────────────────────────────
        stage_num = 8
        log_stage(stage_num, "Writing pipeline report...")
        report["status"] = "completed"
        path = _finalize_report(report, pipeline_start, REPO_ROOT / "output")
        log.info("  Pipeline report: %s", path)

        total_elapsed = time.time() - pipeline_start
        log.info("=" * 60)
        log.info("PIPELINE COMPLETED SUCCESSFULLY in %.1fs (%.1f min)",
                 total_elapsed, total_elapsed / 60)
        log.info("  Matching results : %s", output_dir / "matching_results.tsv")
        log.info("  Candidate pairs  : %s", output_dir / "candidate_pairs.tsv")
        log.info("  Pipeline report  : %s", path)
        log.info("=" * 60)
        return 0

    except KeyboardInterrupt:
        log.warning("\nInterrupted by user (Ctrl+C).")
        log.warning("Progress is saved. Re-run without --force to resume.")
        report["status"] = "interrupted"
        report["interrupted_at_stage"] = stage_num
        _finalize_report(report, pipeline_start, REPO_ROOT / "output")
        return 130

    except (Exception, SystemExit) as exc:  # noqa: BLE001
        # SystemExit derives from BaseException, not Exception, so it must be
        # listed explicitly to be caught here.  We do NOT catch BaseException
        # broadly because that would swallow KeyboardInterrupt.
        exit_code = 1
        if isinstance(exc, SystemExit):
            # Preserve the original exit code if it is a non-zero integer.
            exit_code = exc.code if isinstance(exc.code, int) and exc.code != 0 else 1
        log.error("=" * 60)
        log.error("PIPELINE FAILED at stage [%02d/%02d]", stage_num, n_stages)
        log.error("Error: %s", exc)
        log.error("=" * 60)
        log.exception("Full traceback:")
        report["status"] = "failed"
        report["error"] = str(exc)
        report["failed_at_stage"] = stage_num
        try:
            _finalize_report(report, pipeline_start, REPO_ROOT / "output")
        except Exception:
            pass
        return exit_code


def _check_imports() -> None:
    """Verify all required packages are importable."""
    required = [
        ("pandas",    "pandas"),
        ("numpy",     "numpy"),
        ("sklearn",   "scikit-learn"),
        ("lightgbm",  "lightgbm"),
        ("rapidfuzz", "rapidfuzz"),
    ]
    missing = []
    for module, package in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            f"Required packages not installed: {missing}\n"
            f"  Run: pip install -r requirements.txt"
        )


def _finalize_report(report: dict, pipeline_start: float, output_dir: Path) -> Path:
    """Add timing info and write the JSON report."""
    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["total_duration_seconds"] = round(time.time() - pipeline_start, 2)
    return _write_pipeline_report(report, output_dir)


if __name__ == "__main__":
    sys.exit(main())
