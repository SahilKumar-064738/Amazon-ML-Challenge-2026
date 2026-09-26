"""
pipeline_validation.py
----------------------
Inter-stage validation for the combined pipeline.

Provides validation functions that run:
  1. Before preprocessing  — raw dataset integrity
  2. After preprocessing   — cleaned dataset integrity
  3. Before MLnAWS         — MLnAWS input contract (identical to bridge output validation)
  4. After MLnAWS          — final output integrity

All validation failures raise ValueError with a clear, actionable message.
Non-fatal issues are returned as warnings in the result dict.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1 — Raw dataset validation
# ---------------------------------------------------------------------------

RAW_SOURCE_SCHEMA  = ["entity_id", "business_name", "business_address", "country"]
RAW_GT_SCHEMA      = ["source1_entity_id", "matched_entity_ids"]

_SOURCE_ID_PATTERNS = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}


def validate_raw_dataset(raw_dir: Path, splits: List[str] | None = None) -> dict:
    """Validate that all required raw input files exist and have correct schemas.

    Parameters
    ----------
    raw_dir : Path
        Root raw data directory (e.g. dataset/raw/).
        Expected layout: raw_dir/{split}/{split}_source{1,2,3}.tsv
                         raw_dir/train/train_ground_truth.tsv

    Returns
    -------
    dict
        row_counts per file; raises on any fatal error.
    """
    if splits is None:
        splits = ["train", "test"]

    result: dict = {"splits": {}}

    for split in splits:
        split_dir = raw_dir / split
        if not split_dir.exists():
            raise ValueError(
                f"raw validation: expected directory not found: {split_dir}\n"
                f"  Place raw files in {split_dir}/"
            )

        sources = {
            "source1": split_dir / f"{split}_source1.tsv",
            "source2": split_dir / f"{split}_source2.tsv",
            "source3": split_dir / f"{split}_source3.tsv",
        }
        if split == "train":
            gt_path = split_dir / f"{split}_ground_truth.tsv"

        # Check existence
        for key, path in sources.items():
            if not path.exists():
                raise ValueError(
                    f"raw validation: missing {split}/{path.name}\n"
                    f"  Expected at: {path}"
                )

        if split == "train" and not gt_path.exists():
            raise ValueError(
                f"raw validation: missing {split}/ground_truth: {gt_path}"
            )

        split_result: dict = {}

        # Schema and quick sanity check (read first chunk only — fast)
        for key, path in sources.items():
            chunk = pd.read_csv(path, sep="\t", dtype=str,
                                keep_default_na=False, na_values=[], nrows=5)
            actual = list(chunk.columns)
            if actual != RAW_SOURCE_SCHEMA:
                raise ValueError(
                    f"raw validation: {path.name} has wrong schema.\n"
                    f"  Expected: {RAW_SOURCE_SCHEMA}\n"
                    f"  Got:      {actual}"
                )
            # Check ID prefix
            prefix = _SOURCE_ID_PATTERNS[key]
            bad_ids = chunk[~chunk["entity_id"].str.startswith(prefix)]
            if not bad_ids.empty:
                raise ValueError(
                    f"raw validation: {path.name} has entity_id values not starting with '{prefix}'.\n"
                    f"  Examples: {bad_ids['entity_id'].head(3).tolist()}"
                )

            # Count rows cheaply (line count - 1 for header)
            n_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
            split_result[path.name] = n_rows
            logger.info("  raw: %s/%s — %d rows [OK]", split, path.name, n_rows)

        if split == "train":
            chunk = pd.read_csv(gt_path, sep="\t", dtype=str,
                                keep_default_na=False, na_values=[], nrows=5)
            actual = list(chunk.columns)
            if actual != RAW_GT_SCHEMA:
                raise ValueError(
                    f"raw validation: {gt_path.name} has wrong schema.\n"
                    f"  Expected: {RAW_GT_SCHEMA}\n"
                    f"  Got:      {actual}"
                )
            n_gt = sum(1 for _ in open(gt_path, encoding="utf-8")) - 1
            split_result[gt_path.name] = n_gt
            logger.info("  raw: train/ground_truth — %d rows [OK]", n_gt)

        result["splits"][split] = split_result

    result["status"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Stage 2 — Preprocessed dataset validation
# ---------------------------------------------------------------------------

PREPROCESSED_SOURCE_SCHEMA = [
    "entity_id",
    "business_name", "business_name_normalized", "business_name_canonical",
    "business_address", "business_address_normalized", "business_address_canonical",
    "address_landmark", "address_numbers", "address_postal_code",
    "address_sorted_tokens", "country", "country_normalized",
    "business_name_is_missing", "business_address_is_missing",
    "name_script_class",
]
PREPROCESSED_GT_SCHEMA = ["source1_entity_id", "matched_entity_ids"]


def validate_preprocessed_dataset(preprocessed_dir: Path,
                                   splits: List[str] | None = None) -> dict:
    """Validate MLnNor's output files before the bridge layer runs."""
    if splits is None:
        splits = ["train", "test"]

    result: dict = {"splits": {}}

    for split in splits:
        split_dir = preprocessed_dir / split
        if not split_dir.exists():
            raise ValueError(
                f"preprocessed validation: directory not found: {split_dir}"
            )

        split_result: dict = {}

        for src in ("source1_clean.tsv", "source2_clean.tsv", "source3_clean.tsv"):
            path = split_dir / src
            if not path.exists():
                raise ValueError(
                    f"preprocessed validation: missing {split}/{src}"
                )
            chunk = pd.read_csv(path, sep="\t", dtype=str,
                                keep_default_na=False, na_values=[], nrows=5)
            actual = list(chunk.columns)
            if actual != PREPROCESSED_SOURCE_SCHEMA:
                raise ValueError(
                    f"preprocessed validation: {src} has wrong schema.\n"
                    f"  Expected: {PREPROCESSED_SOURCE_SCHEMA}\n"
                    f"  Got:      {actual}"
                )
            n_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
            split_result[src] = n_rows
            logger.info("  preprocessed: %s/%s — %d rows [OK]", split, src, n_rows)

        if split == "train":
            gt_path = split_dir / "ground_truth_clean.tsv"
            if not gt_path.exists():
                raise ValueError(
                    f"preprocessed validation: missing {split}/ground_truth_clean.tsv"
                )
            chunk = pd.read_csv(gt_path, sep="\t", dtype=str,
                                keep_default_na=False, na_values=[], nrows=5)
            actual = list(chunk.columns)
            if actual != PREPROCESSED_GT_SCHEMA:
                raise ValueError(
                    f"preprocessed validation: ground_truth_clean.tsv wrong schema.\n"
                    f"  Expected: {PREPROCESSED_GT_SCHEMA}\n"
                    f"  Got:      {actual}"
                )
            n_gt = sum(1 for _ in open(gt_path, encoding="utf-8")) - 1
            split_result["ground_truth_clean.tsv"] = n_gt
            logger.info("  preprocessed: train/ground_truth_clean — %d rows [OK]", n_gt)

        result["splits"][split] = split_result

    result["status"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Stage 3 — MLnAWS input contract (repeated here for clarity in the orchestrator)
# ---------------------------------------------------------------------------

MATCHING_INPUT_SCHEMAS = {
    "clean":   ["entity_id", "clean_text"],
    "feature": ["entity_id", "clean_name", "clean_address", "clean_country"],
    "gt":      ["source1_entity_id", "matching_entity_ids"],
}


def validate_matching_inputs(real_dir: Path,
                              splits: List[str] | None = None) -> dict:
    """Validate that all MLnAWS input files exist with correct schemas."""
    if splits is None:
        splits = ["train", "test"]

    result: dict = {"splits": {}}

    for split in splits:
        files: list[tuple[str, list[str]]] = [
            (f"clean_s1_{split}.tsv",     MATCHING_INPUT_SCHEMAS["clean"]),
            (f"clean_s2s3_{split}.tsv",   MATCHING_INPUT_SCHEMAS["clean"]),
            (f"feature_s1_{split}.tsv",   MATCHING_INPUT_SCHEMAS["feature"]),
            (f"feature_s2s3_{split}.tsv", MATCHING_INPUT_SCHEMAS["feature"]),
        ]
        if split == "train":
            files.append(
                (f"ground_truth_{split}.tsv", MATCHING_INPUT_SCHEMAS["gt"])
            )

        split_result: dict = {}
        for filename, expected_cols in files:
            path = real_dir / filename
            if not path.exists():
                raise ValueError(
                    f"matching input validation: required file missing: {path}\n"
                    f"  Run the preprocessing stage first."
                )
            chunk = pd.read_csv(path, sep="\t", dtype=str,
                                keep_default_na=False, na_values=[], nrows=1)
            actual = list(chunk.columns)
            if actual != expected_cols:
                raise ValueError(
                    f"matching input validation: {filename} wrong schema.\n"
                    f"  Expected: {expected_cols}\n"
                    f"  Got:      {actual}"
                )

            # Quick sanity: no empty entity_id
            id_col = expected_cols[0]
            chunk_full = pd.read_csv(path, sep="\t", dtype=str,
                                     keep_default_na=False, na_values=[], nrows=100)
            empty_ids = (chunk_full[id_col].str.strip() == "").sum()
            if empty_ids > 0:
                raise ValueError(
                    f"matching input validation: {filename} has {empty_ids} empty "
                    f"'{id_col}' values in the first 100 rows."
                )

            n_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
            split_result[filename] = n_rows
            logger.info("  contract: %s — %d rows [OK]", filename, n_rows)

        result["splits"][split] = split_result

    result["status"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Stage 4 — Final output validation
# ---------------------------------------------------------------------------

MATCHING_OUTPUT_SCHEMAS = {
    "matching_results.tsv":  ["source1_entity_id", "matched_entity_ids"],
    "candidate_pairs.tsv":   ["source1_entity_id", "candidate_entity_ids"],
}


def validate_final_outputs(output_dir: Path) -> dict:
    """Validate that the MLnAWS final output files exist and are non-trivially populated."""
    result: dict = {}
    warnings: list[str] = []

    for filename, expected_cols in MATCHING_OUTPUT_SCHEMAS.items():
        path = output_dir / filename
        if not path.exists():
            raise ValueError(
                f"final output validation: expected file not found: {path}"
            )
        if path.stat().st_size == 0:
            raise ValueError(
                f"final output validation: output file is empty: {path}"
            )

        chunk = pd.read_csv(path, sep="\t", dtype=str,
                            keep_default_na=False, na_values=[], nrows=5)
        actual = list(chunk.columns)
        if actual != expected_cols:
            raise ValueError(
                f"final output validation: {filename} wrong schema.\n"
                f"  Expected: {expected_cols}\n"
                f"  Got:      {actual}"
            )

        n_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        if n_rows == 0:
            raise ValueError(
                f"final output validation: {filename} has 0 data rows (only header)."
            )

        result[filename] = n_rows
        logger.info("  output: %s — %d rows [OK]", filename, n_rows)

    result["warnings"] = warnings
    result["status"] = "ok"
    return result
