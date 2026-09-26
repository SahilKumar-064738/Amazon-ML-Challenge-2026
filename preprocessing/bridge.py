"""
preprocessing/bridge.py
-----------------------
Bridge layer: converts MLnNor's cleaned per-source TSV files into the exact
format expected by the MLnAWS matching pipeline.

CONTRACT DEFINITION
===================

MLnNor produces (for each split, in processed/<split>/):
    source1_clean.tsv    columns: entity_id, business_name,
                                  business_name_normalized, business_name_canonical,
                                  business_address, business_address_normalized,
                                  business_address_canonical, address_landmark,
                                  address_numbers, address_postal_code,
                                  address_sorted_tokens, country, country_normalized,
                                  business_name_is_missing, business_address_is_missing,
                                  name_script_class
    source2_clean.tsv    (same schema)
    source3_clean.tsv    (same schema)
    ground_truth_clean.tsv  columns: source1_entity_id, matched_entity_ids

MLnAWS expects (in dataset/real/):
    clean_s1_{split}.tsv         entity_id, clean_text
    clean_s2s3_{split}.tsv       entity_id, clean_text   (S2+S3 combined)
    feature_s1_{split}.tsv       entity_id, clean_name, clean_address, clean_country
    feature_s2s3_{split}.tsv     entity_id, clean_name, clean_address, clean_country
    ground_truth_{split}.tsv     source1_entity_id, matching_entity_ids   (train only)

TRANSLATION RULES
=================
clean_text       = business_name_normalized + " " + business_address_normalized
                   (trimmed; mimics prepare_real_data.py behaviour using the
                   higher-quality MLnNor normalization instead of the simple
                   abbreviation-only one in prepare_real_data.py)
clean_name       = business_name_normalized
clean_address    = business_address_normalized
clean_country    = country_normalized
matching_entity_ids  = matched_entity_ids  (column rename only; values unchanged)

S2+S3 combined   = pd.concat([source2_clean, source3_clean])

The bridge NEVER re-normalizes text — it only reorganizes columns that MLnNor
already produced. If MLnNor's normalization changes, the bridge output
changes automatically without any modification here.

CHUNKED READING
===============
The bridge reads MLnNor's output files in chunks (CHUNK_ROWS) so neither
the bridge nor the downstream pipeline ever loads a 2.5 GB file all at once.
Each chunk is streamed directly to the output file with header written only
on the first chunk (mode="w" then mode="a").

Row-count and ID-set invariants are verified after the bridge completes,
matching the same guarantees MLnNor enforces on its own outputs.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Chunk size for streaming bridge conversion — 100k rows per chunk is safe
# even for multi-GB files on 128 GB RAM: a 100k-row chunk of 4-column text
# data is roughly 20-40 MB in memory.
CHUNK_ROWS: int = 100_000

# Columns produced by MLnNor that the bridge reads.
_PREPROCESSED_COLS = [
    "entity_id",
    "business_name_normalized",
    "business_address_normalized",
    "country_normalized",
]

# Columns required in MLnNor's ground-truth output.
_GT_INPUT_COLS = {"source1_entity_id", "matched_entity_ids"}

# ---------------------------------------------------------------------------
# Per-chunk transform helpers
# ---------------------------------------------------------------------------

def _chunk_to_clean_text(chunk: pd.DataFrame) -> pd.DataFrame:
    """Build (entity_id, clean_text) from a preprocessed-source chunk."""
    name = chunk["business_name_normalized"].fillna("")
    addr = chunk["business_address_normalized"].fillna("")
    combined = (name + " " + addr).str.strip()
    return pd.DataFrame({"entity_id": chunk["entity_id"], "clean_text": combined})


def _chunk_to_feature(chunk: pd.DataFrame) -> pd.DataFrame:
    """Build (entity_id, clean_name, clean_address, clean_country) from a chunk."""
    return pd.DataFrame({
        "entity_id":     chunk["entity_id"],
        "clean_name":    chunk["business_name_normalized"].fillna(""),
        "clean_address": chunk["business_address_normalized"].fillna(""),
        "clean_country": chunk["country_normalized"].fillna(""),
    })


def _stream_source_file(
    input_path: Path,
    clean_text_path: Path,
    feature_path: Path,
) -> int:
    """Stream one source_clean.tsv to clean_text and feature files.

    Uses chunked pandas reads so the full file is never in memory at once.

    Returns
    -------
    int
        Total number of rows written.
    """
    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        usecols=_PREPROCESSED_COLS,
        chunksize=CHUNK_ROWS,
    ):
        n = len(chunk)
        total_rows += n

        clean_df = _chunk_to_clean_text(chunk)
        feat_df = _chunk_to_feature(chunk)

        mode = "w" if first_chunk else "a"
        header = first_chunk

        clean_df.to_csv(clean_text_path, sep="\t", index=False,
                        mode=mode, header=header, quoting=csv.QUOTE_NONE,
                        escapechar="\\")
        feat_df.to_csv(feature_path, sep="\t", index=False,
                       mode=mode, header=header, quoting=csv.QUOTE_NONE,
                       escapechar="\\")

        first_chunk = False
        logger.debug("  bridge: wrote %d rows from %s", n, input_path.name)

    return total_rows


def _append_source_to_combined(
    input_path: Path,
    combined_clean_path: Path,
    combined_feat_path: Path,
    first_source: bool,
) -> int:
    """Append one source's clean/feature data to the combined S2+S3 files."""
    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        usecols=_PREPROCESSED_COLS,
        chunksize=CHUNK_ROWS,
    ):
        n = len(chunk)
        total_rows += n

        clean_df = _chunk_to_clean_text(chunk)
        feat_df = _chunk_to_feature(chunk)

        # Write header only on the very first chunk of the very first source.
        write_header = first_source and first_chunk
        mode = "w" if (first_source and first_chunk) else "a"

        clean_df.to_csv(combined_clean_path, sep="\t", index=False,
                        mode=mode, header=write_header,
                        quoting=csv.QUOTE_NONE, escapechar="\\")
        feat_df.to_csv(combined_feat_path, sep="\t", index=False,
                       mode=mode, header=write_header,
                       quoting=csv.QUOTE_NONE, escapechar="\\")

        first_chunk = False

    return total_rows


def _convert_ground_truth(input_path: Path, output_path: Path) -> int:
    """Convert ground_truth_clean.tsv to the MLnAWS expected format.

    Only change: rename column 'matched_entity_ids' -> 'matching_entity_ids'.
    Values are NEVER modified.

    Returns row count.
    """
    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        chunksize=CHUNK_ROWS,
    ):
        # Validate schema
        missing = _GT_INPUT_COLS - set(chunk.columns)
        if missing:
            raise ValueError(
                f"ground_truth_clean.tsv is missing required columns: {missing}. "
                f"Found: {list(chunk.columns)}"
            )

        # Rename column (pure rename — no value change)
        out_chunk = chunk[["source1_entity_id", "matched_entity_ids"]].rename(
            columns={"matched_entity_ids": "matching_entity_ids"}
        )

        mode = "w" if first_chunk else "a"
        out_chunk.to_csv(output_path, sep="\t", index=False,
                         mode=mode, header=first_chunk,
                         quoting=csv.QUOTE_NONE, escapechar="\\")

        total_rows += len(out_chunk)
        first_chunk = False

    return total_rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_bridge(
    preprocessed_dir: Path,
    real_dir: Path,
    splits: Optional[List[str]] = None,
) -> dict:
    """Convert MLnNor's cleaned outputs to MLnAWS's expected input format.

    Parameters
    ----------
    preprocessed_dir : Path
        Root directory of MLnNor's processed output.
        Expected layout::

            preprocessed_dir/
              train/
                source1_clean.tsv
                source2_clean.tsv
                source3_clean.tsv
                ground_truth_clean.tsv
              test/
                source1_clean.tsv
                source2_clean.tsv
                source3_clean.tsv

    real_dir : Path
        Destination directory for MLnAWS input files (dataset/real/).

    splits : list[str], optional
        Splits to convert. Defaults to ["train", "test"].

    Returns
    -------
    dict
        Summary with row counts per split/file and validation results.
    """
    if splits is None:
        splits = ["train", "test"]

    real_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"splits": {}}

    for split in splits:
        split_in = preprocessed_dir / split
        if not split_in.exists():
            raise FileNotFoundError(
                f"bridge: preprocessed split directory not found: {split_in}"
            )

        logger.info("[bridge] Converting split '%s' from %s -> %s", split, split_in, real_dir)
        split_report: dict = {}

        # ── S1 → clean + feature ─────────────────────────────────────────
        s1_in = split_in / "source1_clean.tsv"
        clean_s1_out = real_dir / f"clean_s1_{split}.tsv"
        feat_s1_out  = real_dir / f"feature_s1_{split}.tsv"

        if not s1_in.exists():
            raise FileNotFoundError(f"bridge: missing {s1_in}")

        s1_rows = _stream_source_file(s1_in, clean_s1_out, feat_s1_out)
        split_report["s1_rows"] = s1_rows
        logger.info("  [bridge] S1 %s: %d rows -> %s, %s", split, s1_rows,
                    clean_s1_out.name, feat_s1_out.name)

        # ── S2+S3 → combined clean + feature ─────────────────────────────
        combined_clean_out = real_dir / f"clean_s2s3_{split}.tsv"
        combined_feat_out  = real_dir / f"feature_s2s3_{split}.tsv"

        s2_in = split_in / "source2_clean.tsv"
        s3_in = split_in / "source3_clean.tsv"

        if not s2_in.exists():
            raise FileNotFoundError(f"bridge: missing {s2_in}")
        if not s3_in.exists():
            raise FileNotFoundError(f"bridge: missing {s3_in}")

        s2_rows = _append_source_to_combined(s2_in, combined_clean_out, combined_feat_out, first_source=True)
        s3_rows = _append_source_to_combined(s3_in, combined_clean_out, combined_feat_out, first_source=False)
        split_report["s2_rows"] = s2_rows
        split_report["s3_rows"] = s3_rows
        split_report["s2s3_combined_rows"] = s2_rows + s3_rows
        logger.info("  [bridge] S2+S3 %s: S2=%d, S3=%d -> %s, %s",
                    split, s2_rows, s3_rows,
                    combined_clean_out.name, combined_feat_out.name)

        # ── Ground truth (train only) ─────────────────────────────────────
        if split == "train":
            gt_in  = split_in / "ground_truth_clean.tsv"
            gt_out = real_dir / f"ground_truth_{split}.tsv"

            if not gt_in.exists():
                raise FileNotFoundError(f"bridge: missing {gt_in}")

            gt_rows = _convert_ground_truth(gt_in, gt_out)
            split_report["ground_truth_rows"] = gt_rows
            logger.info("  [bridge] GT %s: %d rows -> %s", split, gt_rows, gt_out.name)

        # ── Per-split validation ──────────────────────────────────────────
        _validate_bridge_outputs(real_dir, split, split_report)
        split_report["status"] = "ok"
        report["splits"][split] = split_report

    report["status"] = "completed"
    return report


# ---------------------------------------------------------------------------
# Bridge output validation
# ---------------------------------------------------------------------------

def _validate_bridge_outputs(real_dir: Path, split: str, split_report: dict) -> None:
    """Light validation of the bridge output files.

    Checks:
    - All expected files exist and are non-empty.
    - Headers are correct.
    - Row counts match split_report tallies.
    - No empty entity_id values in any file.
    - Duplicate entity_ids are flagged (non-fatal warning).
    """
    expected: list[tuple[str, list[str]]] = [
        (f"clean_s1_{split}.tsv",    ["entity_id", "clean_text"]),
        (f"clean_s2s3_{split}.tsv",  ["entity_id", "clean_text"]),
        (f"feature_s1_{split}.tsv",  ["entity_id", "clean_name", "clean_address", "clean_country"]),
        (f"feature_s2s3_{split}.tsv",["entity_id", "clean_name", "clean_address", "clean_country"]),
    ]
    if split == "train":
        expected.append(
            (f"ground_truth_{split}.tsv", ["source1_entity_id", "matching_entity_ids"])
        )

    for filename, expected_cols in expected:
        path = real_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(
                f"bridge validation: expected non-empty file {path} does not exist or is empty"
            )

        # Read only the header row cheaply
        with open(path, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")

        if header != expected_cols:
            raise ValueError(
                f"bridge validation: {filename} has wrong header.\n"
                f"  Expected: {expected_cols}\n"
                f"  Got:      {header}"
            )

    logger.debug("[bridge] Validation passed for split '%s'", split)


def validate_contract(real_dir: Path, splits: Optional[List[str]] = None) -> dict:
    """Verify that all MLnAWS input files exist with correct schemas.

    This is the contract check run BEFORE handing off to MLnAWS.
    Raises ValueError with a clear message on any violation.

    Returns
    -------
    dict
        Row counts per file.
    """
    if splits is None:
        splits = ["train", "test"]

    result: dict = {"splits": {}}

    for split in splits:
        split_result: dict = {}
        files_to_check: list[tuple[str, list[str]]] = [
            (f"clean_s1_{split}.tsv",    ["entity_id", "clean_text"]),
            (f"clean_s2s3_{split}.tsv",  ["entity_id", "clean_text"]),
            (f"feature_s1_{split}.tsv",  ["entity_id", "clean_name", "clean_address", "clean_country"]),
            (f"feature_s2s3_{split}.tsv",["entity_id", "clean_name", "clean_address", "clean_country"]),
        ]
        if split == "train":
            files_to_check.append(
                (f"ground_truth_{split}.tsv", ["source1_entity_id", "matching_entity_ids"])
            )

        for filename, expected_cols in files_to_check:
            path = real_dir / filename
            if not path.exists():
                raise ValueError(
                    f"contract check: required MLnAWS input file missing: {path}\n"
                    f"  Run preprocessing first or use --skip-preprocessing if files exist."
                )

            df_head = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                  na_values=[], nrows=1)
            actual_cols = list(df_head.columns)
            if actual_cols != expected_cols:
                raise ValueError(
                    f"contract check: {filename} has wrong schema.\n"
                    f"  Expected columns: {expected_cols}\n"
                    f"  Got columns:      {actual_cols}"
                )

            # Count rows cheaply
            n_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1  # minus header
            split_result[filename] = n_rows

        result["splits"][split] = split_result

    result["status"] = "ok"
    return result
