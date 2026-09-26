"""Schema, ID, invariant, and ground-truth validation.

Fatal issues raise FatalValidationError and must stop the pipeline (no
partial output written). Non-fatal issues are returned as warning
dictionaries so the caller can record them in the data-quality report.

Ground-truth referential validation is implemented with vectorized set
operations (pandas .isin / set membership), never a per-row Python loop,
so it stays fast on files with millions of rows (Section 11 / 26.6).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Set

import pandas as pd

from .config import GT_MATCH_PATTERN, GT_SOURCE1_PATTERN, SOURCE_ID_PATTERNS


class FatalValidationError(Exception):
    """Raised for any validation failure that must stop the pipeline."""


def validate_schema(df: pd.DataFrame, expected_columns: List[str], name: str) -> None:
    if list(df.columns) != expected_columns:
        raise FatalValidationError(
            f"ERROR: schema mismatch for {name}\n"
            f"  Expected columns: {expected_columns}\n"
            f"  Got columns:      {list(df.columns)}"
        )


def validate_entity_ids(df: pd.DataFrame, source_key: str, name: str) -> None:
    """Fatal on: malformed entity_id (wrong prefix/pattern) or duplicate entity_id."""
    pattern = re.compile(SOURCE_ID_PATTERNS[source_key])
    is_valid = df["entity_id"].str.match(pattern)
    bad = df.loc[~is_valid, "entity_id"]
    if len(bad) > 0:
        raise FatalValidationError(
            f"ERROR: malformed entity_id in {name}\n"
            f"  Expected pattern: {SOURCE_ID_PATTERNS[source_key]}\n"
            f"  Malformed count:  {len(bad)}\n"
            f"  Examples:         {bad.head(5).tolist()}"
        )
    dup_mask = df["entity_id"].duplicated(keep=False)
    dup_count = int(df["entity_id"].duplicated(keep="first").sum())
    if dup_count > 0:
        examples = df.loc[dup_mask, "entity_id"].unique()[:5].tolist()
        raise FatalValidationError(
            f"ERROR: duplicate entity_id in {name}\n"
            f"  Duplicate count: {dup_count}\n"
            f"  Examples:        {examples}"
        )


def validate_row_count_preserved(input_rows: int, output_rows: int, name: str) -> None:
    if input_rows != output_rows:
        raise FatalValidationError(
            f"ERROR: row-count invariant failed for {name}\n"
            f"  Input rows:  {input_rows}\n"
            f"  Output rows: {output_rows}\n"
            f"  Cleaning must never drop or add records."
        )


def validate_id_set_preserved(input_ids: Set[str], output_ids: Set[str], name: str) -> None:
    """Fatal unless the input and output entity_id sets are exactly equal.

    This is stronger than a row-count check: it would catch a swap (same
    count, different actual IDs) that a count-only check could miss.
    """
    if input_ids != output_ids:
        missing = input_ids - output_ids
        added = output_ids - input_ids
        raise FatalValidationError(
            f"ERROR: ID-set invariant failed for {name}\n"
            f"  Input IDs:   {len(input_ids)}\n"
            f"  Output IDs:  {len(output_ids)}\n"
            f"  Missing IDs: {len(missing)} (examples: {list(missing)[:5]})\n"
            f"  Added IDs:   {len(added)} (examples: {list(added)[:5]})"
        )


def validate_ground_truth_schema(gt_df: pd.DataFrame, gt_columns: List[str], name: str) -> None:
    validate_schema(gt_df, gt_columns, name)


def validate_ground_truth_duplicates(gt_df: pd.DataFrame, name: str) -> None:
    dup_count = int(gt_df["source1_entity_id"].duplicated(keep="first").sum())
    if dup_count > 0:
        raise FatalValidationError(
            f"ERROR: duplicate source1_entity_id in {name}\n"
            f"  Duplicate count: {dup_count}"
        )


def _split_match_ids(series: pd.Series) -> pd.Series:
    """Vectorized split of a comma-separated matched_entity_ids column into
    lists, treating '' as an empty list (a valid, non-fatal case)."""
    return series.fillna("").apply(lambda s: [] if s == "" else s.split(","))


def validate_ground_truth_referential(
    gt_df: pd.DataFrame,
    s1_ids: Set[str],
    s2_ids: Set[str],
    s3_ids: Set[str],
    name: str,
) -> Dict[str, object]:
    """Referential validation of ground truth against cleaned source ID sets.

    Fatal on: any source1_entity_id not present in S1; any token in
    matched_entity_ids that is malformed, or well-formed but not present in
    the corresponding S2/S3 id set; any duplicate id within a single match
    list. Uses vectorized/set-based operations throughout — never a
    per-token Python loop across the whole file when it can be avoided at
    this scale, since this file can have >2M rows.
    """
    source1_pattern = re.compile(GT_SOURCE1_PATTERN)
    match_pattern = re.compile(GT_MATCH_PATTERN)

    bad_s1_format = ~gt_df["source1_entity_id"].str.match(source1_pattern)
    if bad_s1_format.any():
        examples = gt_df.loc[bad_s1_format, "source1_entity_id"].head(5).tolist()
        raise FatalValidationError(
            f"ERROR: malformed source1_entity_id in {name}\n"
            f"  Count: {int(bad_s1_format.sum())}\n"
            f"  Examples: {examples}"
        )

    gt_s1_set = set(gt_df["source1_entity_id"])
    unknown_s1 = gt_s1_set - s1_ids
    if unknown_s1:
        raise FatalValidationError(
            f"ERROR: ground truth references unknown source1_entity_id in {name}\n"
            f"  Unknown count: {len(unknown_s1)}\n"
            f"  Examples: {list(unknown_s1)[:5]}"
        )

    # Explode all match tokens once, vectorized, instead of looping per row.
    match_lists = _split_match_ids(gt_df["matched_entity_ids"])
    exploded = match_lists.explode()
    exploded = exploded.dropna()
    exploded = exploded[exploded != ""]

    if len(exploded) > 0:
        malformed_mask = ~exploded.str.match(match_pattern)
        if malformed_mask.any():
            examples = exploded.loc[malformed_mask].head(5).tolist()
            raise FatalValidationError(
                f"ERROR: malformed matched_entity_ids token in {name}\n"
                f"  Count: {int(malformed_mask.sum())}\n"
                f"  Examples: {examples}"
            )

        is_s2 = exploded.str.startswith("S2-")
        is_s3 = exploded.str.startswith("S3-")
        unknown_s2 = set(exploded.loc[is_s2]) - s2_ids
        unknown_s3 = set(exploded.loc[is_s3]) - s3_ids
        if unknown_s2 or unknown_s3:
            raise FatalValidationError(
                f"ERROR: ground truth references unknown S2/S3 ids in {name}\n"
                f"  Unknown S2 count: {len(unknown_s2)} (examples: {list(unknown_s2)[:5]})\n"
                f"  Unknown S3 count: {len(unknown_s3)} (examples: {list(unknown_s3)[:5]})"
            )

    # Duplicate ids inside a single match list.
    dup_within_row = match_lists.apply(lambda ids: len(ids) != len(set(ids)))
    if dup_within_row.any():
        bad_rows = gt_df.loc[dup_within_row, "source1_entity_id"].head(5).tolist()
        raise FatalValidationError(
            f"ERROR: duplicate ids within a single matched_entity_ids cell in {name}\n"
            f"  Affected row count: {int(dup_within_row.sum())}\n"
            f"  Examples (source1_entity_id): {bad_rows}"
        )

    match_counts = match_lists.apply(len)
    return {
        "rows": len(gt_df),
        "zero_match_rows": int((match_counts == 0).sum()),
        "mean_matches_per_entity": float(match_counts.mean()) if len(match_counts) else 0.0,
        "max_matches_for_one_entity": int(match_counts.max()) if len(match_counts) else 0,
        "total_s2_references": int((exploded.str.startswith("S2-")).sum()) if len(exploded) else 0,
        "total_s3_references": int((exploded.str.startswith("S3-")).sum()) if len(exploded) else 0,
    }
