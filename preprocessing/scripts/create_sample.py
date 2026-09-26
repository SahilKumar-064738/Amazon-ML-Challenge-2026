#!/usr/bin/env python3
"""Create a small local sample dataset from the seven full-size raw TSVs,
without loading any full file entirely into memory (Section 15/11).

Reads each source file in chunks (pandas `chunksize`), keeps only the
first N rows of each Source 1/2/3 file, and — for the training ground
truth — keeps only the rows whose source1_entity_id is among the sampled
Source-1 ids, so the sample's ground truth stays internally consistent
(every kept source1_entity_id really is in the sampled source1 file).
Referenced S2/S3 ids in the kept ground-truth rows may not all be present
in the small S2/S3 sample (that's expected for a naive row-prefix sample)
— use --keep-referenced to additionally pull in any S2/S3 rows the sampled
ground truth references, so the sample is fully referentially valid and
will pass validate_ground_truth_referential() unmodified.

Usage:
    python scripts/create_sample.py \\
        --input-dir ./data_full \\
        --output-dir ./data/sample \\
        --rows 500 \\
        --keep-referenced
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CHUNK_SIZE = 50_000


def sample_source_file(path: Path, out_path: Path, n_rows: int) -> pd.DataFrame:
    """Read only enough chunks of `path` to get n_rows, write them out, and
    return the sampled DataFrame (small, safe to keep in memory)."""
    collected = []
    total = 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                              na_values=[], chunksize=CHUNK_SIZE):
        needed = n_rows - total
        if needed <= 0:
            break
        collected.append(chunk.iloc[:needed])
        total += min(needed, len(chunk))
    sampled = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out_path, sep="\t", index=False)
    return sampled


def sample_ground_truth(
    gt_path: Path, out_path: Path, s1_ids: set,
) -> pd.DataFrame:
    """Keep only ground-truth rows whose source1_entity_id is in s1_ids,
    reading in chunks so the full ground-truth file is never fully
    materialized at once."""
    kept_chunks = []
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False,
                              na_values=[], chunksize=CHUNK_SIZE):
        kept_chunks.append(chunk[chunk["source1_entity_id"].isin(s1_ids)])
    result = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(
        columns=["source1_entity_id", "matched_entity_ids"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False)
    return result


def pull_referenced_rows(
    source_path: Path, out_path: Path, existing_df: pd.DataFrame, needed_ids: set,
) -> None:
    """Append any rows from source_path whose entity_id is in needed_ids
    but not already present in existing_df, reading source_path in chunks."""
    have_ids = set(existing_df["entity_id"]) if len(existing_df) else set()
    still_needed = needed_ids - have_ids
    if not still_needed:
        return
    extra_chunks = []
    for chunk in pd.read_csv(source_path, sep="\t", dtype=str, keep_default_na=False,
                              na_values=[], chunksize=CHUNK_SIZE):
        match = chunk[chunk["entity_id"].isin(still_needed)]
        if len(match):
            extra_chunks.append(match)
            still_needed -= set(match["entity_id"])
        if not still_needed:
            break
    if extra_chunks:
        combined = pd.concat([existing_df] + extra_chunks, ignore_index=True)
        combined.to_csv(out_path, sep="\t", index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True,
                         help="Directory containing train/ and test/ subdirs with the full raw TSVs "
                              "(SageMaker-style layout, matching src/preprocess.py's expected input).")
    parser.add_argument("--output-dir", required=True, help="Where to write the sampled train/ and test/ subdirs.")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to keep per source file (default: 500).")
    parser.add_argument("--keep-referenced", action="store_true",
                         help="Also pull in any S2/S3 rows referenced by the sampled ground truth "
                              "that fell outside the row-prefix sample, so the sample passes "
                              "ground-truth referential validation unmodified.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for split, has_gt in (("train", True), ("test", False)):
        split_in = input_dir / split
        split_out = output_dir / split
        if not split_in.exists():
            print(f"WARNING: {split_in} not found, skipping split '{split}'", file=sys.stderr)
            continue

        s1 = sample_source_file(split_in / f"{split}_source1.tsv", split_out / f"{split}_source1.tsv", args.rows)
        s2 = sample_source_file(split_in / f"{split}_source2.tsv", split_out / f"{split}_source2.tsv", args.rows)
        s3 = sample_source_file(split_in / f"{split}_source3.tsv", split_out / f"{split}_source3.tsv", args.rows)
        print(f"[{split}] sampled source1={len(s1)}, source2={len(s2)}, source3={len(s3)} rows")

        if has_gt:
            gt_out_path = split_out / f"{split}_ground_truth.tsv"
            gt = sample_ground_truth(split_in / f"{split}_ground_truth.tsv", gt_out_path, set(s1["entity_id"]))
            print(f"[{split}] sampled ground_truth={len(gt)} rows")

            if args.keep_referenced:
                needed_s2, needed_s3 = set(), set()
                for cell in gt["matched_entity_ids"]:
                    if not cell:
                        continue
                    for token in cell.split(","):
                        if token.startswith("S2-"):
                            needed_s2.add(token)
                        elif token.startswith("S3-"):
                            needed_s3.add(token)
                pull_referenced_rows(split_in / f"{split}_source2.tsv", split_out / f"{split}_source2.tsv", s2, needed_s2)
                pull_referenced_rows(split_in / f"{split}_source3.tsv", split_out / f"{split}_source3.tsv", s3, needed_s3)
                print(f"[{split}] pulled in any additional referenced S2/S3 rows for referential consistency")

    print(f"\nSample written to: {output_dir}")
    print("Run it with:")
    print(f"  python -m src.preprocess --input-dir {output_dir} --output-dir outputs/sample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
