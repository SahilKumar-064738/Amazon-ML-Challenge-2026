#!/usr/bin/env python3
"""Preprocessing entrypoint for Business Entity Resolution.

Scope (implementation.md, Section 30 / task Section 1): raw TSV ingestion ->
validation -> cleaning/normalization/canonicalization -> diagnostics ->
invariant verification -> cleaned TSV files -> metadata/data-quality
reports. Embeddings, candidate generation, similarity scoring, and entity
matching are explicitly NOT implemented here.

This same module is the SageMaker Processing entrypoint AND the local CLI —
there is exactly one preprocessing implementation, run identically in both
environments (only the --input-dir/--output-dir paths differ), per
implementation.md Section 26, item 9.

CLI interface note (documented resolution of a spec inconsistency, see
README.md "Design decisions / resolved inconsistencies"): the outer task
brief suggests a single `--output-dir`, while implementation.md Section 15.4
uses separate `--output-processed-dir` / `--output-metadata-dir`. Both are
supported here: pass `--output-dir` for the common case (creates
`<output-dir>/processed/` and `<output-dir>/metadata/`), or pass the two
`--output-*-dir` flags directly (as SageMaker's ProcessingOutput destinations
do) to control them independently. If both are given, the explicit
`--output-*-dir` flags win.

Exit code: 0 on success, 1 on any FatalValidationError (no partial /
half-written output is left for a failed split; see run()).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Optional Parquet-cache import.
# scripts/prepare_data.py is not part of the src/ package, so we attempt a
# sys.path-relative import.  If it fails for any reason (running inside a
# SageMaker container where scripts/ is absent, pyarrow not installed, etc.)
# we set the flag to False and fall back to plain pd.read_csv — the pipeline
# is fully functional either way.
# ---------------------------------------------------------------------------
try:
    import importlib.util as _ilu
    import os as _os

    _scripts_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scripts")
    _spec = _ilu.spec_from_file_location(
        "prepare_data",
        _os.path.join(_scripts_dir, "prepare_data.py"),
    )
    if _spec is not None and _spec.loader is not None:
        _prepare_data = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_prepare_data)  # type: ignore[union-attr]
        _load_tsv_cached = _prepare_data.load_tsv_cached
        _CACHE_AVAILABLE = True
    else:
        _CACHE_AVAILABLE = False
        _load_tsv_cached = None  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    _CACHE_AVAILABLE = False
    _load_tsv_cached = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional Indic transliterator — loaded at main() time when --indic-dict is
# passed, or when INDIC_DICT env var is set.  Never required at import time.
# ---------------------------------------------------------------------------
try:
    from .normalization import set_transliterator as _set_transliterator
    from .indic_transliteration import IndicTransliterator as _IndicTransliterator
    _INDIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _set_transliterator = None   # type: ignore[assignment]
    _IndicTransliterator = None  # type: ignore[assignment]
    _INDIC_AVAILABLE = False

try:  # pragma: no cover - import shim for `python -m src.preprocess` vs SageMaker container
    from .config import (
        GT_COLUMNS,
        GT_OUTPUT_COLUMNS,
        REQUIRED_OUTPUT_COLUMNS,
        SOURCE_COLUMNS,
        SOURCE_KEYS,
        TEST_SPLIT,
        TRAIN_SPLIT,
    )
    from .diagnostics import (
        exact_duplicate_rows,
        name_address_country_collisions,
        name_canonical_collisions,
        name_country_collisions,
        collision_groups_to_frame,
    )
    from .normalization import (
        canonicalize_address_series,
        canonicalize_name_series,
        classify_name_script_series,
        extract_address_numbers_series,
        extract_address_postal_code_series,
        extract_landmark_series,
        make_address_sorted_tokens_series,
        normalize_address_series,
        normalize_country_series,
        normalize_name_series,
    )
    from .validation import (
        FatalValidationError,
        validate_entity_ids,
        validate_ground_truth_duplicates,
        validate_ground_truth_referential,
        validate_ground_truth_schema,
        validate_id_set_preserved,
        validate_row_count_preserved,
        validate_schema,
    )
except ImportError:  # running as a plain script (e.g. inside a SageMaker container
    # invoked as `python preprocess.py` from within source_dir, with no package
    # context). Fall back to same-directory imports.
    from config import (  # type: ignore
        GT_COLUMNS,
        GT_OUTPUT_COLUMNS,
        REQUIRED_OUTPUT_COLUMNS,
        SOURCE_COLUMNS,
        SOURCE_KEYS,
        TEST_SPLIT,
        TRAIN_SPLIT,
    )
    from diagnostics import (  # type: ignore
        exact_duplicate_rows,
        name_address_country_collisions,
        name_canonical_collisions,
        name_country_collisions,
        collision_groups_to_frame,
    )
    from normalization import (  # type: ignore
        canonicalize_address_series,
        canonicalize_name_series,
        classify_name_script_series,
        extract_address_numbers_series,
        extract_address_postal_code_series,
        extract_landmark_series,
        make_address_sorted_tokens_series,
        normalize_address_series,
        normalize_country_series,
        normalize_name_series,
    )
    from validation import (  # type: ignore
        FatalValidationError,
        validate_entity_ids,
        validate_ground_truth_duplicates,
        validate_ground_truth_referential,
        validate_ground_truth_schema,
        validate_id_set_preserved,
        validate_row_count_preserved,
        validate_schema,
    )


def read_tsv(path: Path, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Read a TSV with explicit dtype=str and no NaN coercion, per spec.

    keep_default_na=False + na_values=[] together ensure pandas never turns
    a literal "NA"/"NULL" cell into a real NaN — missing-value handling is
    done explicitly in normalization.py instead.

    When ``scripts/prepare_data.py`` is importable and pyarrow is installed,
    this function transparently serves the DataFrame from a Snappy-compressed
    Parquet cache (keyed by SHA-256 of the source file).  On the very first
    call the cache is written; every subsequent call skips TSV parsing
    entirely.  If the cache layer is unavailable for any reason, the function
    falls back to a direct ``pd.read_csv`` call — the returned DataFrame is
    identical either way (all columns are ``object`` / ``str`` dtype, no
    ``float NaN`` values).

    Parameters
    ----------
    path:
        Path to the raw ``.tsv`` file.
    cache_dir:
        Directory for Parquet cache files.  ``None`` (default) lets
        ``load_tsv_cached`` choose a ``cache/`` sibling of ``path``'s
        grandparent directory, which keeps the cache out of the input tree.
        Pass an explicit path (e.g. from ``--cache-dir``) to override.
    """
    if _CACHE_AVAILABLE:
        return _load_tsv_cached(path, cache_dir=cache_dir)
    # Fallback: plain read with the same strict NaN-prevention options.
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])


def clean_source(df: pd.DataFrame, source_key: str, name: str) -> "tuple[pd.DataFrame, dict]":
    """Validate and clean one source file (Source 1, 2, or 3).

    Uses vectorized Series-level normalization (no df.apply(axis=1) over
    millions of rows) and enforces the row-count + ID-set invariants before
    returning.
    """
    validate_schema(df, SOURCE_COLUMNS, name)
    validate_entity_ids(df, source_key, name)

    input_rows = len(df)
    input_ids = set(df["entity_id"])

    out = pd.DataFrame({"entity_id": df["entity_id"]})
    out["business_name"] = df["business_name"]

    name_normalized, name_is_missing = normalize_name_series(df["business_name"])
    out["business_name_normalized"] = name_normalized
    out["business_name_canonical"] = canonicalize_name_series(name_normalized)
    out["business_name_is_missing"] = name_is_missing

    out["business_address"] = df["business_address"]
    addr_normalized, addr_is_missing = normalize_address_series(df["business_address"])
    addr_canonical = canonicalize_address_series(addr_normalized)
    out["business_address_normalized"] = addr_normalized
    out["business_address_canonical"] = addr_canonical
    out["address_landmark"] = extract_landmark_series(df["business_address"])
    out["address_numbers"] = extract_address_numbers_series(addr_normalized)
    out["address_postal_code"] = extract_address_postal_code_series(addr_normalized)
    out["address_sorted_tokens"] = make_address_sorted_tokens_series(addr_canonical)
    out["business_address_is_missing"] = addr_is_missing

    out["country"] = df["country"]
    out["country_normalized"] = normalize_country_series(df["country"])

    # name_script_class is computed from the RAW business_name column (not from
    # name_normalized). This is intentional: after Indic transliteration the
    # normalized form would be all-Latin even for a Devanagari input, so the
    # script classification must reflect the ORIGINAL script seen in the source
    # data, not the post-transliteration form.
    out["name_script_class"] = classify_name_script_series(df["business_name"].fillna(""))

    out = out[REQUIRED_OUTPUT_COLUMNS]

    output_rows = len(out)
    output_ids = set(out["entity_id"])
    validate_row_count_preserved(input_rows, output_rows, name)
    validate_id_set_preserved(input_ids, output_ids, name)

    exact_dupes = exact_duplicate_rows(df, SOURCE_COLUMNS)

    stats = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "id_set_equal": True,
        "duplicate_entity_id_count": 0,  # would have raised in validate_entity_ids otherwise
        "name_missing_count": int(name_is_missing.sum()),
        "address_missing_count": int(addr_is_missing.sum()),
        "country_distribution": df["country"].value_counts().to_dict(),
        "exact_duplicate_rows": exact_dupes,
    }
    return out, stats


def clean_ground_truth(
    gt_df: pd.DataFrame,
    s1_ids: set,
    s2_ids: set,
    s3_ids: set,
    name: str,
) -> "tuple[pd.DataFrame, dict]":
    """Validate ground truth referentially. Ground truth is NEVER used to
    alter source records, and its own values are not transformed (its two
    columns are IDs, not free text) — see implementation.md Section 11.
    """
    validate_ground_truth_schema(gt_df, GT_COLUMNS, name)
    validate_ground_truth_duplicates(gt_df, name)
    gt_stats = validate_ground_truth_referential(gt_df, s1_ids, s2_ids, s3_ids, name)
    out = gt_df[GT_OUTPUT_COLUMNS].copy()
    return out, gt_stats


def build_collision_report(cleaned_s1: pd.DataFrame) -> Dict:
    """Bounded, memory-safe collision diagnostics on cleaned Source 1
    (diagnostic only — never used to drop/merge records)."""
    return {
        "business_name_canonical": name_canonical_collisions(cleaned_s1),
        "business_name_canonical__country_normalized": name_country_collisions(cleaned_s1),
        "business_name_canonical__business_address_canonical__country_normalized":
            name_address_country_collisions(cleaned_s1),
    }


def discover_split_files(split_dir: Path, split: str) -> Dict[str, Path]:
    """Locate the source (and, for train, ground-truth) files for a split
    inside split_dir, matching the SageMaker ProcessingInput mount layout
    (input/<split>/<split>_source1.tsv, etc.)."""
    files = {
        "source1": split_dir / f"{split}_source1.tsv",
        "source2": split_dir / f"{split}_source2.tsv",
        "source3": split_dir / f"{split}_source3.tsv",
    }
    if split == TRAIN_SPLIT:
        files["ground_truth"] = split_dir / f"{split}_ground_truth.tsv"
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FatalValidationError(
            f"ERROR: missing expected input file(s) for split '{split}': {missing}\n"
            f"  Looked in: {split_dir}"
        )
    return files


def run(
    input_dir: Path,
    output_processed_dir: Path,
    output_metadata_dir: Path,
    splits: List[str],
    cache_dir: Optional[Path] = None,
) -> Dict:
    output_processed_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_dir.mkdir(parents=True, exist_ok=True)

    report: Dict = {"started_at": time.time(), "splits": {}, "status": "running"}

    for split in splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            raise FatalValidationError(
                f"ERROR: expected input subdirectory not found: {split_dir}\n"
                f"  Input dir must contain a '{split}/' subdirectory (SageMaker-style layout)."
            )
        files = discover_split_files(split_dir, split)
        split_out = output_processed_dir / split
        split_out.mkdir(parents=True, exist_ok=True)
        split_report: Dict = {}

        s1 = read_tsv(files["source1"], cache_dir)
        s2 = read_tsv(files["source2"], cache_dir)
        s3 = read_tsv(files["source3"], cache_dir)

        s1_clean, s1_stats = clean_source(s1, "source1", f"{split}/source1")
        del s1  # free the raw frame before processing the next source (Section 11)
        s2_clean, s2_stats = clean_source(s2, "source2", f"{split}/source2")
        del s2
        s3_clean, s3_stats = clean_source(s3, "source3", f"{split}/source3")
        del s3

        s1_clean.to_csv(split_out / "source1_clean.tsv", sep="\t", index=False)
        s2_clean.to_csv(split_out / "source2_clean.tsv", sep="\t", index=False)
        s3_clean.to_csv(split_out / "source3_clean.tsv", sep="\t", index=False)

        collisions = build_collision_report(s1_clean)
        for key, result in collisions.items():
            frame = collision_groups_to_frame(result)
            frame.to_csv(
                output_metadata_dir / f"canonicalization_collisions_{split}_{key}.tsv",
                sep="\t", index=False,
            )

        split_report.update({
            "source1": s1_stats,
            "source2": s2_stats,
            "source3": s3_stats,
            "canonicalization_collisions": {
                k: {kk: vv for kk, vv in v.items() if kk != "top_groups"}
                for k, v in collisions.items()
            },
        })

        if split == TRAIN_SPLIT:
            gt = read_tsv(files["ground_truth"], cache_dir)
            gt_clean, gt_stats = clean_ground_truth(
                gt,
                set(s1_clean["entity_id"]),
                set(s2_clean["entity_id"]),
                set(s3_clean["entity_id"]),
                f"{split}/ground_truth",
            )
            gt_clean.to_csv(split_out / "ground_truth_clean.tsv", sep="\t", index=False)
            split_report["ground_truth"] = gt_stats

        # Free cleaned frames for this split before moving to the next.
        del s1_clean, s2_clean, s3_clean

        report["splits"][split] = split_report

    report["status"] = "completed"
    report["finished_at"] = time.time()
    report["duration_seconds"] = report["finished_at"] - report["started_at"]

    with open(output_metadata_dir / "preprocessing_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(output_metadata_dir / "preprocessing_report.txt", "w") as f:
        f.write(json.dumps(report, indent=2, default=str))

    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Business Entity Resolution — data ingestion & preprocessing. "
            "Reads raw TSVs, validates, cleans/normalizes/canonicalizes, and "
            "writes cleaned TSVs + a data-quality report. Does NOT perform "
            "embeddings, candidate generation, or entity matching."
        )
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing <split>/ subdirectories (e.g. train/, test/) "
             "with the raw *_source1.tsv / *_source2.tsv / *_source3.tsv / "
             "*_ground_truth.tsv files. Matches the SageMaker ProcessingInput layout.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Convenience flag: creates <output-dir>/processed/ and "
             "<output-dir>/metadata/. Overridden by --output-processed-dir / "
             "--output-metadata-dir if those are also given.",
    )
    parser.add_argument(
        "--output-processed-dir", default=None,
        help="Directory to write cleaned TSVs to (default: <output-dir>/processed).",
    )
    parser.add_argument(
        "--output-metadata-dir", default=None,
        help="Directory to write the data-quality report + diagnostics to "
             "(default: <output-dir>/metadata).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=[TRAIN_SPLIT, TEST_SPLIT],
        choices=[TRAIN_SPLIT, TEST_SPLIT],
        help="Which splits to process (default: both train and test).",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help=(
            "Directory for Parquet cache files (speeds up repeat runs by "
            "skipping TSV parsing).  Defaults to a 'cache/' subdirectory "
            "next to --input-dir.  Has no effect if pyarrow is not installed "
            "or scripts/prepare_data.py is unavailable."
        ),
    )
    parser.add_argument(
        "--indic-dict", default=None, metavar="PATH",
        help=(
            "Path to a JSON file produced by DictionaryLearner.save_dictionary() "
            "or IndicTransliterator.save_dictionary().  When supplied, Indic-script "
            "business names are transliterated to Latin phonetics before the rest of "
            "normalization runs.  Has no effect when --no-transliterate is also given."
        ),
    )
    parser.add_argument(
        "--no-transliterate", action="store_true",
        help=(
            "Disable Indic transliteration entirely, even if --indic-dict is given.  "
            "Useful for ablation experiments."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.output_processed_dir and not args.output_dir:
        parser.error("Provide either --output-dir or --output-processed-dir/--output-metadata-dir.")

    output_dir = Path(args.output_dir) if args.output_dir else None
    output_processed_dir = Path(args.output_processed_dir) if args.output_processed_dir else (
        output_dir / "processed"
    )
    output_metadata_dir = Path(args.output_metadata_dir) if args.output_metadata_dir else (
        output_dir / "metadata"
    )

    # ── Indic transliterator setup ─────────────────────────────────────────
    no_translit = getattr(args, "no_transliterate", False)
    indic_dict_path = getattr(args, "indic_dict", None)
    if _INDIC_AVAILABLE and not no_translit:
        if indic_dict_path:
            dict_path = Path(indic_dict_path)
            if not dict_path.exists():
                print(
                    f"ERROR: --indic-dict path not found: {dict_path}",
                    file=sys.stderr,
                )
                return 1
            transliterator = _IndicTransliterator.load(dict_path)  # type: ignore[union-attr]
            _set_transliterator(transliterator)  # type: ignore[misc]
            print(f"Indic transliterator loaded from {dict_path} "
                  f"({len(transliterator.get_dictionary())} dictionary entries)")
        else:
            # No dictionary file → use character-level transliteration only
            _set_transliterator(_IndicTransliterator())  # type: ignore[misc]

    try:
        report = run(
            Path(args.input_dir),
            output_processed_dir,
            output_metadata_dir,
            args.splits,
            cache_dir=Path(args.cache_dir) if getattr(args, "cache_dir", None) else None,
        )
    except FatalValidationError as e:
        print(str(e), file=sys.stderr)
        print("FATAL: preprocessing failed validation. No further processing performed.",
              file=sys.stderr)
        return 1

    print(f"Preprocessing completed successfully in {report['duration_seconds']:.2f}s")
    print(f"  Splits processed: {list(report['splits'].keys())}")
    print(f"  Cleaned data:     {output_processed_dir}")
    print(f"  Report:           {output_metadata_dir / 'preprocessing_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
