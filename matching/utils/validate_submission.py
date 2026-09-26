#!/usr/bin/env python3
"""
utils/validate_submission.py
----------------------------
Official submission validator.
Checks matching_results.tsv and candidate_pairs.tsv against every rule
in the challenge spec.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Exits 0 (PASS) or 1 (issues found).
"""
import argparse
import csv
import os
import sys


def load_tsv(path, required_cols):
    """Load a TSV and return list-of-dicts. Validates columns and sep."""
    issues = []
    if not os.path.isfile(path):
        return None, [f"File not found: {path}"]
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            return None, [f"{path}: empty file or no header"]
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            return None, [f"{path}: missing columns {missing}; found {reader.fieldnames}"]
        for i, row in enumerate(reader, start=2):
            rows.append(row)
    return rows, issues


def load_source_ids(test_dir):
    """Load all entity_ids from test source files."""
    s1_ids, s2s3_ids = set(), set()
    for fname, target in [
        ("test_source1.tsv", s1_ids),
        ("test_source2.tsv", s2s3_ids),
        ("test_source3.tsv", s2s3_ids),
    ]:
        path = os.path.join(test_dir, fname)
        if not os.path.isfile(path):
            # graceful — mock dir may use different names
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                eid = row.get("entity_id", "").strip()
                if eid:
                    target.add(eid)
    return s1_ids, s2s3_ids


def validate(matching_path, candidate_path, test_dir):
    issues = []
    print(f"\nValidating:")
    print(f"  matching  : {matching_path}")
    print(f"  candidate : {candidate_path}")
    print(f"  test-dir  : {test_dir}\n")

    # ── Load test source IDs ────────────────────────────────────────────
    s1_ids, s2s3_ids = load_source_ids(test_dir)
    if s1_ids:
        print(f"  Test S1 entities   : {len(s1_ids)}")
        print(f"  Test S2+S3 entities: {len(s2s3_ids)}")
    else:
        print("  WARNING: test source files not found — skipping ID-existence checks")

    # ── Load matching_results.tsv ───────────────────────────────────────
    match_rows, load_issues = load_tsv(
        matching_path, ["source1_entity_id", "matched_entity_ids"]
    )
    issues.extend(load_issues)
    if match_rows is None:
        print("FAIL — could not load matching_results.tsv")
        return issues

    # ── Load candidate_pairs.tsv ────────────────────────────────────────
    cand_rows, load_issues = load_tsv(
        candidate_path, ["source1_entity_id", "candidate_entity_ids"]
    )
    issues.extend(load_issues)
    if cand_rows is None:
        print("FAIL — could not load candidate_pairs.tsv")
        return issues

    # ── Build lookup dicts ──────────────────────────────────────────────
    match_by_s1 = {}
    for row in match_rows:
        s1 = row["source1_entity_id"].strip()
        raw = row["matched_entity_ids"].strip()
        ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        if s1 in match_by_s1:
            issues.append(f"matching_results: duplicate source1_entity_id row: {s1!r}")
        match_by_s1[s1] = ids

    cand_by_s1 = {}
    for row in cand_rows:
        s1 = row["source1_entity_id"].strip()
        raw = row["candidate_entity_ids"].strip()
        ids = [x.strip() for x in raw.split(",") if x.strip()] if raw else []
        if s1 in cand_by_s1:
            issues.append(f"candidate_pairs: duplicate source1_entity_id row: {s1!r}")
        cand_by_s1[s1] = ids

    # ── Rule 1: every S1 test entity present exactly once ──────────────
    if s1_ids:
        missing = s1_ids - set(match_by_s1.keys())
        extra   = set(match_by_s1.keys()) - s1_ids
        if missing:
            issues.append(
                f"matching_results: {len(missing)} S1 entities missing "
                f"(e.g. {sorted(missing)[:5]})"
            )
        if extra:
            issues.append(
                f"matching_results: {len(extra)} S1 IDs not in test set "
                f"(e.g. {sorted(extra)[:5]})"
            )
        # same for candidate_pairs
        missing_c = s1_ids - set(cand_by_s1.keys())
        if missing_c:
            issues.append(
                f"candidate_pairs: {len(missing_c)} S1 entities missing "
                f"(e.g. {sorted(missing_c)[:5]})"
            )

    # ── Rule 2: no duplicate IDs within a list ──────────────────────────
    for s1, ids in match_by_s1.items():
        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            issues.append(
                f"matching_results: duplicate IDs in list for {s1!r}: {dupes[:5]}"
            )
    for s1, ids in cand_by_s1.items():
        if len(ids) != len(set(ids)):
            dupes = [x for x in ids if ids.count(x) > 1]
            issues.append(
                f"candidate_pairs: duplicate IDs in list for {s1!r}: {dupes[:5]}"
            )

    # ── Rule 3: matched IDs must be S2-/S3- prefixed ────────────────────
    bad_prefix_count = 0
    for s1, ids in match_by_s1.items():
        for eid in ids:
            if not (eid.startswith("S2-") or eid.startswith("S3-")):
                bad_prefix_count += 1
    if bad_prefix_count:
        issues.append(
            f"matching_results: {bad_prefix_count} matched IDs with invalid "
            "prefix (not S2- or S3-)"
        )

    # ── Rule 4: matched IDs must exist in test set ──────────────────────
    if s2s3_ids:
        bad_ids = []
        for s1, ids in match_by_s1.items():
            for eid in ids:
                if eid not in s2s3_ids:
                    bad_ids.append(eid)
        if bad_ids:
            issues.append(
                f"matching_results: {len(bad_ids)} matched IDs not in test "
                f"S2/S3 set (e.g. {bad_ids[:5]})"
            )

    # ── Rule 5: every matched ID must be in candidate set ───────────────
    orphan_count = 0
    for s1, matched in match_by_s1.items():
        cands = set(cand_by_s1.get(s1, []))
        for eid in matched:
            if eid not in cands:
                orphan_count += 1
    if orphan_count:
        issues.append(
            f"{orphan_count} matched ID(s) not in candidate_pairs for the same S1 "
            "(pipeline bug: matched IDs must be a subset of candidates)"
        )

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"  matching_results rows : {len(match_by_s1)}")
    print(f"  candidate_pairs rows  : {len(cand_by_s1)}")
    print()

    if issues:
        print(f"FAIL — {len(issues)} issue(s) found:\n")
        for i, issue in enumerate(issues, 1):
            print(f"  [{i}] {issue}")
        print()
    else:
        print("PASS")
        print("  Both output files pass all submission rules.")
        print("  Safe to submit.\n")

    return issues


def main():
    p = argparse.ArgumentParser(description="Validate submission files")
    p.add_argument("--matching",  required=True, help="Path to matching_results.tsv")
    p.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    p.add_argument("--test-dir",  required=True, help="Directory with test source TSVs")
    args = p.parse_args()

    issues = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if not issues else 1)


if __name__ == "__main__":
    main()
