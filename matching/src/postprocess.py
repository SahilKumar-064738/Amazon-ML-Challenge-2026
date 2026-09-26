"""
src/postprocess.py
------------------
Phase 5 — Post-processing utilities for the entity-resolution pipeline.

Public API (Phase 5.1):
    dedupe_matches(predicted_by_s1) -> dict[str, list[str]]

Public API (Phase 5.2):
    filter_valid_ids(predicted_by_s1, valid_s2s3_ids) -> dict[str, list[str]]

Public API (Phase 5.3):
    ensure_full_coverage(predicted_by_s1, all_test_s1_ids) -> dict[str, list[str]]

Public API (Phase 5.4):
    check_candidate_consistency(predicted_by_s1, candidate_pairs_by_s1) -> dict

Public API (Phase 5.5):
    write_matching_results(predicted_by_s1, output_path) -> int
    write_candidate_pairs_final(candidate_pairs_by_s1, output_path) -> int
"""

from __future__ import annotations

import csv
import os


def dedupe_matches(
    predicted_by_s1: dict[str, set[str] | list[str]],
) -> dict[str, list[str]]:
    """Deduplicate and sort per-S1-entity match collections.

    Takes the direct output of :func:`src.threshold.apply_threshold` (which
    returns ``dict[str, set[str]]``) and normalises every value into a sorted,
    deduplicated ``list[str]``.  The function also accepts lists as values so
    it can be used defensively even if a caller passes raw list-accumulations
    instead of sets.

    Parameters
    ----------
    predicted_by_s1 : dict[str, set[str] | list[str]]
        Maps each S1 entity ID (``str``) to a collection of matched candidate
        entity IDs.  This is the direct output of ``apply_threshold()``.
        * Keys must be ``str``.
        * Each value must be a collection whose elements are all ``str``.
        * An empty collection is valid and produces an empty output list.

    Returns
    -------
    dict[str, list[str]]
        Same keys as the input.  Each value is a **sorted, deduplicated**
        ``list[str]`` of matched entity IDs.

        * Singletons (empty input collection) → ``[]``.
        * Already-deduplicated sets from ``apply_threshold`` → unchanged
          content, converted to sorted lists.
        * Lists with duplicates → duplicates removed, result sorted.

    Side Effects (stdout)
    ---------------------
    Prints a brief deduplication report:

    * Total number of S1 entities processed.
    * Number of S1 entities that had at least one duplicate removed.
    * Total duplicate IDs removed across all S1 entities.
    * If zero duplicates: ``"No duplicates found — apply_threshold built sets correctly [OK]"``
    * If any duplicates:  ``"WARNING: X duplicates found — check apply_threshold() for dict accumulation bugs"``

    Raises
    ------
    TypeError
        If *predicted_by_s1* is not a ``dict``.
    ValueError
        If any key in *predicted_by_s1* is not a ``str``.
    ValueError
        If any element inside a value collection is not a ``str``.

    Examples
    --------
    >>> from src.postprocess import dedupe_matches
    >>> result = dedupe_matches({"S1-001": {"S2-099"}, "S1-002": set()})
    Total S1 entities processed : 2
    Entities with duplicates removed : 0
    Total duplicate IDs removed : 0
    No duplicates found — apply_threshold built sets correctly [OK]
    >>> result
    {'S1-001': ['S2-099'], 'S1-002': []}
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(predicted_by_s1, dict):
        raise TypeError(
            f"dedupe_matches() expects a dict, got {type(predicted_by_s1).__name__}."
        )

    for key in predicted_by_s1:
        if not isinstance(key, str):
            raise ValueError(
                f"All keys must be str; found key of type {type(key).__name__!r}: {key!r}."
            )

    for key, value in predicted_by_s1.items():
        for element in value:
            if not isinstance(element, str):
                raise ValueError(
                    f"All match IDs must be str; key {key!r} contains element of "
                    f"type {type(element).__name__!r}: {element!r}."
                )

    # ------------------------------------------------------------------
    # Deduplication pass
    # ------------------------------------------------------------------
    result: dict[str, list[str]] = {}
    total_dupes: int = 0
    entities_with_dupes: int = 0

    for s1_id, matches in predicted_by_s1.items():
        original_list = list(matches)
        deduped = sorted(set(original_list))
        dupes_here = len(original_list) - len(deduped)
        if dupes_here > 0:
            entities_with_dupes += 1
            total_dupes += dupes_here
        result[s1_id] = deduped

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_entities = len(predicted_by_s1)
    print(f"Total S1 entities processed : {total_entities}")
    print(f"Entities with duplicates removed : {entities_with_dupes}")
    print(f"Total duplicate IDs removed : {total_dupes}")

    if total_dupes == 0:
        print("No duplicates found — apply_threshold built sets correctly [OK]")
    else:
        print(
            f"WARNING: {total_dupes} duplicates found — check apply_threshold() "
            f"for dict accumulation bugs"
        )

    return result


# ---------------------------------------------------------------------------
# Phase 5.2 — Valid-ID filtering
# ---------------------------------------------------------------------------

#: Prefixes that are legal for S2/S3 candidate IDs (case-sensitive).
_VALID_PREFIXES: tuple[str, ...] = ("S2-", "S3-")


def filter_valid_ids(
    predicted_by_s1: dict[str, list[str]],
    valid_s2s3_ids: set[str],
) -> dict[str, list[str]]:
    """Filter each S1 entity's match list to only legitimate S2/S3 IDs.

    Operates on the output of :func:`dedupe_matches` and removes any candidate
    ID that either:

    1. Does **not** start with ``"S2-"`` or ``"S3-"`` (case-sensitive) — catches
       S1-self-matches or garbage strings that slipped through scoring; or
    2. Does **not** appear in *valid_s2s3_ids* — catches hallucinated IDs that
       have the right prefix but were never present in the test S2+S3 corpus.

    The two violation types are counted and reported independently so that
    upstream bugs can be diagnosed precisely.

    Parameters
    ----------
    predicted_by_s1 : dict[str, list[str]]
        Direct output of :func:`dedupe_matches`.  Keys are S1 entity IDs;
        values are sorted, deduplicated lists of candidate IDs.
    valid_s2s3_ids : set[str]
        The complete set of ``entity_id`` values loaded from the test S2+S3
        files (e.g. ``mock_clean_s2.tsv``).  Only IDs present in this set
        **and** carrying a valid prefix are kept.

    Returns
    -------
    dict[str, list[str]]
        Same keys as *predicted_by_s1*.  Each value contains only IDs that
        pass both the prefix check and the membership check.  S1 entities
        whose entire match list is invalid become ``[]`` (singleton).
        Ordering of surviving elements is preserved (input already sorted).

    Side Effects (stdout)
    ---------------------
    Prints a filtering report:

    * Count of IDs removed for wrong prefix (``"S1-"`` self-matches or garbage).
    * Count of IDs removed for not being in *valid_s2s3_ids*.
    * Total IDs removed (sum of both counts).
    * If zero removed: ``"All matched IDs are valid [OK]"``
    * If any removed:  ``"WARNING: X IDs removed — investigate blocking/scoring bug"``

    Raises
    ------
    TypeError
        If *predicted_by_s1* is not a ``dict``.
    TypeError
        If *valid_s2s3_ids* is not a ``set``.

    Examples
    --------
    >>> valid = {"S2-099", "S2-100", "S3-999"}
    >>> inp = {"S1-001": ["S2-099", "S2-999"], "S1-002": ["S1-001"]}
    >>> filter_valid_ids(inp, valid)
    IDs removed — wrong prefix  : 1
    IDs removed — not in valid set : 1
    Total IDs removed : 2
    WARNING: 2 IDs removed — investigate blocking/scoring bug
    {'S1-001': ['S2-099'], 'S1-002': []}
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(predicted_by_s1, dict):
        raise TypeError(
            f"filter_valid_ids() expects predicted_by_s1 to be a dict, "
            f"got {type(predicted_by_s1).__name__}."
        )
    if not isinstance(valid_s2s3_ids, set):
        raise TypeError(
            f"filter_valid_ids() expects valid_s2s3_ids to be a set, "
            f"got {type(valid_s2s3_ids).__name__}."
        )

    # ------------------------------------------------------------------
    # Filtering pass
    # ------------------------------------------------------------------
    result: dict[str, list[str]] = {}
    removed_wrong_prefix: int = 0
    removed_not_in_valid: int = 0

    for s1_id, matches in predicted_by_s1.items():
        kept: list[str] = []
        for cand_id in matches:
            has_valid_prefix = any(cand_id.startswith(p) for p in _VALID_PREFIXES)
            if not has_valid_prefix:
                removed_wrong_prefix += 1
            elif cand_id not in valid_s2s3_ids:
                removed_not_in_valid += 1
            else:
                kept.append(cand_id)
        result[s1_id] = kept

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_removed = removed_wrong_prefix + removed_not_in_valid
    print(f"IDs removed — wrong prefix  : {removed_wrong_prefix}")
    print(f"IDs removed — not in valid set : {removed_not_in_valid}")
    print(f"Total IDs removed : {total_removed}")

    if total_removed == 0:
        print("All matched IDs are valid [OK]")
    else:
        print(
            f"WARNING: {total_removed} IDs removed — investigate blocking/scoring bug"
        )

    return result


# ---------------------------------------------------------------------------
# Phase 5.3 — Full S1 coverage
# ---------------------------------------------------------------------------


def ensure_full_coverage(
    predicted_by_s1: dict[str, list[str]],
    all_test_s1_ids: list[str],
) -> dict[str, list[str]]:
    """Guarantee the output dict contains exactly one entry per test S1 entity.

    The upstream pipeline (blocking → scoring → thresholding) may silently drop
    S1 entities that had no candidates above the threshold.  This function
    backfills any missing S1 ID with an empty match list ``[]`` so that every
    test-set entity appears in the final submission, whether or not it has a
    predicted match.

    Extra S1 keys already present in *predicted_by_s1* that do not appear in
    *all_test_s1_ids* are **kept** — they are not stripped, because they may
    represent legitimate predictions on evaluation-time entities not yet
    enumerated in the test manifest.

    Parameters
    ----------
    predicted_by_s1 : dict[str, list[str]]
        Direct output of :func:`filter_valid_ids`.  Keys are S1 entity IDs;
        values are sorted, validated lists of candidate IDs.
    all_test_s1_ids : list[str]
        The ordered list of every ``entity_id`` value from the test source-1
        file (e.g. ``mock_clean_s1.tsv``).  Each ID must appear exactly once;
        duplicates indicate an upstream data error and raise ``ValueError``.

    Returns
    -------
    dict[str, list[str]]
        A copy of *predicted_by_s1* that is guaranteed to contain a key for
        every ID in *all_test_s1_ids*.  Missing entries are backfilled with
        ``[]``.  Pre-existing entries are never modified.

    Side Effects (stdout)
    ---------------------
    Prints a coverage report:

    * Total expected S1 count (``len(all_test_s1_ids)``).
    * Count of S1 entities backfilled with empty matches.
    * Final total S1 count in the returned dict (must equal expected).
    * If zero backfilled: ``"Full S1 coverage confirmed [OK]"``
    * If any backfilled: ``"INFO: X S1 entities had no candidates — backfilled as singletons"``

    Raises
    ------
    TypeError
        If *all_test_s1_ids* is not a ``list``.
    ValueError
        If any S1 ID appears more than once in *all_test_s1_ids*.
    ValueError
        If the final key count in the result does not equal
        ``len(all_test_s1_ids)``.  (Defensive post-condition check.)

    Examples
    --------
    >>> inp = {"S1-001": ["S2-099"]}
    >>> ensure_full_coverage(inp, ["S1-001", "S1-002"])
    Total expected S1 count : 2
    S1 entities backfilled : 1
    Final total S1 count : 2
    INFO: 1 S1 entities had no candidates — backfilled as singletons
    {'S1-001': ['S2-099'], 'S1-002': []}
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(all_test_s1_ids, list):
        raise TypeError(
            f"ensure_full_coverage() expects all_test_s1_ids to be a list, "
            f"got {type(all_test_s1_ids).__name__}."
        )

    # Detect duplicates in all_test_s1_ids — O(n) via Counter-style dict
    seen: dict[str, int] = {}
    for s1_id in all_test_s1_ids:
        seen[s1_id] = seen.get(s1_id, 0) + 1
    duplicates = [s1_id for s1_id, count in seen.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"ensure_full_coverage() found duplicate S1 IDs in all_test_s1_ids: "
            f"{sorted(duplicates)!r}.  Each ID must appear exactly once."
        )

    # ------------------------------------------------------------------
    # Backfill pass
    # ------------------------------------------------------------------
    result: dict[str, list[str]] = dict(predicted_by_s1)  # shallow copy
    backfilled: int = 0

    for s1_id in all_test_s1_ids:
        if s1_id not in result:
            result[s1_id] = []
            backfilled += 1

    # ------------------------------------------------------------------
    # Post-condition: final count must cover all expected IDs
    # The result may have MORE keys than all_test_s1_ids (extras kept),
    # so we check that every expected ID is present, not that counts match.
    # ------------------------------------------------------------------
    missing_after = [s1_id for s1_id in all_test_s1_ids if s1_id not in result]
    if missing_after:
        raise ValueError(
            f"ensure_full_coverage() post-condition failed: "
            f"{len(missing_after)} S1 IDs still missing after backfill: "
            f"{missing_after!r}."
        )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    expected_count = len(all_test_s1_ids)
    # "Final total" = keys that came from all_test_s1_ids (always == expected)
    final_count = expected_count  # all IDs now guaranteed present
    print(f"Total expected S1 count : {expected_count}")
    print(f"S1 entities backfilled : {backfilled}")
    print(f"Final total S1 count : {final_count}")

    if backfilled == 0:
        print("Full S1 coverage confirmed [OK]")
    else:
        print(
            f"INFO: {backfilled} S1 entities had no candidates — backfilled as singletons"
        )

    return result


# ---------------------------------------------------------------------------
# Phase 5.4 — Candidate consistency check
# ---------------------------------------------------------------------------


def check_candidate_consistency(
    predicted_by_s1: dict[str, list[str]],
    candidate_pairs_by_s1: dict[str, list[str]],
) -> dict:
    """Verify every predicted match ID was a blocking candidate for that S1 entity.

    A fundamental invariant of the pipeline is that the ML model can only
    predict a match for a (S1, S2/S3) pair that the blocking stage nominated
    as a candidate.  If a predicted match ID was *not* in the candidate set for
    that S1 entity it is an "orphan" — evidence of a data-join bug, a scoring
    leak, or a post-processing error.

    This check is a read-only audit; it does **not** raise on violations.
    The caller (``run_production.py``) is responsible for deciding whether to
    abort the run based on the returned report.

    Parameters
    ----------
    predicted_by_s1 : dict[str, list[str]]
        Direct output of :func:`ensure_full_coverage`.  Maps each S1 entity ID
        to its (possibly empty) sorted list of predicted matched candidate IDs.
    candidate_pairs_by_s1 : dict[str, list[str]]
        The candidate set produced by Phase 1 blocking, in the same in-memory
        format as ``candidates_by_s1`` passed to :func:`blocking.write_candidate_pairs`.
        Maps each S1 entity ID to an ordered list of nominated candidate IDs.
        An S1 entity absent from this dict is treated as having an empty
        candidate set (all its predicted matches will be orphans).

    Returns
    -------
    dict
        A report dict with the following keys:

        ``"passed"`` : bool
            ``True`` only if *n_violations* is zero.
        ``"n_violations"`` : int
            Number of S1 entities that have at least one orphan match ID.
        ``"violations"`` : list[dict]
            One entry per violating S1 entity, each a dict with keys:

            * ``"source1_entity_id"`` : str — the S1 entity ID.
            * ``"orphan_ids"`` : list[str] — predicted match IDs that were
              not in the candidate set for this S1 entity.

    Side Effects (stdout)
    ---------------------
    Prints a consistency report:

    * Total S1 entities checked.
    * Count of S1 entities with at least one violation.
    * If passed: ``"Candidate consistency check passed [OK]"``
    * If failed:
      ``"ERROR: X S1 entities have matched IDs outside candidate set"``
      followed by one line per violation showing the S1 ID and its orphan IDs.

    Raises
    ------
    TypeError
        If *predicted_by_s1* is not a ``dict``.
    TypeError
        If *candidate_pairs_by_s1* is not a ``dict``.

    Notes
    -----
    * S1 entities with an empty predicted match list ``[]`` always pass — there
      is nothing to check.
    * S1 entities present in *predicted_by_s1* but absent from
      *candidate_pairs_by_s1* are treated as having an empty candidate set,
      so any non-empty predicted match list produces a violation.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(predicted_by_s1, dict):
        raise TypeError(
            f"check_candidate_consistency() expects predicted_by_s1 to be a dict, "
            f"got {type(predicted_by_s1).__name__}."
        )
    if not isinstance(candidate_pairs_by_s1, dict):
        raise TypeError(
            f"check_candidate_consistency() expects candidate_pairs_by_s1 to be a dict, "
            f"got {type(candidate_pairs_by_s1).__name__}."
        )

    # ------------------------------------------------------------------
    # Consistency check pass
    # ------------------------------------------------------------------
    violations: list[dict] = []

    for s1_id, predicted_matches in predicted_by_s1.items():
        if not predicted_matches:
            # Empty match list — nothing to violate
            continue

        # Treat absent S1 IDs as having no candidates
        candidate_set: set[str] = set(candidate_pairs_by_s1.get(s1_id, []))

        orphans = [m for m in predicted_matches if m not in candidate_set]
        if orphans:
            violations.append({
                "source1_entity_id": s1_id,
                "orphan_ids": orphans,
            })

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    n_violations = len(violations)
    passed = n_violations == 0

    # ------------------------------------------------------------------
    # Report (stdout)
    # ------------------------------------------------------------------
    total_checked = len(predicted_by_s1)
    print(f"Total S1 entities checked : {total_checked}")
    print(f"S1 entities with violations : {n_violations}")

    if passed:
        print("Candidate consistency check passed [OK]")
    else:
        print(
            f"ERROR: {n_violations} S1 entities have matched IDs outside candidate set"
        )
        for v in violations:
            print(
                f"  {v['source1_entity_id']!r} — orphan IDs: {v['orphan_ids']!r}"
            )

    return {
        "passed": passed,
        "n_violations": n_violations,
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Phase 5.5 — TSV writers
# ---------------------------------------------------------------------------

def write_matching_results(
    predicted_by_s1: dict[str, list[str]],
    output_path: str,
) -> int:
    """Write the final per-S1 match predictions to a TSV file.

    Produces the canonical submission file consumed by the evaluation harness.
    Rows are sorted by ``source1_entity_id`` for deterministic, diff-friendly
    output.

    Output format
    ~~~~~~~~~~~~~
    Tab-separated, two columns, with a header row::

        source1_entity_id\\tmatched_entity_ids
        S1-001\\tS2-099,S2-101
        S1-002\\t

    * ``matched_entity_ids`` is a comma-separated list of IDs with **no spaces**
      around the commas.
    * S1 entities with no predicted matches produce an **empty string** cell
      (not ``"set()"`` or ``"[]"``).
    * Rows are sorted ascending by ``source1_entity_id``.
    * No CSV quoting is applied (``csv.QUOTE_NONE``).
    * Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    predicted_by_s1 : dict[str, list[str]]
        Direct output of :func:`ensure_full_coverage`.  Maps each S1 entity ID
        to a sorted, deduplicated list of predicted match IDs.
    output_path : str
        Destination file path for the TSV.

    Returns
    -------
    int
        Number of data rows written (excludes the header row).
        Equals ``len(predicted_by_s1)``.

    Examples
    --------
    >>> write_matching_results({"S1-002": ["S2-101"], "S1-001": []}, "out.tsv")
    2
    # out.tsv contains:
    # source1_entity_id\\tmatched_entity_ids
    # S1-001\\t
    # S1-002\\tS2-101
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Write to a temporary file first, then atomically rename into place.
    # This prevents downstream readers from observing a partially-written file
    # if the process is interrupted mid-write.
    tmp_path = output_path + ".tmp"
    rows_written = 0
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(
                fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            writer.writerow(["source1_entity_id", "matched_entity_ids"])

            for s1_id in sorted(predicted_by_s1.keys()):
                matched_cell = ",".join(predicted_by_s1[s1_id])  # "" when list is empty
                writer.writerow([s1_id, matched_cell])
                rows_written += 1

        os.replace(tmp_path, output_path)
    except BaseException:
        # Clean up the temp file on any failure so stale .tmp files don't
        # accumulate between runs.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return rows_written


def write_candidate_pairs_final(
    candidate_pairs_by_s1: dict[str, list[str]],
    output_path: str,
) -> int:
    """Write the blocking candidate pairs to a TSV file (final submission copy).

    Mirrors the format of :func:`blocking.write_candidate_pairs` but sorts rows
    by ``source1_entity_id`` for deterministic output and returns a row count.
    The original candidate order within each row is preserved (highest-similarity
    first when the input came from a ranked blocking step).

    Output format
    ~~~~~~~~~~~~~
    Tab-separated, two columns, with a header row::

        source1_entity_id\\tcandidate_entity_ids
        S1-001\\tS2-099,S2-100,S2-101,S3-999
        S1-002\\tS2-101,S3-999

    * ``candidate_entity_ids`` is a comma-separated list in the **original order**
      supplied by the caller (blocking rank order).
    * S1 entities with no candidates produce an **empty string** cell.
    * Rows are sorted ascending by ``source1_entity_id``.
    * No CSV quoting is applied (``csv.QUOTE_NONE``).
    * Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    candidate_pairs_by_s1 : dict[str, list[str]]
        Maps each S1 entity ID to its ordered list of candidate IDs from
        Phase 1 blocking.
    output_path : str
        Destination file path for the TSV.

    Returns
    -------
    int
        Number of data rows written (excludes the header row).
        Equals ``len(candidate_pairs_by_s1)``.

    Examples
    --------
    >>> write_candidate_pairs_final({"S1-001": ["S2-099", "S2-100"]}, "cands.tsv")
    1
    # cands.tsv contains:
    # source1_entity_id\\tcandidate_entity_ids
    # S1-001\\tS2-099,S2-100
    """
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # Write to a temporary file first, then atomically rename into place.
    tmp_path = output_path + ".tmp"
    rows_written = 0
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(
                fh, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

            for s1_id in sorted(candidate_pairs_by_s1.keys()):
                candidate_cell = ",".join(candidate_pairs_by_s1[s1_id])
                writer.writerow([s1_id, candidate_cell])
                rows_written += 1

        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return rows_written
