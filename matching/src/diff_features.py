"""
diff_features.py
----------------
Phase 2 — Difference Features (V6 addition).

Merges three feature groups from Sid-techweb/AmazonML-New into the existing
Business Entity Resolution pipeline.

SOURCE (Sid's repo):
    business_entity_resolution/src/features.py  — compute_features()
        - tok_inter, tok_jacc, tok_q_cov, tok_s_cov
        - nm_q_only, nm_s_only, nm_diff_ratio
        - num_q, num_s, num_inter, num_q_only, num_s_only, num_q_cov,
          num_conflict, num_set_eq, num_first_eq, num_rel_diff
        - ad_q_only

    business_entity_resolution/src/v2/featx.py  — numbers()
        - prem_eq, prem_in_s, s_prem_in_q, q_nums_subset, prem_close

Adaptation notes:
    * Sid uses Polars + rapidfuzz.process.cpdist (batch vectorised).
      We use pandas + rapidfuzz.fuzz (per-pair) to match the existing pipeline.
    * Sid's "q" side = our candidate (S2/S3); Sid's "s" / S1 side = our S1.
      Feature naming keeps the semantic: _s1_ = S1, _cand_ = candidate.
    * Sid uses name_core (key-words only) for token features; we use clean_name
      (already normalised by our pipeline).  The semantic is equivalent after
      our normalisation.
    * address house-number extraction follows the same first-number logic as
      Sid's addr_nums column (``re.findall(r"\\d+", addr)``), which is simpler
      than our address_components.py parser.  We keep address_components.py
      for the full component features (V3) and add first-number features here
      as a separate, cheaper computation.
    * num_rel_diff: Sid computes over first number of each address.  We compute
      the same way.  -1.0 sentinel when one or both sides have no first number.

Feature columns added (15 new features):
    --- Token Difference: name ---
    name_common_token_count       int  count of tokens in both names
    name_s1_only_token_count      int  count of tokens only in S1 name
    name_cand_only_token_count    int  count of tokens only in candidate name
    name_token_difference_ratio   float  (s1_only + cand_only) / (common + s1_only + cand_only + ε)
    name_diff_token_sim           float  fuzzy ratio of the leftover tokens (0..1); 1.0 if no diff tokens

    --- Token Difference: address ---
    addr_common_token_count       int  count of tokens in both addresses
    addr_s1_only_token_count      int  count of tokens only in S1 address
    addr_cand_only_token_count    int  count of tokens only in candidate address

    --- Numeric Difference: address numbers ---
    numeric_common_count          int  digit-sequences in both addresses
    numeric_s1_only_count         int  digit-sequences only in S1
    numeric_cand_only_count       int  digit-sequences only in candidate
    numeric_overlap_ratio         float  common / max(s1_count, cand_count)  (0 when neither has nums)
    numeric_conflict              int  {0,1}: both have nums AND no overlap
    numeric_set_equal             int  {0,1}: sets of digit-sequences are identical
    numeric_rel_diff              float  |first_num_s1 - first_num_cand| / max(abs); -1 if missing

All values:
    - Numeric dtype (int or float)
    - Never NaN
    - numeric_rel_diff uses -1.0 as sentinel when a first number is absent
    - All float features in [-1, 1] (numeric_rel_diff can be -1.0)
    - All count features ≥ 0

Public API
~~~~~~~~~~
    DIFF_FEATURE_COLS : list[str]  — 15 feature column names
    add_diff_features(pair_df)     — appends DIFF_FEATURE_COLS to pair table
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# ---------------------------------------------------------------------------
# Feature column names (stable, used in FEATURE_COLUMNS_V6)
# ---------------------------------------------------------------------------

#: 15 difference feature columns added in V6.
DIFF_FEATURE_COLS: list[str] = [
    # Token difference — name
    "name_common_token_count",
    "name_s1_only_token_count",
    "name_cand_only_token_count",
    "name_token_difference_ratio",
    "name_diff_token_sim",
    # Token difference — address
    "addr_common_token_count",
    "addr_s1_only_token_count",
    "addr_cand_only_token_count",
    # Numeric difference — address
    "numeric_common_count",
    "numeric_s1_only_count",
    "numeric_cand_only_count",
    "numeric_overlap_ratio",
    "numeric_conflict",
    "numeric_set_equal",
    "numeric_rel_diff",
]

# ---------------------------------------------------------------------------
# Internal helpers (adapted from Sid's compute_features / numbers())
# ---------------------------------------------------------------------------

_RE_DIGITS = re.compile(r"\d+")


def _to_str(val: Any) -> str:
    """Coerce a possibly-missing value to a stripped string."""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:   # NaN check
        return ""
    return str(val).strip()


def _tokenise(text: str) -> list[str]:
    """Whitespace tokenise; returns empty list for blank strings."""
    return text.split() if text else []


def _extract_nums(text: str) -> list[str]:
    """Extract all digit sequences from address text (Sid: addr_nums).

    Replicates Sid's: ``pl.col(col).str.extract_all(r"\\d+")``
    Returns a list of string digit-sequences in order of appearance.
    Duplicates are preserved (dedup handled by callers using sets).
    """
    return _RE_DIGITS.findall(text)


def _token_diff_features_name(s1: str, cand: str) -> dict:
    """Token difference features for a name pair.

    Adapted from Sid's compute_features():
        tok_inter = |qt ∩ st|
        nm_q_only  = |qt \\ st|   (Sid: query-only tokens)
        nm_s_only  = |st \\ qt|   (Sid: S1-only tokens)
        nm_diff_ratio = fuzz.ratio(sorted(qt\\st), sorted(st\\qt)) / 100
            → 100.0 when both diff-sets are empty (identical names)

    In our naming convention:
        - S1 = Sid's "s" (S1 catalogue side)
        - candidate = Sid's "q" (query side, S2/S3)
    """
    toks_s1   = set(_tokenise(s1))
    toks_cand = set(_tokenise(cand))

    common      = toks_s1 & toks_cand
    s1_only     = toks_s1 - toks_cand
    cand_only   = toks_cand - toks_s1

    n_common    = len(common)
    n_s1_only   = len(s1_only)
    n_cand_only = len(cand_only)
    n_total     = n_common + n_s1_only + n_cand_only

    # token difference ratio: fraction of total unique tokens that differ
    diff_ratio = float(n_s1_only + n_cand_only) / float(n_total) if n_total > 0 else 0.0

    # Fuzzy similarity of the differing portions (Sid: nm_diff_ratio)
    # Sort to stabilise ordering (Sid uses list.sort().list.join())
    s1_diff_str   = " ".join(sorted(s1_only))
    cand_diff_str = " ".join(sorted(cand_only))
    if not s1_diff_str and not cand_diff_str:
        # both empty → identical names → full similarity
        diff_sim = 1.0
    else:
        diff_sim = fuzz.ratio(s1_diff_str, cand_diff_str) / 100.0

    return {
        "name_common_token_count":    n_common,
        "name_s1_only_token_count":   n_s1_only,
        "name_cand_only_token_count": n_cand_only,
        "name_token_difference_ratio": diff_ratio,
        "name_diff_token_sim":        diff_sim,
    }


def _token_diff_features_addr(s1: str, cand: str) -> dict:
    """Token difference features for an address pair.

    Adapted from Sid's tok DataFrame construction:
        ad_inter  = |qa ∩ sa|
        ad_q_only = |qa \\ sa|   (Sid: query's extra tokens)

    Here we add the symmetric s1_only count as well.
    """
    toks_s1   = set(_tokenise(s1))
    toks_cand = set(_tokenise(cand))

    n_common    = len(toks_s1 & toks_cand)
    n_s1_only   = len(toks_s1 - toks_cand)
    n_cand_only = len(toks_cand - toks_s1)

    return {
        "addr_common_token_count":    n_common,
        "addr_s1_only_token_count":   n_s1_only,
        "addr_cand_only_token_count": n_cand_only,
    }


def _numeric_diff_features(s1_addr: str, cand_addr: str) -> dict:
    """Numeric difference features over digit sequences in two address strings.

    Adapted from Sid's compute_features() tok block:
        num_q  = |unique(qn)|        (Sid: query numeric count)
        num_s  = |unique(sn)|        (Sid: S1 numeric count)
        num_inter = |unique(qn) ∩ unique(sn)|
        num_q_only = |qn \\ sn|
        num_s_only = |sn \\ qn|
        num_q_cov  = num_inter / max(num_q, 1)
        num_conflict = (num_q > 0) & (num_s > 0) & (num_inter == 0)
        num_set_eq   = sorted(unique(qn)) == sorted(unique(sn))
        num_first_eq = qn[0] == sn[0]   (first digit sequence)

    And from Sid's compute_features() final .with_columns():
        num_rel_diff = |qn1 - sn1| / max(qn1, sn1, 1)
            (relative difference of first number; -1.0 if either is missing)
    """
    nums_s1   = list(dict.fromkeys(_extract_nums(s1_addr)))    # unique, order-preserved
    nums_cand = list(dict.fromkeys(_extract_nums(cand_addr)))

    set_s1   = set(nums_s1)
    set_cand = set(nums_cand)

    common      = set_s1 & set_cand
    s1_only     = set_s1 - set_cand
    cand_only   = set_cand - set_s1

    n_common    = len(common)
    n_s1_only   = len(s1_only)
    n_cand_only = len(cand_only)

    # Numeric overlap ratio: shared / max(s1_count, cand_count)
    max_count = max(len(set_s1), len(set_cand))
    overlap_ratio = float(n_common) / float(max_count) if max_count > 0 else 0.0

    # Conflict: both sides have numbers but share none
    conflict = int(bool(set_s1) and bool(set_cand) and n_common == 0)

    # Set equality: same digit-sequence sets
    set_equal = int(set_s1 == set_cand)

    # Relative difference of first number (Sid: num_rel_diff)
    # -1.0 sentinel when either side has no first number
    if nums_s1 and nums_cand:
        try:
            v_s1   = float(nums_s1[0])
            v_cand = float(nums_cand[0])
            denom  = max(abs(v_s1), abs(v_cand), 1.0)
            rel_diff = abs(v_s1 - v_cand) / denom
        except (ValueError, OverflowError):
            rel_diff = -1.0
    else:
        rel_diff = -1.0   # sentinel: one or both sides have no number

    return {
        "numeric_common_count":    n_common,
        "numeric_s1_only_count":   n_s1_only,
        "numeric_cand_only_count": n_cand_only,
        "numeric_overlap_ratio":   overlap_ratio,
        "numeric_conflict":        conflict,
        "numeric_set_equal":       set_equal,
        "numeric_rel_diff":        rel_diff,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_diff_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 15 difference features for every candidate pair.

    Reads ``s1_clean_name``, ``candidate_clean_name``, ``s1_clean_address``,
    and ``candidate_clean_address`` from *pair_df*.

    Appends 15 new columns in the order defined by :data:`DIFF_FEATURE_COLS`.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table.  Must contain at least:
        - ``source1_entity_id``
        - ``candidate_entity_id``
        - ``s1_clean_name``
        - ``candidate_clean_name``
        - ``s1_clean_address``
        - ``candidate_clean_address``

    Returns
    -------
    pd.DataFrame
        Copy of *pair_df* with 15 new columns appended.
        All values are numeric, finite, and never NaN.
        Count columns have integer dtype.
        Float columns have float64 dtype.
        ``numeric_rel_diff`` uses -1.0 as a sentinel meaning "one or both
        sides have no numeric token in the address".

    Raises
    ------
    ValueError
        If required columns are missing from *pair_df*.
    """
    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "s1_clean_name",
        "candidate_clean_name",
        "s1_clean_address",
        "candidate_clean_address",
    }
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(
            f"add_diff_features(): pair_df is missing required column(s): "
            f"{sorted(missing)}.  Found: {list(pair_df.columns)}"
        )

    out_df = pair_df.copy()

    # Handle empty table
    if out_df.empty:
        int_cols = {
            "name_common_token_count", "name_s1_only_token_count",
            "name_cand_only_token_count",
            "addr_common_token_count", "addr_s1_only_token_count",
            "addr_cand_only_token_count",
            "numeric_common_count", "numeric_s1_only_count",
            "numeric_cand_only_count",
            "numeric_conflict", "numeric_set_equal",
        }
        for col in DIFF_FEATURE_COLS:
            dtype = int if col in int_cols else float
            out_df[col] = pd.Series(dtype=dtype)
        return out_df

    # Pre-extract all name/address strings
    s1_names   = [_to_str(x) for x in out_df["s1_clean_name"]]
    cand_names = [_to_str(x) for x in out_df["candidate_clean_name"]]
    s1_addrs   = [_to_str(x) for x in out_df["s1_clean_address"]]
    cand_addrs = [_to_str(x) for x in out_df["candidate_clean_address"]]

    # Accumulate results
    results: dict[str, list] = {col: [] for col in DIFF_FEATURE_COLS}

    for s1_n, cand_n, s1_a, cand_a in zip(s1_names, cand_names, s1_addrs, cand_addrs):
        # Token difference — name
        nf = _token_diff_features_name(s1_n, cand_n)
        results["name_common_token_count"].append(nf["name_common_token_count"])
        results["name_s1_only_token_count"].append(nf["name_s1_only_token_count"])
        results["name_cand_only_token_count"].append(nf["name_cand_only_token_count"])
        results["name_token_difference_ratio"].append(nf["name_token_difference_ratio"])
        results["name_diff_token_sim"].append(nf["name_diff_token_sim"])

        # Token difference — address
        af = _token_diff_features_addr(s1_a, cand_a)
        results["addr_common_token_count"].append(af["addr_common_token_count"])
        results["addr_s1_only_token_count"].append(af["addr_s1_only_token_count"])
        results["addr_cand_only_token_count"].append(af["addr_cand_only_token_count"])

        # Numeric difference
        ndf = _numeric_diff_features(s1_a, cand_a)
        results["numeric_common_count"].append(ndf["numeric_common_count"])
        results["numeric_s1_only_count"].append(ndf["numeric_s1_only_count"])
        results["numeric_cand_only_count"].append(ndf["numeric_cand_only_count"])
        results["numeric_overlap_ratio"].append(ndf["numeric_overlap_ratio"])
        results["numeric_conflict"].append(ndf["numeric_conflict"])
        results["numeric_set_equal"].append(ndf["numeric_set_equal"])
        results["numeric_rel_diff"].append(ndf["numeric_rel_diff"])

    # Assign columns with correct dtypes
    int_cols = {
        "name_common_token_count", "name_s1_only_token_count",
        "name_cand_only_token_count",
        "addr_common_token_count", "addr_s1_only_token_count",
        "addr_cand_only_token_count",
        "numeric_common_count", "numeric_s1_only_count",
        "numeric_cand_only_count",
        "numeric_conflict", "numeric_set_equal",
    }
    for col in DIFF_FEATURE_COLS:
        if col in int_cols:
            out_df[col] = np.array(results[col], dtype=np.int32)
        else:
            out_df[col] = np.array(results[col], dtype=np.float64)

    return out_df
