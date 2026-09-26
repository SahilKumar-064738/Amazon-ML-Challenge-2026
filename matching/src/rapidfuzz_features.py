"""
rapidfuzz_features.py
---------------------
Phase 3 — Sid RapidFuzz Feature Family.

Merges the proven RapidFuzz feature group from Sid-techweb/AmazonML-New
into the Business Entity Resolution pipeline.

SOURCE (Sid's repo):
    business_entity_resolution/src/features.py (lines 68–86, 120–122):
        f = {
            "nm_ratio":     _cp(fuzz.ratio, qc, sc),
            "nm_tset":      _cp(fuzz.token_set_ratio, qc, sc),
            "nm_tsort":     _cp(fuzz.token_sort_ratio, qc, sc),
            "nm_partial":   _cp(fuzz.partial_ratio, qc, sc),
            "key_ratio":    _cp(fuzz.ratio, qk, sk),
            "key_partial":  _cp(fuzz.partial_ratio, qk, sk),
            "key_jw":       _cp(JaroWinkler.normalized_similarity, qk, sk),
            "full_ratio":   _cp(fuzz.ratio, Q["name_norm"], S["name_norm"]),
            "ad_ratio":     _cp(fuzz.ratio, qa, sa),
            "ad_tset":      _cp(fuzz.token_set_ratio, qa, sa),
            "ad_tsort":     _cp(fuzz.token_sort_ratio, qa, sa),
            "ad_partial":   _cp(fuzz.partial_ratio, qa, sa),
        }
        ad_qonly_in_s = _cp(fuzz.partial_ratio, tok["ad_q_diff"], S["addr_norm"])

MAPPING & DEDUPLICATION:
    - nm_tsort and nm_tset are already in Phase 2.2 (name_token_sort_ratio, name_token_set_ratio) -> Skipped.
    - name_jaro_winkler is in Phase 2.2 -> Skipped.
    - name_levenshtein is in Phase 2.2 -> Kept as-is.
    - Genuinely new RapidFuzz signals added here (10 features):
        1. name_fuzz_ratio          -- fuzz.ratio on clean names in [0, 1]
        2. name_partial_ratio       -- fuzz.partial_ratio on clean names in [0, 1]
        3. name_key_ratio           -- fuzz.ratio on space-free key strings in [0, 1]
        4. name_key_partial_ratio   -- fuzz.partial_ratio on space-free key strings in [0, 1]
        5. name_key_jaro_winkler    -- JaroWinkler on space-free key strings in [0, 1]
        6. addr_fuzz_ratio          -- fuzz.ratio on clean addresses in [0, 1]
        7. addr_partial_ratio       -- fuzz.partial_ratio on clean addresses in [0, 1]
        8. addr_token_sort_ratio    -- fuzz.token_sort_ratio on addresses in [0, 1]
        9. addr_token_set_ratio     -- fuzz.token_set_ratio on addresses in [0, 1]
       10. addr_diff_token_sim      -- fuzz.partial_ratio of candidate diff-address tokens in S1 address in [0, 1]
                                       (1.0 if candidate has no differing address tokens)

ALL FEATURES:
    - Bounded in [0.0, 1.0]
    - Strictly NaN-free and Inf-free
    - Deterministic
    - Pure pair-level computation: chunkable, memory-safe, no global matrices
    - Respects existing normalization and data cleaning

Public API:
    RAPIDFUZZ_FEATURE_COLS : list[str]
    add_rapidfuzz_features(pair_df) -> pd.DataFrame
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


# ---------------------------------------------------------------------------
# Feature Column Names (10 features)
# ---------------------------------------------------------------------------

RAPIDFUZZ_FEATURE_COLS: list[str] = [
    # Name RapidFuzz features
    "name_fuzz_ratio",
    "name_partial_ratio",
    "name_key_ratio",
    "name_key_partial_ratio",
    "name_key_jaro_winkler",
    # Address RapidFuzz features
    "addr_fuzz_ratio",
    "addr_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_diff_token_sim",
]


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

_RE_WHITESPACE = re.compile(r"\s+")


def _safe_str(val: Any) -> str:
    """Coerce value to a non-null stripped string."""
    if val is None or (isinstance(val, float) and val != val):
        return ""
    return str(val).strip()


def _make_key(text: str) -> str:
    """Remove all spaces to produce a compound-resilient key string."""
    return _RE_WHITESPACE.sub("", text)


def _token_diff(s_main: str, s_sub: str) -> str:
    """Return a string of tokens present in s_main but absent from s_sub."""
    if not s_main:
        return ""
    toks_main = set(s_main.split())
    toks_sub = set(s_sub.split()) if s_sub else set()
    diff = toks_main - toks_sub
    return " ".join(sorted(diff)) if diff else ""


# ---------------------------------------------------------------------------
# Public Feature Extraction Function
# ---------------------------------------------------------------------------

def add_rapidfuzz_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 10 RapidFuzz similarity features for candidate pairs.

    Reads ``s1_clean_name``, ``candidate_clean_name``, ``s1_clean_address``,
    and ``candidate_clean_address`` from `pair_df`.

    Appends 10 new columns in the order defined by :data:`RAPIDFUZZ_FEATURE_COLS`.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table. Must contain:
        - ``source1_entity_id``
        - ``candidate_entity_id``
        - ``s1_clean_name``
        - ``candidate_clean_name``
        - ``s1_clean_address``
        - ``candidate_clean_address``

    Returns
    -------
    pd.DataFrame
        Copy of `pair_df` with the 10 RapidFuzz feature columns appended.
        All values are finite numeric floats in [0.0, 1.0]. Never NaN.
    """
    if not isinstance(pair_df, pd.DataFrame):
        raise TypeError(f"add_rapidfuzz_features: expected pd.DataFrame, got {type(pair_df).__name__}")

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
            f"add_rapidfuzz_features: pair_df missing required column(s): {sorted(missing)}. "
            f"Found: {list(pair_df.columns)}"
        )

    out_df = pair_df.copy()

    # Empty table edge case
    if out_df.empty:
        for col in RAPIDFUZZ_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=np.float64)
        return out_df

    # Extract clean text representations
    s1_names = [_safe_str(x) for x in out_df["s1_clean_name"]]
    cand_names = [_safe_str(x) for x in out_df["candidate_clean_name"]]
    s1_addrs = [_safe_str(x) for x in out_df["s1_clean_address"]]
    cand_addrs = [_safe_str(x) for x in out_df["candidate_clean_address"]]

    # Space-free key representations (for compound words like "Wall Mart" vs "Walmart")
    s1_name_keys = [_make_key(n) for n in s1_names]
    cand_name_keys = [_make_key(n) for n in cand_names]

    # Pre-allocate feature arrays
    n = len(out_df)
    name_fuzz_ratio = np.zeros(n, dtype=np.float64)
    name_partial_ratio = np.zeros(n, dtype=np.float64)
    name_key_ratio = np.zeros(n, dtype=np.float64)
    name_key_partial_ratio = np.zeros(n, dtype=np.float64)
    name_key_jaro_winkler = np.zeros(n, dtype=np.float64)
    addr_fuzz_ratio = np.zeros(n, dtype=np.float64)
    addr_partial_ratio = np.zeros(n, dtype=np.float64)
    addr_token_sort_ratio = np.zeros(n, dtype=np.float64)
    addr_token_set_ratio = np.zeros(n, dtype=np.float64)
    addr_diff_token_sim = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s1_n = s1_names[i]
        c_n = cand_names[i]
        s1_nk = s1_name_keys[i]
        c_nk = cand_name_keys[i]
        s1_a = s1_addrs[i]
        c_a = cand_addrs[i]

        # 1. Name fuzz ratio
        name_fuzz_ratio[i] = float(fuzz.ratio(s1_n, c_n) / 100.0)

        # 2. Name partial ratio
        name_partial_ratio[i] = float(fuzz.partial_ratio(s1_n, c_n) / 100.0)

        # 3. Name space-free key ratio
        name_key_ratio[i] = float(fuzz.ratio(s1_nk, c_nk) / 100.0)

        # 4. Name space-free key partial ratio
        name_key_partial_ratio[i] = float(fuzz.partial_ratio(s1_nk, c_nk) / 100.0)

        # 5. Name space-free key Jaro-Winkler
        name_key_jaro_winkler[i] = float(JaroWinkler.normalized_similarity(s1_nk, c_nk))

        # 6. Address fuzz ratio
        addr_fuzz_ratio[i] = float(fuzz.ratio(s1_a, c_a) / 100.0)

        # 7. Address partial ratio
        addr_partial_ratio[i] = float(fuzz.partial_ratio(s1_a, c_a) / 100.0)

        # 8. Address token sort ratio
        addr_token_sort_ratio[i] = float(fuzz.token_sort_ratio(s1_a, c_a) / 100.0)

        # 9. Address token set ratio
        addr_token_set_ratio[i] = float(fuzz.token_set_ratio(s1_a, c_a) / 100.0)

        # 10. Address difference token similarity (Sid's ad_qonly_in_s)
        # Differing address tokens in candidate: check if approximately present in S1 address
        cand_diff_addr = _token_diff(c_a, s1_a)
        if not cand_diff_addr:
            addr_diff_token_sim[i] = 1.0  # no differing tokens -> full compatibility
        elif not s1_a:
            addr_diff_token_sim[i] = 0.0
        else:
            addr_diff_token_sim[i] = float(fuzz.partial_ratio(cand_diff_addr, s1_a) / 100.0)

    # Attach columns in exact order
    out_df["name_fuzz_ratio"] = name_fuzz_ratio
    out_df["name_partial_ratio"] = name_partial_ratio
    out_df["name_key_ratio"] = name_key_ratio
    out_df["name_key_partial_ratio"] = name_key_partial_ratio
    out_df["name_key_jaro_winkler"] = name_key_jaro_winkler
    out_df["addr_fuzz_ratio"] = addr_fuzz_ratio
    out_df["addr_partial_ratio"] = addr_partial_ratio
    out_df["addr_token_sort_ratio"] = addr_token_sort_ratio
    out_df["addr_token_set_ratio"] = addr_token_set_ratio
    out_df["addr_diff_token_sim"] = addr_diff_token_sim

    return out_df
