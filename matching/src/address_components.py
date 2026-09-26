"""
address_components.py
---------------------
Custom address-component parser for the Business Entity Resolution pipeline.

Extracts structured address sub-fields from free-text address strings
WITHOUT using geocoding, external APIs, or external address databases.
All logic is self-contained regex/heuristic parsing, safe for challenge use.

Supported component extraction:
  - Indian PIN code (6 digits)
  - US ZIP code (5-digit or ZIP+4: 12345 or 12345-6789)
  - French postal code (5 digits starting 0–9, first digit 0–9)
  - House / building number
  - Unit / apartment number
  - Floor
  - Locality (city-like token after known keywords)
  - Street number (same as house number by alternate extraction path)
  - Landmark context (near/opp/behind/opposite/adj/next to ...)

Pairwise comparison features derived from extracted components:
  1. postal_code_exact_match      -- exact match on any postal code found  {0.0, 1.0}
  2. postal_code_mismatch         -- both have codes AND they differ  {0.0, 1.0}
  3. house_number_match           -- house/building number exact match  {0.0, 1.0}
  4. house_number_conflict        -- both have numbers AND they differ  {0.0, 1.0}
  5. unit_match                   -- unit/apt number exact match  {0.0, 1.0}
  6. floor_match                  -- floor number exact match  {0.0, 1.0}
  7. locality_token_overlap       -- Jaccard over locality tokens  [0.0, 1.0]
  8. street_number_match          -- numeric overlap on street numbers  {0.0, 1.0}
  9. component_agreement_count    -- count of matching components (0–5)  [0.0, 1.0] normalised

All features are:
  - Deterministic (same input → same output every run)
  - NaN-free (edge-case defaults are always finite floats)
  - Challenge-safe (no external dependencies beyond stdlib regex)

Public API
~~~~~~~~~~
    parse_address_components(text: str) -> dict
    compare_address_components(a: dict, b: dict) -> dict
    add_address_component_features(pair_df: pd.DataFrame) -> pd.DataFrame

Component dict keys (all values are str, empty string if absent):
    postal_code, house_number, unit, floor, locality, landmark
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Feature columns produced by this module (9 features).
ADDRESS_COMPONENT_FEATURE_COLS: list[str] = [
    "postal_code_exact_match",
    "postal_code_mismatch",
    "house_number_match",
    "house_number_conflict",
    "unit_match",
    "floor_match",
    "locality_token_overlap",
    "street_number_match",
    "component_agreement_count",
]

# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

# Indian PIN code: exactly 6 consecutive digits
# Anchored to word boundaries so "1234567" doesn't match
_RE_PIN_IN = re.compile(r"\b([1-9]\d{5})\b")

# US ZIP code: 5 digits, optionally followed by -4 digits
# Exclude 6-digit sequences (those are Indian PINs above)
_RE_ZIP_US = re.compile(r"\b(\d{5})(?:-\d{4})?\b")

# French postal code: 5 digits, first digit 0-9 (de facto covers 01–99 depts)
_RE_PC_FR = re.compile(r"\b(0[1-9]\d{3}|[1-9]\d{4})\b")

# House / building number at start of address or after comma/newline
# Matches patterns like: "12", "12A", "12-14", "No. 12", "No 12", "#12"
_RE_HOUSE_NUMBER = re.compile(
    r"(?:^|,\s*|\bno\.?\s*|#\s*)(\d+[a-z]?(?:[/-]\d+[a-z]?)?)\b",
    re.IGNORECASE,
)

# Unit / apartment: "apt 12", "unit 5", "suite 200", "flat 3A", "#12"
_RE_UNIT = re.compile(
    r"\b(?:apt\.?|apartment|unit|ste\.?|suite|flat|room|rm\.?)\s*#?\s*([0-9a-z]+)\b",
    re.IGNORECASE,
)

# Floor: "floor 3", "3rd floor", "2nd floor", "ground floor", "3/f"
_RE_FLOOR = re.compile(
    r"\b(?:(?:(\d+)(?:st|nd|rd|th)?)\s*(?:floor|fl\.?|/f))|(?:floor\s+(\d+))|(?:ground\s+floor)\b",
    re.IGNORECASE,
)

# Landmark: text immediately after "near", "opp", "opposite", "behind",
# "adj", "adjacent to", "next to", "across from"
_RE_LANDMARK = re.compile(
    r"\b(?:near|opp(?:osite)?|behind|adj(?:acent\s+to)?|next\s+to|across\s+from)\s+([^,;\n]{2,40})",
    re.IGNORECASE,
)

# Locality keywords: city/locality prefix words
_RE_LOCALITY = re.compile(
    r"\b(?:city|town|village|district|dist\.?|area|locality|nagar|sector|phase|block|colony|"
    r"taluk|tehsil|mandal|ward)\b\s*([a-z0-9][^,;\n]{1,30})",
    re.IGNORECASE,
)

# Whitespace normaliser
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(text: Any) -> str:
    """Coerce to string, lower-case, normalise whitespace."""
    if text is None or (isinstance(text, float) and text != text):
        return ""
    s = str(text).strip().lower()
    return _RE_WHITESPACE.sub(" ", s)


def _extract_postal_code(text: str) -> str:
    """Return the first postal code found in *text* (empty string if none).

    Priority order: Indian PIN (6-digit) > French postal code > US ZIP.
    Returns the raw matched string so callers can compare directly.
    """
    # Try Indian PIN first (most specific — 6 digits)
    m = _RE_PIN_IN.search(text)
    if m:
        return m.group(1)

    # Try French postal code next (5 digits with specific first-digit range)
    m = _RE_PC_FR.search(text)
    if m:
        return m.group(1)

    # Fall back to US ZIP
    m = _RE_ZIP_US.search(text)
    if m:
        return m.group(1)

    return ""


def _extract_house_number(text: str) -> str:
    """Return the first house / building number found (empty string if none)."""
    m = _RE_HOUSE_NUMBER.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_unit(text: str) -> str:
    """Return unit / apartment identifier (empty string if none)."""
    m = _RE_UNIT.search(text)
    if m:
        return m.group(1).strip()
    return ""


def _extract_floor(text: str) -> str:
    """Return floor token (empty string if none)."""
    m = _RE_FLOOR.search(text)
    if m:
        # Pattern has two capture groups; pick whichever matched
        g1, g2 = m.group(1), m.group(2)
        if g1:
            return g1.strip()
        if g2:
            return g2.strip()
        # ground floor case — no numeric group
        return "0"
    return ""


def _extract_locality(text: str) -> str:
    """Return locality token(s) (empty string if none)."""
    m = _RE_LOCALITY.search(text)
    if m:
        raw = m.group(1).strip()
        # Trim at comma or semicolon
        raw = re.split(r"[,;]", raw)[0].strip()
        return raw
    return ""


def _extract_landmark(text: str) -> str:
    """Return first landmark context string (empty string if none)."""
    m = _RE_LANDMARK.search(text)
    if m:
        raw = m.group(1).strip()
        # Trim at comma or semicolon
        raw = re.split(r"[,;]", raw)[0].strip()
        return raw
    return ""


# ---------------------------------------------------------------------------
# Public API — component parser
# ---------------------------------------------------------------------------

def parse_address_components(text: Any) -> dict[str, str]:
    """Parse an address string into structured sub-field components.

    Parameters
    ----------
    text : str or any
        Raw / pre-normalised address string.  Non-string values are coerced
        to string; None and NaN become empty dict values.

    Returns
    -------
    dict with keys:
        postal_code   -- first postal code found (PIN / ZIP / FR code), or ""
        house_number  -- house / building number, or ""
        unit          -- unit / apartment identifier, or ""
        floor         -- floor number as string, or ""
        locality      -- locality / city / colony text, or ""
        landmark      -- landmark context (near/opp/behind ...), or ""

    All values are lowercase strings.  Empty string means the component
    was not found in *text*.  No value is None or NaN.
    """
    cleaned = _clean(text)

    return {
        "postal_code":  _extract_postal_code(cleaned),
        "house_number": _extract_house_number(cleaned),
        "unit":         _extract_unit(cleaned),
        "floor":        _extract_floor(cleaned),
        "locality":     _extract_locality(cleaned),
        "landmark":     _extract_landmark(cleaned),
    }


# ---------------------------------------------------------------------------
# Public API — pairwise comparison
# ---------------------------------------------------------------------------

def compare_address_components(
    a: dict[str, str],
    b: dict[str, str],
) -> dict[str, float]:
    """Compute 9 pairwise comparison features from two parsed component dicts.

    Parameters
    ----------
    a, b : dict[str, str]
        Output of :func:`parse_address_components`.

    Returns
    -------
    dict[str, float]
        Keys are the 9 entries in :data:`ADDRESS_COMPONENT_FEATURE_COLS`.
        All values are finite floats in [0.0, 1.0].  No NaN, no Inf.

    Feature semantics
    -----------------
    postal_code_exact_match
        1.0 if both have a non-empty postal code AND they are identical.
        0.0 otherwise.

    postal_code_mismatch
        1.0 if both have non-empty postal codes AND they differ.
        0.0 otherwise.  Mutually exclusive with postal_code_exact_match.

    house_number_match
        1.0 if both have a non-empty house number AND they are identical.
        0.0 otherwise.

    house_number_conflict
        1.0 if both have non-empty house numbers AND they differ.
        0.0 otherwise.  Mutually exclusive with house_number_match.

    unit_match
        1.0 if both have a non-empty unit AND they are identical.
        0.0 otherwise.

    floor_match
        1.0 if both have a non-empty floor AND they are identical.
        0.0 otherwise.

    locality_token_overlap
        Token-level Jaccard similarity between locality strings.
        Jaccard({tokens(a.locality)}, {tokens(b.locality)}).
        Both empty → 0.0 (unknown locality is not informative).

    street_number_match
        1.0 if the digit sequences in both house_number strings intersect.
        0.0 otherwise.  More lenient than house_number_match: matches
        "12A" with "12" since both contain "12".

    component_agreement_count
        Number of component fields that agree (both non-empty AND equal)
        among {postal_code, house_number, unit, floor, locality},
        normalised to [0.0, 1.0] by dividing by 5.
        Reflects overall structural agreement across all address components.
    """
    # 1. Postal code
    pc_a = a.get("postal_code", "")
    pc_b = b.get("postal_code", "")

    if pc_a and pc_b:
        postal_exact = 1.0 if pc_a == pc_b else 0.0
        postal_mismatch = 0.0 if pc_a == pc_b else 1.0
    else:
        postal_exact = 0.0
        postal_mismatch = 0.0

    # 2. House number
    hn_a = a.get("house_number", "")
    hn_b = b.get("house_number", "")

    if hn_a and hn_b:
        hn_match = 1.0 if hn_a == hn_b else 0.0
        hn_conflict = 0.0 if hn_a == hn_b else 1.0
    else:
        hn_match = 0.0
        hn_conflict = 0.0

    # 3. Unit
    unit_a = a.get("unit", "")
    unit_b = b.get("unit", "")
    unit_match = 1.0 if (unit_a and unit_b and unit_a == unit_b) else 0.0

    # 4. Floor
    fl_a = a.get("floor", "")
    fl_b = b.get("floor", "")
    floor_match = 1.0 if (fl_a and fl_b and fl_a == fl_b) else 0.0

    # 5. Locality token overlap (Jaccard)
    loc_a = set(a.get("locality", "").split()) if a.get("locality") else set()
    loc_b = set(b.get("locality", "").split()) if b.get("locality") else set()
    if loc_a and loc_b:
        intersection = loc_a & loc_b
        union = loc_a | loc_b
        locality_overlap = float(len(intersection)) / float(len(union))
    else:
        locality_overlap = 0.0

    # 6. Street number match (digit-sequence intersection in house_number)
    nums_a = set(re.findall(r"\d+", hn_a))
    nums_b = set(re.findall(r"\d+", hn_b))
    street_num_match = 1.0 if (nums_a and nums_b and nums_a & nums_b) else 0.0

    # 7. Component agreement count
    #    Count fields that both have values AND are identical
    agreement = 0
    # postal code
    if pc_a and pc_b and pc_a == pc_b:
        agreement += 1
    # house number
    if hn_a and hn_b and hn_a == hn_b:
        agreement += 1
    # unit
    if unit_a and unit_b and unit_a == unit_b:
        agreement += 1
    # floor
    if fl_a and fl_b and fl_a == fl_b:
        agreement += 1
    # locality (at least one token in common)
    if loc_a and loc_b and (loc_a & loc_b):
        agreement += 1

    component_agreement_norm = float(agreement) / 5.0

    return {
        "postal_code_exact_match":   postal_exact,
        "postal_code_mismatch":      postal_mismatch,
        "house_number_match":        hn_match,
        "house_number_conflict":     hn_conflict,
        "unit_match":                unit_match,
        "floor_match":               floor_match,
        "locality_token_overlap":    locality_overlap,
        "street_number_match":       street_num_match,
        "component_agreement_count": component_agreement_norm,
    }


# ---------------------------------------------------------------------------
# Public API — DataFrame-level feature adder
# ---------------------------------------------------------------------------

def add_address_component_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 9 address-component features for every candidate pair.

    Reads ``s1_clean_address`` and ``candidate_clean_address`` from *pair_df*,
    parses each into structured components via :func:`parse_address_components`,
    and derives 9 pairwise comparison features via
    :func:`compare_address_components`.

    Parameters
    ----------
    pair_df : pd.DataFrame
        Candidate pair table.  Must contain:
        - ``source1_entity_id``
        - ``candidate_entity_id``
        - ``s1_clean_address``
        - ``candidate_clean_address``

    Returns
    -------
    pd.DataFrame
        A copy of *pair_df* with 9 new columns appended in the order
        defined by :data:`ADDRESS_COMPONENT_FEATURE_COLS`.

    All output feature values are:
    - Finite floats in [0.0, 1.0]
    - Never NaN
    - Deterministic

    Raises
    ------
    ValueError
        If required columns are missing from *pair_df*.
    """
    required = {"source1_entity_id", "candidate_entity_id",
                "s1_clean_address", "candidate_clean_address"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(
            f"add_address_component_features(): pair_df is missing required "
            f"column(s): {sorted(missing)}.  Found: {list(pair_df.columns)}"
        )

    out_df = pair_df.copy()

    # Initialise all feature columns to 0.0 for the empty-table case
    if out_df.empty:
        for col in ADDRESS_COMPONENT_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=float)
        return out_df

    # Parse components for every row
    s1_addrs = [
        "" if pd.isna(x) else str(x)
        for x in out_df["s1_clean_address"]
    ]
    cand_addrs = [
        "" if pd.isna(x) else str(x)
        for x in out_df["candidate_clean_address"]
    ]

    s1_components = [parse_address_components(a) for a in s1_addrs]
    cand_components = [parse_address_components(a) for a in cand_addrs]

    # Compute comparison features for each pair
    comparisons = [
        compare_address_components(a, b)
        for a, b in zip(s1_components, cand_components)
    ]

    # Append feature columns in canonical order
    for col in ADDRESS_COMPONENT_FEATURE_COLS:
        out_df[col] = [float(cmp[col]) for cmp in comparisons]

    return out_df
