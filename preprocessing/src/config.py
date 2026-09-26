"""Shared configuration for the preprocessing pipeline.

All mappings here were validated against the seven attached dataset files
before inclusion (see implementation.md, Sections 2-3, 8-9). Nothing here
is applied "because it's a common ER trick" without evidence — and nothing
here performs entity matching or deduplication.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

SOURCE_ID_PATTERNS = {
    "source1": r"^S1-\d+$",
    "source2": r"^S2-\d+$",
    "source3": r"^S3-\d+$",
}

# Which source key is expected for each canonical source file name.
SOURCE_KEYS = ("source1", "source2", "source3")

GT_SOURCE1_PATTERN = r"^S1-\d+$"
GT_MATCH_PATTERN = r"^S[23]-\d+$"

# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------
# Whole-field NA-like tokens (business_name). Matched case-insensitively
# after stripping whitespace.
NA_LIKE_TOKENS = {"na", "n/a", "null", "none", "nan", "<null>", "<na>"}

# ---------------------------------------------------------------------------
# Business-name legal-suffix canonicalization
# ---------------------------------------------------------------------------
# Token-boundary-only mapping applied to business_name_canonical. Keys are
# matched against a token with trailing punctuation stripped (e.g. "ltd."
# and "ltd" both match the "ltd" key).
#
# Design: every entry maps ONE surface form to ONE canonical form within
# the same legal-entity family.  The raw business_name and the
# business_name_normalized columns are NEVER modified, so no information
# is lost — the canonical form is an ADDITIONAL view only.
#
# Families and their canonical targets:
#   limited-company family  → "limited"
#   private family          → "private"
#   corporation family      → "corporation"
#   incorporated family     → "incorporated"
#   company family          → "company"
#   llc family              → "llc"      (no single English expansion agreed
#                                         upon across jurisdictions; keep the
#                                         abbreviation as the canonical form
#                                         so all LLC-style variants unify)
#   llp family              → "llp"
#   pllc family             → "pllc"
#   pc family               → "pc"       (professional corporation; keep
#                                         abbreviated to avoid collision with
#                                         "personal computer" in names)
#   sarl/sas/sci/sa family  → each maps to its own canonical abbreviation
#                             (French legal forms; full expansions are
#                              foreign-language strings that would look
#                              strange in an otherwise-English canonical
#                              field and would create new collision risk)
LEGAL_SUFFIX_MAP = {
    # --- English forms (original) ---
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
    # --- LLC / LLP family ---
    "llc": "llc",
    "l.l.c": "llc",
    "llp": "llp",
    "l.l.p": "llp",
    # --- Professional / other US forms ---
    "pllc": "pllc",
    "p.l.l.c": "pllc",
    "pc": "pc",
    "p.c": "pc",
    # --- French legal forms (appear in test data: France rows) ---
    "sarl": "sarl",
    "s.a.r.l": "sarl",
    "sas": "sas",
    "s.a.s": "sas",
    "sci": "sci",
    "s.c.i": "sci",
    "sa": "sa",
    "s.a": "sa",
}

# ---------------------------------------------------------------------------
# Address abbreviation canonicalization
# ---------------------------------------------------------------------------
# Token-boundary-only mapping applied to business_address_canonical.
# Every key is matched against a token with leading/trailing [.,] stripped
# (e.g. "rd." and "rd" both match the "rd" key).
#
# Design constraints:
#   - "st" ↔ "saint" ambiguity: "st" is mapped to "street" here because
#     the dataset contains no observed city/place names where "st" is an
#     abbreviation for "saint" that would be confused (e.g. "St Louis" would
#     become "street louis", but "st" in an address token position is
#     overwhelmingly a street-type abbreviation in US/India/France data).
#     Raw and normalized addresses are always preserved, so any downstream
#     stage can use the raw field if saint/street ambiguity is a concern.
#   - "sq" ↔ square: safe in street-address context; "sq" as a standalone
#     address token is almost always "Square" (Connaught Sq, etc.).
#   - "pl" ↔ place: same rationale; "pl" as a street-type token.
#   - "dr" ↔ drive: "dr" also abbreviates "doctor" but only as a personal
#     title, never as a standalone address-type token in these sources.
ADDRESS_ABBREV_MAP = {
    # --- original entries ---
    "rd":   "road",
    "st":   "street",
    "ave":  "avenue",
    "ste":  "suite",
    "fl":   "floor",
    # --- new entries (audit P1b) ---
    "dr":   "drive",
    "ct":   "court",
    "ln":   "lane",
    "blvd": "boulevard",
    "pl":   "place",
    "cir":  "circle",
    "ter":  "terrace",
    "hwy":  "highway",
    "pkwy": "parkway",
    "sq":   "square",
}

# Landmark prefixes (word-boundary matched — see normalization.py). Order
# matters only in that "opp." is checked before "opp" would be redundant
# since both are listed explicitly; longer/more-specific tokens are checked
# first to avoid a short prefix masking a more specific one.
LANDMARK_PREFIXES = ["opposite", "opp.", "opp", "behind", "near"]

# Decorative leading characters stripped from the START of a name/address
# only (never from the middle) — verified against the "leading special
# character" pattern observed in up to ~577k Source 2 rows.
DECORATIVE_LEADING_CHARS = r"\-\#\[\]<>@\.\*\s"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
REQUIRED_OUTPUT_COLUMNS = [
    "entity_id",
    "business_name", "business_name_normalized", "business_name_canonical",
    "business_address", "business_address_normalized", "business_address_canonical",
    "address_landmark",
    "address_numbers",
    "address_postal_code",
    "address_sorted_tokens",
    "country", "country_normalized",
    "business_name_is_missing", "business_address_is_missing",
    "name_script_class",
]

GT_OUTPUT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
# Bound how much detail the collision report keeps in memory / on disk.
MAX_COLLISION_GROUPS_REPORTED = 200
MAX_SAMPLE_IDS_PER_GROUP = 10

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
ALL_SPLITS = (TRAIN_SPLIT, TEST_SPLIT)


@dataclass
class Paths:
    input_dir: str
    output_processed_dir: str
    output_metadata_dir: str
