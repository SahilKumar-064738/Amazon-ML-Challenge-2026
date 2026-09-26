"""Deterministic, conservative text normalization functions.

Every transformation here was validated against the actual seven attached
TSV files (see implementation.md Sections 2-3, 8-9) before being included.
Nothing here performs entity deduplication or matching.

Design notes (see implementation.md Section 26 "known issues to fix"):
  * Landmark detection is WORD-BOUNDARY aware (a substring like "near"
    inside "nearby" or "shearer" must never be treated as a landmark).
  * Legal-suffix and address-abbreviation mapping are TOKEN-boundary only
    (never a substring replace inside a longer word).
  * Bulk (DataFrame-column) functions are provided alongside the
    single-value functions so callers can avoid `df.apply(axis=1)` over
    millions of rows (Section 11). The single-value functions remain the
    source of truth and are exercised directly by the unit tests; the bulk
    functions are thin vectorized wrappers around the same logic.

Production-grade additions (upgrade pass):
  * Accent folding   — NFD decompose + strip Mn (combining diacritics).
                       LATIN-SCRIPT-GATED: never applied to Devanagari,
                       Tamil, or any other non-Latin script where combining
                       marks are semantically meaningful.
  * URL / email strip — removes embedded http(s) URLs, bare domain tokens
                        (e.g. "www.acme.com", "acme.in"), and email handles
                        (@user) from business names before any other step.
  * Leet-speak repair — digit-for-letter substitutions (0→o, 1→i, 3→e,
                        4→a, 5→s, 7→t) fixed ONLY inside mixed
                        alphanumeric tokens — never in pure-digit tokens or
                        standalone numbers.
  * DBA / T/A strip  — removes "doing business as", "dba", "d/b/a",
                       "trading as", "t/a", "also known as", "aka" tags
                       that split the canonical name from an alias.  The
                       portion BEFORE the tag is kept (it is always the
                       legal/primary name).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

import pandas as pd

from .config import (
    ADDRESS_ABBREV_MAP,
    DECORATIVE_LEADING_CHARS,
    LANDMARK_PREFIXES,
    LEGAL_SUFFIX_MAP,
    NA_LIKE_TOKENS,
)

# ---------------------------------------------------------------------------
# Optional Indic transliteration — imported lazily so the module loads cleanly
# even in environments where indic_transliteration.py is absent.
# ---------------------------------------------------------------------------
try:
    from .indic_transliteration import IndicTransliterator as _IndicTransliteratorClass
    _INDIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _IndicTransliteratorClass = None  # type: ignore[assignment,misc]
    _INDIC_AVAILABLE = False

# Module-level transliterator instance.  None by default — the pipeline is
# identical to the pre-transliteration baseline when this is None.
_TRANSLITERATOR: Optional[object] = None  # type: ignore[type-arg]


def set_transliterator(t: object) -> None:  # type: ignore[type-arg]
    """Install an ``IndicTransliterator`` for use by ``normalize_name_basic``.

    Call this once before processing begins (e.g. in ``preprocess.main()``
    or in a test fixture).  Pass ``None`` to disable transliteration and
    restore the default behaviour.

    Parameters
    ----------
    t:
        An ``IndicTransliterator`` instance, or ``None`` to disable.
    """
    global _TRANSLITERATOR
    _TRANSLITERATOR = t


def get_transliterator() -> Optional[object]:  # type: ignore[type-arg]
    """Return the currently-installed transliterator, or ``None``."""
    return _TRANSLITERATOR

_WS_RE = re.compile(r"\s+")
_DECORATIVE_LEADING_RE = re.compile(rf"^[{DECORATIVE_LEADING_CHARS}]+")

# Component equal to (optionally bracketed) null/NULL/<NULL>, case-insensitive,
# once stripped of surrounding whitespace.
_NULL_COMPONENT_RE = re.compile(r"(?i)^\s*<?\s*null\s*>?\s*$")

# Landmark prefixes compiled as WORD-BOUNDARY patterns so "near" never
# matches inside "nearby"/"shearer" etc. Sorted longest-first so a more
# specific phrase ("opposite") is preferred over a shorter one that could
# also match ("opp").
_LANDMARK_ALTERNATION = "|".join(
    re.escape(p) for p in sorted(LANDMARK_PREFIXES, key=len, reverse=True)
)
# The alternation MUST be wrapped in a non-capturing group: without it, the
# lookbehind/lookahead below would only bind to the first/last alternative
# respectively, leaving every middle alternative (e.g. "behind", "opp.")
# completely unchecked for word boundaries.
_LANDMARK_RE = re.compile(rf"(?i)(?<![a-z0-9])(?:{_LANDMARK_ALTERNATION})(?![a-z0-9])")

# Token-boundary punctuation strip used before legal-suffix / abbreviation
# lookup (does not affect the stored normalized value, only the lookup key).
_TOKEN_TRIM_RE = re.compile(r"^[.,]+|[.,]+$")

# ---------------------------------------------------------------------------
# Production additions — module-level compiled patterns
# ---------------------------------------------------------------------------

# Email addresses: user@domain.tld  (strip the whole token)
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)

# URLs: http(s)://... or www.something or bare domain-like tokens
# (e.g. "acme.com", "shop.acme.in", "www.acme.org").
# Strategy: match anything that looks like a domain token, i.e. contains
# a dot followed by a known TLD-like suffix (2-6 alpha chars).
# Anchored with word boundaries so "U.S.A." or "S.A.R.L" (legal suffixes
# that use dots) are NOT caught — they have uppercase letters and are
# handled by the suffix map, not here.
_URL_RE = re.compile(
    r"""(?ix)
    (?:https?://\S+)           # full URL with scheme
    |
    (?:www\.[A-Za-z0-9\-]+     # www. prefix
       (?:\.[A-Za-z]{2,6})+    # one or more .tld segments
       (?:/\S*)?)               # optional path
    |
    (?<!\w)                    # not preceded by a word char (prevents
                               # matching "first.street" or "st.john")
    [A-Za-z0-9]                # starts with alnum
    [A-Za-z0-9\-]*             # middle alnum/hyphen
    \.                         # a dot
    (?:com|net|org|edu|gov|io|co|in|fr|de|uk|biz|info|me|us|eu)
    (?:/\S*)?                  # optional path
    (?!\w)                     # not followed by a word char
    """
)

# DBA / T/A / trading-as patterns.
# Captures the entire "dba …" suffix starting at the tag so we can
# discard it, keeping only the portion before the tag.
# Applied case-insensitively.
_DBA_RE = re.compile(
    r"""(?ix)
    [\s,/\-]*          # optional separator before the tag
    \b(?:
        doing\s+business\s+as
      | d[./]?\s*b[./]?\s*a\.?      # dba / d.b.a / d/b/a / d b a
      | trading\s+as
      | t[./]?\s*/?\s*a\.?          # t/a / t.a. / ta
      | also\s+known\s+as
      | a[./]?\s*k[./]?\s*a\.?      # aka / a.k.a
    )\b
    .*$                # everything after the tag (the alias)
    """,
    re.DOTALL,
)

# Leet-speak digit-to-letter substitutions.
# ONLY applied inside mixed alphanumeric tokens (tokens containing both
# letters AND digits).  Pure-digit tokens (building numbers, ZIP codes,
# phone numbers) are never touched.
#
# Substitution map (digit → most unambiguous Latin letter replacement):
#   0 → o   1 → i   3 → e   4 → a   5 → s   7 → t
#
# Deliberately excluded:
#   2 (→ z or to): too ambiguous; "2" as "to" is common English
#   6 (→ g or b): too ambiguous; "6" inside a normal word is unusual
#   8 (→ b): uncommon
_LEET_MAP: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
}

# Pre-compiled translation table for str.translate — faster than repeated
# re.sub calls for single-character substitutions.
_LEET_TABLE = str.maketrans(_LEET_MAP)

# A "mixed alphanumeric token" must contain at least one ASCII letter AND
# at least one of the leet digits.
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")
_HAS_LEET_DIGIT_RE = re.compile(r"[013457]")


def _repair_leet_token(token: str) -> str:
    """Repair leet-speak digits in a single token IFF it is mixed
    alphanumeric.  Pure-digit tokens are returned unchanged.

    "Preparat0ry"  → "Preparatory"
    "4cme"         → "acme"
    "123"          → "123"     (pure digit — untouched)
    "42"           → "42"      (pure digit — untouched)
    "studio54"     → "studio54"  (no leet digits 01357 present — untouched)
    """
    if not _HAS_LETTER_RE.search(token):
        return token  # pure-digit token: leave completely untouched
    if not _HAS_LEET_DIGIT_RE.search(token):
        return token  # no leet digits present
    return token.translate(_LEET_TABLE)


def _repair_leet(text: str) -> str:
    """Apply leet-speak repair to every whitespace-separated token in text."""
    return " ".join(_repair_leet_token(tok) for tok in text.split())


# ---------------------------------------------------------------------------
# Accent folding helper — Latin-gated
# ---------------------------------------------------------------------------

def _is_latin_only(text: str) -> bool:
    """Return True if all *letter* characters in text are Latin-script.

    Digits, punctuation, whitespace are ignored.  An empty string (or a
    string with no letters) returns True so that accent folding is applied
    to numeric/punctuation-only strings without risk.

    This is the gate that prevents accent folding from destroying combining
    marks in Devanagari, Tamil, Gujarati, etc.
    """
    for ch in text:
        cat = unicodedata.category(ch)
        if not cat.startswith("L"):
            continue
        cp = ord(ch)
        # Latin ranges: Basic Latin + Latin-1 Supplement + Latin Extended A/B
        # + IPA Extensions + Latin Extended Additional
        if not (
            (0x0041 <= cp <= 0x007A)   # A-Z a-z
            or (0x00C0 <= cp <= 0x024F)  # Latin-1 Supplement + Extended A/B
            or (0x1E00 <= cp <= 0x1EFF)  # Latin Extended Additional
        ):
            return False
    return True


def fold_accents(text: str) -> str:
    """Strip combining diacritical marks from Latin-script text.

    Applies NFD decomposition (separates base letter from combining mark)
    then discards all Unicode category Mn (Mark, Nonspacing) characters.

    "café"    → "cafe"
    "naïve"   → "naive"
    "résumé"  → "resume"
    "Ångström" → "Angstrom"

    NOT applied (returns text unchanged) when text contains non-Latin
    letters (Devanagari, Tamil, Arabic, CJK …) — combining marks in those
    scripts are phonemically essential.
    """
    if not _is_latin_only(text):
        return text
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


# ---------------------------------------------------------------------------
# Single-value functions (source of truth; used directly by unit tests)
# ---------------------------------------------------------------------------

def nfkc(text: str) -> str:
    """Unicode NFKC normalization. Does not transliterate scripts."""
    return unicodedata.normalize("NFKC", text)


def collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace to a single space and strip ends."""
    return _WS_RE.sub(" ", text).strip()


def is_na_like(text: str) -> bool:
    """True if text is a whole-field NA-like literal (e.g. 'NA', 'N/A', 'NULL')."""
    if text is None:
        return True
    return text.strip().lower() in NA_LIKE_TOKENS


def normalize_name_basic(name: str) -> Tuple[str, bool]:
    """Conservative normalization -> business_name_normalized.

    Returns (normalized_value, is_missing_flag). A name is "missing" if it
    is empty/whitespace-only, an NA-like literal token, or normalizes down
    to an empty string once decorative leading characters are stripped.
    The row itself is never dropped for this — see preprocess.py.

    Cleaning pipeline (in order):
      0. Indic transliteration — if an IndicTransliterator is installed via
                               set_transliterator(), Indic-script tokens are
                               converted to Latin phonetics before any other
                               step.  Latin-only names are untouched.
                               When no transliterator is installed (default),
                               this step is a no-op.
      1. Email stripping     — remove user@domain.tld tokens
      2. URL / domain strip  — remove http://…, www.…, bare domain tokens
      3. DBA / T/A strip     — keep only the primary name before any
                               "doing business as" / "t/a" / "aka" tag
      4. NA-like check       — after stripping, the residual may now be empty
                               or an NA literal
      5. NFKC               — Unicode compatibility normalisation
      6. Lowercase
      7. Accent folding      — NFD + strip Mn, Latin-gated only
      8. Decorative-leading strip
      9. Leet-speak repair   — digit-for-letter in mixed-alnum tokens only
     10. Whitespace collapse
    """
    if name is None or name == "" or is_na_like(name):
        return "", True

    text = name

    # 0. Indic transliteration (optional — no-op when _TRANSLITERATOR is None).
    if _TRANSLITERATOR is not None:
        text = _TRANSLITERATOR.transliterate(text)  # type: ignore[union-attr]

    # 1. Strip email addresses (must run before URL stripping so the @domain
    #    part is not mistaken for a bare domain token).
    text = _EMAIL_RE.sub(" ", text)

    # 2. Strip URLs and bare domain tokens.
    text = _URL_RE.sub(" ", text)

    # 3. Strip DBA / T/A / AKA suffixes — keep only the primary name.
    text = _DBA_RE.sub("", text).strip().rstrip(",;/")

    # 4. Re-check after stripping: the residual may now be empty or NA.
    text = text.strip()
    if not text or is_na_like(text):
        return "", True

    # 5. NFKC Unicode normalization.
    text = nfkc(text)

    # 6. Lowercase.
    text = text.lower()

    # 7. Accent folding (Latin-gated — Devanagari/Tamil/etc. untouched).
    text = fold_accents(text)

    # 8. Strip decorative leading characters.
    text = _DECORATIVE_LEADING_RE.sub("", text)

    # 9. Leet-speak digit-for-letter repair (mixed tokens only).
    text = _repair_leet(text)

    # 10. Collapse whitespace.
    text = collapse_whitespace(text)
    return text, (text == "")


def canonicalize_name(name_normalized: str) -> str:
    """Stronger canonical form -> business_name_canonical.

    Applies '&' -> 'and' and the validated legal-suffix mapping, at token
    boundaries only (never inside a longer word). Order: ampersand
    expansion happens before tokenization, so "X & Y Inc" and "X and Y
    Incorporated" canonicalize identically and consistently.
    """
    if not name_normalized:
        return ""
    text = name_normalized.replace("&", " and ")
    tokens = text.split()
    out_tokens = []
    for tok in tokens:
        key = _TOKEN_TRIM_RE.sub("", tok)
        out_tokens.append(LEGAL_SUFFIX_MAP.get(key, tok))
    return collapse_whitespace(" ".join(out_tokens))


def extract_landmark(address: str) -> str:
    """Return the landmark phrase found in `address`, or '' if none.

    Word-boundary aware: a substring like "near" embedded inside another
    word (e.g. "nearby", "shearer") is never treated as a landmark. The
    landmark text is only *copied out*; callers must not delete it from the
    address itself (see normalize_address_basic).
    """
    if not address:
        return ""
    match = _LANDMARK_RE.search(address)
    if not match:
        return ""
    start = match.start()
    rest = address[start:]
    end = rest.find(",")
    phrase = rest if end == -1 else rest[:end]
    return phrase.strip()


def normalize_address_basic(address: str) -> Tuple[str, bool]:
    """business_address_normalized: Unicode + case + whitespace + null-component strip.

    Whole-field empty string -> ("", True) (is_missing).
    A component (comma-separated segment) that is exactly a null-like token
    (e.g. "NULL", "<null>") is removed from the joined string, since it is a
    missing-value marker, not real address text. If every component was a
    null marker, the result is treated as missing.
    Landmark phrases (near/opp/opposite/behind ...) are NEVER removed here.

    Accent folding (Latin-gated) is applied so that "Résidence du Parc" and
    "Residence du Parc" normalise identically. URL/leet/DBA stripping is
    intentionally NOT applied to addresses — street names and building
    numbers must be preserved exactly.
    """
    if address is None or address == "":
        return "", True
    components = [c.strip() for c in address.split(",")]
    kept = [c for c in components if c != "" and not _NULL_COMPONENT_RE.match(c)]
    if not kept:
        return "", True
    text = ", ".join(kept)
    text = nfkc(text)
    text = text.lower()
    # Accent folding: strip combining diacritics from Latin-script addresses
    # (e.g. French "résidence" → "residence", "allée" → "allee") so that
    # abbreviated and accented variants canonicalise identically.
    text = fold_accents(text)
    text = collapse_whitespace(text)
    return text, False


def canonicalize_address(address_normalized: str) -> str:
    """business_address_canonical: apply the validated abbreviation mapping.

    Token-boundary only (splits each comma-separated component on
    whitespace). Never touches digits, building/unit numbers, PIN codes, or
    state names — those are not in ADDRESS_ABBREV_MAP.
    """
    if not address_normalized:
        return ""
    parts = address_normalized.split(",")
    out_parts = []
    for part in parts:
        tokens = part.strip().split(" ")
        out_tokens = []
        for tok in tokens:
            key = _TOKEN_TRIM_RE.sub("", tok)
            out_tokens.append(ADDRESS_ABBREV_MAP.get(key, tok))
        out_parts.append(" ".join(out_tokens))
    return collapse_whitespace(", ".join(out_parts))


def normalize_country(country: str) -> str:
    """Open-set country normalization: NFKC + lowercase + whitespace strip.

    Deliberately NO alias/lookup table — see implementation.md Section 8.7.
    A training-only alias table would be an assumption unsupported by the
    data and would risk silently mishandling unseen values (e.g. "France",
    which appears only in test data). This function must work correctly on
    any string, seen or unseen.
    """
    if country is None:
        return ""
    text = nfkc(country).lower()
    return collapse_whitespace(text)


# ---------------------------------------------------------------------------
# Vectorized (bulk) wrappers — avoid df.apply(axis=1) over millions of rows.
# These implement the exact same logic as the single-value functions above,
# using pandas' vectorized .str accessor instead of a Python-level loop.
# ---------------------------------------------------------------------------

def normalize_name_series(names: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Vectorized equivalent of normalize_name_basic over a whole column.

    Delegates to normalize_name_basic per-value so that all cleaning steps
    (accent folding, URL/email strip, DBA strip, leet repair) are applied
    identically to the Series path and the single-value path.

    Performance: still uses a list comprehension (not df.apply with a lambda)
    and short-circuits the expensive single-value call for cells that are
    trivially empty or NA-literal via a fast vectorized pre-check.
    """
    s = names.fillna("")
    # Fast vectorized pre-pass to identify trivially missing rows so we don't
    # spend time running the full pipeline on them.
    is_na_literal = s.str.strip().str.lower().isin(NA_LIKE_TOKENS)
    is_empty_input = s == ""
    needs_processing = ~(is_na_literal | is_empty_input)

    # Run the full single-value pipeline only on rows that need it.
    norm_values = [""] * len(s)
    missing_flags = [True] * len(s)
    for idx in s.index[needs_processing]:
        norm_val, is_miss = normalize_name_basic(s.loc[idx])
        norm_values[s.index.get_loc(idx)] = norm_val
        missing_flags[s.index.get_loc(idx)] = is_miss

    normalized = pd.Series(norm_values, index=s.index)
    is_missing = pd.Series(missing_flags, index=s.index)
    return normalized, is_missing


def canonicalize_name_series(names_normalized: pd.Series) -> pd.Series:
    """Vectorized equivalent of canonicalize_name over a whole column."""
    return names_normalized.map(canonicalize_name)


def normalize_address_series(addresses: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Vectorized-ish equivalent of normalize_address_basic over a column.

    Component splitting/filtering is inherently per-row (variable number of
    comma components), so this uses a list comprehension over the column
    rather than df.apply(axis=1) with a lambda — meaningfully faster in
    practice and avoids constructing an intermediate row-wise DataFrame.
    """
    s = addresses.fillna("")
    results = [normalize_address_basic(v) for v in s]
    normalized = pd.Series([r[0] for r in results], index=addresses.index)
    is_missing = pd.Series([r[1] for r in results], index=addresses.index)
    return normalized, is_missing


def canonicalize_address_series(addresses_normalized: pd.Series) -> pd.Series:
    """Vectorized equivalent of canonicalize_address over a whole column."""
    return addresses_normalized.map(canonicalize_address)


def extract_landmark_series(addresses: pd.Series) -> pd.Series:
    """Vectorized equivalent of extract_landmark over a whole column."""
    s = addresses.fillna("")
    return s.map(extract_landmark)


def normalize_country_series(countries: pd.Series) -> pd.Series:
    """Vectorized equivalent of normalize_country over a whole column."""
    s = countries.fillna("")
    out = [nfkc(v) if v != "" else "" for v in s]
    out_series = pd.Series(out, index=countries.index).str.lower()
    out_series = out_series.str.replace(_WS_RE, " ", regex=True).str.strip()
    return out_series


# ---------------------------------------------------------------------------
# New derived representations (P1c additions)
# ---------------------------------------------------------------------------
# All four functions below are ADDITIVE views — they never modify the
# existing normalized/canonical columns.  Each is a deterministic, pure
# function that depends only on stdlib / already-imported modules.

# Regex for digit-runs (one or more consecutive digits).
_DIGIT_RUN_RE = re.compile(r"\d+")

# Candidate postal/PIN code: standalone token that is exactly 5 or 6 digits.
# 5 digits covers US ZIP and French postal codes.
# 6 digits covers Indian PIN codes.
# The token must be standalone (word-boundary) to avoid matching partial
# strings like the "12345" inside a phone-number run "01234567890".
_POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")


def extract_address_numbers(address_normalized: str) -> str:
    """Return all digit-runs from the normalized address, space-joined in
    original left-to-right order.  Preserves leading zeros (string not int).

    "Shop 12, MG Road"        -> "12"
    "Sector 17, Chandigarh"   -> "17"
    "Plot 42, Sector 17"      -> "42 17"
    "160017, Chandigarh"      -> "160017"
    ""                        -> ""
    """
    if not address_normalized:
        return ""
    runs = _DIGIT_RUN_RE.findall(address_normalized)
    return " ".join(runs)


def extract_address_postal_code(address_normalized: str) -> str:
    """Extract the first plausible postal / PIN / ZIP code from the
    normalized address.  Returns the matched string (preserving leading
    zeros) or '' if none found.

    Strategy: scan left-to-right for the first standalone 5- or 6-digit
    token.  5-digit = US ZIP / French postal; 6-digit = Indian PIN.
    Open-set: works on any country because it relies only on digit-run
    length, not on a country-specific format table.

    "Chandigarh 160017"   -> "160017"
    "560001 Bangalore"    -> "560001"
    "Peoria, IL 61602"    -> "61602"
    "1064 Newton Rd"      -> ""   (4 digits: too short)
    ""                    -> ""
    """
    if not address_normalized:
        return ""
    match = _POSTAL_RE.search(address_normalized)
    return match.group(1) if match else ""


def make_address_sorted_tokens(address_canonical: str) -> str:
    """Return the canonical address tokens in sorted (alphabetical) order,
    space-joined.  Deterministic; handles the common case where the same
    address has components in different order across sources.

    "mg road sector 17"   -> "17 mg road sector"
    "sector 17 mg road"   -> "17 mg road sector"   (same output)
    ""                    -> ""
    """
    if not address_canonical:
        return ""
    return " ".join(sorted(address_canonical.split()))


def classify_name_script(name_normalized: str) -> str:
    """Return the dominant Unicode script class of the business name.

    Uses only unicodedata (stdlib) — no external dependencies.
    Classification is deterministic and does not modify the name.

    Return values (fixed vocabulary):
      "empty"              — name is empty or whitespace-only
      "latin"              — all letter characters are Latin (ASCII a-z,
                             extended Latin, etc.)
      "devanagari"         — dominant non-Latin script is Devanagari
                             (Hindi, Marathi, Nepali, …)
      "tamil"              — dominant script is Tamil
      "telugu"             — dominant script is Telugu
      "gujarati"           — dominant script is Gujarati
      "gurmukhi"           — dominant script is Gurmukhi (Punjabi)
      "malayalam"          — dominant script is Malayalam
      "non_latin"          — non-Latin script not in the above list
      "latin_and_non_latin"— name contains BOTH Latin letters and at least
                             one letter from a non-Latin script (mixed)

    Digits, punctuation, and whitespace are ignored (not counted toward any
    script); classification is based on letter characters only.
    """
    if not name_normalized or not name_normalized.strip():
        return "empty"

    # Unicode block ranges for the scripts we need to distinguish.
    # Each tuple: (start_codepoint, end_codepoint, label)
    _SCRIPT_RANGES = (
        (0x0900, 0x097F, "devanagari"),
        (0x0980, 0x09FF, "bengali"),
        (0x0A00, 0x0A7F, "gurmukhi"),
        (0x0A80, 0x0AFF, "gujarati"),
        (0x0B00, 0x0B7F, "oriya"),
        (0x0B80, 0x0BFF, "tamil"),
        (0x0C00, 0x0C7F, "telugu"),
        (0x0C80, 0x0CFF, "kannada"),
        (0x0D00, 0x0D7F, "malayalam"),
        (0x0600, 0x06FF, "arabic"),
        (0x4E00, 0x9FFF, "cjk"),
        (0x3040, 0x30FF, "japanese"),
        (0xAC00, 0xD7AF, "korean"),
    )

    latin_count = 0
    non_latin_script_counts: dict = {}

    for ch in name_normalized:
        cat = unicodedata.category(ch)
        if not cat.startswith("L"):
            # Skip digits (Nd), punctuation (P*), separators (Z*), etc.
            continue
        cp = ord(ch)
        # Latin: Basic Latin letters + Latin Extended blocks
        if (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
            latin_count += 1
        else:
            script = "non_latin"
            for start, end, name in _SCRIPT_RANGES:
                if start <= cp <= end:
                    script = name
                    break
            non_latin_script_counts[script] = non_latin_script_counts.get(script, 0) + 1

    total_letters = latin_count + sum(non_latin_script_counts.values())
    if total_letters == 0:
        return "empty"

    if non_latin_script_counts and latin_count > 0:
        return "latin_and_non_latin"
    if latin_count > 0:
        return "latin"
    # Pure non-Latin: return the dominant script name
    dominant = max(non_latin_script_counts, key=non_latin_script_counts.get)
    return dominant


# ---------------------------------------------------------------------------
# Vectorized (bulk) wrappers for the new derived-representation functions
# ---------------------------------------------------------------------------

def extract_address_numbers_series(addresses_normalized: pd.Series) -> pd.Series:
    """Vectorized equivalent of extract_address_numbers over a column."""
    return addresses_normalized.fillna("").map(extract_address_numbers)


def extract_address_postal_code_series(addresses_normalized: pd.Series) -> pd.Series:
    """Vectorized equivalent of extract_address_postal_code over a column."""
    return addresses_normalized.fillna("").map(extract_address_postal_code)


def make_address_sorted_tokens_series(addresses_canonical: pd.Series) -> pd.Series:
    """Vectorized equivalent of make_address_sorted_tokens over a column."""
    return addresses_canonical.fillna("").map(make_address_sorted_tokens)


def classify_name_script_series(names_normalized: pd.Series) -> pd.Series:
    """Vectorized equivalent of classify_name_script over a column."""
    return names_normalized.fillna("").map(classify_name_script)
