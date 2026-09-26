"""Indic-script transliteration and dictionary-learning for entity resolution.

PURPOSE
-------
This module provides two independent building blocks:

1. ``IndicTransliterator``
   Character-level phonetic mapping from nine Indic scripts
   (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
   Kannada, Malayalam) to a unified Latin phonetic representation.
   Uses only Python stdlib (``unicodedata``) — no external packages.

2. ``DictionaryLearner``
   Learns token-level transliteration-to-English corrections from
   (Indic name, Latin name) training pairs — e.g.:

       "praivet"   → "private"
       "likhmited" → "limited"
       "injiniyaring" → "engineering"

   The learned dictionary is applied *after* character-level
   transliteration to fix systematic phonetic divergences that a
   generic character map cannot capture.

INTEGRATION
-----------
``normalization.py`` exposes ``set_transliterator(t)`` /
``get_transliterator()``.  When a transliterator is installed, every
call to ``normalize_name_basic()`` passes Indic text through
``IndicTransliterator.transliterate()`` before the rest of the pipeline
runs (step 0.5 in the documented 10-step sequence).

Latin-only names are returned unchanged.  Mixed Latin + Indic names have
only their Indic tokens transliterated.

DESIGN INVARIANTS
-----------------
* No row is ever dropped.
* The raw ``business_name`` column is always preserved verbatim.
* ``name_script_class`` is computed from the raw name (before
  transliteration) so the metadata column always reflects the original
  script.
* When ``IndicTransliterator`` is ``None`` (default), the pipeline is
  bit-for-bit identical to the pre-transliteration pipeline — zero
  regression risk.
* No external dependencies.  stdlib only.

SCRIPT COVERAGE
---------------
The 9 Indic scripts share a common phonological structure inherited from
the Brahmic family: consonants carry an inherent /a/ vowel, matras
(vowel signs) override it, and the virama suppresses it.  The character
maps below encode this shared structure for each script.

Unicode block assignments (ISO 15924):
  Devanagari  U+0900–U+097F   used by Hindi, Marathi, Nepali, Sanskrit
  Bengali     U+0980–U+09FF   used by Bengali, Assamese
  Gurmukhi    U+0A00–U+0A7F   used by Punjabi
  Gujarati    U+0A80–U+0AFF   used by Gujarati
  Oriya       U+0B00–U+0B7F   used by Odia
  Tamil       U+0B80–U+0BFF   used by Tamil
  Telugu      U+0C00–U+0C7F   used by Telugu
  Kannada     U+0C80–U+0CFF   used by Kannada
  Malayalam   U+0D00–U+0D7F   used by Malayalam
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shared Unicode constants
# ---------------------------------------------------------------------------

# Combining marks that are phonemically essential in Indic scripts (do NOT
# strip these — they are the vowel signs / matras).  Unicode category "Mc"
# (Mark, Spacing Combining) and "Mn" (Mark, Nonspacing).
_MC = "Mc"
_MN = "Mn"

# Virama (vowel-suppressor / halant).  Each script has its own codepoint but
# they share the same phonetic role: suppress the inherent /a/ of the
# preceding consonant.  Listed per-script for the transliterator.
_VIRAMA: dict[str, str] = {
    "devanagari": "\u094D",  # ् DEVANAGARI SIGN VIRAMA
    "bengali":    "\u09CD",  # ্ BENGALI SIGN VIRAMA
    "gurmukhi":   "\u0A4D",  # ੍ GURMUKHI SIGN VIRAMA
    "gujarati":   "\u0ACD",  # ્ GUJARATI SIGN VIRAMA
    "oriya":      "\u0B4D",  # ୍ ORIYA SIGN VIRAMA
    "tamil":      "\u0BCD",  # ் TAMIL SIGN VIRAMA
    "telugu":     "\u0C4D",  # ్ TELUGU SIGN VIRAMA
    "kannada":    "\u0CCD",  # ್ KANNADA SIGN VIRAMA
    "malayalam":  "\u0D4D",  # ് MALAYALAM SIGN VIRAMA
}

# Inherent vowel appended to a consonant when no matra or virama follows.
_INHERENT_VOWEL = "a"


# ---------------------------------------------------------------------------
# Per-script character maps
# ---------------------------------------------------------------------------
# Each map is a dict { unicode_char: latin_string }.
# Keys cover:
#   - Independent vowels (used at word/syllable start)
#   - Vowel signs / matras (Mc/Mn combining marks)
#   - Consonants (with inherent /a/ — handled by transliterate_syllable)
#   - Numerals (0-9)
#   - Common punctuation / anusvara / visarga / chandrabindu / nukta
#
# Romanization follows a simplified ITRANS-like scheme designed for
# entity-resolution matching (not scholarly transliteration): the goal is
# that two names that sound the same produce the same Latin string.  Hard
# consonant clusters and retroflex/dental distinctions are collapsed to their
# most common English spelling.
# ---------------------------------------------------------------------------

# ── Devanagari ─────────────────────────────────────────────────────────────
_DEVANAGARI_MAP: dict[str, str] = {
    # Independent vowels
    "अ": "a",  "आ": "aa", "इ": "i",  "ई": "ii", "उ": "u",  "ऊ": "uu",
    "ए": "e",  "ऐ": "ai", "ओ": "o",  "औ": "au", "ऋ": "ri", "ॠ": "rii",
    "ऌ": "li", "ॡ": "lii","अं": "am","अः": "ah",
    # Vowel signs (matras)
    "\u093E": "aa",  # ा  aa
    "\u093F": "i",   # ि  i
    "\u0940": "ii",  # ी  ii
    "\u0941": "u",   # ु  u
    "\u0942": "uu",  # ू  uu
    "\u0943": "ri",  # ृ  ri
    "\u0944": "rii", # ॄ  rii
    "\u0947": "e",   # े  e
    "\u0948": "ai",  # ै  ai
    "\u094B": "o",   # ो  o
    "\u094C": "au",  # ौ  au
    "\u094D": "",    # ्  virama — suppresses inherent vowel (handled separately)
    "\u0902": "n",   # ं  anusvara (nasal)
    "\u0903": "h",   # ः  visarga
    "\u0901": "n",   # ँ  chandrabindu
    "\u0945": "e",   # ॅ  candra e (short e, rare)
    "\u0949": "o",   # ॉ  candra o (loan-word short o, e.g. "kॉrp" → "korp")
    "\u094A": "o",   # ॊ  short o
    # Consonants
    "क": "k",  "ख": "kh", "ग": "g",  "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh","ज": "j",  "झ": "jh", "ञ": "ny",
    "ट": "t",  "ठ": "th", "ड": "d",  "ढ": "dh", "ण": "n",
    "त": "t",  "थ": "th", "द": "d",  "ध": "dh", "न": "n",
    "प": "p",  "फ": "ph", "ब": "b",  "भ": "bh", "म": "m",
    "य": "y",  "र": "r",  "ल": "l",  "व": "v",  "श": "sh",
    "ष": "sh", "स": "s",  "ह": "h",
    "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gya",
    # Nukta consonants (loan-sound adaptations)
    "क़": "q",  "ख़": "kh", "ग़": "gh", "ज़": "z",  "ड़": "r",
    "ढ़": "rh", "फ़": "f",  "य़": "y",  "ऱ": "r",
    # Numerals
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
    # Punctuation / danda
    "।": ".", "॥": ".",
}

# ── Bengali ────────────────────────────────────────────────────────────────
_BENGALI_MAP: dict[str, str] = {
    # Independent vowels
    "অ": "a",  "আ": "aa", "ই": "i",  "ঈ": "ii", "উ": "u",  "ঊ": "uu",
    "এ": "e",  "ঐ": "oi", "ও": "o",  "ঔ": "ou", "ঋ": "ri",
    # Vowel signs
    "\u09BE": "aa", "\u09BF": "i",  "\u09C0": "ii", "\u09C1": "u",
    "\u09C2": "uu", "\u09C3": "ri", "\u09C7": "e",  "\u09C8": "oi",
    "\u09CB": "o",  "\u09CC": "ou", "\u09CD": "",   # virama
    "\u0982": "n",  "\u0983": "h",  "\u0981": "n",  # anusvara / visarga / chandrabindu
    # Consonants
    "ক": "k",  "খ": "kh", "গ": "g",  "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh","জ": "j",  "ঝ": "jh", "ঞ": "ny",
    "ট": "t",  "ঠ": "th", "ড": "d",  "ঢ": "dh", "ণ": "n",
    "ত": "t",  "থ": "th", "দ": "d",  "ধ": "dh", "ন": "n",
    "প": "p",  "ফ": "ph", "ব": "b",  "ভ": "bh", "ম": "m",
    "য": "y",  "র": "r",  "ল": "l",  "শ": "sh",
    "ষ": "sh", "স": "s",  "হ": "h",  "ড়": "r", "ঢ়": "rh",
    "য়": "y", "ৎ": "t",
    # Numerals
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}

# ── Gurmukhi ───────────────────────────────────────────────────────────────
_GURMUKHI_MAP: dict[str, str] = {
    # Independent vowels
    "ਅ": "a",  "ਆ": "aa", "ਇ": "i",  "ਈ": "ii", "ਉ": "u",  "ਊ": "uu",
    "ਏ": "e",  "ਐ": "ai", "ਓ": "o",  "ਔ": "au",
    # Vowel signs
    "\u0A3E": "aa", "\u0A3F": "i",  "\u0A40": "ii", "\u0A41": "u",
    "\u0A42": "uu", "\u0A47": "e",  "\u0A48": "ai", "\u0A4B": "o",
    "\u0A4C": "au", "\u0A4D": "",   # virama
    "\u0A02": "n",  "\u0A70": "n",  "\u0A71": "h",
    # Consonants
    "ਕ": "k",  "ਖ": "kh", "ਗ": "g",  "ਘ": "gh", "ਙ": "ng",
    "ਚ": "ch", "ਛ": "chh","ਜ": "j",  "ਝ": "jh", "ਞ": "ny",
    "ਟ": "t",  "ਠ": "th", "ਡ": "d",  "ਢ": "dh", "ਣ": "n",
    "ਤ": "t",  "ਥ": "th", "ਦ": "d",  "ਧ": "dh", "ਨ": "n",
    "ਪ": "p",  "ਫ": "ph", "ਬ": "b",  "ਭ": "bh", "ਮ": "m",
    "ਯ": "y",  "ਰ": "r",  "ਲ": "l",  "ਵ": "v",  "ਸ਼": "sh",
    "ਸ": "s",  "ਹ": "h",  "ਲ਼": "l", "ਖ਼": "kh", "ਗ਼": "gh",
    "ਜ਼": "z", "ਫ਼": "f",
    # Numerals
    "੦": "0", "੧": "1", "੨": "2", "੩": "3", "੪": "4",
    "੫": "5", "੬": "6", "੭": "7", "੮": "8", "੯": "9",
}

# ── Gujarati ───────────────────────────────────────────────────────────────
_GUJARATI_MAP: dict[str, str] = {
    # Independent vowels
    "અ": "a",  "આ": "aa", "ઇ": "i",  "ઈ": "ii", "ઉ": "u",  "ઊ": "uu",
    "એ": "e",  "ઐ": "ai", "ઓ": "o",  "ઔ": "au", "ઋ": "ri",
    # Vowel signs
    "\u0ABE": "aa", "\u0ABF": "i",  "\u0AC0": "ii", "\u0AC1": "u",
    "\u0AC2": "uu", "\u0AC3": "ri", "\u0AC7": "e",  "\u0AC8": "ai",
    "\u0ACB": "o",  "\u0ACC": "au", "\u0ACD": "",   # virama
    "\u0A82": "n",  "\u0A83": "h",  "\u0A81": "n",
    # Consonants
    "ક": "k",  "ખ": "kh", "ગ": "g",  "ઘ": "gh", "ઙ": "ng",
    "ચ": "ch", "છ": "chh","જ": "j",  "ઝ": "jh", "ઞ": "ny",
    "ટ": "t",  "ઠ": "th", "ડ": "d",  "ઢ": "dh", "ણ": "n",
    "ત": "t",  "થ": "th", "દ": "d",  "ધ": "dh", "ન": "n",
    "પ": "p",  "ફ": "ph", "બ": "b",  "ભ": "bh", "મ": "m",
    "ય": "y",  "ર": "r",  "લ": "l",  "વ": "v",  "શ": "sh",
    "ષ": "sh", "સ": "s",  "હ": "h",  "ળ": "l",
    # Numerals
    "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
    "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
}

# ── Oriya (Odia) ───────────────────────────────────────────────────────────
_ORIYA_MAP: dict[str, str] = {
    # Independent vowels
    "ଅ": "a",  "ଆ": "aa", "ଇ": "i",  "ଈ": "ii", "ଉ": "u",  "ଊ": "uu",
    "ଏ": "e",  "ଐ": "ai", "ଓ": "o",  "ଔ": "au", "ଋ": "ri",
    # Vowel signs
    "\u0B3E": "aa", "\u0B3F": "i",  "\u0B40": "ii", "\u0B41": "u",
    "\u0B42": "uu", "\u0B43": "ri", "\u0B47": "e",  "\u0B48": "ai",
    "\u0B4B": "o",  "\u0B4C": "au", "\u0B4D": "",   # virama
    "\u0B02": "n",  "\u0B03": "h",  "\u0B01": "n",
    # Consonants
    "କ": "k",  "ଖ": "kh", "ଗ": "g",  "ଘ": "gh", "ଙ": "ng",
    "ଚ": "ch", "ଛ": "chh","ଜ": "j",  "ଝ": "jh", "ଞ": "ny",
    "ଟ": "t",  "ଠ": "th", "ଡ": "d",  "ଢ": "dh", "ଣ": "n",
    "ତ": "t",  "ଥ": "th", "ଦ": "d",  "ଧ": "dh", "ନ": "n",
    "ପ": "p",  "ଫ": "ph", "ବ": "b",  "ଭ": "bh", "ମ": "m",
    "ଯ": "y",  "ର": "r",  "ଲ": "l",  "ଵ": "v",  "ଶ": "sh",
    "ଷ": "sh", "ସ": "s",  "ହ": "h",  "ଳ": "l",  "ଡ଼": "r",
    # Numerals
    "୦": "0", "୧": "1", "୨": "2", "୩": "3", "୪": "4",
    "୫": "5", "୬": "6", "୭": "7", "୮": "8", "୯": "9",
}

# ── Tamil ──────────────────────────────────────────────────────────────────
# Tamil has a more restricted consonant inventory than other Brahmic scripts.
# The same glyph is used for both voiced and unvoiced variants in many
# positions; the romanization reflects the most common spoken realisation.
_TAMIL_MAP: dict[str, str] = {
    # Independent vowels
    "அ": "a",  "ஆ": "aa", "இ": "i",  "ஈ": "ii", "உ": "u",  "ஊ": "uu",
    "எ": "e",  "ஏ": "ee", "ஐ": "ai", "ஒ": "o",  "ஓ": "oo", "ஔ": "au",
    # Vowel signs
    "\u0BBE": "aa", "\u0BBF": "i",  "\u0BC0": "ii", "\u0BC1": "u",
    "\u0BC2": "uu", "\u0BC6": "e",  "\u0BC7": "ee", "\u0BC8": "ai",
    "\u0BCA": "o",  "\u0BCB": "oo", "\u0BCC": "au", "\u0BCD": "",  # virama
    "\u0B82": "n",  "\u0B83": "h",
    # Consonants
    "க": "k",  "ங": "ng", "ச": "ch", "ஞ": "ny", "ட": "t",  "ண": "n",
    "த": "th", "ந": "n",  "ப": "p",  "ம": "m",  "ய": "y",  "ர": "r",
    "ல": "l",  "வ": "v",  "ழ": "zh", "ள": "l",  "ற": "r",  "ன": "n",
    "ஜ": "j",  "ஷ": "sh", "ஸ": "s",  "ஹ": "h",  "க்ஷ": "ksh",
    # Numerals
    "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
    "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
}

# ── Telugu ─────────────────────────────────────────────────────────────────
_TELUGU_MAP: dict[str, str] = {
    # Independent vowels
    "అ": "a",  "ఆ": "aa", "ఇ": "i",  "ఈ": "ii", "ఉ": "u",  "ఊ": "uu",
    "ఎ": "e",  "ఏ": "ee", "ఐ": "ai", "ఒ": "o",  "ఓ": "oo", "ఔ": "au",
    "ఋ": "ri",
    # Vowel signs
    "\u0C3E": "aa", "\u0C3F": "i",  "\u0C40": "ii", "\u0C41": "u",
    "\u0C42": "uu", "\u0C43": "ri", "\u0C46": "e",  "\u0C47": "ee",
    "\u0C48": "ai", "\u0C4A": "o",  "\u0C4B": "oo", "\u0C4C": "au",
    "\u0C4D": "",   # virama
    "\u0C02": "n",  "\u0C03": "h",  "\u0C01": "n",
    # Consonants
    "క": "k",  "ఖ": "kh", "గ": "g",  "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh","జ": "j",  "ఝ": "jh", "ఞ": "ny",
    "ట": "t",  "ఠ": "th", "డ": "d",  "ఢ": "dh", "ణ": "n",
    "త": "t",  "థ": "th", "ద": "d",  "ధ": "dh", "న": "n",
    "ప": "p",  "ఫ": "ph", "బ": "b",  "భ": "bh", "మ": "m",
    "య": "y",  "ర": "r",  "ల": "l",  "వ": "v",  "శ": "sh",
    "ష": "sh", "స": "s",  "హ": "h",  "ళ": "l",  "ఱ": "r",
    # Numerals
    "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
    "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
}

# ── Kannada ────────────────────────────────────────────────────────────────
_KANNADA_MAP: dict[str, str] = {
    # Independent vowels
    "ಅ": "a",  "ಆ": "aa", "ಇ": "i",  "ಈ": "ii", "ಉ": "u",  "ಊ": "uu",
    "ಎ": "e",  "ಏ": "ee", "ಐ": "ai", "ಒ": "o",  "ಓ": "oo", "ಔ": "au",
    "ಋ": "ri",
    # Vowel signs
    "\u0CBE": "aa", "\u0CBF": "i",  "\u0CC0": "ii", "\u0CC1": "u",
    "\u0CC2": "uu", "\u0CC3": "ri", "\u0CC6": "e",  "\u0CC7": "ee",
    "\u0CC8": "ai", "\u0CCA": "o",  "\u0CCB": "oo", "\u0CCC": "au",
    "\u0CCD": "",   # virama
    "\u0C82": "n",  "\u0C83": "h",  "\u0C81": "n",
    # Consonants
    "ಕ": "k",  "ಖ": "kh", "ಗ": "g",  "ಘ": "gh", "ಙ": "ng",
    "ಚ": "ch", "ಛ": "chh","ಜ": "j",  "ಝ": "jh", "ಞ": "ny",
    "ಟ": "t",  "ಠ": "th", "ಡ": "d",  "ಢ": "dh", "ಣ": "n",
    "ತ": "t",  "ಥ": "th", "ದ": "d",  "ಧ": "dh", "ನ": "n",
    "ಪ": "p",  "ಫ": "ph", "ಬ": "b",  "ಭ": "bh", "ಮ": "m",
    "ಯ": "y",  "ರ": "r",  "ಲ": "l",  "ವ": "v",  "ಶ": "sh",
    "ಷ": "sh", "ಸ": "s",  "ಹ": "h",  "ಳ": "l",  "ಱ": "r",
    # Numerals
    "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
    "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
}

# ── Malayalam ──────────────────────────────────────────────────────────────
_MALAYALAM_MAP: dict[str, str] = {
    # Independent vowels
    "അ": "a",  "ആ": "aa", "ഇ": "i",  "ഈ": "ii", "ഉ": "u",  "ഊ": "uu",
    "എ": "e",  "ഏ": "ee", "ഐ": "ai", "ഒ": "o",  "ഓ": "oo", "ഔ": "au",
    "ഋ": "ri",
    # Vowel signs
    "\u0D3E": "aa", "\u0D3F": "i",  "\u0D40": "ii", "\u0D41": "u",
    "\u0D42": "uu", "\u0D43": "ri", "\u0D46": "e",  "\u0D47": "ee",
    "\u0D48": "ai", "\u0D4A": "o",  "\u0D4B": "oo", "\u0D4C": "au",
    "\u0D4D": "",   # virama
    "\u0D02": "n",  "\u0D03": "h",  "\u0D01": "n",
    # Consonants
    "ക": "k",  "ഖ": "kh", "ഗ": "g",  "ഘ": "gh", "ങ": "ng",
    "ച": "ch", "ഛ": "chh","ജ": "j",  "ഝ": "jh", "ഞ": "ny",
    "ട": "t",  "ഠ": "th", "ഡ": "d",  "ഢ": "dh", "ണ": "n",
    "ത": "t",  "ഥ": "th", "ദ": "d",  "ധ": "dh", "ന": "n",
    "പ": "p",  "ഫ": "ph", "ബ": "b",  "ഭ": "bh", "മ": "m",
    "യ": "y",  "ര": "r",  "ല": "l",  "വ": "v",  "ശ": "sh",
    "ഷ": "sh", "സ": "s",  "ഹ": "h",  "ള": "l",  "ഴ": "zh",
    "റ": "r",
    # Numerals
    "൦": "0", "൧": "1", "൨": "2", "൩": "3", "൪": "4",
    "൫": "5", "൬": "6", "൭": "7", "൮": "8", "൯": "9",
}

# ---------------------------------------------------------------------------
# Unified lookup — merge all scripts into one flat dict.
# Conflicts (same Unicode char in two scripts) don't exist because each
# Brahmic script occupies its own disjoint Unicode block.
# ---------------------------------------------------------------------------
_ALL_MAPS: dict[str, dict[str, str]] = {
    "devanagari": _DEVANAGARI_MAP,
    "bengali":    _BENGALI_MAP,
    "gurmukhi":   _GURMUKHI_MAP,
    "gujarati":   _GUJARATI_MAP,
    "oriya":      _ORIYA_MAP,
    "tamil":      _TAMIL_MAP,
    "telugu":     _TELUGU_MAP,
    "kannada":    _KANNADA_MAP,
    "malayalam":  _MALAYALAM_MAP,
}

# Flat merged map: char → latin string.  Built once at import time.
_FLAT_MAP: dict[str, str] = {}
for _script_map in _ALL_MAPS.values():
    _FLAT_MAP.update(_script_map)

# Virama set — a frozenset of all virama codepoints for fast membership test.
_ALL_VIRAMA: frozenset[str] = frozenset(_VIRAMA.values())

# ---------------------------------------------------------------------------
# Indic codepoint range detection (mirrors classify_name_script logic)
# ---------------------------------------------------------------------------
_INDIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
)


def _has_indic(text: str) -> bool:
    """Return True if *text* contains at least one Indic-script codepoint."""
    for ch in text:
        cp = ord(ch)
        for lo, hi in _INDIC_RANGES:
            if lo <= cp <= hi:
                return True
    return False


# ---------------------------------------------------------------------------
# Core transliteration engine
# ---------------------------------------------------------------------------

def _transliterate_word(word: str) -> str:
    """Transliterate a single word (no whitespace) from any Indic script to
    Latin using the flat map and Brahmic inherent-vowel rules.

    Algorithm
    ---------
    We iterate character-by-character.  Each character falls into one of
    four categories:

    1. **Virama** (vowel suppressor): means the preceding consonant has NO
       following vowel.  We peek back and strip the inherent /a/ we would
       otherwise have emitted.

    2. **Consonant** (has a map entry that is a *letter sequence*): we emit
       the Latin equivalent and append the inherent vowel "a".  If the
       *next* character is a virama or another matra (vowel sign), we will
       fix the vowel in the next iteration.

    3. **Matra / vowel sign** (combining mark Mc or Mn with a map entry):
       we need to replace the inherent "a" we just appended for the
       preceding consonant with the correct vowel.  We pop the last
       character(s) from the output buffer and append the matra's Latin.

    4. **Any other character** (Latin, digit, punctuation, space): passed
       through unchanged.

    The result is a best-effort phonetic Latin rendering.  It is not a
    scholarly transliteration — the goal is entity-resolution matching, so
    dental/retroflex distinctions and long/short vowel distinctions in
    clusters are collapsed.
    """
    if not word:
        return word

    buf: list[str] = []
    chars = list(word)
    n = len(chars)

    i = 0
    while i < n:
        ch = chars[i]
        cp = ord(ch)

        # ── Check for a 2-char ligature first (e.g. "क्ष", "த்") ─────────
        if i + 1 < n:
            pair = ch + chars[i + 1]
            if pair in _FLAT_MAP:
                latin = _FLAT_MAP[pair]
                buf.append(latin)
                # Ligatures are treated as consonants — append inherent vowel
                # unless next char is virama or matra.
                if i + 2 < n:
                    nxt = chars[i + 2]
                    nxt_latin = _FLAT_MAP.get(nxt)
                    nxt_cat = unicodedata.category(nxt)
                    if nxt in _ALL_VIRAMA:
                        # Virama follows: no inherent vowel needed
                        i += 3  # consume pair + virama
                        continue
                    if nxt_cat in (_MC, _MN) and nxt_latin is not None:
                        # Matra follows: replace inherent vowel
                        if nxt_latin:  # non-empty (not virama mapping)
                            buf.append(nxt_latin)
                        i += 3  # consume pair + matra
                        continue
                buf.append(_INHERENT_VOWEL)
                i += 2
                continue

        # ── Single character ──────────────────────────────────────────────
        if ch in _FLAT_MAP:
            latin = _FLAT_MAP[ch]
            cat = unicodedata.category(ch)

            if ch in _ALL_VIRAMA:
                # Virama: strip the inherent "a" we appended for the preceding
                # consonant.  The buf ends with the consonant + "a"; pop "a".
                if buf and buf[-1] == _INHERENT_VOWEL:
                    buf.pop()
                i += 1
                continue

            if cat in (_MC, _MN):
                # Matra (vowel sign): replaces inherent vowel of preceding consonant.
                # Pop the trailing "a" if we just added it.
                if buf and buf[-1] == _INHERENT_VOWEL:
                    buf.pop()
                if latin:  # empty string = virama (already handled above, but safe)
                    buf.append(latin)
                i += 1
                continue

            # Regular consonant or independent vowel
            buf.append(latin)
            # Check whether the next character will override the inherent vowel.
            if i + 1 < n:
                nxt = chars[i + 1]
                nxt_latin = _FLAT_MAP.get(nxt)
                nxt_cat = unicodedata.category(nxt)
                if nxt not in _ALL_VIRAMA and not (
                    nxt_cat in (_MC, _MN) and nxt_latin is not None
                ):
                    # Next char won't modify this consonant's vowel → add inherent
                    buf.append(_INHERENT_VOWEL)
            else:
                # Last char in word → add inherent vowel for a final consonant
                buf.append(_INHERENT_VOWEL)

        else:
            # Not in any Indic map (Latin, digit, punctuation…) — pass through.
            buf.append(ch)

        i += 1

    result = "".join(buf)
    # Collapse any double-inherent-vowels that can arise from ligature handling
    result = re.sub(r"aa+", "aa", result)
    return result


def transliterate(text: str) -> str:
    """Transliterate *text* from any Indic script to Latin phonetics.

    Processes the text token by token (split on whitespace), transliterating
    each token independently.  Tokens that contain no Indic characters are
    returned unchanged (so a mixed Latin+Indic name has only its Indic
    portions transliterated).

    Parameters
    ----------
    text:
        Raw business name string, possibly containing Indic characters.

    Returns
    -------
    str
        Phonetic Latin rendering.  Case is mixed (consonants emit lowercase,
        independent vowels emit lowercase).  The subsequent ``normalize_name_basic``
        pipeline will lowercase everything uniformly.

    Examples
    --------
    >>> transliterate("प्राइवेट लिमिटेड")
    'praiveta limiteda'
    >>> transliterate("Raj नमस्ते")
    'Raj namaste'   # Latin token "Raj" is untouched
    >>> transliterate("hello world")
    'hello world'   # No Indic content — untouched
    """
    if not text or not _has_indic(text):
        return text

    tokens = text.split(" ")
    out_tokens: list[str] = []
    for tok in tokens:
        if _has_indic(tok):
            out_tokens.append(_transliterate_word(tok))
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


# ---------------------------------------------------------------------------
# IndicTransliterator — stateful wrapper with optional learned dictionary
# ---------------------------------------------------------------------------

class IndicTransliterator:
    """Wrapper around the character-level transliteration engine.

    Optionally enhanced by a ``DictionaryLearner`` to apply post-hoc
    token-level corrections (e.g. "praivet" → "private").

    Parameters
    ----------
    dictionary:
        Optional pre-built correction dictionary
        ``{transliterated_token: english_token}``.  If supplied, it is
        applied as a post-processing step after character-level
        transliteration.
    """

    def __init__(
        self,
        dictionary: Optional[Dict[str, str]] = None,
    ) -> None:
        self._dict: Dict[str, str] = dict(dictionary) if dictionary else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transliterate(self, text: str) -> str:
        """Transliterate *text* and apply learned dictionary corrections.

        1. Character-level transliteration (``transliterate()`` function).
        2. Token-level dictionary correction (if dictionary is non-empty).
        """
        result = transliterate(text)
        if self._dict:
            result = self._apply_dict(result)
        return result

    def update_dictionary(self, corrections: Dict[str, str]) -> None:
        """Merge *corrections* into the internal dictionary.

        Later calls to ``transliterate()`` will apply the updated dictionary.
        Existing entries are overwritten by the new ones.
        """
        self._dict.update(corrections)

    def get_dictionary(self) -> Dict[str, str]:
        """Return a copy of the current correction dictionary."""
        return dict(self._dict)

    def save_dictionary(self, path: Path) -> None:
        """Persist the dictionary to a JSON file at *path*."""
        Path(path).write_text(
            json.dumps(self._dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "IndicTransliterator":
        """Load an ``IndicTransliterator`` from a JSON dictionary file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(dictionary=data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_dict(self, text: str) -> str:
        """Replace transliterated tokens with dictionary corrections.

        Operates at whitespace-split token level so that "praivet limiteda
        kampani" → "private limited company" if all three entries are in
        the dictionary.
        """
        tokens = text.split(" ")
        return " ".join(self._dict.get(tok, tok) for tok in tokens)


# ---------------------------------------------------------------------------
# DictionaryLearner — build correction maps from training pairs
# ---------------------------------------------------------------------------

class DictionaryLearner:
    """Learn token-level transliteration→English corrections from training data.

    The learner is given a list of (Indic name, corresponding Latin/English
    name) pairs.  It:

    1. Transliterates each Indic name with the character-level engine.
    2. Aligns tokens between the transliteration and the Latin name by
       position (1-to-1 alignment of whitespace-split token lists).
    3. Records (transliterated_token, latin_token) pairs.
    4. Keeps the most common Latin target for each transliterated source
       (majority vote), filtering out:
         - Pairs where the transliterated token is already a perfect
           ASCII match (no correction needed).
         - Pairs where the Latin token is empty or numeric.
         - Rare mappings seen fewer than ``min_count`` times.

    Parameters
    ----------
    min_count:
        Minimum number of times a (src, tgt) pair must be seen to be
        included in the learned dictionary.  Default 1 (include everything
        seen at least once — appropriate for small datasets; raise for
        production-scale data to avoid noise).
    """

    def __init__(self, min_count: int = 1) -> None:
        self._min_count = min_count
        # src_token → Counter({tgt_token: count})
        self._counts: Dict[str, Counter] = {}

    def learn_from_pairs(
        self,
        indic_names: List[str],
        latin_names: List[str],
    ) -> None:
        """Accumulate token-pair observations from a list of name pairs.

        Parameters
        ----------
        indic_names:
            List of raw Indic-script business names (from source1/2/3).
        latin_names:
            List of corresponding Latin-script names (parallel list, same
            length).  Typically the romanized name from the *other* source
            file for the same ground-truth entity.
        """
        if len(indic_names) != len(latin_names):
            raise ValueError(
                f"indic_names and latin_names must have the same length "
                f"({len(indic_names)} vs {len(latin_names)})"
            )

        for indic_raw, latin_raw in zip(indic_names, latin_names):
            if not indic_raw or not latin_raw:
                continue
            if not _has_indic(indic_raw):
                continue  # skip already-Latin source names

            transliterated = transliterate(indic_raw).lower().strip()
            latin_norm = latin_raw.lower().strip()

            src_tokens = transliterated.split()
            tgt_tokens = latin_norm.split()

            # Positional 1-to-1 alignment (zip stops at the shorter list).
            for src_tok, tgt_tok in zip(src_tokens, tgt_tokens):
                if not src_tok or not tgt_tok:
                    continue
                if not tgt_tok.isalpha():
                    continue  # skip numeric / punctuation targets
                if src_tok == tgt_tok:
                    continue  # transliteration already correct — no correction needed
                if src_tok not in self._counts:
                    self._counts[src_tok] = Counter()
                self._counts[src_tok][tgt_tok] += 1

    def build_dictionary(self) -> Dict[str, str]:
        """Return the learned correction dictionary.

        For each transliterated token that has been observed, picks the most
        frequent Latin target (majority vote).  Filters out pairs seen fewer
        than ``min_count`` times.

        Returns
        -------
        dict
            ``{transliterated_token: english_correction}``
        """
        result: Dict[str, str] = {}
        for src_tok, counter in self._counts.items():
            best_tgt, best_count = counter.most_common(1)[0]
            if best_count >= self._min_count:
                result[src_tok] = best_tgt
        return result

    def build_transliterator(self) -> IndicTransliterator:
        """Convenience: build and return an ``IndicTransliterator`` with the
        learned dictionary already installed."""
        return IndicTransliterator(dictionary=self.build_dictionary())
