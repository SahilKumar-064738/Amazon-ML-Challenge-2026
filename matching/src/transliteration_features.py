"""
transliteration_features.py
----------------------------
Phase 4 — Sid Indic Transliteration Dictionary & Features.

Implements the learned transliteration dictionary capability adapted from
Sid-techweb/AmazonML-New (indic_dictionary.py, apply_indic_dict.py).

SOURCE IN SID REPO:
    business_entity_resolution/src/indic_dictionary.py:
        - Aligns matched training pairs with equal token counts:
          qt, st = qn.split(), sn.split()
          if len(qt) == len(st):
              for a, b in zip(qt, st):
                  tok_counts[a] += 1
                  pair_counts[(a, b)] += 1
        - Retains dominant mappings where a != b, count >= MIN_COUNT,
          and count / tok_counts[a] >= MIN_SHARE (dominant probability >= 60%).
    business_entity_resolution/src/apply_indic_dict.py:
        - Remaps name tokens using the learned mapping:
          toks = [mapping.get(t, t) for t in name.split()]

CHALLENGE COMPLIANCE & LEAKAGE PREVENTION:
    - Zero external APIs, zero external databases, zero external web requests.
    - Dictionary is learned solely from training matched pairs.
    - When evaluating validation splits, only the training entity split is used.
    - Preserves existing immutable normalization/cleaning pipeline.

FEATURES (8 features in V9):
    1. translit_token_mapped_count     (int32): Number of tokens remapped in candidate name.
    2. translit_token_mapped_ratio     (float64 in [0, 1]): Ratio of remapped tokens to candidate tokens.
    3. translit_name_fuzz_ratio        (float64 in [0, 1]): fuzz.ratio(s1_name, remapped_candidate_name).
    4. translit_name_token_sort_ratio  (float64 in [0, 1]): fuzz.token_sort_ratio(s1_name, remapped_candidate_name).
    5. translit_name_token_set_ratio   (float64 in [0, 1]): fuzz.token_set_ratio(s1_name, remapped_candidate_name).
    6. translit_similarity_gain        (float64 in [0, 1]): max(0.0, translit_name_fuzz_ratio - name_fuzz_ratio).
    7. translit_addr_mapped_count      (int32): Number of tokens remapped in candidate address.
    8. translit_addr_fuzz_ratio        (float64 in [0, 1]): fuzz.ratio(s1_addr, remapped_candidate_addr).

Public API:
    TRANSLITERATION_FEATURE_COLS : list[str]
    TransliterationDictionary    : class for learning, saving, loading, and remapping
    get_default_transliteration_dict() -> TransliterationDictionary
    add_transliteration_features(pair_df, dictionary=None) -> pd.DataFrame
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# ---------------------------------------------------------------------------
# Feature Column Names (8 features)
# ---------------------------------------------------------------------------

TRANSLITERATION_FEATURE_COLS: list[str] = [
    "translit_token_mapped_count",
    "translit_token_mapped_ratio",
    "translit_name_fuzz_ratio",
    "translit_name_token_sort_ratio",
    "translit_name_token_set_ratio",
    "translit_similarity_gain",
    "translit_addr_mapped_count",
    "translit_addr_fuzz_ratio",
]


# ---------------------------------------------------------------------------
# Curated competition phonetic transliteration priors (seed dictionary)
# ---------------------------------------------------------------------------
# Common English business words frequently transliterated in Indian entity data.
_SEED_TRANSLIT_PAIRS: dict[str, str] = {
    "praivet": "private",
    "praivheta": "private",
    "praaivet": "private",
    "injiniyaring": "engineering",
    "injiniyaringa": "engineering",
    "injiniyari": "engineering",
    "kampani": "company",
    "kampanii": "company",
    "intaraneshanal": "international",
    "intaraneshnal": "international",
    "udhyog": "udyog",
    "bhavan": "bhawan",
    "bharat": "bharat",
    "shreshtha": "shrestha",
    "kendra": "kendra",
    "kendr": "kendra",
    "sewa": "seva",
    "sansthan": "sansthan",
    "vidyalaya": "vidyalaya",
    "vidyalay": "vidyalaya",
    "mahavidyalaya": "mahavidyalaya",
    "vishwa": "vishwa",
    "vishva": "vishwa",
    "nagara": "nagar",
    "nagaram": "nagar",
    "rasta": "road",
    "rastaa": "road",
    "marg": "marga",
}


# ---------------------------------------------------------------------------
# TransliterationDictionary class
# ---------------------------------------------------------------------------

class TransliterationDictionary:
    """Learned transliterated-token -> canonical-token dictionary.

    Learned strictly from training matched pairs without external APIs.
    Thread-safe and immutable once built.
    """

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        token_stats: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._mapping: dict[str, str] = dict(mapping or {})
        self._token_stats: dict[str, dict[str, Any]] = dict(token_stats or {})

    def __len__(self) -> int:
        return len(self._mapping)

    def __contains__(self, token: str) -> bool:
        return token.lower() in self._mapping

    def get(self, token: str, default: str | None = None) -> str:
        """Lookup canonical target token. Defaults to input token."""
        clean = token.lower()
        if default is None:
            default = token
        return self._mapping.get(clean, default)

    def to_dict(self) -> dict[str, str]:
        """Return a copy of the mapping dictionary."""
        return dict(self._mapping)

    def remap_text(self, text: str) -> tuple[str, int]:
        """Remap words in `text` using the dictionary.

        Returns
        -------
        remapped_text : str
            Text with known transliterated tokens replaced.
        mapped_count : int
            Number of tokens replaced.
        """
        if not text:
            return "", 0
        toks = text.split()
        mapped_count = 0
        remapped = []
        for t in toks:
            t_low = t.lower()
            if t_low in self._mapping:
                target = self._mapping[t_low]
                if t.isupper():
                    target = target.upper()
                elif t.istitle():
                    target = target.title()
                remapped.append(target)
                mapped_count += 1
            else:
                remapped.append(t)
        return " ".join(remapped), mapped_count

    @classmethod
    def learn_from_pairs(
        cls,
        pairs_df: pd.DataFrame,
        s1_name_col: str = "s1_clean_name",
        cand_name_col: str = "candidate_clean_name",
        label_col: str = "label",
        entity_id_col: str = "source1_entity_id",
        allowed_entity_ids: set[str] | frozenset[str] | None = None,
        min_count: int = 3,
        min_share: float = 0.60,
        include_seeds: bool = True,
    ) -> "TransliterationDictionary":
        """Learn token mappings from matched training pairs.

        Follows Sid's alignment logic:
        For matched pairs with equal token counts, align tokens by position.
        Count token occurrences and co-occurrences.
        Keep mapping if count >= min_count and dominant share >= min_share.

        Parameters
        ----------
        pairs_df : pd.DataFrame
            DataFrame of pairs containing text and label.
        s1_name_col : str
            Column name for S1 clean name.
        cand_name_col : str
            Column name for candidate clean name.
        label_col : str
            Binary match label column (only label==1 is used).
        entity_id_col : str
            S1 entity ID column for leak-free filtering.
        allowed_entity_ids : set-like, optional
            If provided, only pairs belonging to these S1 entities are used.
            Crucial for train/val split hygiene.
        min_count : int, default 3
            Minimum co-occurrence count to accept mapping.
        min_share : float, default 0.60
            Minimum dominant target share (c / total_tok_count).
        include_seeds : bool, default True
            Whether to include high-confidence seed priors.
        """
        df = pairs_df
        if label_col in df.columns:
            df = df[df[label_col] == 1]

        if allowed_entity_ids is not None and entity_id_col in df.columns:
            df = df[df[entity_id_col].isin(allowed_entity_ids)]

        pair_counts: Counter[tuple[str, str]] = Counter()
        tok_counts: Counter[str] = Counter()

        s1_col = s1_name_col if s1_name_col in df.columns else "clean_name"
        cand_col = cand_name_col if cand_name_col in df.columns else "candidate_name"

        if s1_col in df.columns and cand_col in df.columns:
            for s1_n, cand_n in zip(df[s1_col], df[cand_col]):
                if not isinstance(s1_n, str) or not isinstance(cand_n, str):
                    continue
                qt = cand_n.lower().split()
                st = s1_n.lower().split()
                # Positional alignment on equal-length token sequences (Sid's alignment rule)
                if len(qt) == len(st) and len(qt) > 0:
                    for a, b in zip(qt, st):
                        tok_counts[a] += 1
                        pair_counts[(a, b)] += 1

        # Find best candidate target for each source token
        best_targets: dict[str, tuple[str, int]] = defaultdict(lambda: ("", 0))
        for (a, b), c in pair_counts.items():
            if c > best_targets[a][1]:
                best_targets[a] = (b, c)

        mapping: dict[str, str] = {}
        token_stats: dict[str, dict[str, Any]] = {}

        if include_seeds:
            for a, b in _SEED_TRANSLIT_PAIRS.items():
                mapping[a] = b
                token_stats[a] = {"count": 10, "share": 1.0, "is_seed": True}

        # Filter according to frequency and dominance thresholds
        for a, (b, c) in best_targets.items():
            total_a = tok_counts[a]
            share = c / total_a if total_a > 0 else 0.0
            if a != b and c >= min_count and share >= min_share:
                mapping[a] = b
                token_stats[a] = {"count": c, "share": share, "is_seed": False}

        return cls(mapping=mapping, token_stats=token_stats)

    def save(self, file_path: str) -> None:
        """Save dictionary mapping to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mapping": self._mapping,
                    "token_stats": self._token_stats,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, file_path: str) -> "TransliterationDictionary":
        """Load dictionary mapping from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "mapping" in data:
            return cls(mapping=data["mapping"], token_stats=data.get("token_stats"))
        return cls(mapping=data)


# ---------------------------------------------------------------------------
# Default singleton instance
# ---------------------------------------------------------------------------

_DEFAULT_DICT: TransliterationDictionary | None = None


def get_default_transliteration_dict() -> TransliterationDictionary:
    """Return default TransliterationDictionary initialized with seed pairs."""
    global _DEFAULT_DICT
    if _DEFAULT_DICT is None:
        _DEFAULT_DICT = TransliterationDictionary(mapping=_SEED_TRANSLIT_PAIRS)
    return _DEFAULT_DICT


def set_default_transliteration_dict(dictionary: TransliterationDictionary) -> None:
    """Set global default TransliterationDictionary."""
    global _DEFAULT_DICT
    _DEFAULT_DICT = dictionary


# ---------------------------------------------------------------------------
# Feature extraction function
# ---------------------------------------------------------------------------

def add_transliteration_features(
    pair_df: pd.DataFrame,
    dictionary: TransliterationDictionary | None = None,
    translit_dict: TransliterationDictionary | None = None,
) -> pd.DataFrame:
    """Compute 8 Indic transliteration features for candidate pairs.

    Remaps candidate entity names using the transliteration dictionary and
    evaluates similarity against S1 reference strings.

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
    dictionary : TransliterationDictionary, optional
        Dictionary instance. If None, uses :func:`get_default_transliteration_dict`.
    translit_dict : TransliterationDictionary, optional
        Alias for `dictionary`.

    Returns
    -------
    pd.DataFrame
        Copy of `pair_df` with 8 transliteration feature columns appended.
        All values are finite numeric floats or ints in [0.0, 1.0]. Zero NaNs.
    """
    if not isinstance(pair_df, pd.DataFrame):
        raise TypeError(f"add_transliteration_features: expected pd.DataFrame, got {type(pair_df).__name__}")

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
            f"add_transliteration_features: pair_df missing required column(s): {sorted(missing)}. "
            f"Found: {list(pair_df.columns)}"
        )

    out_df = pair_df.copy()

    # Empty table edge case
    if out_df.empty:
        int_cols = {"translit_token_mapped_count", "translit_addr_mapped_count"}
        for col in TRANSLITERATION_FEATURE_COLS:
            dtype = np.int32 if col in int_cols else np.float64
            out_df[col] = pd.Series(dtype=dtype)
        return out_df

    t_dict = dictionary if dictionary is not None else (translit_dict if translit_dict is not None else get_default_transliteration_dict())

    n = len(out_df)
    mapped_counts = np.zeros(n, dtype=np.int32)
    mapped_ratios = np.zeros(n, dtype=np.float64)
    translit_fuzz_ratios = np.zeros(n, dtype=np.float64)
    translit_sort_ratios = np.zeros(n, dtype=np.float64)
    translit_set_ratios = np.zeros(n, dtype=np.float64)
    similarity_gains = np.zeros(n, dtype=np.float64)
    addr_mapped_counts = np.zeros(n, dtype=np.int32)
    translit_addr_fuzz_ratios = np.zeros(n, dtype=np.float64)

    s1_names = [str(x).strip() if pd.notna(x) else "" for x in out_df["s1_clean_name"]]
    cand_names = [str(x).strip() if pd.notna(x) else "" for x in out_df["candidate_clean_name"]]
    s1_addrs = [str(x).strip() if pd.notna(x) else "" for x in out_df["s1_clean_address"]]
    cand_addrs = [str(x).strip() if pd.notna(x) else "" for x in out_df["candidate_clean_address"]]

    # Pre-existing name_fuzz_ratio if already computed in pair_df, else compute on the fly
    has_orig_fuzz = "name_fuzz_ratio" in out_df.columns
    orig_fuzz = (
        out_df["name_fuzz_ratio"].to_numpy(dtype=np.float64)
        if has_orig_fuzz
        else np.zeros(n, dtype=np.float64)
    )

    for i in range(n):
        s1_n = s1_names[i]
        c_n = cand_names[i]
        s1_a = s1_addrs[i]
        c_a = cand_addrs[i]

        # 1. Remap candidate name
        remapped_c_n, n_mapped = t_dict.remap_text(c_n)
        c_toks = c_n.split()
        n_toks = len(c_toks) if c_toks else 1

        mapped_counts[i] = n_mapped
        mapped_ratios[i] = float(n_mapped / n_toks)

        # Transliteration name similarities (case-insensitive)
        s1_n_l = s1_n.lower()
        remap_c_n_l = remapped_c_n.lower()
        fuzz_sim = float(fuzz.ratio(s1_n_l, remap_c_n_l) / 100.0)
        sort_sim = float(fuzz.token_sort_ratio(s1_n_l, remap_c_n_l) / 100.0)
        set_sim = float(fuzz.token_set_ratio(s1_n_l, remap_c_n_l) / 100.0)

        translit_fuzz_ratios[i] = fuzz_sim
        translit_sort_ratios[i] = sort_sim
        translit_set_ratios[i] = set_sim

        # Gain over baseline fuzz ratio
        base_fuzz = orig_fuzz[i] if has_orig_fuzz else float(fuzz.ratio(s1_n_l, c_n.lower()) / 100.0)
        similarity_gains[i] = max(0.0, fuzz_sim - base_fuzz)

        # 2. Remap candidate address
        remapped_c_a, n_addr_mapped = t_dict.remap_text(c_a)
        addr_mapped_counts[i] = n_addr_mapped
        translit_addr_fuzz_ratios[i] = float(fuzz.ratio(s1_a.lower(), remapped_c_a.lower()) / 100.0)

    out_df["translit_token_mapped_count"] = mapped_counts
    out_df["translit_token_mapped_ratio"] = mapped_ratios
    out_df["translit_name_fuzz_ratio"] = translit_fuzz_ratios
    out_df["translit_name_token_sort_ratio"] = translit_sort_ratios
    out_df["translit_name_token_set_ratio"] = translit_set_ratios
    out_df["translit_similarity_gain"] = similarity_gains
    out_df["translit_addr_mapped_count"] = addr_mapped_counts
    out_df["translit_addr_fuzz_ratio"] = translit_addr_fuzz_ratios

    return out_df
