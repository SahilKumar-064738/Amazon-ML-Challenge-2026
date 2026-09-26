"""
frequency_features.py
---------------------
Frequency-aware matching features for the Business Entity Resolution pipeline.

Corpus statistics are built EXCLUSIVELY from the S2+S3 searchable corpus.
S1 (the query source) is NEVER used to compute corpus statistics.
Validation S1 and test S1 are NEVER included.

This module addresses the "common name collision" failure mode: when many
entities share the same name (e.g. "Reliance Store", "Pizza Hut"), the
standard string similarity features assign high scores to many false
positives.  Rarity-weighted features down-weight matches on common tokens
and up-weight matches on rare, discriminative tokens.

Features produced (7 features):
    1.  name_freq_s1              -- how common is the S1 name in the corpus
                                     (IDF-like score: low = common, high = rare)
    2.  name_freq_cand            -- how common is the candidate name in corpus
    3.  token_freq_max            -- max(token frequency) over shared name tokens
                                     (high = shared tokens are common → less discriminative)
    4.  token_freq_min            -- min(token frequency) over shared name tokens
    5.  rare_token_overlap        -- fraction of shared tokens that are rare
                                     (doc_freq ≤ RARE_TOKEN_THRESHOLD)
    6.  rare_shared_token_count   -- raw count of shared rare tokens
    7.  rarity_weighted_name_sim  -- name token similarity weighted by IDF scores
                                     (rare shared tokens contribute more)
    8.  rarity_weighted_addr_sim  -- address token similarity weighted by IDF

NOTE: name_freq_s1 and name_freq_cand are computed from the CORPUS (S2+S3).
      If an S1 entity's name does not appear in the corpus, it receives
      the maximum rarity score (1.0 — it is a unique name).

All output values:
    - Finite floats in [0.0, 1.0]
    - Never NaN
    - Deterministic

Public API
~~~~~~~~~~
    CorpusStats                           -- class holding corpus statistics
        CorpusStats.build(corpus_df)      -- build from S2+S3 feature DataFrame
        CorpusStats.name_document_freq(name) -> int
        CorpusStats.token_document_freq(token) -> int

    add_frequency_features(pair_df, corpus_stats) -> pd.DataFrame
        Appends the 8 frequency-aware features to a candidate pair table.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Feature columns produced by this module (8 features).
FREQUENCY_FEATURE_COLS: list[str] = [
    "name_freq_s1",
    "name_freq_cand",
    "token_freq_max",
    "token_freq_min",
    "rare_token_overlap",
    "rare_shared_token_count",
    "rarity_weighted_name_sim",
    "rarity_weighted_addr_sim",
]

#: A token is "rare" if it appears in at most this many corpus documents.
RARE_TOKEN_THRESHOLD: int = 5

#: Minimum IDF smoothing to avoid division by zero.
_IDF_SMOOTH_EPSILON: float = 1e-6


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenise(text: Any) -> list[str]:
    """Lower-case whitespace tokenisation; returns empty list for blank input."""
    if text is None or (isinstance(text, float) and text != text):
        return []
    s = str(text).strip().lower()
    if not s:
        return []
    return s.split()


def _normalise(x: float) -> float:
    """Clamp to [0, 1] and assert finite."""
    if x != x or x == float("inf") or x == float("-inf"):
        return 0.0
    return max(0.0, min(1.0, float(x)))


# ---------------------------------------------------------------------------
# CorpusStats — statistics built from S2+S3 only
# ---------------------------------------------------------------------------

class CorpusStats:
    """Holds token and entity name frequency statistics from the S2+S3 corpus.

    Build once from the combined S2+S3 feature DataFrame and reuse across
    all candidate pair feature computations.

    Usage
    -----
    ::

        stats = CorpusStats.build(s2s3_feature_df)
        enriched_df = add_frequency_features(pair_df, stats)

    The feature DataFrame must contain at minimum:
        entity_id, clean_name, clean_address
    """

    def __init__(self) -> None:
        # Number of corpus documents
        self._n_corpus: int = 0

        # name_doc_freq[normalised_name] = number of corpus documents with that name
        self._name_doc_freq: dict[str, int] = {}

        # token_doc_freq[token] = number of corpus documents containing that token
        self._token_doc_freq: dict[str, int] = {}

        # addr_token_doc_freq[token] = number of corpus docs containing address token
        self._addr_token_doc_freq: dict[str, int] = {}

        self._is_built: bool = False

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, corpus_df: pd.DataFrame) -> "CorpusStats":
        """Build corpus statistics from a S2+S3 feature DataFrame.

        Parameters
        ----------
        corpus_df : pd.DataFrame
            Must contain ``entity_id``, ``clean_name``, ``clean_address``.
            S1 records MUST NOT be passed here.

        Returns
        -------
        CorpusStats
            Fitted statistics object.

        Raises
        ------
        ValueError
            If required columns are absent or the DataFrame is empty.
        """
        required = {"entity_id", "clean_name", "clean_address"}
        missing = required - set(corpus_df.columns)
        if missing:
            raise ValueError(
                f"CorpusStats.build(): corpus_df is missing columns: "
                f"{sorted(missing)}. Found: {list(corpus_df.columns)}"
            )
        if corpus_df.empty:
            raise ValueError("CorpusStats.build(): corpus_df is empty.")

        stats = cls()
        stats._n_corpus = len(corpus_df)

        name_doc_freq: dict[str, int] = defaultdict(int)
        token_doc_freq: dict[str, int] = defaultdict(int)
        addr_token_doc_freq: dict[str, int] = defaultdict(int)

        for _, row in corpus_df.iterrows():
            # --- Name ---
            name = str(row["clean_name"]).strip().lower() if not _is_missing(row["clean_name"]) else ""
            if name:
                name_doc_freq[name] += 1
            name_tokens = set(_tokenise(name))
            for tok in name_tokens:
                token_doc_freq[tok] += 1

            # --- Address ---
            addr = str(row["clean_address"]).strip().lower() if not _is_missing(row["clean_address"]) else ""
            addr_tokens = set(_tokenise(addr))
            for tok in addr_tokens:
                addr_token_doc_freq[tok] += 1

        stats._name_doc_freq = dict(name_doc_freq)
        stats._token_doc_freq = dict(token_doc_freq)
        stats._addr_token_doc_freq = dict(addr_token_doc_freq)
        stats._is_built = True
        return stats

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def name_document_freq(self, name: str) -> int:
        """Return the number of corpus documents with this exact name."""
        self._assert_built()
        key = str(name).strip().lower() if not _is_missing(name) else ""
        return self._name_doc_freq.get(key, 0)

    def token_document_freq(self, token: str) -> int:
        """Return number of corpus documents containing this name token."""
        self._assert_built()
        return self._token_doc_freq.get(str(token).lower().strip(), 0)

    def addr_token_document_freq(self, token: str) -> int:
        """Return number of corpus documents containing this address token."""
        self._assert_built()
        return self._addr_token_doc_freq.get(str(token).lower().strip(), 0)

    def n_corpus(self) -> int:
        """Total number of documents in the corpus."""
        self._assert_built()
        return self._n_corpus

    def idf(self, token: str, addr: bool = False) -> float:
        """Smooth IDF score for a name token.

        IDF(t) = log((N + 1) / (df(t) + 1))

        High IDF = rare token = more discriminative.
        The result is NOT normalised to [0,1] here; callers normalise
        via max_idf or by dividing by log(N+1).

        Parameters
        ----------
        token : str
        addr : bool
            If True, use address token document frequencies.

        Returns
        -------
        float  ≥ 0
        """
        self._assert_built()
        N = self._n_corpus
        if addr:
            df = self._addr_token_doc_freq.get(str(token).lower().strip(), 0)
        else:
            df = self._token_doc_freq.get(str(token).lower().strip(), 0)
        return float(np.log((N + 1.0) / (df + 1.0)))

    def max_idf(self, addr: bool = False) -> float:
        """Maximum possible IDF = log((N+1) / 1) for a hapax legomenon."""
        self._assert_built()
        return float(np.log(self._n_corpus + 1.0))

    def _assert_built(self) -> None:
        if not self._is_built:
            raise RuntimeError(
                "CorpusStats must be built via CorpusStats.build(corpus_df) "
                "before calling any lookup method."
            )


def _is_missing(val: Any) -> bool:
    """Return True if value is None or float NaN."""
    if val is None:
        return True
    if isinstance(val, float) and val != val:
        return True
    return False


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _name_rarity_score(name: str, stats: CorpusStats) -> float:
    """Normalised rarity of an entity name in the corpus.

    Returns a value in [0, 1]:
    - 1.0 = name never seen in corpus (maximally rare / unique)
    - 0.0 = name appears in every corpus document (extremely common)

    Formula:
        rarity = 1 - (doc_freq(name) / n_corpus)
    """
    n = stats.n_corpus()
    df = stats.name_document_freq(name)
    if n == 0:
        return 1.0
    return _normalise(1.0 - df / n)


def _shared_token_features(
    toks_a: list[str],
    toks_b: list[str],
    stats: CorpusStats,
    addr: bool = False,
) -> dict[str, float]:
    """Compute token-frequency features over shared tokens between two token lists."""
    set_a = set(toks_a)
    set_b = set(toks_b)
    shared = set_a & set_b

    max_idf = stats.max_idf(addr=addr)
    if max_idf < _IDF_SMOOTH_EPSILON:
        max_idf = 1.0

    if not shared:
        return {
            "token_freq_max":          0.0,
            "token_freq_min":          0.0,
            "rare_token_overlap":      0.0,
            "rare_shared_token_count": 0.0,
            "rarity_weighted_sim":     0.0,
        }

    # Token frequencies of shared tokens
    shared_freqs = [stats.token_document_freq(tok) if not addr
                    else stats.addr_token_document_freq(tok)
                    for tok in shared]

    n = stats.n_corpus()

    # Max / min frequency normalised to [0,1] by corpus size
    freq_max = max(shared_freqs) / n if n > 0 else 0.0
    freq_min = min(shared_freqs) / n if n > 0 else 0.0

    # Rare token overlap: fraction of shared tokens that are "rare"
    n_rare = sum(1 for f in shared_freqs if f <= RARE_TOKEN_THRESHOLD)
    n_shared = len(shared)
    rare_overlap = n_rare / n_shared if n_shared > 0 else 0.0

    # Rare shared token count normalised by union size
    union = set_a | set_b
    rare_count_norm = n_rare / len(union) if union else 0.0

    # Rarity-weighted similarity (IDF-weighted Jaccard over shared tokens)
    # w_sim = sum(IDF(t) for t in shared) / sum(IDF(t) for t in union)
    shared_idf_sum = sum(stats.idf(tok, addr=addr) for tok in shared)
    union_idf_sum = sum(stats.idf(tok, addr=addr) for tok in union)
    rarity_sim = shared_idf_sum / union_idf_sum if union_idf_sum > _IDF_SMOOTH_EPSILON else 0.0

    return {
        "token_freq_max":          _normalise(freq_max),
        "token_freq_min":          _normalise(freq_min),
        "rare_token_overlap":      _normalise(rare_overlap),
        "rare_shared_token_count": _normalise(rare_count_norm),
        "rarity_weighted_sim":     _normalise(rarity_sim),
    }


# ---------------------------------------------------------------------------
# Public API — DataFrame-level feature adder
# ---------------------------------------------------------------------------

def add_frequency_features(
    pair_df: pd.DataFrame,
    corpus_stats: CorpusStats,
) -> pd.DataFrame:
    """Compute 8 frequency-aware features for every candidate pair.

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
    corpus_stats : CorpusStats
        Pre-built corpus statistics (from S2+S3 only).

    Returns
    -------
    pd.DataFrame
        Copy of *pair_df* with 8 new columns appended:
        :data:`FREQUENCY_FEATURE_COLS`.

    Raises
    ------
    ValueError
        If required columns are missing.
    RuntimeError
        If *corpus_stats* has not been built.
    """
    required = {
        "source1_entity_id", "candidate_entity_id",
        "s1_clean_name", "candidate_clean_name",
        "s1_clean_address", "candidate_clean_address",
    }
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(
            f"add_frequency_features(): pair_df is missing column(s): "
            f"{sorted(missing)}.  Found: {list(pair_df.columns)}"
        )

    # Trigger built check early
    _ = corpus_stats.n_corpus()

    out_df = pair_df.copy()

    if out_df.empty:
        for col in FREQUENCY_FEATURE_COLS:
            out_df[col] = pd.Series(dtype=float)
        return out_df

    # Pre-compute name rarity for all rows
    name_freq_s1_vals = []
    name_freq_cand_vals = []
    token_freq_max_vals = []
    token_freq_min_vals = []
    rare_token_overlap_vals = []
    rare_shared_count_vals = []
    rarity_weighted_name_vals = []
    rarity_weighted_addr_vals = []

    for _, row in out_df.iterrows():
        s1_name = "" if _is_missing(row["s1_clean_name"]) else str(row["s1_clean_name"])
        cand_name = "" if _is_missing(row["candidate_clean_name"]) else str(row["candidate_clean_name"])
        s1_addr = "" if _is_missing(row["s1_clean_address"]) else str(row["s1_clean_address"])
        cand_addr = "" if _is_missing(row["candidate_clean_address"]) else str(row["candidate_clean_address"])

        # 1 & 2: Name rarity scores
        name_freq_s1_vals.append(_name_rarity_score(s1_name, corpus_stats))
        name_freq_cand_vals.append(_name_rarity_score(cand_name, corpus_stats))

        # 3-7: Token-level frequency features (name)
        toks_s1 = _tokenise(s1_name)
        toks_cand = _tokenise(cand_name)
        name_feats = _shared_token_features(toks_s1, toks_cand, corpus_stats, addr=False)

        token_freq_max_vals.append(name_feats["token_freq_max"])
        token_freq_min_vals.append(name_feats["token_freq_min"])
        rare_token_overlap_vals.append(name_feats["rare_token_overlap"])
        rare_shared_count_vals.append(name_feats["rare_shared_token_count"])
        rarity_weighted_name_vals.append(name_feats["rarity_weighted_sim"])

        # 8: Rarity-weighted address similarity
        toks_s1_addr = _tokenise(s1_addr)
        toks_cand_addr = _tokenise(cand_addr)
        addr_feats = _shared_token_features(toks_s1_addr, toks_cand_addr, corpus_stats, addr=True)
        rarity_weighted_addr_vals.append(addr_feats["rarity_weighted_sim"])

    out_df["name_freq_s1"] = name_freq_s1_vals
    out_df["name_freq_cand"] = name_freq_cand_vals
    out_df["token_freq_max"] = token_freq_max_vals
    out_df["token_freq_min"] = token_freq_min_vals
    out_df["rare_token_overlap"] = rare_token_overlap_vals
    out_df["rare_shared_token_count"] = rare_shared_count_vals
    out_df["rarity_weighted_name_sim"] = rarity_weighted_name_vals
    out_df["rarity_weighted_addr_sim"] = rarity_weighted_addr_vals

    return out_df
