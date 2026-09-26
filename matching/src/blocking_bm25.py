"""
blocking_bm25.py
----------------
BM25 candidate retrieval for the Business Entity Resolution pipeline.

Provides a lightweight BM25 index built on top of ``rank_bm25`` (Apache 2.0)
as a **second retrieval channel** alongside the existing Char-TF-IDF blocking
in ``blocking.py``.

Public API
~~~~~~~~~~
    BM25Index                              — index class
        BM25Index.fit(corpus_df)           — build index from S2+S3 DataFrame
        BM25Index.search(query_df, top_k)  — retrieve top-k candidates

    search_candidates_bm25(               — convenience one-shot function
        query_df, corpus_df, top_k=20
    ) -> pd.DataFrame

Design constraints
~~~~~~~~~~~~~~~~~~
* The index is ALWAYS fitted on the S2+S3 searchable corpus.
  S1 (the query source) is NEVER passed to fit().
  Validation S1 or test S1 NEVER enters the corpus.

* Tokenisation: whitespace split on the pre-normalised ``clean_text`` column.
  No stemming, no stopword removal — consistent with the upstream
  normalisation contract.

* Empty query strings yield an empty candidate list for that entity (no crash).

* Duplicate corpus texts are handled safely; IDs are preserved 1-to-1 with
  corpus rows.

* Output DataFrame mirrors ``search_candidates_mock()`` column names so the
  caller can UNION the two result frames without renaming:

      source1_entity_id   – entity_id from the query (S1) DataFrame
      candidate_entity_id – entity_id from the corpus (S2+S3) DataFrame
      bm25_score          – raw BM25Okapi score (float ≥ 0)
      bm25_rank           – 1-based rank within the S1 query (1 = best)

* Output is sorted by (source1_entity_id, bm25_rank) for deterministic order.

* No country hardcoding.  Country filtering is deliberately omitted; the
  existing Char-TF-IDF path handles same_country_filter when desired.

Dependencies
~~~~~~~~~~~~
    rank-bm25==0.2.2   (Apache 2.0, pure Python, no GPU/PyTorch)
    numpy>=1.24.0
    pandas>=2.0.0

License note
~~~~~~~~~~~~
    rank-bm25 is distributed under the Apache License 2.0, which is
    compatible with this project.  Source: https://github.com/dorianbrown/rank_bm25
"""

import warnings
from typing import List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns required in both the corpus and the query DataFrames.
_REQUIRED_COLS = {"entity_id", "clean_text"}

# Sentinel BM25 score used when a query has no tokens (empty text).
_EMPTY_SCORE: float = 0.0

# Default number of candidates to return per query entity.
DEFAULT_BM25_TOP_K: int = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Whitespace-tokenise a pre-normalised text string.

    Returns an empty list for blank / whitespace-only strings so that
    empty queries produce zero candidates rather than crashing.
    """
    if not isinstance(text, str):
        text = "" if (text is None or (isinstance(text, float) and np.isnan(text))) else str(text)
    return text.split()


def _validate_df(df: pd.DataFrame, label: str) -> None:
    """Raise ValueError if *df* lacks the required columns or entity_id uniqueness."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{label} must be a pandas DataFrame, got {type(df).__name__}."
        )
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"BM25Index: {label} is missing required column(s): "
            f"{sorted(missing)}.  Found: {list(df.columns)}"
        )
    if df["entity_id"].str.strip().eq("").any():
        raise ValueError(
            f"BM25Index: {label} contains one or more blank 'entity_id' values."
        )


# ---------------------------------------------------------------------------
# BM25Index — the retrieval index class
# ---------------------------------------------------------------------------

class BM25Index:
    """BM25Okapi index over a searchable corpus (S2+S3).

    Usage
    -----
    ::

        # Build once on S2+S3
        index = BM25Index()
        index.fit(s2s3_df)

        # Query for each batch of S1 entities
        candidates_df = index.search(s1_df, top_k=20)

    The returned DataFrame has columns:
    ``source1_entity_id``, ``candidate_entity_id``, ``bm25_score``, ``bm25_rank``.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._corpus_ids: List[str] = []
        self._n_corpus: int = 0

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, corpus_df: pd.DataFrame) -> "BM25Index":
        """Build the BM25 index from *corpus_df* (S2+S3 records).

        Parameters
        ----------
        corpus_df : pd.DataFrame
            Must contain ``entity_id`` and ``clean_text`` columns.
            S1 records MUST NOT be passed here — doing so would leak
            query-side information into the retrieval IDF statistics.

        Returns
        -------
        BM25Index
            Returns *self* for method chaining.

        Raises
        ------
        TypeError
            If *corpus_df* is not a DataFrame.
        ValueError
            If required columns are absent, entity_ids are blank, or
            the corpus is empty.
        """
        _validate_df(corpus_df, "corpus_df")

        if len(corpus_df) == 0:
            raise ValueError(
                "BM25Index.fit(): corpus_df is empty.  "
                "Cannot build a BM25 index on zero documents."
            )

        self._corpus_ids = corpus_df["entity_id"].tolist()
        self._n_corpus = len(self._corpus_ids)

        # Tokenise each corpus document.
        # Duplicate texts are fine — each row keeps its own entity_id.
        tokenized_corpus = [
            _tokenize(str(text)) for text in corpus_df["clean_text"]
        ]

        # Rows with zero tokens produce a no-op document; BM25Okapi handles
        # empty token lists without errors.
        n_empty = sum(1 for toks in tokenized_corpus if not toks)
        if n_empty > 0:
            warnings.warn(
                f"BM25Index.fit(): {n_empty} corpus document(s) have empty "
                "'clean_text' and will score 0.0 for all queries.",
                UserWarning,
                stacklevel=2,
            )

        self._bm25 = BM25Okapi(tokenized_corpus)
        return self

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query_df: pd.DataFrame,
        top_k: int = DEFAULT_BM25_TOP_K,
    ) -> pd.DataFrame:
        """Retrieve the top-*k* BM25 candidates from the corpus for each query row.

        Parameters
        ----------
        query_df : pd.DataFrame
            S1 query entities.  Must contain ``entity_id`` and ``clean_text``.
            S1 records must NOT have been used to build the index.
        top_k : int, default 20
            Maximum number of candidates per query entity.  If the corpus
            contains fewer than *top_k* documents, all documents are returned.

        Returns
        -------
        pd.DataFrame
            One row per (query entity, candidate) pair with columns:

            * ``source1_entity_id``   – entity_id from *query_df*
            * ``candidate_entity_id`` – entity_id from the corpus
            * ``bm25_score``          – raw BM25Okapi score (float ≥ 0)
            * ``bm25_rank``           – 1-based rank within query (1 = best)

            Sorted by (source1_entity_id, bm25_rank).
            Entities with empty ``clean_text`` yield zero candidate rows.

        Raises
        ------
        RuntimeError
            If the index has not been fitted yet (``fit()`` not called).
        TypeError / ValueError
            For invalid *query_df* or *top_k*.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25Index.search(): the index has not been fitted.  "
                "Call BM25Index.fit(corpus_df) on the S2+S3 corpus first."
            )

        _validate_df(query_df, "query_df")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(
                f"BM25Index.search(): top_k must be an integer >= 1, got {top_k!r}."
            )

        effective_k = min(top_k, self._n_corpus)
        corpus_ids = self._corpus_ids  # local reference for speed

        rows = []
        for s1_id, raw_text in zip(
            query_df["entity_id"], query_df["clean_text"]
        ):
            tokens = _tokenize(str(raw_text))
            if not tokens:
                # Empty query → no candidates; safe no-op
                continue

            scores: np.ndarray = self._bm25.get_scores(tokens)
            # scores has length == n_corpus; dtype float64

            if effective_k >= len(scores):
                # Return all (sorted)
                top_indices = np.argsort(scores)[::-1]
            else:
                # Partial sort: argpartition then sort the partition
                partition_idx = np.argpartition(scores, -effective_k)[-effective_k:]
                top_indices = partition_idx[np.argsort(scores[partition_idx])[::-1]]

            rank = 1
            for idx in top_indices:
                score = float(scores[idx])
                if score <= 0.0:
                    # BM25 scores ≤ 0 carry no signal; stop adding candidates
                    break
                rows.append(
                    {
                        "source1_entity_id":   str(s1_id),
                        "candidate_entity_id": corpus_ids[idx],
                        "bm25_score":          score,
                        "bm25_rank":           rank,
                    }
                )
                rank += 1

        result_df = pd.DataFrame(
            rows,
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "bm25_score",
                "bm25_rank",
            ],
        )

        # Deterministic sort: primary = s1 entity_id (preserves corpus order
        # tie-break), secondary = bm25_rank
        result_df = result_df.sort_values(
            ["source1_entity_id", "bm25_rank"],
            kind="stable",
        ).reset_index(drop=True)

        return result_df

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True once ``fit()`` has been called successfully."""
        return self._bm25 is not None

    @property
    def corpus_size(self) -> int:
        """Number of documents in the fitted corpus."""
        return self._n_corpus


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------

def search_candidates_bm25(
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    top_k: int = DEFAULT_BM25_TOP_K,
) -> pd.DataFrame:
    """Fit a BM25 index on *corpus_df* and return top-*k* candidates for *query_df*.

    This is a convenience wrapper around :class:`BM25Index` for callers that
    do not need to reuse the index across multiple search calls.

    Parameters
    ----------
    query_df : pd.DataFrame
        S1 query entities (``entity_id``, ``clean_text``).
        MUST NOT be the same DataFrame as *corpus_df*.
    corpus_df : pd.DataFrame
        S2+S3 searchable corpus (``entity_id``, ``clean_text``).
        S1 records MUST NOT be included here.
    top_k : int, default 20
        Candidates per query entity.

    Returns
    -------
    pd.DataFrame
        Same schema as :meth:`BM25Index.search`.

    Examples
    --------
    ::

        from src.ingestion import load_clean_tsv
        from src.blocking_bm25 import search_candidates_bm25

        s1_df    = load_clean_tsv("dataset/mock/mock_clean_s1.tsv")
        s2s3_df  = load_clean_tsv("dataset/mock/mock_clean_s2.tsv")

        # Corpus = S2+S3; query = S1.  Never swap these!
        candidates = search_candidates_bm25(s1_df, s2s3_df, top_k=20)
    """
    index = BM25Index()
    index.fit(corpus_df)
    return index.search(query_df, top_k=top_k)
