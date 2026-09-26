"""
ingestion.py
------------
Loads pre-cleaned TSV files produced by an upstream cleaning/normalization
process.  Each file must contain exactly the columns:

    entity_id   – unique string identifier for the record
    clean_text  – already-normalized text; loaded as-is without modification

This module is intentionally narrow: it reads and validates data.
It does NOT normalize text, does NOT implement blocking, and does NOT
import sklearn or any ML library.
"""

import csv
import pandas as pd


# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {"entity_id", "clean_text"}


def load_clean_tsv(path: str) -> pd.DataFrame:
    """Load a pre-cleaned TSV file and return a validated DataFrame.

    Parameters
    ----------
    path : str
        File-system path to the TSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at minimum the columns ``entity_id`` and
        ``clean_text``.  All values are preserved as strings exactly as
        they appear in the file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist or is not readable.
    ValueError
        If required columns are absent, if any ``entity_id`` value is
        missing / empty, or if duplicate ``entity_id`` values are found.

    Notes
    -----
    * ``dtype=str``          – every column is read as a plain string.
    * ``quoting=QUOTE_NONE`` – no quote-character interpretation, so
                               quoted fields are preserved verbatim.
    * ``keep_default_na=False`` – empty cells become ``""`` instead of
                                   ``NaN``, preventing silent coercion.
    * The file is expected to use ``\\t`` as the separator.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
    )

    _validate(df, path)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate(df: pd.DataFrame, path: str) -> None:
    """Run all data-quality checks and raise on the first violation."""

    # 1. Required columns present?
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"File '{path}' is missing required column(s): "
            f"{sorted(missing_cols)}.  "
            f"Found columns: {list(df.columns)}"
        )

    # 2. Missing or empty entity_id values?
    empty_id_mask = df["entity_id"].str.strip() == ""
    n_empty_ids = int(empty_id_mask.sum())
    if n_empty_ids > 0:
        raise ValueError(
            f"File '{path}' contains {n_empty_ids} row(s) with a missing "
            f"or empty 'entity_id'."
        )

    # 3. Duplicate entity_id values?
    duplicates = df["entity_id"][df["entity_id"].duplicated(keep=False)]
    if not duplicates.empty:
        dup_list = sorted(duplicates.unique().tolist())
        raise ValueError(
            f"File '{path}' contains duplicate 'entity_id' values: "
            f"{dup_list}"
        )

    # 4. Warn (non-fatal) about empty clean_text values — callers can
    #    decide how to handle them; ingestion itself does not drop rows.
    empty_text_mask = df["clean_text"].str.strip() == ""
    n_empty_text = int(empty_text_mask.sum())
    if n_empty_text > 0:
        import warnings
        warnings.warn(
            f"File '{path}' contains {n_empty_text} row(s) with an empty "
            f"'clean_text'.  Those rows are retained but flagged.",
            UserWarning,
            stacklevel=3,
        )
