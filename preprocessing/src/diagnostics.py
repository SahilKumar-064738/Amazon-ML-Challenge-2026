"""Memory-safe diagnostic reports.

Diagnostics ONLY — nothing here deletes, merges, or deduplicates records.
"Same canonical name, different entity_id" is expected and meaningful in
entity resolution (see implementation.md Section 9); these functions exist
to report on it, never to act on it.

Bounded-memory design (Section 11 / 26.4): group sizes are computed with
`groupby(...).size()` (an int count per group, not a Python list of every
member), and only the sample IDs for the largest MAX_COLLISION_GROUPS_REPORTED
groups are ever materialized as lists, each capped at
MAX_SAMPLE_IDS_PER_GROUP entries.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import MAX_COLLISION_GROUPS_REPORTED, MAX_SAMPLE_IDS_PER_GROUP


def _collision_summary(df: pd.DataFrame, group_cols: List[str], id_col: str = "entity_id") -> Dict:
    """Summarize collision groups on `group_cols` without materializing a
    full id-list per group for the whole dataset.

    Returns a bounded dict: group_count, rows_involved, max_group_size, and
    a capped list of the largest groups with a small sample of entity_ids
    each (fetched only for those top groups, not for every group).
    """
    non_empty = df[(df[group_cols] != "").all(axis=1)] if group_cols else df
    sizes = non_empty.groupby(group_cols, sort=False).size()
    collision_sizes = sizes[sizes > 1]

    if len(collision_sizes) == 0:
        return {
            "group_count": 0,
            "rows_involved": 0,
            "max_group_size": 0,
            "top_groups": [],
        }

    top = collision_sizes.sort_values(ascending=False).head(MAX_COLLISION_GROUPS_REPORTED)

    top_groups = []
    for key, size in top.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=non_empty.index)
        for col, val in zip(group_cols, key_tuple):
            mask &= non_empty[col] == val
        sample_ids = non_empty.loc[mask, id_col].head(MAX_SAMPLE_IDS_PER_GROUP).tolist()
        top_groups.append({
            "group_key": dict(zip(group_cols, key_tuple)),
            "group_size": int(size),
            "sample_entity_ids": sample_ids,
        })

    return {
        "group_count": int(len(collision_sizes)),
        "rows_involved": int(collision_sizes.sum()),
        "max_group_size": int(collision_sizes.max()),
        "top_groups": top_groups,
    }


def name_canonical_collisions(df: pd.DataFrame) -> Dict:
    """Collisions on business_name_canonical alone (Section 3 / 26)."""
    return _collision_summary(df, ["business_name_canonical"])


def name_country_collisions(df: pd.DataFrame) -> Dict:
    """Collisions on business_name_canonical + country_normalized."""
    return _collision_summary(df, ["business_name_canonical", "country_normalized"])


def name_address_country_collisions(df: pd.DataFrame) -> Dict:
    """Collisions on business_name_canonical + business_address_canonical + country_normalized.

    This is the tightest grouping and typically has far fewer/smaller
    groups than name-only collisions.
    """
    return _collision_summary(
        df, ["business_name_canonical", "business_address_canonical", "country_normalized"]
    )


def collision_groups_to_frame(collision_result: Dict) -> pd.DataFrame:
    """Flatten a _collision_summary() result's top_groups into a small,
    bounded DataFrame suitable for writing to a diagnostic TSV file."""
    rows = []
    for group in collision_result.get("top_groups", []):
        row = dict(group["group_key"])
        row["group_size"] = group["group_size"]
        row["sample_entity_ids"] = ",".join(group["sample_entity_ids"])
        rows.append(row)
    if not rows:
        # Still return a DataFrame with the right shape (no rows).
        return pd.DataFrame(columns=["group_size", "sample_entity_ids"])
    return pd.DataFrame(rows)


def exact_duplicate_rows(df: pd.DataFrame, columns: List[str]) -> Dict:
    """Count exact duplicate rows (all `columns` byte-identical).

    Reported diagnostically only — never removed (Section 9). None were
    found in the analyzed dataset, but production batches could differ.
    """
    dup_mask = df.duplicated(subset=columns, keep=False)
    dup_count = int(dup_mask.sum())
    groups_count = 0
    if dup_count > 0:
        groups_count = int(df.loc[dup_mask, columns].drop_duplicates().shape[0])
    return {
        "exact_duplicate_row_count": dup_count,
        "exact_duplicate_group_count": groups_count,
    }
