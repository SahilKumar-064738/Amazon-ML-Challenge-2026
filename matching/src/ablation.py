"""
ablation.py
-----------
Ablation runner and diagnostic framework for the Business Entity Resolution pipeline.

This module provides:

1. MISS CLASSIFIER
   classify_misses(scored_pairs_df, ground_truth_df, threshold)
   → Categorises every missed ground-truth pair into one of 11 failure modes:
     1  candidate_retrieval_miss    -- GT pair never reached the candidate set
     2  wrong_candidate_ranking     -- pair retrieved but scored too low
     3  lgbm_false_negative         -- pair scored above threshold but below decision
     4  lgbm_false_positive         -- non-match pair predicted above threshold
     5  common_name_collision       -- missed because name is too common
     6  address_mismatch            -- addresses are dissimilar; may cause miss
     7  cross_script_transliteration-- different scripts / transliteration variants
     8  ocr_typo                    -- likely OCR error or typo in one record
     9  abbreviation_legal_suffix   -- abbreviated or different legal suffix
    10  empty_missing_address       -- one or both addresses are empty
    11  singleton_no_match_error    -- S1 entity has no GT match but was predicted

2. ABLATION TABLE RUNNER
   run_ablation(experiments, ...) → pd.DataFrame
   Produces a structured ablation table matching the requested format:

   | Experiment | Candidate Recall | Macro F0.5 | Precision | Recall |
   |            | Singleton F0.5   | Runtime    | RAM       |       |

3. EXPERIMENT REGISTRY
   Simple dict-based registry so each experiment is a callable that
   returns (scored_pairs_df, candidate_recall) when given the inputs.

All measurements are ACTUAL (not estimated).  If an experiment cannot run
(e.g. no GPU), it records None for that metric.

Design constraints:
  - Never modifies ground truth
  - Never leaks validation entities into training
  - Classifier uses only the pair table + ground truth (no heuristic tuning)
  - Runtime measured via time.perf_counter(); RAM via tracemalloc
  - Ablation table emitted as both pd.DataFrame and printed Markdown

Public API
~~~~~~~~~~
    MissCategory                  -- string constants for the 11 miss types
    classify_misses(...)          -- classify every failure for one threshold
    run_ablation(...)             -- run all experiments and return table
    print_ablation_table(df)      -- print as Markdown
"""

from __future__ import annotations

import gc
import re
import time
import tracemalloc
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Miss category constants
# ---------------------------------------------------------------------------

class MissCategory:
    """String constants for the 11 miss categories."""
    CANDIDATE_RETRIEVAL_MISS    = "candidate_retrieval_miss"
    WRONG_RANKING               = "wrong_candidate_ranking"
    LGBM_FALSE_NEGATIVE         = "lgbm_false_negative"
    LGBM_FALSE_POSITIVE         = "lgbm_false_positive"
    COMMON_NAME_COLLISION       = "common_name_collision"
    ADDRESS_MISMATCH            = "address_mismatch"
    CROSS_SCRIPT                = "cross_script_transliteration"
    OCR_TYPO                    = "ocr_typo"
    ABBREVIATION_SUFFIX         = "abbreviation_legal_suffix"
    EMPTY_ADDRESS               = "empty_missing_address"
    SINGLETON_ERROR             = "singleton_no_match_error"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_gt(ground_truth_df: pd.DataFrame) -> dict[str, set[str]]:
    """Parse ground-truth DataFrame into s1_id → set of true match IDs."""
    gt: dict[str, set[str]] = {}
    match_col = None
    for col in ("matching_entity_ids", "match_entity_ids"):
        if col in ground_truth_df.columns:
            match_col = col
            break
    if match_col is None:
        raise ValueError(
            "classify_misses(): ground_truth_df must have "
            "'matching_entity_ids' or 'match_entity_ids'."
        )
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        raw = str(row[match_col]).strip()
        if raw and raw.lower() not in ("nan", "none", ""):
            gt[s1_id] = {m.strip() for m in raw.split(",") if m.strip()}
        else:
            gt[s1_id] = set()
    return gt


def _has_non_latin(text: str) -> bool:
    """Return True if *text* contains characters outside the Latin block."""
    return bool(re.search(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]", text))


def _looks_like_typo(a: str, b: str) -> bool:
    """Heuristic: strings look like OCR / typo pairs if edit distance is small
    but they differ in 1-3 characters AND share at least 60% of their length."""
    if not a or not b:
        return False
    la, lb = len(a), len(b)
    if la < 3 or lb < 3:
        return False
    # Quick Hamming-style check for equal-length strings
    if la == lb:
        diffs = sum(1 for x, y in zip(a, b) if x != y)
        return 1 <= diffs <= 3
    # For unequal lengths, compare common prefix length to shorter
    common_prefix = 0
    for x, y in zip(a, b):
        if x == y:
            common_prefix += 1
        else:
            break
    return common_prefix / min(la, lb) >= 0.6


_LEGAL_SUFFIXES = re.compile(
    r"\b(ltd|llc|llp|inc|corp|co|pvt|plc|gmbh|ag|sa|sas|bv|nv|"
    r"limited|incorporated|corporation|company|private|public)\b",
    re.IGNORECASE,
)


def _strip_legal(name: str) -> str:
    """Remove common legal suffixes from a name."""
    return _LEGAL_SUFFIXES.sub("", name).strip()


# ---------------------------------------------------------------------------
# Miss classifier
# ---------------------------------------------------------------------------

def classify_misses(
    scored_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    candidate_pairs_df: Optional[pd.DataFrame],
    threshold: float,
    pair_feature_df: Optional[pd.DataFrame] = None,
    name_freq_threshold: float = 0.3,
) -> dict:
    """Classify every failure (missed GT pair or false positive) into categories.

    Parameters
    ----------
    scored_pairs_df : pd.DataFrame
        Model-scored candidate pairs.  Must have:
        source1_entity_id, candidate_entity_id, prob_match.
    ground_truth_df : pd.DataFrame
        Ground truth.  Must have source1_entity_id and matching_entity_ids.
    candidate_pairs_df : pd.DataFrame or None
        The full candidate set produced by blocking (before model scoring).
        If provided, used to distinguish retrieval misses from model misses.
        Must have source1_entity_id, candidate_entity_ids (comma-separated).
    threshold : float
        Decision threshold used for predictions.
    pair_feature_df : pd.DataFrame or None
        Candidate pair table with s1_clean_name, candidate_clean_name,
        s1_clean_address, candidate_clean_address.  Used for detailed
        miss categorisation.  If None, some categories are approximated.
    name_freq_threshold : float
        name_freq_s1 value below which a name is "common".
        Only used if pair_feature_df contains name_freq_s1.

    Returns
    -------
    dict
        "miss_counts"   : dict[str, int]   -- count per category
        "total_gt_pairs": int
        "total_misses"  : int
        "total_fp"      : int
        "miss_list"     : list[dict]       -- per-pair miss records
        "fp_list"       : list[dict]       -- per-pair FP records
        "category_pct"  : dict[str, float] -- percentage of total misses per category
        "top_categories": list[str]        -- categories sorted by count desc
    """
    gt = _parse_gt(ground_truth_df)

    # Build candidate set (if provided)
    candidate_set: set[tuple[str, str]] = set()
    if candidate_pairs_df is not None and not candidate_pairs_df.empty:
        if "candidate_entity_ids" in candidate_pairs_df.columns:
            for _, row in candidate_pairs_df.iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_cands = str(row["candidate_entity_ids"]).strip()
                if raw_cands and raw_cands.lower() not in ("nan", "none"):
                    for cid in raw_cands.split(","):
                        cid = cid.strip()
                        if cid:
                            candidate_set.add((s1_id, cid))
        elif "candidate_entity_id" in candidate_pairs_df.columns:
            # Long format
            for _, row in candidate_pairs_df.iterrows():
                candidate_set.add((
                    str(row["source1_entity_id"]).strip(),
                    str(row["candidate_entity_id"]).strip(),
                ))

    # Build scored pairs lookup: (s1_id, cand_id) → prob_match
    prob_lookup: dict[tuple[str, str], float] = {}
    for _, row in scored_pairs_df.iterrows():
        key = (str(row["source1_entity_id"]).strip(),
               str(row["candidate_entity_id"]).strip())
        prob_lookup[key] = float(row["prob_match"])

    # Build predictions: s1_id → set of predicted matches
    preds: dict[str, set[str]] = {}
    for (s1_id, cand_id), prob in prob_lookup.items():
        if prob >= threshold:
            preds.setdefault(s1_id, set()).add(cand_id)

    # Build feature lookup if available
    feat_lookup: dict[tuple[str, str], dict] = {}
    if pair_feature_df is not None and not pair_feature_df.empty:
        for _, row in pair_feature_df.iterrows():
            key = (str(row.get("source1_entity_id", "")).strip(),
                   str(row.get("candidate_entity_id", "")).strip())
            feat_lookup[key] = row.to_dict()

    miss_counts: dict[str, int] = {
        cat: 0 for cat in [
            MissCategory.CANDIDATE_RETRIEVAL_MISS,
            MissCategory.WRONG_RANKING,
            MissCategory.LGBM_FALSE_NEGATIVE,
            MissCategory.LGBM_FALSE_POSITIVE,
            MissCategory.COMMON_NAME_COLLISION,
            MissCategory.ADDRESS_MISMATCH,
            MissCategory.CROSS_SCRIPT,
            MissCategory.OCR_TYPO,
            MissCategory.ABBREVIATION_SUFFIX,
            MissCategory.EMPTY_ADDRESS,
            MissCategory.SINGLETON_ERROR,
        ]
    }

    miss_list: list[dict] = []
    fp_list: list[dict] = []
    total_gt_pairs = 0

    # --- False Negatives: GT pairs that were not predicted ---
    for s1_id, true_ids in gt.items():
        if not true_ids:
            continue  # singleton — handled separately below
        total_gt_pairs += len(true_ids)
        pred_ids = preds.get(s1_id, set())

        for cand_id in true_ids:
            if cand_id in pred_ids:
                continue  # correctly predicted — not a miss

            pair = (s1_id, cand_id)
            feat = feat_lookup.get(pair, {})

            # Classify this miss
            category = _classify_fn(
                s1_id=s1_id,
                cand_id=cand_id,
                pair=pair,
                candidate_set=candidate_set,
                prob_lookup=prob_lookup,
                threshold=threshold,
                feat=feat,
                name_freq_threshold=name_freq_threshold,
            )

            miss_counts[category] += 1
            miss_list.append({
                "source1_entity_id":   s1_id,
                "candidate_entity_id": cand_id,
                "category":            category,
                "prob_match":          prob_lookup.get(pair, None),
                "in_candidates":       pair in candidate_set if candidate_set else None,
            })

    # --- False Positives: predicted pairs that are not GT ---
    for s1_id, pred_ids in preds.items():
        true_ids = gt.get(s1_id, set())
        for cand_id in pred_ids:
            if cand_id not in true_ids:
                miss_counts[MissCategory.LGBM_FALSE_POSITIVE] += 1
                fp_list.append({
                    "source1_entity_id":   s1_id,
                    "candidate_entity_id": cand_id,
                    "prob_match":          prob_lookup.get((s1_id, cand_id), None),
                })

    # --- Singleton errors: entities with no GT match that were predicted ---
    for s1_id, true_ids in gt.items():
        if true_ids:
            continue  # not singleton
        pred_ids = preds.get(s1_id, set())
        if pred_ids:
            miss_counts[MissCategory.SINGLETON_ERROR] += len(pred_ids)
            for cand_id in pred_ids:
                fp_list.append({
                    "source1_entity_id":   s1_id,
                    "candidate_entity_id": cand_id,
                    "prob_match":          prob_lookup.get((s1_id, cand_id), None),
                    "is_singleton_fp":     True,
                })

    total_misses = sum(
        v for k, v in miss_counts.items()
        if k not in (MissCategory.LGBM_FALSE_POSITIVE, MissCategory.SINGLETON_ERROR)
    )
    total_fp = miss_counts[MissCategory.LGBM_FALSE_POSITIVE] + miss_counts[MissCategory.SINGLETON_ERROR]

    denom = max(total_misses, 1)
    category_pct = {k: round(v / denom * 100, 1) for k, v in miss_counts.items()}
    top_categories = sorted(miss_counts.keys(), key=lambda k: -miss_counts[k])

    return {
        "miss_counts":    miss_counts,
        "total_gt_pairs": total_gt_pairs,
        "total_misses":   total_misses,
        "total_fp":       total_fp,
        "miss_list":      miss_list,
        "fp_list":        fp_list,
        "category_pct":   category_pct,
        "top_categories": top_categories,
    }


def _classify_fn(
    s1_id: str,
    cand_id: str,
    pair: tuple[str, str],
    candidate_set: set,
    prob_lookup: dict,
    threshold: float,
    feat: dict,
    name_freq_threshold: float,
) -> str:
    """Classify a single false-negative pair into one miss category."""

    # 1. Was the pair never retrieved by blocking at all?
    if candidate_set and pair not in candidate_set:
        return MissCategory.CANDIDATE_RETRIEVAL_MISS

    prob = prob_lookup.get(pair, None)

    # 2. Empty / missing address
    s1_addr = str(feat.get("s1_clean_address", "")).strip()
    cand_addr = str(feat.get("candidate_clean_address", "")).strip()
    if not s1_addr or not cand_addr:
        return MissCategory.EMPTY_ADDRESS

    # 3. Cross-script / transliteration
    s1_name = str(feat.get("s1_clean_name", "")).strip()
    cand_name = str(feat.get("candidate_clean_name", "")).strip()
    if _has_non_latin(s1_name) or _has_non_latin(cand_name):
        return MissCategory.CROSS_SCRIPT

    # 4. OCR / typo
    if s1_name and cand_name and _looks_like_typo(s1_name, cand_name):
        return MissCategory.OCR_TYPO

    # 5. Abbreviation / legal suffix difference
    stripped_s1 = _strip_legal(s1_name)
    stripped_cand = _strip_legal(cand_name)
    if stripped_s1 and stripped_cand and stripped_s1 == stripped_cand:
        return MissCategory.ABBREVIATION_SUFFIX

    # 6. Common name collision (only if freq feature available)
    if "name_freq_s1" in feat:
        nf = float(feat["name_freq_s1"])
        if nf < name_freq_threshold:
            return MissCategory.COMMON_NAME_COLLISION

    # 7. Address mismatch (address similarity features low)
    addr_jac = feat.get("address_jaccard", None)
    if addr_jac is not None and float(addr_jac) < 0.1:
        return MissCategory.ADDRESS_MISMATCH

    # 8. Model false negative (pair was retrieved but scored below threshold)
    if prob is not None:
        if threshold > 0 and prob < threshold:
            return MissCategory.LGBM_FALSE_NEGATIVE
        elif prob >= threshold:
            # Should not be a miss then — but classified as wrong ranking
            return MissCategory.WRONG_RANKING

    # Default: wrong ranking
    return MissCategory.WRONG_RANKING


# ---------------------------------------------------------------------------
# Ablation table row builder
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Result record for one ablation experiment."""
    name: str
    candidate_recall: Optional[float] = None
    macro_f05: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    singleton_f05: Optional[float] = None
    runtime_sec: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    notes: str = ""


def build_ablation_table(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convert a list of ExperimentResult objects to a formatted DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per experiment with the standard ablation column schema.
    """
    rows = []
    for r in results:
        rows.append({
            "Experiment":       r.name,
            "Candidate Recall": _fmt(r.candidate_recall, ".4f"),
            "Macro F0.5":       _fmt(r.macro_f05, ".4f"),
            "Precision":        _fmt(r.precision, ".4f"),
            "Recall":           _fmt(r.recall, ".4f"),
            "Singleton F0.5":   _fmt(r.singleton_f05, ".4f"),
            "Runtime (s)":      _fmt(r.runtime_sec, ".1f"),
            "Peak RAM (MB)":    _fmt(r.peak_ram_mb, ".0f"),
            "Notes":            r.notes,
        })
    return pd.DataFrame(rows)


def _fmt(val: Optional[float], fmt: str) -> str:
    if val is None:
        return "—"
    try:
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)


def print_ablation_table(df: pd.DataFrame) -> None:
    """Print the ablation table as a Markdown-formatted table."""
    if df.empty:
        print("(empty ablation table)")
        return

    cols = list(df.columns)
    # Column widths
    widths = {col: max(len(col), df[col].astype(str).str.len().max()) for col in cols}

    # Header
    header = "| " + " | ".join(col.ljust(widths[col]) for col in cols) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in cols) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        line = "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |"
        print(line)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    experiment_fn: Callable[[], dict],
) -> ExperimentResult:
    """Run a single experiment function and capture timing + RAM.

    Parameters
    ----------
    name : str
        Display name for the experiment row.
    experiment_fn : Callable[[], dict]
        Zero-argument callable that returns a dict with keys:
        candidate_recall, macro_f05, precision, recall, singleton_f05.
        All values may be None if not measured.

    Returns
    -------
    ExperimentResult
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        result_dict = experiment_fn()
    except Exception as exc:
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return ExperimentResult(
            name=name,
            runtime_sec=round(t1 - t0, 2),
            peak_ram_mb=round(peak / 1024 / 1024, 1),
            notes=f"ERROR: {exc}",
        )

    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return ExperimentResult(
        name=name,
        candidate_recall=result_dict.get("candidate_recall"),
        macro_f05=result_dict.get("macro_f05"),
        precision=result_dict.get("precision"),
        recall=result_dict.get("recall"),
        singleton_f05=result_dict.get("singleton_f05"),
        runtime_sec=round(t1 - t0, 2),
        peak_ram_mb=round(peak / 1024 / 1024, 1),
        notes=result_dict.get("notes", ""),
    )


def run_ablation(
    experiments: list[tuple[str, Callable[[], dict]]],
    print_table: bool = True,
) -> pd.DataFrame:
    """Run all experiments in sequence and return an ablation table.

    Parameters
    ----------
    experiments : list of (name, callable)
        Each callable must return a dict with keys:
        candidate_recall, macro_f05, precision, recall, singleton_f05, notes.
    print_table : bool
        If True, print the Markdown table after running.

    Returns
    -------
    pd.DataFrame
        Ablation table with one row per experiment.
    """
    results = []
    for name, fn in experiments:
        print(f"  Running: {name} ...", flush=True)
        result = run_experiment(name, fn)
        results.append(result)
        print(
            f"    → F0.5={result.macro_f05 or '—'}, "
            f"CandRecall={result.candidate_recall or '—'}, "
            f"Runtime={result.runtime_sec}s",
            flush=True,
        )

    df = build_ablation_table(results)

    if print_table:
        print("\n=== ABLATION TABLE ===")
        print_ablation_table(df)

    return df


# ---------------------------------------------------------------------------
# Metrics helper (for use inside experiment_fn callables)
# ---------------------------------------------------------------------------

def compute_metrics_from_scored_pairs(
    scored_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    candidate_pairs_df: Optional[pd.DataFrame] = None,
    beta: float = 0.5,
) -> dict:
    """Compute macro F0.5, precision, recall, singleton F0.5, candidate recall.

    This is a convenience wrapper that orchestrates threshold sweep +
    singleton report in one call.  Used inside experiment_fn callables.

    Parameters
    ----------
    scored_pairs_df : pd.DataFrame
        source1_entity_id, candidate_entity_id, prob_match.
    ground_truth_df : pd.DataFrame
        source1_entity_id, matching_entity_ids.
    candidate_pairs_df : pd.DataFrame or None
        Optional candidate set for candidate recall measurement.
    beta : float
        F-beta parameter.

    Returns
    -------
    dict
        candidate_recall, macro_f05, precision, recall, singleton_f05.
    """
    from src.threshold import sweep_thresholds, singleton_report, apply_threshold

    # Candidate recall
    cand_recall = None
    if candidate_pairs_df is not None and not candidate_pairs_df.empty:
        try:
            from src.blocking import compute_candidate_recall
            cands_by_s1: dict[str, list[str]] = {}
            if "candidate_entity_ids" in candidate_pairs_df.columns:
                for _, row in candidate_pairs_df.iterrows():
                    s1_id = str(row["source1_entity_id"]).strip()
                    raw = str(row["candidate_entity_ids"]).strip()
                    if raw and raw.lower() not in ("nan", "none"):
                        cands_by_s1[s1_id] = [c.strip() for c in raw.split(",") if c.strip()]
            elif "candidate_entity_id" in candidate_pairs_df.columns:
                for s1_id, grp in candidate_pairs_df.groupby("source1_entity_id"):
                    cands_by_s1[str(s1_id)] = grp["candidate_entity_id"].astype(str).tolist()

            # Adapt ground_truth_df to match_entity_ids format for compute_candidate_recall
            gt_for_recall = ground_truth_df.copy()
            if "matching_entity_ids" in gt_for_recall.columns and "match_entity_ids" not in gt_for_recall.columns:
                gt_for_recall = gt_for_recall.rename(columns={"matching_entity_ids": "match_entity_ids"})

            recall_result = compute_candidate_recall(cands_by_s1, gt_for_recall)
            cand_recall = recall_result.get("recall")
        except Exception:
            cand_recall = None

    # Threshold sweep
    sweep_result = sweep_thresholds(scored_pairs_df, ground_truth_df, beta=beta)
    best_threshold = sweep_result["best_threshold"]
    best_f05 = sweep_result["best_macro_f_beta"]

    # Precision / recall at best threshold
    from src.threshold import apply_threshold, _parse_ground_truth_df
    preds = apply_threshold(scored_pairs_df, best_threshold)
    gt_dict = _parse_ground_truth_df(ground_truth_df)

    prec_scores, rec_scores = [], []
    for s1_id, true_ids in gt_dict.items():
        pred_ids = preds.get(str(s1_id), set())
        n_tp = len(pred_ids & set(true_ids))
        n_pred = len(pred_ids)
        n_true = len(true_ids)
        prec_scores.append(n_tp / n_pred if n_pred > 0 else (1.0 if n_true == 0 else 0.0))
        rec_scores.append(n_tp / n_true if n_true > 0 else (1.0 if n_pred == 0 else 0.0))

    macro_prec = float(np.mean(prec_scores)) if prec_scores else None
    macro_rec = float(np.mean(rec_scores)) if rec_scores else None

    # Singleton report
    s_report = singleton_report(preds, ground_truth_df, beta=beta)
    singleton_f05 = s_report["singleton"]["macro_f_beta"]

    return {
        "candidate_recall": cand_recall,
        "macro_f05":        best_f05,
        "precision":        macro_prec,
        "recall":           macro_rec,
        "singleton_f05":    singleton_f05,
        "best_threshold":   best_threshold,
    }
