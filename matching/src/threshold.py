"""
threshold.py
------------
Phase 4 -- Threshold selection utilities.

Phase 4.1: entity-level F-beta metric              (entity_f_beta).
Phase 4.2: threshold application infrastructure     (apply_threshold).
Phase 4.3: macro F-beta + threshold sweep           (macro_f_beta,
                                                      sweep_thresholds).

No optimal threshold is *selected* here; that belongs to Phase 4.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default beta value for the F-beta score (precision-weighted).
DEFAULT_BETA: float = 0.5


# ---------------------------------------------------------------------------
# Phase 4.1 -- Entity-level F-beta metric
# ---------------------------------------------------------------------------

def entity_f_beta(
    predicted_ids: "set[str] | frozenset[str] | list[str]",
    true_ids: "set[str] | frozenset[str] | list[str]",
    beta: float = DEFAULT_BETA,
) -> float:
    """Compute the entity-level F-beta score for a single query entity.

    Given the set of S2 entity IDs that a model predicted as matches
    (*predicted_ids*) and the ground-truth set of matching S2 entity IDs
    (*true_ids*), this function returns the F-beta score evaluated at the
    **entity level** (i.e. treating each ID as a discrete element, not a
    ranked list).

    The formula is the standard weighted harmonic mean of precision and
    recall::

        precision = |predicted ∩ true| / |predicted|   (if |predicted| > 0)
        recall    = |predicted ∩ true| / |true|         (if |true|    > 0)

        F_beta = (1 + beta^2) * precision * recall
                 ----------------------------------
                  beta^2 * precision  +  recall

    When beta < 1, precision is weighted more heavily than recall.
    The canonical entity-resolution choice is beta = 0.5.

    Parameters
    ----------
    predicted_ids : set-like or list of str
        S2 entity IDs predicted as matches by the model for one S1 entity.
        Duplicate values are collapsed; order is ignored.
    true_ids : set-like or list of str
        Ground-truth matching S2 entity IDs for the same S1 entity.
        Duplicate values are collapsed; order is ignored.
    beta : float, optional
        The beta parameter that controls the precision/recall trade-off.
        Must be strictly positive (``beta > 0``).
        Default is ``0.5`` (precision-weighted).

    Returns
    -------
    float
        F-beta score in the closed interval ``[0.0, 1.0]``.

        Special cases:

        * If both *predicted_ids* and *true_ids* are empty -> ``1.0``
          (a model that predicts nothing for an entity with no true matches
          is considered perfect).
        * If *predicted_ids* is empty and *true_ids* is non-empty -> ``0.0``
          (recall is zero; no matches were predicted).
        * If *predicted_ids* is non-empty and *true_ids* is empty -> ``0.0``
          (precision is zero; all predictions are false positives).
        * If precision = 0 and recall = 0 -> ``0.0`` (harmonic mean is 0).

    Raises
    ------
    TypeError
        * If *predicted_ids* is a bare string (not a collection of strings).
        * If *true_ids* is a bare string (not a collection of strings).
        * If *predicted_ids* or *true_ids* contains non-string elements.
    ValueError
        * If *beta* is not strictly positive (``beta <= 0``).

    Notes
    -----
    * Duplicate IDs within either input are collapsed via ``frozenset``
      conversion before any computation.
    * The function is commutative in precision/recall only when beta = 1.
      For beta != 1, swapping inputs changes the score.
    * No model is accessed; no threshold is applied; this function is
      purely a metric computation.

    Examples
    --------
    Perfect match (all and only true IDs predicted):

    >>> entity_f_beta({"A", "B"}, {"A", "B"})
    1.0

    No predictions for a non-empty ground truth (recall = 0):

    >>> entity_f_beta(set(), {"A", "B"})
    0.0

    All-false-positive predictions for an empty ground truth:

    >>> entity_f_beta({"A"}, set())
    0.0

    Both empty (trivially correct):

    >>> entity_f_beta(set(), set())
    1.0

    Partial match -- 1 of 2 true IDs predicted, no false positives:

    >>> round(entity_f_beta({"A"}, {"A", "B"}, beta=0.5), 6)
    0.833333
    """

    # ------------------------------------------------------------------
    # 1. Validate beta
    # ------------------------------------------------------------------
    if beta <= 0.0:
        raise ValueError(
            f"beta must be strictly positive (beta > 0), got {beta!r}."
        )

    # ------------------------------------------------------------------
    # 2. Validate and normalise predicted_ids
    # ------------------------------------------------------------------
    if isinstance(predicted_ids, (str, bytes)):
        raise TypeError(
            "predicted_ids must be a collection of string IDs, "
            f"not a bare string: {predicted_ids!r}"
        )

    try:
        pred_set: frozenset[str] = frozenset(predicted_ids)
    except TypeError as exc:
        raise TypeError(
            "predicted_ids must be an iterable of strings, "
            f"got {type(predicted_ids).__name__}: {exc}"
        ) from exc

    if pred_set and not all(isinstance(eid, str) for eid in pred_set):
        bad = [eid for eid in pred_set if not isinstance(eid, str)]
        raise TypeError(
            "predicted_ids must contain only str values; "
            f"found non-string element(s): {bad[:5]!r}"
        )

    # ------------------------------------------------------------------
    # 3. Validate and normalise true_ids
    # ------------------------------------------------------------------
    if isinstance(true_ids, (str, bytes)):
        raise TypeError(
            "true_ids must be a collection of string IDs, "
            f"not a bare string: {true_ids!r}"
        )

    try:
        true_set: frozenset[str] = frozenset(true_ids)
    except TypeError as exc:
        raise TypeError(
            "true_ids must be an iterable of strings, "
            f"got {type(true_ids).__name__}: {exc}"
        ) from exc

    if true_set and not all(isinstance(eid, str) for eid in true_set):
        bad = [eid for eid in true_set if not isinstance(eid, str)]
        raise TypeError(
            "true_ids must contain only str values; "
            f"found non-string element(s): {bad[:5]!r}"
        )

    # ------------------------------------------------------------------
    # 4. Special cases
    # ------------------------------------------------------------------
    # Both empty -> trivially perfect (model correctly predicts no matches)
    if not pred_set and not true_set:
        return 1.0

    # One side empty -> score is 0 (either recall=0 or precision=0)
    if not pred_set or not true_set:
        return 0.0

    # ------------------------------------------------------------------
    # 5. Core metric computation
    # ------------------------------------------------------------------
    n_pred = len(pred_set)
    n_true = len(true_set)
    n_tp = len(pred_set & true_set)

    # Both non-empty but no intersection -> precision=0, recall=0 -> F=0
    if n_tp == 0:
        return 0.0

    precision: float = n_tp / n_pred
    recall: float = n_tp / n_true

    beta_sq: float = beta * beta
    f_beta: float = (
        (1.0 + beta_sq) * precision * recall
        / (beta_sq * precision + recall)
    )

    return f_beta


# ---------------------------------------------------------------------------
# Phase 4.2 -- Threshold application infrastructure
# ---------------------------------------------------------------------------

#: Column in the scored pair table that holds the S1 entity identifier.
#: Must match src.model.PAIR_S1_ENTITY_COL.
_S1_COL: str = "source1_entity_id"

#: Column in the scored pair table that holds the raw match probability.
#: Must match src.model.PROB_MATCH_COL.
_PROB_COL: str = "prob_match"

#: Column in the scored pair table that holds the candidate S2 entity ID.
_CAND_COL: str = "candidate_entity_id"

#: Default decision threshold (inclusive lower bound on prob_match).
DEFAULT_THRESHOLD: float = 0.5


def apply_threshold(
    scored_pairs_df: pd.DataFrame,
    threshold: float,
    s1_col: str = _S1_COL,
    prob_col: str = _PROB_COL,
    cand_col: str = _CAND_COL,
    margin_threshold: float = 0.2,
    relaxed_threshold: float = 0.35,
) -> dict[str, set[str]]:
    """Convert a scored pair table into per-S1-entity sets of predicted matches.

    Applies probability thresholds and top1-top2 margin logic optimized for Macro F0.5.
    If the margin between the best candidate and the second best candidate is >= margin_threshold,
    we relax the probability threshold for the top candidate to `relaxed_threshold`.
    Otherwise, we use the base `threshold`. We do not enforce 1-to-1 matching.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(scored_pairs_df, pd.DataFrame):
        raise TypeError(
            f"apply_threshold() expects a pd.DataFrame as scored_pairs_df, "
            f"got {type(scored_pairs_df).__name__}."
        )

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError(
            f"threshold must be a numeric float, got {type(threshold).__name__}: {threshold!r}"
        )

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"threshold must be in [0.0, 1.0], got {threshold}."
        )

    for col, name in ((s1_col, "s1_col"), (cand_col, "cand_col"), (prob_col, "prob_col")):
        if col not in scored_pairs_df.columns:
            raise ValueError(
                f"apply_threshold(): required column '{col}' not found in scored_pairs_df. "
                f"Available columns: {list(scored_pairs_df.columns)}"
            )

    if scored_pairs_df.empty:
        return {}

    prob_series = scored_pairs_df[prob_col]
    if prob_series.isnull().any():
        raise ValueError(
            f"apply_threshold(): column '{prob_col}' contains NaN values. "
            "All probability scores must be finite."
        )

    import numpy as _np
    prob_vals = prob_series.to_numpy(dtype=float)
    if not _np.isfinite(prob_vals).all():
        raise ValueError(
            f"apply_threshold(): column '{prob_col}' contains inf values. "
            "All probability scores must be finite."
        )

    if (prob_vals < 0.0).any() or (prob_vals > 1.0).any():
        bad_min = float(prob_vals.min())
        bad_max = float(prob_vals.max())
        raise ValueError(
            f"apply_threshold(): prob_match values must be in [0.0, 1.0], "
            f"got range [{bad_min}, {bad_max}]."
        )

    predictions: dict[str, set[str]] = {
        str(s1_id): set() for s1_id in scored_pairs_df[s1_col].unique()
    }
    
    # Sort to compute top1-top2 margins efficiently
    df = scored_pairs_df.sort_values([s1_col, prob_col], ascending=[True, False]).copy()
    
    # Compute rank per S1 entity
    df['rank'] = df.groupby(s1_col).cumcount()
    
    # Extract top 1 and top 2 probabilities
    top1 = df[df['rank'] == 0].set_index(s1_col)[prob_col]
    top2 = df[df['rank'] == 1].set_index(s1_col)[prob_col]
    
    # Compute margin, filling 0 if no top 2
    margin = (top1 - top2.reindex(top1.index, fill_value=0.0))
    
    df['margin'] = df[s1_col].map(margin)
    
    # Apply logic
    # Condition 1: Base threshold
    cond1 = df[prob_col] >= threshold
    
    # Condition 2: Relaxed threshold for top1 if margin is high
    cond2 = (df['rank'] == 0) & (df['margin'] >= margin_threshold) & (df[prob_col] >= relaxed_threshold)
    
    matched_rows = df[cond1 | cond2]
    
    for s1_id, cand_id in zip(matched_rows[s1_col].to_numpy(), matched_rows[cand_col].to_numpy()):
        predictions[str(s1_id)].add(str(cand_id))
        
    return predictions


# ---------------------------------------------------------------------------
# Phase 4.3 -- Macro F-beta + threshold sweep
# ---------------------------------------------------------------------------

#: Both column-name variants accepted by _parse_ground_truth_df for ground truth.
_GT_COL_PRIMARY: str = "matching_entity_ids"   # canonical name (Phase 2 convention)
_GT_COL_ALT: str = "match_entity_ids"          # legacy alias (backward compat only)
_GT_S1_COL: str = "source1_entity_id"

#: Default threshold grid used by sweep_thresholds when none is supplied.
DEFAULT_THRESHOLDS: tuple[float, ...] = tuple(
    round(v / 100, 2) for v in range(1, 100)   # 0.01, 0.02, …, 0.99
)


def macro_f_beta(
    predictions: "dict[str, set[str]]",
    ground_truth: "dict[str, set[str]]",
    beta: float = DEFAULT_BETA,
) -> float:
    """Compute the macro-averaged entity-level F-beta score.

    Evaluates :func:`entity_f_beta` independently for every S1 entity that
    appears in *ground_truth*, then returns the unweighted arithmetic mean
    of those per-entity scores (macro average).

    Only S1 entities present in *ground_truth* are scored.  S1 entities
    that appear in *predictions* but not in *ground_truth* are ignored
    (they cannot be evaluated without a reference).

    Parameters
    ----------
    predictions : dict[str, set[str]]
        Output of :func:`apply_threshold`.  Maps each S1 entity ID to the
        set of predicted matching candidate IDs.  S1 entities not present
        in this dict are treated as predicting an empty set for that entity.
    ground_truth : dict[str, set[str]]
        Maps each S1 entity ID to the **frozenset or set** of true matching
        S2/S3 candidate IDs.  Every key in *ground_truth* is evaluated;
        entities absent from *predictions* score 0.0 (empty prediction).
    beta : float, optional
        Beta parameter forwarded to :func:`entity_f_beta`.
        Must be strictly positive.  Default: ``0.5``.

    Returns
    -------
    float
        Macro-averaged F-beta score in ``[0.0, 1.0]``.

        * If *ground_truth* is empty → ``0.0`` (no entities to evaluate).
        * Otherwise the arithmetic mean of per-entity :func:`entity_f_beta`
          scores across all S1 entities in *ground_truth*.

    Raises
    ------
    TypeError
        * If *predictions* or *ground_truth* is not a ``dict``.
    ValueError
        * If *beta* is not strictly positive.

    Notes
    -----
    * The macro average treats every S1 entity equally regardless of how
      many true matches it has.  This matches the competition scoring
      convention where each query entity contributes one data point.
    * Ground-truth entries with an empty true-match set (zero-match entities)
      are still scored via ``entity_f_beta`` (which returns ``1.0`` when
      both sides are empty, and ``0.0`` when the prediction is non-empty).
    * No threshold is applied here; *predictions* must already be binary
      (sets of predicted IDs from :func:`apply_threshold`).
    """
    # ------------------------------------------------------------------
    # 1. Type validation
    # ------------------------------------------------------------------
    if not isinstance(predictions, dict):
        raise TypeError(
            f"macro_f_beta(): predictions must be a dict, "
            f"got {type(predictions).__name__}."
        )
    if not isinstance(ground_truth, dict):
        raise TypeError(
            f"macro_f_beta(): ground_truth must be a dict, "
            f"got {type(ground_truth).__name__}."
        )
    # beta is validated inside entity_f_beta

    # ------------------------------------------------------------------
    # 2. Edge case: no ground truth entities to evaluate
    # ------------------------------------------------------------------
    if not ground_truth:
        return 0.0

    # ------------------------------------------------------------------
    # 3. Per-entity scoring
    # ------------------------------------------------------------------
    scores: list[float] = []
    for s1_id, true_ids in ground_truth.items():
        pred_ids: set[str] = predictions.get(str(s1_id), set())
        score = entity_f_beta(pred_ids, true_ids, beta=beta)
        scores.append(score)

    return float(np.mean(scores))


def _parse_ground_truth_df(
    ground_truth_df: pd.DataFrame,
) -> dict[str, frozenset[str]]:
    """Internal helper: parse a ground-truth DataFrame into a lookup dict.

    Accepts both column-name conventions used across the project:

    * ``matching_entity_ids`` (canonical — used by ``features.load_ground_truth``
      and ``blocking.compute_candidate_recall``)
    * ``match_entity_ids``    (legacy alias — accepted for backward compatibility)

    The match-ID column may contain a comma-separated list of IDs (e.g.
    ``"S2-001,S2-002"``), a single ID, or an empty string (zero-match entity).

    Parameters
    ----------
    ground_truth_df : pd.DataFrame
        Must contain ``source1_entity_id`` and one of the two match-ID
        columns listed above.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping of S1 entity ID → frozenset of true matching IDs.
        Zero-match entities map to an empty frozenset.

    Raises
    ------
    ValueError
        If the DataFrame is missing ``source1_entity_id`` or both accepted
        match-ID column names.
    """
    if _GT_S1_COL not in ground_truth_df.columns:
        raise ValueError(
            f"ground_truth_df must contain a '{_GT_S1_COL}' column. "
            f"Found: {list(ground_truth_df.columns)}"
        )

    # Detect which match-ID column is present
    if _GT_COL_PRIMARY in ground_truth_df.columns:
        match_col = _GT_COL_PRIMARY
    elif _GT_COL_ALT in ground_truth_df.columns:
        match_col = _GT_COL_ALT
    else:
        raise ValueError(
            f"ground_truth_df must contain '{_GT_COL_PRIMARY}' or "
            f"'{_GT_COL_ALT}'. Found: {list(ground_truth_df.columns)}"
        )

    gt_lookup: dict[str, frozenset[str]] = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = str(row[_GT_S1_COL]).strip()
        raw = str(row[match_col]).strip()

        if not raw or raw.lower() == "nan":
            gt_lookup[s1_id] = frozenset()
        else:
            gt_lookup[s1_id] = frozenset(
                mid.strip() for mid in raw.split(",") if mid.strip()
            )

    return gt_lookup


def sweep_thresholds(
    scored_pairs_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    thresholds: "list[float] | tuple[float, ...] | None" = None,
    beta: float = DEFAULT_BETA,
) -> dict:
    """Evaluate a grid of probability thresholds and return per-threshold metrics.

    For each candidate threshold in *thresholds*, this function:

    1. Applies the threshold to *scored_pairs_df* via :func:`apply_threshold`
       to produce per-S1-entity prediction sets.
    2. Computes the macro-averaged entity-level F-beta score across all S1
       entities present in *ground_truth_df* via :func:`macro_f_beta`.
    3. Records per-threshold precision, recall, and F-beta.

    The function does **not** select a threshold.  It returns a structured
    result dict containing the full sweep table so that Phase 4.4 can make
    the selection decision.

    Parameters
    ----------
    scored_pairs_df : pd.DataFrame
        Scored candidate pair table from Phase 3.4/3.5.  Must contain:

        * ``source1_entity_id`` -- S1 entity ID.
        * ``candidate_entity_id`` -- S2/S3 candidate entity ID.
        * ``prob_match`` -- match probability in ``[0, 1]``.

    ground_truth_df : pd.DataFrame
        Ground-truth table.  Must contain ``source1_entity_id`` and one of:

        * ``matching_entity_ids`` (primary; comma-separated S2/S3 IDs), or
        * ``match_entity_ids`` (alternate accepted format).

    thresholds : list or tuple of float, optional
        Candidate thresholds to evaluate.  Each value must be in ``[0, 1]``.
        If ``None`` (default), uses :data:`DEFAULT_THRESHOLDS`
        (0.01, 0.02, …, 0.99 — 99 evenly spaced values).
    beta : float, optional
        Beta for the F-beta metric.  Default: ``0.5`` (precision-weighted).

    Returns
    -------
    dict with the following keys:

    ``"rows"`` : list[dict]
        One dict per evaluated threshold, sorted in ascending threshold order.
        Each row dict has keys:

        * ``"threshold"``  (float) -- the threshold value evaluated.
        * ``"macro_f_beta"`` (float) -- macro-averaged F-beta across GT entities.
        * ``"macro_precision"`` (float) -- macro-averaged precision.
        * ``"macro_recall"`` (float) -- macro-averaged recall.
        * ``"n_gt_entities"`` (int) -- number of S1 entities in ground truth.
        * ``"n_predicted_total"`` (int) -- total candidate IDs predicted across
          all S1 entities at this threshold.

    ``"best_threshold"``  : float
        The threshold value with the highest ``macro_f_beta``.  Ties are
        broken by selecting the **lowest** threshold (most recall-friendly).
        **Note**: this is reported for transparency only; Phase 4.4 owns
        the authoritative selection decision.

    ``"best_macro_f_beta"`` : float
        The macro F-beta score at ``"best_threshold"``.

    ``"n_thresholds_evaluated"`` : int
        Number of unique thresholds evaluated.

    ``"beta"`` : float
        The beta value used throughout the sweep.

    Raises
    ------
    TypeError
        * If *scored_pairs_df* is not a ``pd.DataFrame``.
        * If *ground_truth_df* is not a ``pd.DataFrame``.
        * If *thresholds* is not ``None``, a list, or a tuple.
    ValueError
        * If *scored_pairs_df* is missing required columns.
        * If *ground_truth_df* is missing ``source1_entity_id`` or both
          accepted match-ID columns.
        * If *thresholds* is empty.
        * If any threshold value is outside ``[0, 1]``.
        * If *beta* is not strictly positive.
        * If ``prob_match`` contains NaN, infinite, or out-of-range values
          (delegated to :func:`apply_threshold`).

    Notes
    -----
    * Ground-truth entities absent from *scored_pairs_df* are scored with
      an empty prediction set (contributing 0.0 to the macro average if
      they have non-empty true matches).
    * S1 entities in *scored_pairs_df* that are absent from *ground_truth_df*
      are excluded from metric computation (they are predicted over, but
      not evaluated).
    * The ``"best_threshold"`` value is determined strictly from the sweep
      results; no external assumption (e.g. 0.5 or 0.8) is made.
    * No threshold is persisted or applied to any production data here.
    """
    # ------------------------------------------------------------------
    # 1. Type validation
    # ------------------------------------------------------------------
    if not isinstance(scored_pairs_df, pd.DataFrame):
        raise TypeError(
            f"sweep_thresholds(): scored_pairs_df must be a pd.DataFrame, "
            f"got {type(scored_pairs_df).__name__}."
        )
    if not isinstance(ground_truth_df, pd.DataFrame):
        raise TypeError(
            f"sweep_thresholds(): ground_truth_df must be a pd.DataFrame, "
            f"got {type(ground_truth_df).__name__}."
        )
    if thresholds is not None and not isinstance(thresholds, (list, tuple)):
        raise TypeError(
            f"sweep_thresholds(): thresholds must be None, a list, or a tuple, "
            f"got {type(thresholds).__name__}."
        )
    if beta <= 0.0:
        raise ValueError(
            f"sweep_thresholds(): beta must be strictly positive, got {beta!r}."
        )

    # ------------------------------------------------------------------
    # 2. Resolve threshold grid
    # ------------------------------------------------------------------
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    thresholds = list(thresholds)

    if len(thresholds) == 0:
        raise ValueError(
            "sweep_thresholds(): thresholds must be a non-empty sequence."
        )

    bad = [t for t in thresholds if not isinstance(t, (int, float)) or t < 0.0 or t > 1.0]
    if bad:
        raise ValueError(
            f"sweep_thresholds(): all threshold values must be in [0.0, 1.0]. "
            f"Invalid value(s): {bad[:5]!r}"
        )

    # Deduplicate and sort ascending
    thresholds_sorted = sorted(set(float(t) for t in thresholds))

    # ------------------------------------------------------------------
    # 3. Parse ground truth into a dict lookup
    # ------------------------------------------------------------------
    gt_lookup: dict[str, frozenset[str]] = _parse_ground_truth_df(ground_truth_df)
    n_gt_entities = len(gt_lookup)

    # ------------------------------------------------------------------
    # 4. Validate scored_pairs_df has the required columns
    #    (delegated to apply_threshold for prob quality; we check
    #    column presence here to give a clear early error)
    # ------------------------------------------------------------------
    required = {_S1_COL, _CAND_COL, _PROB_COL}
    missing = required - set(scored_pairs_df.columns)
    if missing:
        raise ValueError(
            f"sweep_thresholds(): scored_pairs_df is missing required "
            f"column(s): {sorted(missing)}. "
            f"Found: {list(scored_pairs_df.columns)}"
        )

    # ------------------------------------------------------------------
    # 5. Sweep: evaluate each threshold
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for thresh in thresholds_sorted:
        # 5a. Apply threshold -> per-S1 prediction sets
        preds: dict[str, set[str]] = apply_threshold(
            scored_pairs_df, threshold=thresh
        )

        # 5b. Compute per-entity metrics against ground truth
        entity_f_scores: list[float] = []
        entity_precisions: list[float] = []
        entity_recalls: list[float] = []
        n_predicted_total: int = 0

        for s1_id, true_ids in gt_lookup.items():
            pred_ids: set[str] = preds.get(str(s1_id), set())
            n_predicted_total += len(pred_ids)

            # Per-entity precision and recall (raw, before F-beta)
            n_tp = len(pred_ids & set(true_ids))
            n_pred = len(pred_ids)
            n_true = len(true_ids)

            prec = n_tp / n_pred if n_pred > 0 else (1.0 if n_true == 0 else 0.0)
            rec  = n_tp / n_true if n_true > 0 else (1.0 if n_pred == 0 else 0.0)

            entity_precisions.append(prec)
            entity_recalls.append(rec)

            # Entity F-beta via the validated Phase 4.1 implementation
            f = entity_f_beta(pred_ids, true_ids, beta=beta)
            entity_f_scores.append(f)

        macro_f = float(np.mean(entity_f_scores)) if entity_f_scores else 0.0
        macro_p = float(np.mean(entity_precisions)) if entity_precisions else 0.0
        macro_r = float(np.mean(entity_recalls)) if entity_recalls else 0.0

        rows.append({
            "threshold": thresh,
            "macro_f_beta": macro_f,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "n_gt_entities": n_gt_entities,
            "n_predicted_total": n_predicted_total,
        })

    # ------------------------------------------------------------------
    # 6. Identify best threshold (highest macro F-beta; ties -> lowest t)
    # ------------------------------------------------------------------
    best_row = max(rows, key=lambda r: (r["macro_f_beta"], -r["threshold"]))
    best_threshold: float = best_row["threshold"]
    best_macro_f_beta: float = best_row["macro_f_beta"]

    return {
        "rows": rows,
        "best_threshold": best_threshold,
        "best_macro_f_beta": best_macro_f_beta,
        "n_thresholds_evaluated": len(rows),
        "beta": beta,
    }


# ---------------------------------------------------------------------------
# Phase 4.4 -- Singleton-specific performance breakdown
# ---------------------------------------------------------------------------

def singleton_report(
    predicted_by_s1: "dict[str, set[str]]",
    ground_truth_df: pd.DataFrame,
    beta: float = DEFAULT_BETA,
) -> dict:
    """Produce a diagnostic breakdown of performance on singleton vs non-singleton S1 entities.

    Splits the S1 entities present in *ground_truth_df* into two groups:

    * **Singleton** -- entities whose ground-truth match set is **empty**
      (no matching S2/S3 entity exists; the correct prediction is an empty set).
    * **Non-singleton** -- entities whose ground-truth match set contains
      **one or more** matching S2/S3 entity IDs.

    For each group, the function computes macro-averaged entity-level
    F-beta, precision, and recall using the same logic as
    :func:`macro_f_beta`.  This is a **read-only diagnostic**; it never
    alters *predicted_by_s1*, the model, or any threshold.

    Parameters
    ----------
    predicted_by_s1 : dict[str, set[str]]
        Per-S1-entity prediction sets, typically produced by
        :func:`apply_threshold`.  S1 entities absent from this dict are
        treated as predicting an empty set.
    ground_truth_df : pd.DataFrame
        Ground-truth table.  Must contain ``source1_entity_id`` and one of:

        * ``matching_entity_ids`` (primary column name), or
        * ``match_entity_ids`` (alternate column name).

        Comma-separated ID strings, single IDs, and empty strings are all
        accepted (same parsing as :func:`sweep_thresholds`).
    beta : float, optional
        Beta parameter forwarded to :func:`entity_f_beta` for all
        per-entity scores.  Must be strictly positive.
        Default: ``0.5`` (precision-weighted, matching the pipeline default).

    Returns
    -------
    dict with the following keys:

    ``"singleton"`` : dict
        Metrics for S1 entities with **empty** ground-truth match sets:

        * ``"n_entities"`` (int) -- count of singleton GT entities.
        * ``"macro_f_beta"`` (float) -- macro-averaged F-beta (0.0 if 0 entities).
        * ``"macro_precision"`` (float) -- macro-averaged precision.
        * ``"macro_recall"`` (float) -- macro-averaged recall.
        * ``"n_false_positive_predictions"`` (int) -- total predicted IDs
          across all singleton entities (every prediction is a FP for these
          entities since their true match set is empty).

    ``"non_singleton"`` : dict
        Metrics for S1 entities with **non-empty** ground-truth match sets:

        * ``"n_entities"`` (int) -- count of non-singleton GT entities.
        * ``"macro_f_beta"`` (float) -- macro-averaged F-beta (0.0 if 0 entities).
        * ``"macro_precision"`` (float) -- macro-averaged precision.
        * ``"macro_recall"`` (float) -- macro-averaged recall.
        * ``"n_predicted_total"`` (int) -- total predicted IDs across all
          non-singleton entities.

    ``"overall"`` : dict
        Metrics across **all** GT entities (singleton + non-singleton combined):

        * ``"n_entities"`` (int) -- total GT entities.
        * ``"macro_f_beta"`` (float) -- macro-averaged F-beta across all GT entities.
        * ``"macro_precision"`` (float) -- macro-averaged precision.
        * ``"macro_recall"`` (float) -- macro-averaged recall.

    ``"beta"`` : float
        The beta value used for all F-beta computations.

    Raises
    ------
    TypeError
        * If *predicted_by_s1* is not a ``dict``.
        * If *ground_truth_df* is not a ``pd.DataFrame``.
    ValueError
        * If *ground_truth_df* is missing ``source1_entity_id`` or both
          accepted match-ID columns (delegated to :func:`_parse_ground_truth_df`).
        * If *beta* is not strictly positive.

    Notes
    -----
    * This function is **purely diagnostic**.  It does not modify
      *predicted_by_s1*, select a threshold, or alter any model state.
    * S1 entities present in *predicted_by_s1* but absent from
      *ground_truth_df* are ignored (they cannot be evaluated without a
      reference).
    * A singleton entity that is correctly predicted as empty scores
      ``entity_f_beta(set(), set()) == 1.0``.
    * A singleton entity that receives any predicted IDs scores
      ``entity_f_beta(non_empty, set()) == 0.0`` (all predictions are FPs).

    Examples
    --------
    >>> import pandas as pd
    >>> gt = pd.DataFrame({
    ...     "source1_entity_id": ["S1", "S2", "S3"],
    ...     "matching_entity_ids": ["", "C1", "C2,C3"],
    ... })
    >>> preds = {"S1": set(), "S2": {"C1"}, "S3": {"C2"}}
    >>> report = singleton_report(preds, gt)
    >>> report["singleton"]["n_entities"]
    1
    >>> report["non_singleton"]["n_entities"]
    2
    """

    # ------------------------------------------------------------------
    # 1. Type and value validation
    # ------------------------------------------------------------------
    if not isinstance(predicted_by_s1, dict):
        raise TypeError(
            f"singleton_report(): predicted_by_s1 must be a dict, "
            f"got {type(predicted_by_s1).__name__}."
        )
    if not isinstance(ground_truth_df, pd.DataFrame):
        raise TypeError(
            f"singleton_report(): ground_truth_df must be a pd.DataFrame, "
            f"got {type(ground_truth_df).__name__}."
        )
    if beta <= 0.0:
        raise ValueError(
            f"singleton_report(): beta must be strictly positive, got {beta!r}."
        )

    # ------------------------------------------------------------------
    # 2. Parse ground truth into a dict lookup
    #    (raises ValueError for missing columns, delegated to helper)
    # ------------------------------------------------------------------
    gt_lookup: dict[str, frozenset[str]] = _parse_ground_truth_df(ground_truth_df)

    # ------------------------------------------------------------------
    # 3. Partition GT entities into singleton / non-singleton groups
    # ------------------------------------------------------------------
    singleton_ids: list[str] = []
    non_singleton_ids: list[str] = []

    for s1_id, true_ids in gt_lookup.items():
        if len(true_ids) == 0:
            singleton_ids.append(s1_id)
        else:
            non_singleton_ids.append(s1_id)

    # ------------------------------------------------------------------
    # 4. Helper: compute group-level metrics
    # ------------------------------------------------------------------
    def _group_metrics(entity_ids: list[str]) -> dict:
        """Compute macro metrics for a given list of S1 entity IDs."""
        if not entity_ids:
            return {
                "n_entities": 0,
                "macro_f_beta": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
            }

        f_scores: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []

        for s1_id in entity_ids:
            true_ids = gt_lookup[s1_id]
            pred_ids: set[str] = predicted_by_s1.get(str(s1_id), set())

            n_tp = len(pred_ids & set(true_ids))
            n_pred = len(pred_ids)
            n_true = len(true_ids)

            prec = n_tp / n_pred if n_pred > 0 else (1.0 if n_true == 0 else 0.0)
            rec  = n_tp / n_true if n_true > 0 else (1.0 if n_pred == 0 else 0.0)

            precisions.append(prec)
            recalls.append(rec)
            f_scores.append(entity_f_beta(pred_ids, true_ids, beta=beta))

        return {
            "n_entities": len(entity_ids),
            "macro_f_beta": float(np.mean(f_scores)),
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
        }

    # ------------------------------------------------------------------
    # 5. Compute per-group metrics
    # ------------------------------------------------------------------
    singleton_metrics = _group_metrics(singleton_ids)
    non_singleton_metrics = _group_metrics(non_singleton_ids)
    overall_metrics = _group_metrics(list(gt_lookup.keys()))

    # ------------------------------------------------------------------
    # 6. Augment singleton section with false-positive prediction count
    # ------------------------------------------------------------------
    n_false_positive_predictions: int = sum(
        len(predicted_by_s1.get(str(s1_id), set()))
        for s1_id in singleton_ids
    )
    singleton_metrics["n_false_positive_predictions"] = n_false_positive_predictions

    # ------------------------------------------------------------------
    # 7. Augment non-singleton section with total prediction count
    # ------------------------------------------------------------------
    n_predicted_total: int = sum(
        len(predicted_by_s1.get(str(s1_id), set()))
        for s1_id in non_singleton_ids
    )
    non_singleton_metrics["n_predicted_total"] = n_predicted_total

    # ------------------------------------------------------------------
    # 8. Return structured report
    # ------------------------------------------------------------------
    return {
        "singleton": singleton_metrics,
        "non_singleton": non_singleton_metrics,
        "overall": overall_metrics,
        "beta": beta,
    }
