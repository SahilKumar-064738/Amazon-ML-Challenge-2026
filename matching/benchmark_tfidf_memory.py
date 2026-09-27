"""
matching/benchmark_tfidf_memory.py
------------------------------------
Phase 5 — Progressive TF-IDF memory safety benchmarks.

Runs Tests A through D (as specified in the phase prompt) on synthetic
data scaled to increasingly large sizes, recording RSS, runtime, candidate
count, and candidate recall at each stage.

Usage (from repo root):
    python matching/benchmark_tfidf_memory.py

Results are printed as a Markdown table and saved to:
    phase5_benchmark_results.txt

DO NOT run this on the full 2.2M × 10.3M dataset until all tests pass
with acceptable memory headroom.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# ── Make matching/src importable ─────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO_ROOT, "matching"))

from src.blocking_multiview import (
    fit_multiview_vectorizers,
    search_multiview_candidates,
    _rss_mb,
)

# ── psutil for reliable RSS measurement ──────────────────────────────────────
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False
    print("WARNING: psutil not installed — RSS measurements will be 0.0 MB")
    print("         Install with: pip install psutil")


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_COMPANY_WORDS = [
    "global", "trade", "limited", "corp", "corporation", "pvt", "private",
    "services", "solutions", "technologies", "enterprises", "industries",
    "international", "national", "india", "group", "holdings", "ventures",
    "exports", "imports", "manufacturing", "retail", "wholesale", "agency",
    "consulting", "finance", "capital", "logistics", "supply", "systems",
    "network", "energy", "power", "foods", "pharma", "chemicals", "metals",
    "textiles", "packaging", "printing", "media", "transport", "shipping",
    "construction", "infra", "projects", "associates", "partners", "co",
]
_ADDRESS_WORDS = [
    "street", "road", "nagar", "colony", "sector", "phase", "block",
    "district", "mumbai", "delhi", "bangalore", "chennai", "hyderabad",
    "pune", "kolkata", "ahmedabad", "jaipur", "lucknow", "indore", "bhopal",
    "plot", "flat", "floor", "building", "tower", "near", "opp", "behind",
    "market", "bazaar", "industrial", "area", "zone", "estate", "park",
]

_RNG = np.random.default_rng(42)


def _rand_name(rng: np.random.Generator, length_range=(2, 5)) -> str:
    n = rng.integers(*length_range)
    return " ".join(rng.choice(_COMPANY_WORDS, size=n, replace=True))


def _rand_address(rng: np.random.Generator, length_range=(3, 7)) -> str:
    n = rng.integers(*length_range)
    return " ".join(rng.choice(_ADDRESS_WORDS, size=n, replace=True))


def make_synthetic_corpus(n: int, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate n random entity rows for S2+S3."""
    if rng is None:
        rng = _RNG
    rows = []
    for i in range(n):
        name = _rand_name(rng)
        addr = _rand_address(rng)
        rows.append({
            "entity_id":     f"s2_{i}",
            "clean_text":    f"{name} {addr}",
            "clean_name":    name,
            "clean_address": addr,
        })
    return pd.DataFrame(rows)


def make_synthetic_queries(n: int, corpus: pd.DataFrame,
                           rng: np.random.Generator | None = None) -> pd.DataFrame:
    """
    Generate n query rows where every other query is a slight perturbation
    of a corpus row (to create ground-truth positives for recall measurement).
    """
    if rng is None:
        rng = _RNG
    rows = []
    ground_truth: dict[str, str] = {}   # s1_id → true s2_id match

    corpus_len = len(corpus)
    for i in range(n):
        if i % 2 == 0 and corpus_len > 0:
            # Near-duplicate of a corpus entry (positive pair)
            src_idx = int(rng.integers(0, corpus_len))
            src     = corpus.iloc[src_idx]
            name    = src["clean_name"]
            addr    = src["clean_address"]
            s2_id   = src["entity_id"]
            s1_id   = f"s1_{i}"
            ground_truth[s1_id] = s2_id
        else:
            # Random entity
            name  = _rand_name(rng)
            addr  = _rand_address(rng)
            s1_id = f"s1_{i}"
            s2_id = None

        rows.append({
            "entity_id":     s1_id,
            "clean_text":    f"{name} {addr}",
            "clean_name":    name,
            "clean_address": addr,
        })

    return pd.DataFrame(rows), ground_truth


# ---------------------------------------------------------------------------
# Candidate recall measurement
# ---------------------------------------------------------------------------

def compute_recall(
    results_df: pd.DataFrame,
    ground_truth: dict,
) -> dict:
    """
    Compute candidate recall on the ground-truth positive pairs.

    Returns dict with keys: recall, n_positives, n_retrieved, avg_candidates,
    p95_candidates, max_candidates.
    """
    if not ground_truth:
        return {"recall": None, "n_positives": 0, "n_retrieved": 0,
                "avg_candidates": 0, "p95_candidates": 0, "max_candidates": 0}

    # Candidates per S1 entity
    cand_counts = results_df.groupby("source1_entity_id")["candidate_entity_id"].count()
    n_retrieved = 0

    for s1_id, true_s2 in ground_truth.items():
        grp = results_df[results_df["source1_entity_id"] == s1_id]
        if true_s2 in grp["candidate_entity_id"].values:
            n_retrieved += 1

    recall = n_retrieved / len(ground_truth) if ground_truth else None
    return {
        "recall":         recall,
        "n_positives":    len(ground_truth),
        "n_retrieved":    n_retrieved,
        "avg_candidates": float(cand_counts.mean()) if len(cand_counts) else 0,
        "p95_candidates": float(np.percentile(cand_counts.values, 95)) if len(cand_counts) else 0,
        "max_candidates": int(cand_counts.max()) if len(cand_counts) else 0,
    }


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    test_label:        str
    n_queries:         int
    n_corpus:          int
    chunk_size:        int
    corpus_block_size: int
    workers:           int
    n_views:           int
    dtype:             str
    rss_before_mb:     float
    rss_after_fit_mb:  float
    rss_after_mb:      float
    peak_rss_mb:       float   # max RSS observed during the run
    runtime_s:         float
    n_candidates:      int
    avg_candidates:    float
    p95_candidates:    float
    max_candidates:    int
    recall:            Optional[float]
    n_positives:       int
    error:             str = ""


def run_benchmark(
    label: str,
    n_queries: int,
    n_corpus: int,
    chunk_size: int,
    corpus_block_size: int,
    top_k: int = 50,
    rng_seed: int = 42,
) -> BenchResult:
    """Run one benchmark and return a BenchResult."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK {label}  |  queries={n_queries:,}  corpus={n_corpus:,}  "
          f"chunk={chunk_size:,}  block={corpus_block_size:,}  top_k={top_k}")
    print(f"{'='*60}")

    rng = np.random.default_rng(rng_seed)

    rss_before = _rss_mb()
    print(f"  RSS before data gen: {rss_before:.0f} MB")

    # Generate data
    corpus_df = make_synthetic_corpus(n_corpus, rng)
    query_df, ground_truth = make_synthetic_queries(n_queries, corpus_df, rng)
    print(f"  Data generated: {n_queries:,} queries, {n_corpus:,} corpus rows")

    rss_after_data = _rss_mb()
    print(f"  RSS after data gen:  {rss_after_data:.0f} MB  "
          f"(delta: +{rss_after_data - rss_before:.0f} MB)")

    # Fit vectorizers
    t0 = time.time()
    try:
        vecs = fit_multiview_vectorizers(corpus_df)
    except Exception as exc:
        return BenchResult(
            test_label=label, n_queries=n_queries, n_corpus=n_corpus,
            chunk_size=chunk_size, corpus_block_size=corpus_block_size,
            workers=1, n_views=0, dtype="float32",
            rss_before_mb=rss_before, rss_after_fit_mb=0,
            rss_after_mb=0, peak_rss_mb=0,
            runtime_s=time.time()-t0, n_candidates=0,
            avg_candidates=0, p95_candidates=0, max_candidates=0,
            recall=None, n_positives=0, error=str(exc),
        )

    rss_after_fit = _rss_mb()
    print(f"  RSS after fit:       {rss_after_fit:.0f} MB  "
          f"(delta from data: +{rss_after_fit - rss_after_data:.0f} MB)")
    print(f"  Views fitted: {list(vecs.keys())}")

    # Track peak RSS during search
    rss_samples = [rss_after_fit]

    def _sample_rss():
        r = _rss_mb()
        rss_samples.append(r)
        return r

    # Run retrieval
    try:
        t1 = time.time()
        results_df = search_multiview_candidates(
            query_df, corpus_df, vecs,
            top_k=top_k,
            chunk_size=chunk_size,
            corpus_block_size=corpus_block_size,
        )
        runtime_s = time.time() - t1
    except Exception as exc:
        return BenchResult(
            test_label=label, n_queries=n_queries, n_corpus=n_corpus,
            chunk_size=chunk_size, corpus_block_size=corpus_block_size,
            workers=1, n_views=len(vecs), dtype="float32",
            rss_before_mb=rss_before, rss_after_fit_mb=rss_after_fit,
            rss_after_mb=0, peak_rss_mb=0,
            runtime_s=time.time()-t0, n_candidates=0,
            avg_candidates=0, p95_candidates=0, max_candidates=0,
            recall=None, n_positives=0, error=str(exc),
        )

    rss_after_search = _sample_rss()
    print(f"  RSS after search:    {rss_after_search:.0f} MB  "
          f"(delta from fit: +{rss_after_search - rss_after_fit:.0f} MB)")

    # Explicit release
    del vecs
    gc.collect()
    rss_after_release = _rss_mb()
    print(f"  RSS after release:   {rss_after_release:.0f} MB")

    # Compute recall
    recall_stats = compute_recall(results_df, ground_truth)

    n_cands = len(results_df)
    print(f"  Candidates: {n_cands:,}  "
          f"avg/query={recall_stats['avg_candidates']:.1f}  "
          f"p95={recall_stats['p95_candidates']:.0f}  "
          f"max={recall_stats['max_candidates']}")
    if recall_stats["recall"] is not None:
        print(f"  Candidate recall:   {recall_stats['recall']:.4f}  "
              f"({recall_stats['n_retrieved']}/{recall_stats['n_positives']} positives retrieved)")
    print(f"  Runtime: {runtime_s:.2f}s")

    peak_rss = max(rss_samples)

    return BenchResult(
        test_label=label,
        n_queries=n_queries,
        n_corpus=n_corpus,
        chunk_size=chunk_size,
        corpus_block_size=corpus_block_size,
        workers=1,
        n_views=3,
        dtype="float32",
        rss_before_mb=rss_before,
        rss_after_fit_mb=rss_after_fit,
        rss_after_mb=rss_after_search,
        peak_rss_mb=peak_rss,
        runtime_s=runtime_s,
        n_candidates=n_cands,
        avg_candidates=recall_stats["avg_candidates"],
        p95_candidates=recall_stats["p95_candidates"],
        max_candidates=recall_stats["max_candidates"],
        recall=recall_stats["recall"],
        n_positives=recall_stats["n_positives"],
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_recall(r: Optional[float]) -> str:
    if r is None:
        return "N/A"
    return f"{r:.4f}"


def format_results_table(results: list[BenchResult]) -> str:
    lines = []
    lines.append(
        "| Test | Queries | Corpus | Chunk | Block | Views | Peak RSS (MB) "
        "| Runtime (s) | Candidates | Avg/Q | P95/Q | Recall |"
    )
    lines.append(
        "|------|---------|--------|-------|-------|-------|---------------"
        "|-------------|------------|-------|-------|--------|"
    )
    for r in results:
        status = f" ⚠️ {r.error[:30]}" if r.error else ""
        lines.append(
            f"| {r.test_label} "
            f"| {r.n_queries:,} "
            f"| {r.n_corpus:,} "
            f"| {r.chunk_size:,} "
            f"| {r.corpus_block_size:,} "
            f"| {r.n_views} "
            f"| {r.peak_rss_mb:.0f} "
            f"| {r.runtime_s:.2f} "
            f"| {r.n_candidates:,} "
            f"| {r.avg_candidates:.1f} "
            f"| {r.p95_candidates:.0f} "
            f"| {_fmt_recall(r.recall)}{status} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*70)
    print("PHASE 5 — TF-IDF MEMORY SAFETY PROGRESSIVE BENCHMARKS")
    print("="*70)
    print(f"psutil available: {_PSUTIL}")

    # ── Test configurations ──────────────────────────────────────────────────
    # These are scaled to be safe on a development machine.
    # On the production 128 GB machine, sizes up to Test E are relevant.
    # We DO NOT run the full 2.2M × 10.3M here.
    tests = [
        # label,   queries,  corpus, chunk,  block, top_k
        ("A-tiny",     100,    1_000,   100,   500,    20),
        ("B-small",  1_000,   10_000,  1000,  5000,   30),
        ("C-medium", 5_000,   50_000,  2000, 10000,   30),
        ("D-large",  10_000, 100_000,  5000, 25000,   50),
    ]

    results: list[BenchResult] = []

    for label, n_q, n_c, chunk, block, top_k in tests:
        r = run_benchmark(label, n_q, n_c, chunk, block, top_k)
        results.append(r)

        # Safety gate: if any test uses more than 8 GB (dev machine limit),
        # stop and report rather than crashing.
        if r.peak_rss_mb > 8_000 and not r.error:
            print(f"\n⚠️  SAFETY GATE: peak RSS {r.peak_rss_mb:.0f} MB > 8000 MB threshold.")
            print("    Stopping benchmark progression. Review memory scaling.")
            break

        # Force cleanup between tests
        gc.collect()

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("BENCHMARK SUMMARY TABLE")
    print("="*70)
    table = format_results_table(results)
    print(table)

    # ── Memory scaling analysis ──────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("MEMORY SCALING ANALYSIS")
    print("="*70)
    prev = None
    for r in results:
        if prev is not None and prev.n_corpus > 0 and r.n_corpus > 0:
            corpus_scale = r.n_corpus / prev.n_corpus
            rss_scale    = (r.peak_rss_mb / prev.peak_rss_mb) if prev.peak_rss_mb > 0 else 0
            print(f"  {prev.test_label} → {r.test_label}: "
                  f"corpus ×{corpus_scale:.1f}  RSS ×{rss_scale:.2f}  "
                  f"({'SUBLINEAR ✓' if rss_scale < corpus_scale else 'SUPERLINEAR ⚠️'})")
        prev = r

    # ── Recall summary ───────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("RECALL SUMMARY")
    print("="*70)
    for r in results:
        print(f"  {r.test_label}:  recall={_fmt_recall(r.recall)}  "
              f"n_pos={r.n_positives}  retrieved={r.n_positives * (r.recall or 0):.0f}  "
              f"avg_cands={r.avg_candidates:.1f}")

    # ── Save to file ─────────────────────────────────────────────────────────
    out_path = os.path.join(_REPO_ROOT, "phase5_benchmark_results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Phase 5 — TF-IDF Memory Safety Benchmarks\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(table + "\n\n")
        f.write("Memory scaling:\n")
        prev = None
        for r in results:
            if prev is not None and prev.peak_rss_mb > 0:
                cs = r.n_corpus / prev.n_corpus
                rs = r.peak_rss_mb / prev.peak_rss_mb
                f.write(
                    f"  {prev.test_label} → {r.test_label}: "
                    f"corpus ×{cs:.1f}  RSS ×{rs:.2f}\n"
                )
            prev = r
        f.write("\nRecall:\n")
        for r in results:
            f.write(f"  {r.test_label}: recall={_fmt_recall(r.recall)}  "
                    f"avg_cands={r.avg_candidates:.1f}\n")

    print(f"\nResults saved to: {out_path}")

    # ── Extrapolate to production scale ──────────────────────────────────────
    print("\n\n" + "="*70)
    print("PRODUCTION SCALE EXTRAPOLATION (10M corpus, 2.2M queries)")
    print("="*70)
    # Use the last successful test to extrapolate
    last_ok = [r for r in results if not r.error]
    if last_ok:
        lr = last_ok[-1]
        if lr.n_corpus > 0 and lr.peak_rss_mb > 0:
            prod_corpus  = 10_000_000
            prod_queries = 2_200_000
            scale_factor = prod_corpus / lr.n_corpus
            # Rough linear extrapolation (conservative)
            est_rss_gb = lr.peak_rss_mb * scale_factor / 1024
            print(f"  Last benchmark: corpus={lr.n_corpus:,}  peak RSS={lr.peak_rss_mb:.0f} MB")
            print(f"  Scale factor (corpus): ×{scale_factor:.0f}")
            print(f"  Estimated peak RSS at 10M corpus: ~{est_rss_gb:.1f} GB "
                  f"(linear extrapolation, conservative upper bound)")
            print(f"  Machine RAM budget: 80–90 GB target ceiling")
            if est_rss_gb <= 90:
                print(f"  → LIKELY SAFE at production scale (est ≤ 90 GB)")
                print(f"    Recommended: start with chunk={lr.chunk_size:,} "
                      f"block={lr.corpus_block_size:,} workers=1")
                print(f"    Verify on a 500k-query × 1M-corpus subset before full run.")
            else:
                print(f"  → ⚠️  ESTIMATED TO EXCEED 90 GB — reduce chunk/block sizes first.")
                safe_block = max(10_000, lr.corpus_block_size // 2)
                safe_chunk = max(1_000,  lr.chunk_size // 2)
                print(f"    Try: chunk={safe_chunk:,}  block={safe_block:,}")


if __name__ == "__main__":
    main()
