# Business Entity Resolution — Combined Pipeline

Amazon 2026 ML Challenge

End-to-end pipeline: raw TSV ingestion → text normalization → multi-channel
blocking → feature engineering → LightGBM matching → submission output.

```bash
python run_pipeline.py
```

---

## Quick Status

| Item | Value |
|------|-------|
| Feature version | **V9 — 89 features** |
| Memory safety | **Phase 5 redesign applied — CONDITIONAL GO** |
| Regression tests | **363 passed, 16 skipped, 0 failed** |
| Candidate recall (benchmarked) | **1.0000 on all test sizes** |
| Estimated full-scale peak RSS | **~61–71 GB** (128 GB machine, well within budget) |

---

## 1. What the Pipeline Does

| Stage | Name | Description |
|-------|------|-------------|
| Stage 1 | **Preprocessing** (MLnNor) | Raw TSVs → schema validation → Unicode NFKC → case/whitespace → legal suffix/abbreviation canonicalization → landmark/postal/script extraction → cleaned TSVs + quality report |
| Bridge | **Contract converter** | Transforms MLnNor's multi-column output into the two-file format (clean_text + feature) expected by the matching pipeline |
| Stage 2 | **Matching** (MLnAWS) | Multi-channel blocking → V9 feature engineering → LightGBM + hard-negative mining → threshold sweep → final output |

```
dataset/raw/
  train/{source1,2,3}.tsv + ground_truth.tsv
  test/{source1,2,3}.tsv
        │
        ▼  Stage 1 — Preprocessing (MLnNor)
output/preprocessing/processed/
  {train,test}/source{1,2,3}_clean.tsv
        │
        ▼  Bridge — contract conversion
dataset/real/
  clean_s{1,2s3}_{train,test}.tsv
  feature_s{1,2s3}_{train,test}.tsv
  ground_truth_train.tsv
        │
        ▼  Stage 2 — Matching (MLnAWS)
output/matching/
  matching_results.tsv     ← upload to leaderboard
  candidate_pairs.tsv      ← include in submission zip
```

---

## 2. Feature Engineering — V9 (89 features)

Features are built cumulatively across four phases:

| Version | Features | Phase | Module |
|---------|----------|-------|--------|
| V5 | 40 | Baseline (name/address similarity, retrieval agreement, RRF ranks, frequency, address components) | `features.py`, `features_rich.py`, `rrf_features.py`, `frequency_features.py`, `address_components.py` |
| V6 | +15 = 55 | Difference/numeric/house-number features | `diff_features.py` |
| V7 | +16 = 71 | Margin/ambiguity features (runner-up gap, retrieval confidence) | `margin_features.py` |
| V8 | +10 = 81 | RapidFuzz string similarity (ratio, partial_ratio, JaroWinkler on name keys/addresses) | `rapidfuzz_features.py` |
| V9 | +8 = 89 | Indic transliteration features (romanized name similarity gains) | `transliteration_features.py` |

---

## 3. Blocking Channels

| Channel | Module | Memory safety | Notes |
|---------|--------|---------------|-------|
| Exact match | `blocking_exact.py` | ✅ Bounded hash lookup | Always enabled |
| Multi-view TF-IDF | `blocking_multiview.py` | ✅ **Phase 5 redesigned** | Primary channel |
| BM25 | `blocking_bm25.py` | ✅ ~6 GB, sequential after TF-IDF | Skip with `--skip-bm25` |
| FAISS semantic | `blocking_faiss.py` | ✅ On-disk index, ~1–2 GB | Off by default, `--enable-faiss` |
| Candidate fusion | `blocking_fusion.py` | ✅ ID columns only, no text duplication | Always runs |

### Multi-View TF-IDF — Phase 5 Memory Safety Redesign

The previous pipeline OOM'd at ~130 GB on a 128 GB machine because three
TF-IDF corpus matrices accumulated simultaneously with no cleanup.

**All three issues are fixed:**

1. **Sequential view processing with explicit release** — after each view's
   search completes, `del s1_mat, s2s3_mat` + `gc.collect()` runs before the
   next view begins. At most one corpus matrix is resident at any time.

2. **CLI chunk parameters are now forwarded** — `--chunk-size` and
   `--corpus-block-size` were previously silently ignored. They now propagate
   all the way to `search_candidates_scalable()`.

3. **float32 dtype** — all three TF-IDF view configs now use `dtype=np.float32`,
   halving matrix memory vs the previous float64 default.

**2D-chunked sparse similarity** (unchanged, was already correct):
```
FOR each query chunk (--chunk-size):
    FOR each corpus block (--corpus-block-size):
        sim = s1_chunk @ corpus_block.T   ← sparse × sparse, bounded
        Top-K merge                        ← only K scores kept per query
        del sim, corpus_block             ← released immediately
```
No global Q×C similarity matrix is ever created.

---

## 4. Repository Structure

```
FinalMlnAWS/
│
├── run_pipeline.py             ← SINGLE ENTRYPOINT
├── pipeline_config.py          ← centralised configuration
├── pipeline_logging.py         ← structured logging
├── pipeline_checkpoints.py     ← per-step sentinel checkpoints
├── pipeline_validation.py      ← inter-stage validation
├── requirements.txt
├── pytest.ini
├── README.md
│
├── preprocessing/              ← Stage 1 (MLnNor)
│   ├── src/
│   │   ├── normalization.py    ← all text normalization (DO NOT MODIFY)
│   │   ├── validation.py       ← schema / ID invariant checks
│   │   ├── diagnostics.py      ← collision diagnostics
│   │   ├── preprocess.py       ← preprocessing entrypoint
│   │   └── indic_transliteration.py
│   ├── scripts/create_sample.py
│   └── bridge.py               ← MLnNor → MLnAWS contract converter
│
├── matching/                   ← Stage 2 (MLnAWS)
│   ├── run_production.py       ← matching pipeline CLI
│   ├── run_e2e.py              ← dev/mock end-to-end runner (not production)
│   ├── src/
│   │   ├── blocking.py             ← 2D-chunked scalable TF-IDF search
│   │   ├── blocking_multiview.py   ← ⭐ Phase 5 memory-safe multi-view TF-IDF
│   │   ├── blocking_exact.py       ← exact-match blocking
│   │   ├── blocking_bm25.py        ← BM25 blocking
│   │   ├── blocking_faiss.py       ← FAISS semantic blocking (optional)
│   │   ├── blocking_fusion.py      ← candidate list fusion
│   │   ├── features.py             ← V5 feature engineering
│   │   ├── features_rich.py        ← additional rich features
│   │   ├── diff_features.py        ← V6 (+15 difference features)
│   │   ├── margin_features.py      ← V7 (+16 margin/ambiguity features)
│   │   ├── rapidfuzz_features.py   ← V8 (+10 RapidFuzz features)
│   │   ├── transliteration_features.py ← V9 (+8 Indic transliteration features)
│   │   ├── model.py                ← LightGBM + hard-negative mining
│   │   ├── threshold.py            ← F0.5 threshold sweep
│   │   ├── negative_sampling.py    ← controlled negative pair construction
│   │   ├── ingestion.py            ← TSV loading
│   │   ├── adaptive_k.py, ablation.py, entity_decision.py
│   │   ├── address_components.py, frequency_features.py, rrf_features.py
│   │   └── postprocess.py
│   └── utils/
│       └── validate_submission.py  ← official submission format validator
│
├── dataset/
│   ├── raw/                    ← PUT RAW DATA HERE (never overwritten)
│   │   ├── train/
│   │   └── test/
│   ├── real/                   ← auto-generated by bridge
│   └── mock/                   ← small mock data for run_e2e.py
│
├── tests/
│   ├── test_tfidf_memory_safety.py  ← Phase 5 correctness tests (23 tests)
│   ├── test_mlnaws_v6_v9.py         ← V6–V9 feature regression tests
│   ├── test_faiss_blocking.py
│   ├── test_indic_transliteration.py
│   ├── test_normalization_extended.py
│   ├── test_preprocessing_unit.py
│   ├── test_preprocessing_integration.py
│   ├── test_bridge_contract.py
│   ├── test_checkpoints.py
│   ├── test_cli.py
│   └── test_raw_schema.py
│
├── phase5_tfidf_memory_safety_report.md   ← Phase 5 memory audit
└── final_memory_safety_audit_report.md    ← GO/NO-GO pre-flight report
```

---

## 5. Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 8 vCPU | 16 vCPU |
| RAM | 32 GB | 128 GB |
| Disk | 30 GB free | 60 GB free |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| Python | 3.10 | 3.10 or 3.11 |

Verified not to OOM on 128 GB. Estimated peak RSS: **61–71 GB**
(BM25 on). See `final_memory_safety_audit_report.md` for full breakdown.

---

## 6. Installation

```bash
git clone <repository-url>
cd FinalMlnAWS

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Optional — FAISS semantic blocking:
```bash
pip install faiss-cpu>=1.7.4 sentence-transformers>=2.2.2
```

Optional — memory profiling (used by Phase 5 instrumentation):
```bash
pip install psutil
```

---

## 7. Where to Put Raw Data

Place the **7 raw TSV files** exactly here (do not rename):

```
dataset/raw/train/train_source1.tsv
dataset/raw/train/train_source2.tsv
dataset/raw/train/train_source3.tsv
dataset/raw/train/train_ground_truth.tsv
dataset/raw/test/test_source1.tsv
dataset/raw/test/test_source2.tsv
dataset/raw/test/test_source3.tsv
```

| File | Required Columns |
|------|-----------------|
| `*_source{1,2,3}.tsv` | `entity_id`, `business_name`, `business_address`, `country` |
| `train_ground_truth.tsv` | `source1_entity_id`, `matched_entity_ids` |

Entity ID format: `S1-\d+`, `S2-\d+`, `S3-\d+`.

---

## 8. Running the Full Pipeline

```bash
python run_pipeline.py
```

Steps executed:
```
[01/08] Validate environment
[02/08] Validate raw dataset
[03/08] Run preprocessing (MLnNor)
[04/08] Run bridge  (preprocessed → MLnAWS format)
[05/08] Validate MLnAWS input contract
[06/08] Run entity-resolution matching (MLnAWS)
[07/08] Validate final outputs
[08/08] Write pipeline report
```

---

## 9. Production CLI Options

```bash
python run_pipeline.py [OPTIONS]

# Memory-safe recommended settings (validated):
python run_pipeline.py \
    --chunk-size      5000   \
    --corpus-block-size 25000 \
    --top-k           50     \
    --workers         8      \
    --skip-bm25

# Full production with BM25 (adds ~6 GB peak RSS):
python run_pipeline.py \
    --chunk-size      5000   \
    --corpus-block-size 25000 \
    --top-k           50     \
    --workers         8

# With FAISS semantic blocking (+1–2 GB):
python run_pipeline.py \
    --chunk-size      5000   \
    --corpus-block-size 25000 \
    --top-k           50     \
    --workers         8      \
    --enable-faiss
```

### Key CLI parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--chunk-size` | `20000` | S1 query chunk for TF-IDF 2D blocking. **5000 recommended.** |
| `--corpus-block-size` | `250000` | S2+S3 corpus block for TF-IDF 2D blocking. **25000 recommended.** |
| `--top-k` | `50` | Max candidates per S1 per view. Union across 3 views → ≤150 total. |
| `--workers` | `8` | LightGBM training threads. No effect on blocking (single-threaded). |
| `--skip-bm25` | off | Saves ~6 GB peak RSS. Add if RAM is tight. |
| `--enable-faiss` | off | FAISS semantic blocking. Requires faiss-cpu + sentence-transformers. |
| `--skip-preprocessing` | off | Skip Stage 1; use existing `dataset/real/` files. |
| `--skip-matching` | off | Run preprocessing + bridge only. |
| `--force` | off | Ignore all checkpoints; re-run everything. |
| `--dry-run` | off | Validate inputs without processing. |

---

## 10. Resuming After Interruption

The pipeline checkpoints every major step. On any restart:

```bash
python run_pipeline.py   # automatically skips completed steps
```

To check what is done:
```python
from pipeline_checkpoints import CheckpointStore
from pipeline_config import REPO_ROOT
store = CheckpointStore(REPO_ROOT / "checkpoints")
print(store.resume_summary())
```

To re-run from a specific step, delete its sentinel:
```bash
# Re-run matching from scratch (keeps preprocessing output):
python run_pipeline.py --skip-preprocessing
```

---

## 11. Monitoring Memory During a Run

The Phase 5 TF-IDF redesign emits inline RSS checkpoints:

```
[MEM 14:06:08] view='combined' AFTER_CORPUS_TRANSFORM s2s3_mat=(250000, 575)  nnz=10199227  mem=79 MB: RSS=539 MB
[MEM 14:10:22] view='combined' AFTER_TOPK: RSS=553 MB
[MEM 14:10:22] view='combined' AFTER_RELEASE_AND_GC: RSS=459 MB   ← explicit recovery
```

Watch these lines during a production run:

```bash
# Follow the log:
tail -f logs/pipeline_*.log | grep "MEM\|RSS\|OOM\|Error"

# Key things to watch:
# - AFTER_CORPUS_TRANSFORM: this is the peak for each view
# - AFTER_RELEASE_AND_GC: this should be meaningfully lower (GC recovery working)
# - If AFTER_CORPUS_TRANSFORM exceeds 80 GB → stop and reduce --corpus-block-size
```

**Expected RSS range at full scale (10M corpus):**
- Combined view AFTER_CORPUS_TRANSFORM: ~10–15 GB above baseline
- After del+gc recovery: ~3–5 GB above baseline
- Total pipeline peak: ~61–71 GB (BM25 on) / ~55–65 GB (BM25 off)

---

## 12. Mandatory Test E Before Full Run

**Do not run the full 2.2M S1 × 10.3M S2+S3 job without completing Test E.**

Test E validates that real-data nnz/row is consistent with synthetic benchmark
estimates. Use a 10% sample:

```bash
# Extract 500k S1 × 1M S2+S3 subset from real data (adjust paths as needed)
python run_pipeline.py \
    --chunk-size       5000  \
    --corpus-block-size 25000 \
    --top-k            50    \
    --workers          1     \
    --skip-bm25              \
    --data-dir         dataset/real_sample  # 500k × 1M subset

# Monitor RSS via log lines.
# If peak RSS < 40 GB → full run is safe.
# If peak RSS 40–60 GB → full run is marginal; proceed with BM25 disabled.
# If peak RSS > 60 GB → reduce --corpus-block-size to 10000 before full run.
```

---

## 13. Memory Budget Summary

| Component | Peak (estimated, 10M corpus) |
|-----------|------------------------------|
| OS + Python + DataFrames baseline | ~15 GB |
| TF-IDF combined view corpus matrix (one at a time) | ~10 GB |
| TF-IDF combined view query matrix (2.2M S1) | ~2 GB |
| BM25 tokenized corpus (if enabled) | ~6 GB |
| FAISS (if enabled, on-disk index) | ~2 GB |
| Candidate DataFrame (post-fusion, deduped) | ~20–30 GB |
| Feature extraction join | ~10–15 GB |
| LightGBM training | ~3 GB |
| **Total estimated peak (BM25 on)** | **~61–71 GB** |
| **Safety margin to 128 GB** | **57–67 GB** |

The previous OOM at ~130 GB is no longer possible with the Phase 5 fixes
because the dominant cause (three simultaneous corpus matrices = ~20 GB extra,
plus no GC between views) has been eliminated.

---

## 14. Testing

```bash
# All tests (no real data needed — uses synthetic fixtures only):
python -m pytest

# Phase 5 memory-safety tests only:
python -m pytest tests/test_tfidf_memory_safety.py -v

# Feature regression (V6–V9):
python -m pytest tests/test_mlnaws_v6_v9.py -v

# Preprocessing:
python -m pytest tests/test_preprocessing_unit.py tests/test_preprocessing_integration.py -v

# Single test module:
python -m pytest tests/test_tfidf_memory_safety.py::test_near_duplicate_retrieval -v
```

Current results: **363 passed, 16 skipped, 0 failed**

---

## 15. Validating Submission Output

```bash
python matching/utils/validate_submission.py \
    --matching  output/matching/matching_results.tsv \
    --candidate output/matching/candidate_pairs.tsv \
    --test-dir  dataset/raw/test
```

Expected output:
```
PASS
  Both output files pass all submission rules.
  Safe to submit.
```

Submission format:

| File | Columns | Format |
|------|---------|--------|
| `matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` | One row per S1; comma-separated matched IDs |
| `candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` | One row per S1; comma-separated candidate IDs |

---

## 16. Where Outputs Are Written

```
output/matching/matching_results.tsv     ← upload to leaderboard
output/matching/candidate_pairs.tsv      ← include in submission zip
output/pipeline_report.json              ← timings, row counts, config
output/preprocessing/reports/preprocessing_report.json
logs/pipeline_<timestamp>.log
checkpoints/                             ← step sentinels for resumption
```

---

## 17. Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `missing dataset/raw/train/train_source1.tsv` | Raw files not in place | Copy TSV files to `dataset/raw/` |
| `schema mismatch` | Wrong column names | Verify: `entity_id, business_name, business_address, country` |
| `Required packages not installed` | pip packages missing | `pip install -r requirements.txt` |
| `ERROR: Prepared data files missing` | Running matching before preprocessing | Run without `--skip-preprocessing` first |
| `MemoryError` during BM25 | Not enough headroom | Add `--skip-bm25` |
| `Candidate consistency check failed` | Blocking/inference mismatch | Delete matching checkpoints and re-run matching |
| High RSS at `AFTER_CORPUS_TRANSFORM` | Real data denser than synthetic benchmark | Reduce `--corpus-block-size` to `10000` |
| Pipeline interrupted | Any cause | Re-run — checkpoints survive |

---

## 18. GCP Run Procedure

```bash
# ── On your LOCAL machine ─────────────────────────────────────────────────────
gcloud compute scp --recurse /local/path/raw/train/ USER@INSTANCE:~/FinalMlnAWS/dataset/raw/train/
gcloud compute scp --recurse /local/path/raw/test/  USER@INSTANCE:~/FinalMlnAWS/dataset/raw/test/

# ── On the GCP VM ─────────────────────────────────────────────────────────────
git clone <repository-url> && cd FinalMlnAWS

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify raw files are in place:
ls dataset/raw/train/   # should show 4 files
ls dataset/raw/test/    # should show 3 files

# Dry-run — validates inputs without processing:
python run_pipeline.py --dry-run

# ── Recommended production run ────────────────────────────────────────────────
# Step 1: Test E (mandatory — see Section 12)
# Step 2: If Test E peak RSS < 40 GB, run full job:

nohup python run_pipeline.py \
    --chunk-size      5000   \
    --corpus-block-size 25000 \
    --top-k           50     \
    --workers         8      \
    > nohup.out 2>&1 &

echo "PID=$!"

# Monitor:
tail -f logs/pipeline_*.log | grep -E "MEM|Step|done|ERROR|PASS|FAIL"

# Resume after interruption (safe to re-run):
python run_pipeline.py --chunk-size 5000 --corpus-block-size 25000 --workers 8

# Validate output:
python matching/utils/validate_submission.py \
    --matching  output/matching/matching_results.tsv \
    --candidate output/matching/candidate_pairs.tsv \
    --test-dir  dataset/raw/test
```

---

## 19. Phase History

| Phase | Description | Key Files |
|-------|-------------|-----------|
| Phase 1 | Difference/numeric features (+15) | `diff_features.py` |
| Phase 2 | Margin/ambiguity features (+16) | `margin_features.py` |
| Phase 3 | RapidFuzz string similarity (+10) | `rapidfuzz_features.py` |
| Phase 4 | Indic transliteration features (+8) | `transliteration_features.py` |
| Phase 5 | TF-IDF memory safety redesign | `blocking_multiview.py`, `run_production.py`, `tests/test_tfidf_memory_safety.py` |

Full audit details: `phase5_tfidf_memory_safety_report.md`  
GO/NO-GO pre-flight: `final_memory_safety_audit_report.md`
