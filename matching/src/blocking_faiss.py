"""
src/blocking_faiss.py
---------------------
FAISS-backed semantic retrieval for the entity-resolution pipeline.

Public API
----------
    ensure_faiss_index(s2s3_df, index_base, split_label, *, model_name,
                       batch_size, add_chunk_size)
        Build (or reuse) the FAISS index for a corpus DataFrame.

    search_faiss_candidates(s1_df, s2s3_df, index_base, *, top_k,
                            model_name, batch_size)
        Query the FAISS index and return a candidate DataFrame.

    get_or_build_and_search(s2s3_df, s1_df, index_base, split_label, *, ...)
        Convenience wrapper: ensure index exists then search.

Design constraints (see integration spec)
------------------------------------------
* FAISS is OPTIONAL.  If ``faiss-cpu`` or ``sentence-transformers`` are not
  installed, calling any function raises ``RuntimeError`` with an actionable
  message — callers must check this only when FAISS is explicitly enabled.
* Embeddings are L2-normalised → inner-product == cosine similarity.
* Index type: IVFFlat+IP for N >= nlist; falls back to IndexFlatIP for small
  datasets (avoids "n_train < nlist" crash on synthetic / test data).
* Candidate provenance column: ``retrieved_by_faiss`` (int, 0/1).
  NOT ``retrieved_by_embedding`` — that name is not in the production schema.
* ``retrieval_agreement_count`` is computed by ``blocking_fusion.fuse_candidates``
  and intentionally excludes ``retrieved_by_faiss`` to keep the {1,2} contract
  that ``features.build_feature_matrix`` validates against.
* Metadata sidecar (``<base>_metadata.json``) binds an index file to the
  exact configuration that built it.  Any mismatch → rebuild.
* Index build is atomic: all files written to ``<base>.tmp.*`` first, then
  os.replace'd into their final names so an interrupted build never leaves a
  valid-looking but incomplete index.
* Memory-safe: embeddings are generated in ``add_chunk_size`` batches, each
  batch released before the next is allocated.

Indexed text representation
-----------------------------
    "<clean_name> <clean_address>"   (if both columns present)
    "<clean_text>"                   (fallback if clean_name/address absent)

The same representation is used for both indexed corpus vectors and query
vectors so there is no asymmetry.

FAISS index specification (10 M corpus)
-----------------------------------------
    Model       : intfloat/multilingual-e5-small  (117 MB, 384-d, CPU)
    Dimension   : 384
    Normalization: L2 (cosine via IndexFlatIP)
    Index type  : IVFFlat (IndexIVFFlat + IndexFlatIP quantizer)
                  nlist=1024, nprobe=16
                  Falls back to IndexFlatIP for N < nlist
    Storage     : index metadata + on-disk inverted lists (OnDiskInvertedLists)
    float32     : yes (sentence-transformers default)
    Top-K       : configurable (default mirrors pipeline DEFAULT_TOP_K = 50)
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass  # avoid circular imports

# ---------------------------------------------------------------------------
# Lazy imports — raise clearly if faiss / sentence_transformers absent
# ---------------------------------------------------------------------------

def _require_faiss():
    """Import faiss or raise RuntimeError with an actionable install message."""
    try:
        import faiss  # noqa: F401
        return faiss
    except ImportError:
        raise RuntimeError(
            "FAISS was explicitly enabled (--enable-faiss) but 'faiss-cpu' "
            "is not installed in this environment.\n"
            "  Fix: pip install faiss-cpu>=1.7.4\n"
            "  Or:  pip install -r requirements.txt\n"
            "FAISS cannot proceed without faiss-cpu."
        ) from None


def _require_sentence_transformers():
    """Import SentenceTransformer or raise RuntimeError."""
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return SentenceTransformer
    except ImportError:
        raise RuntimeError(
            "FAISS was explicitly enabled (--enable-faiss) but "
            "'sentence-transformers' is not installed in this environment.\n"
            "  Fix: pip install sentence-transformers>=2.2.2\n"
            "  Or:  pip install -r requirements.txt\n"
            "FAISS cannot proceed without sentence-transformers."
        ) from None


# ---------------------------------------------------------------------------
# Text representation
# ---------------------------------------------------------------------------

def _build_embed_texts(df: pd.DataFrame) -> list[str]:
    """Return the canonical embedding text for each row (vectorized, 24M-safe).

    Uses ``clean_name + ' ' + clean_address`` when both columns present;
    otherwise falls back to ``clean_text``.
    """
    has_name    = "clean_name"    in df.columns
    has_address = "clean_address" in df.columns
    has_text    = "clean_text"    in df.columns

    if has_name and has_address:
        names     = df["clean_name"].fillna("").astype(str).str.strip()
        addresses = df["clean_address"].fillna("").astype(str).str.strip()
        combined  = (names + " " + addresses).str.strip()
        # Fall back to clean_text for rows where both name and address are empty
        if has_text:
            fallback = df["clean_text"].fillna("").astype(str).str.strip()
            combined = combined.where(combined != "", fallback)
        return combined.tolist()
    elif has_text:
        return df["clean_text"].fillna("").astype(str).str.strip().tolist()
    else:
        return [""] * len(df)


# ---------------------------------------------------------------------------
# Metadata sidecar helpers
# ---------------------------------------------------------------------------

_METADATA_FIELDS = (
    "model_name",
    "embedding_dim",
    "normalization",
    "index_type",
    "nlist",
    "nprobe",
    "split_label",
    "row_count",
    "entity_ids_hash",
    "add_chunk_size",
)


def _compute_entity_ids_hash(entity_ids: list[str]) -> str:
    """SHA-256 of sorted entity IDs to detect corpus changes."""
    h = hashlib.sha256()
    for eid in sorted(entity_ids):
        h.update(eid.encode("utf-8"))
    return h.hexdigest()[:16]


def _build_metadata(
    *,
    model_name: str,
    embedding_dim: int,
    normalization: str,
    index_type: str,
    nlist: int,
    nprobe: int,
    split_label: str,
    entity_ids: list[str],
    add_chunk_size: int,
) -> dict:
    return {
        "model_name":       model_name,
        "embedding_dim":    embedding_dim,
        "normalization":    normalization,
        "index_type":       index_type,
        "nlist":            nlist,
        "nprobe":           nprobe,
        "split_label":      split_label.upper(),
        "row_count":        len(entity_ids),
        "entity_ids_hash":  _compute_entity_ids_hash(entity_ids),
        "add_chunk_size":   add_chunk_size,
    }


def _metadata_path(index_base: "Path | str") -> Path:
    return Path(str(index_base) + "_metadata.json")


def _index_path(index_base: "Path | str") -> Path:
    return Path(str(index_base) + ".idx")


def _ivfdata_path(index_base: "Path | str") -> Path:
    return Path(str(index_base) + ".idx.ivfdata")


def _load_existing_metadata(index_base: "Path | str") -> dict | None:
    """Load the sidecar JSON; return None if absent or unreadable."""
    p = _metadata_path(index_base)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _index_is_valid(
    index_base: "Path | str",
    expected_meta: dict,
) -> bool:
    """Return True iff the index files exist AND metadata matches expected."""
    idx_p    = _index_path(index_base)
    # ivfdata file only present for IVFFlat; flat index has no .ivfdata
    # We only require the .idx file to exist for a valid cache.
    if not idx_p.exists():
        return False

    existing = _load_existing_metadata(index_base)
    if existing is None:
        return False

    # Compare all fields that affect index correctness
    for field in _METADATA_FIELDS:
        if existing.get(field) != expected_meta.get(field):
            return False
    return True


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------

def _encode_batch(
    model,
    texts: list[str],
    batch_size: int,
    show_progress: bool = True,
) -> "np.ndarray":
    """Encode a list of texts and return a float32 numpy array."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we L2-normalize ourselves after
    )
    return embeddings.astype(np.float32, copy=False)


def build_faiss_index(
    s2s3_df: pd.DataFrame,
    index_base: "Path | str",
    split_label: str = "CORPUS",
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
    add_chunk_size: int | None = None,
    nlist: int | None = None,
    nprobe: int | None = None,
    force: bool = False,
) -> None:
    """Build and persist a FAISS index for *s2s3_df*.

    The build is skipped (cache reuse) if an up-to-date index already exists
    unless *force* is True.

    Parameters
    ----------
    s2s3_df : pd.DataFrame
        Corpus DataFrame with at least ``entity_id`` and text columns.
    index_base : Path | str
        Base path for index files (no extension).  Files written:
            <base>.idx, <base>.idx.ivfdata (IVF only), <base>_metadata.json
    split_label : str
        Human-readable label embedded in metadata ("TRAIN" / "TEST").
    model_name : str | None
        Sentence-transformers model name.  Defaults to
        ``pipeline_config.FAISS_EMBEDDING_MODEL``.
    batch_size : int | None
        Encoding batch size.  Defaults to ``pipeline_config.FAISS_BATCH_SIZE``.
    add_chunk_size : int | None
        Vector-add chunk size.  Defaults to ``pipeline_config.FAISS_ADD_CHUNK_SIZE``.
    nlist : int | None
        IVF cell count.  Defaults to ``pipeline_config.FAISS_NLIST``.
    nprobe : int | None
        Query probe count (stored in metadata; applied at search time).
        Defaults to ``pipeline_config.FAISS_NPROBE``.
    force : bool
        If True, rebuild even when the cache is valid.
    """
    # Resolve defaults from pipeline_config (imported here to avoid circular
    # dependency at module level)
    try:
        from pipeline_config import (
            FAISS_EMBEDDING_MODEL, FAISS_EMBEDDING_DIM,
            FAISS_BATCH_SIZE, FAISS_ADD_CHUNK_SIZE,
            FAISS_NLIST, FAISS_NPROBE,
            FAISS_NORMALIZATION, FAISS_INDEX_TYPE,
        )
    except ImportError:
        # Fallback defaults when running outside the combined-pipeline tree
        FAISS_EMBEDDING_MODEL  = "intfloat/multilingual-e5-small"
        FAISS_EMBEDDING_DIM    = 384
        FAISS_BATCH_SIZE       = 1_000
        FAISS_ADD_CHUNK_SIZE   = 500_000
        FAISS_NLIST            = 1_024
        FAISS_NPROBE           = 16
        FAISS_NORMALIZATION    = "l2"
        FAISS_INDEX_TYPE       = "IVFFlat_IP"

    model_name     = model_name     or FAISS_EMBEDDING_MODEL
    batch_size     = batch_size     or FAISS_BATCH_SIZE
    add_chunk_size = add_chunk_size or FAISS_ADD_CHUNK_SIZE
    nlist          = nlist          or FAISS_NLIST
    nprobe         = nprobe         or FAISS_NPROBE

    faiss = _require_faiss()
    SentenceTransformer = _require_sentence_transformers()

    entity_ids = s2s3_df["entity_id"].tolist()
    n_records  = len(entity_ids)

    expected_meta = _build_metadata(
        model_name      = model_name,
        embedding_dim   = FAISS_EMBEDDING_DIM,
        normalization   = FAISS_NORMALIZATION,
        index_type      = FAISS_INDEX_TYPE,
        nlist           = nlist,
        nprobe          = nprobe,
        split_label     = split_label,
        entity_ids      = entity_ids,
        add_chunk_size  = add_chunk_size,
    )

    index_base = Path(index_base)
    index_base.parent.mkdir(parents=True, exist_ok=True)

    if not force and _index_is_valid(index_base, expected_meta):
        print(f"    [FAISS] Reusing existing index at {index_base}.idx "
              f"({n_records:,} records, split={split_label.upper()})")
        return

    if _index_path(index_base).exists() or _metadata_path(index_base).exists():
        print(f"    [FAISS] Existing index is stale or incomplete — rebuilding.")

    print(f"    [FAISS] Building index for {n_records:,} records "
          f"(split={split_label.upper()}) ...")
    t0 = time.time()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"    [FAISS] Loading model '{model_name}' ...")
    model = SentenceTransformer(model_name)
    d = model.get_sentence_embedding_dimension()

    # ── Prepare texts ───────────────────────────────────────────────────────
    texts = _build_embed_texts(s2s3_df)

    # ── Choose index type based on corpus size ───────────────────────────────
    # IVFFlat requires at least nlist training vectors.
    # Fall back to exact IndexFlatIP for tiny corpora (tests, mock data).
    use_ivf = n_records >= nlist

    # ── Paths ────────────────────────────────────────────────────────────────
    # OnDiskInvertedLists bakes the ivfdata file path into the .idx file at
    # faiss.write_index() time.  We MUST use the final target path for the
    # ivfdata file so that faiss.read_index() can find it on resume.
    # Only the .idx and _metadata.json are written atomically (tmp → rename).
    # The ivfdata file is written incrementally by FAISS and is already at its
    # final location; it is protected against a stale-read by the metadata
    # sidecar check (invalid sidecar → full rebuild wipes the old ivfdata).
    final_ivfdata_path = _ivfdata_path(index_base)
    tmp_idx_path       = Path(str(index_base) + ".tmp.idx")
    tmp_meta_path      = Path(str(index_base) + ".tmp_metadata.json")

    # Clean up any previous failed build artefacts
    for p in (tmp_idx_path, tmp_meta_path):
        if p.exists():
            p.unlink()

    if use_ivf:
        # ── IVFFlat ─────────────────────────────────────────────────────────
        # Train on a sample (up to 100 k) to determine cluster centroids.
        # Rule of thumb: need at least 39 × nlist training vectors for good
        # centroids; 100k / 1024 ≈ 97 per centroid — adequate for nlist=1024.
        sample_size = min(100_000, n_records)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n_records, size=sample_size, replace=False)
        sample_texts = [texts[i] for i in sample_idx]

        print(f"    [FAISS] Encoding {sample_size:,} samples for IVF training ...")
        sample_emb = _encode_batch(model, sample_texts, batch_size, show_progress=False)
        faiss.normalize_L2(sample_emb)

        quantizer  = faiss.IndexFlatIP(d)
        index      = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        print(f"    [FAISS] Training IVF index (nlist={nlist}) ...")
        index.train(sample_emb)
        del sample_emb, sample_texts
        gc.collect()

        # Attach on-disk inverted lists BEFORE adding vectors.
        # Use the FINAL target path — FAISS bakes this path into the .idx file.
        # own=True → FAISS C++ layer manages the OnDiskInvertedLists lifetime.
        if final_ivfdata_path.exists():
            final_ivfdata_path.unlink()  # wipe stale data from a prior build
        invlists = faiss.OnDiskInvertedLists(
            index.nlist, index.code_size, str(final_ivfdata_path)
        )
        index.replace_invlists(invlists, True)

        # ── Add vectors in chunks ────────────────────────────────────────────
        print(f"    [FAISS] Adding {n_records:,} vectors in chunks of "
              f"{add_chunk_size:,} ...")
        for start in range(0, n_records, add_chunk_size):
            end        = min(start + add_chunk_size, n_records)
            chunk_txt  = texts[start:end]
            chunk_ids  = np.arange(start, end, dtype=np.int64)
            chunk_emb  = _encode_batch(model, chunk_txt, batch_size,
                                        show_progress=(n_records > 50_000))
            faiss.normalize_L2(chunk_emb)
            index.add_with_ids(chunk_emb, chunk_ids)
            del chunk_emb, chunk_txt
            gc.collect()

        # Write index metadata to tmp path — rename atomically after
        faiss.write_index(index, str(tmp_idx_path))

    else:
        # ── IndexFlatIP (exact, for small datasets / tests) ──────────────────
        print(f"    [FAISS] Corpus size {n_records} < nlist {nlist}; "
              f"using IndexFlatIP (exact search, no ivfdata file) ...")
        index = faiss.IndexFlatIP(d)

        for start in range(0, n_records, add_chunk_size):
            end       = min(start + add_chunk_size, n_records)
            chunk_txt = texts[start:end]
            chunk_emb = _encode_batch(model, chunk_txt, batch_size,
                                       show_progress=False)
            faiss.normalize_L2(chunk_emb)
            index.add(chunk_emb)
            del chunk_emb, chunk_txt
            gc.collect()

        faiss.write_index(index, str(tmp_idx_path))
        # No .ivfdata file for flat index — remove any stale file from a prior IVF build
        if final_ivfdata_path.exists():
            final_ivfdata_path.unlink()

    # ── Write metadata sidecar to tmp path ───────────────────────────────────
    with open(tmp_meta_path, "w", encoding="utf-8") as fh:
        json.dump(expected_meta, fh, indent=2)

    # ── Atomic rename (.idx and _metadata.json only) ─────────────────────────
    # The ivfdata is already at its final path (written by FAISS directly).
    # We rename .idx last so that if the process is killed between writes,
    # the sidecar check will see a missing/stale .idx and rebuild cleanly.
    os.replace(str(tmp_meta_path), str(_metadata_path(index_base)))
    os.replace(str(tmp_idx_path),  str(_index_path(index_base)))

    elapsed = time.time() - t0
    print(f"    [FAISS] Index built in {elapsed:.1f}s  -> {_index_path(index_base)}")


# ---------------------------------------------------------------------------
# Index search
# ---------------------------------------------------------------------------

def search_faiss_candidates(
    s1_df: pd.DataFrame,
    s2s3_df: pd.DataFrame,
    index_base: "Path | str",
    *,
    top_k: int | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    nprobe: int | None = None,
) -> pd.DataFrame:
    """Query the FAISS index and return a candidate DataFrame.

    Parameters
    ----------
    s1_df : pd.DataFrame
        Query entities (source 1).  Must have ``entity_id`` column.
    s2s3_df : pd.DataFrame
        Corpus entities (source 2/3).  Must have ``entity_id`` column.
        Used to map integer FAISS row indices back to entity IDs.
    index_base : Path | str
        Base path for index files (no extension).
    top_k : int | None
        Candidates to return per S1 entity.  Defaults to
        ``pipeline_config.FAISS_TOP_K``.
    model_name : str | None
        Sentence-transformers model.  Must match the model used at build time.
        Defaults to ``pipeline_config.FAISS_EMBEDDING_MODEL``.
    batch_size : int | None
        Query batch size.  Defaults to ``pipeline_config.FAISS_BATCH_SIZE``.
    nprobe : int | None
        IVF cells to probe.  Defaults to metadata value (``pipeline_config.FAISS_NPROBE``).

    Returns
    -------
    pd.DataFrame
        Columns:
            source1_entity_id    : str
            candidate_entity_id  : str
            score                : float   (cosine similarity, higher = more similar)
            rank                 : int     (1-based rank within S1 entity)
            retrieved_by_faiss   : int     (always 1 — provenance flag)
    """
    try:
        from pipeline_config import FAISS_EMBEDDING_MODEL, FAISS_BATCH_SIZE, FAISS_TOP_K, FAISS_NPROBE
    except ImportError:
        FAISS_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
        FAISS_BATCH_SIZE      = 1_000
        FAISS_TOP_K           = 50
        FAISS_NPROBE          = 16

    model_name = model_name or FAISS_EMBEDDING_MODEL
    batch_size = batch_size or FAISS_BATCH_SIZE
    top_k      = top_k      or FAISS_TOP_K
    nprobe     = nprobe     or FAISS_NPROBE

    faiss = _require_faiss()
    SentenceTransformer = _require_sentence_transformers()

    idx_path = _index_path(index_base)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {idx_path}. "
            "Call build_faiss_index() or ensure_faiss_index() first."
        )

    # Load index
    index = faiss.read_index(str(idx_path))
    # Set nprobe if the index supports it (IVF variants)
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe

    s1_ids   = s1_df["entity_id"].tolist()
    s2s3_ids = s2s3_df["entity_id"].tolist()
    n_s1     = len(s1_ids)

    # Load model
    model = SentenceTransformer(model_name)
    s1_texts = _build_embed_texts(s1_df)

    rows: list[dict] = []
    query_chunk = batch_size * 50  # encode up to 50 encode-batches at once

    for q_start in range(0, n_s1, query_chunk):
        q_end    = min(q_start + query_chunk, n_s1)
        q_texts  = s1_texts[q_start:q_end]
        q_emb    = _encode_batch(model, q_texts, batch_size, show_progress=False)
        faiss.normalize_L2(q_emb)

        # Clamp top_k to corpus size (avoids FAISS assertion errors)
        effective_k = min(top_k, index.ntotal)
        if effective_k == 0:
            del q_emb
            gc.collect()
            continue

        scores, indices = index.search(q_emb, effective_k)

        for local_i, (score_row, idx_row) in enumerate(zip(scores, indices)):
            global_i = q_start + local_i
            s1_id    = s1_ids[global_i]
            for rank, (sc, corpus_idx) in enumerate(zip(score_row, idx_row), start=1):
                if corpus_idx < 0 or corpus_idx >= len(s2s3_ids):
                    continue  # FAISS returns -1 for empty cells
                rows.append({
                    "source1_entity_id":   s1_id,
                    "candidate_entity_id": s2s3_ids[corpus_idx],
                    "score":               float(sc),
                    "rank":                rank,
                    "retrieved_by_faiss":  1,
                })

        del q_emb
        gc.collect()

    if not rows:
        return pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "score", "rank", "retrieved_by_faiss",
        ])

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: ensure index then search
# ---------------------------------------------------------------------------

def ensure_faiss_index(
    s2s3_df: pd.DataFrame,
    index_base: "Path | str",
    split_label: str = "CORPUS",
    *,
    model_name: str | None = None,
    batch_size: int | None = None,
    add_chunk_size: int | None = None,
    nlist: int | None = None,
    nprobe: int | None = None,
    force: bool = False,
) -> None:
    """Build the FAISS index if it doesn't exist or is stale; no-op otherwise."""
    build_faiss_index(
        s2s3_df      = s2s3_df,
        index_base   = index_base,
        split_label  = split_label,
        model_name   = model_name,
        batch_size   = batch_size,
        add_chunk_size = add_chunk_size,
        nlist        = nlist,
        nprobe       = nprobe,
        force        = force,
    )


def get_or_build_and_search(
    s2s3_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    index_base: "Path | str",
    split_label: str = "CORPUS",
    *,
    top_k: int | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    add_chunk_size: int | None = None,
    nlist: int | None = None,
    nprobe: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Ensure the FAISS index exists, then return search results.

    This is the primary entry point called by ``run_production.run_blocking``
    when ``--enable-faiss`` is active.

    Returns the same candidate DataFrame as ``search_faiss_candidates``.
    """
    ensure_faiss_index(
        s2s3_df        = s2s3_df,
        index_base     = index_base,
        split_label    = split_label,
        model_name     = model_name,
        batch_size     = batch_size,
        add_chunk_size = add_chunk_size,
        nlist          = nlist,
        nprobe         = nprobe,
        force          = force,
    )
    return search_faiss_candidates(
        s1_df      = s1_df,
        s2s3_df    = s2s3_df,
        index_base = index_base,
        top_k      = top_k,
        model_name = model_name,
        batch_size = batch_size,
        nprobe     = nprobe,
    )
