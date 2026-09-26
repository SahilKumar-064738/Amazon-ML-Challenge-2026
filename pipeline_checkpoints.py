"""
pipeline_checkpoints.py
-----------------------
Lightweight checkpoint management for the combined pipeline.

A checkpoint is a sentinel file whose presence indicates a stage completed
successfully. Stages are not marked complete until their outputs have been
validated.

Usage:
    from pipeline_checkpoints import Checkpoints, CheckpointStore

    store = CheckpointStore(REPO_ROOT / "checkpoints")
    if not store.is_done(store.PREPROCESSING):
        run_preprocessing()
        store.mark_done(store.PREPROCESSING)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CheckpointStore:
    """Manages sentinel files in a checkpoints directory."""

    # Sentinel file names (stage identifiers)
    PREPROCESSING       = "stage_1_preprocessing_done"
    BRIDGE              = "stage_2_bridge_done"
    CONTRACT_VALIDATED  = "stage_3_contract_validated"
    MATCHING_VECTORIZER = "stage_4a_vectorizer_fitted"
    MATCHING_TRAIN      = "stage_4b_matching_train_done"
    MATCHING_COMPLETE   = "stage_4c_matching_complete"
    FINAL_VALIDATED     = "stage_5_final_outputs_validated"

    ALL_STAGES = [
        PREPROCESSING, BRIDGE, CONTRACT_VALIDATED,
        MATCHING_VECTORIZER, MATCHING_TRAIN, MATCHING_COMPLETE, FINAL_VALIDATED,
    ]

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, stage: str) -> Path:
        return self.checkpoint_dir / stage

    def is_done(self, stage: str, force: bool = False) -> bool:
        """Return True if stage is complete and --force is not set."""
        if force:
            return False
        done = self._path(stage).exists()
        if done:
            logger.info("  [SKIP] Stage '%s' already complete (checkpoint found)", stage)
        return done

    def mark_done(self, stage: str, meta: Optional[str] = None) -> None:
        """Write the checkpoint sentinel file for a stage."""
        p = self._path(stage)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(f"completed_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if meta:
                fh.write(f"meta={meta}\n")
        logger.info("  [DONE] Stage '%s' checkpoint written: %s", stage, p.name)

    def clear(self, stage: str) -> None:
        """Remove the sentinel file for a stage (force re-run)."""
        p = self._path(stage)
        if p.exists():
            p.unlink()
            logger.info("  [CLEARED] Checkpoint '%s' removed", stage)

    def clear_all(self) -> None:
        """Remove ALL checkpoint sentinel files (full re-run)."""
        for stage in self.ALL_STAGES:
            self.clear(stage)
        # Also clear the matching-internal checkpoints if present
        matching_dir = self.checkpoint_dir / "matching"
        if matching_dir.exists():
            for f in matching_dir.iterdir():
                if f.is_file():
                    f.unlink()
        logger.info("  [CLEARED] All checkpoints removed")

    def list_completed(self) -> list[str]:
        """Return list of completed stage names."""
        return [s for s in self.ALL_STAGES if self._path(s).exists()]

    def resume_summary(self) -> str:
        """Return a human-readable summary of checkpoint state."""
        completed = self.list_completed()
        lines = ["Checkpoint status:"]
        for stage in self.ALL_STAGES:
            symbol = "✓" if stage in completed else "○"
            lines.append(f"  {symbol} {stage}")
        return "\n".join(lines)
