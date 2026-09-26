"""
Checkpoint / resume system tests.
"""
import sys
from pathlib import Path
import pytest
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline_checkpoints import CheckpointStore


class TestCheckpointStore:
    def test_new_stage_not_done(self, tmp_path):
        store = CheckpointStore(tmp_path)
        assert not store.is_done(CheckpointStore.PREPROCESSING)

    def test_mark_done_creates_sentinel(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.mark_done(CheckpointStore.PREPROCESSING)
        assert (tmp_path / CheckpointStore.PREPROCESSING).exists()

    def test_is_done_after_mark(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.mark_done(CheckpointStore.PREPROCESSING)
        assert store.is_done(CheckpointStore.PREPROCESSING)

    def test_force_ignores_checkpoint(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.mark_done(CheckpointStore.PREPROCESSING)
        # With force=True, is_done returns False even if sentinel exists
        assert not store.is_done(CheckpointStore.PREPROCESSING, force=True)

    def test_clear_removes_sentinel(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.mark_done(CheckpointStore.PREPROCESSING)
        store.clear(CheckpointStore.PREPROCESSING)
        assert not store.is_done(CheckpointStore.PREPROCESSING)

    def test_clear_all_removes_all(self, tmp_path):
        store = CheckpointStore(tmp_path)
        for stage in [CheckpointStore.PREPROCESSING, CheckpointStore.BRIDGE,
                      CheckpointStore.CONTRACT_VALIDATED]:
            store.mark_done(stage)
        store.clear_all()
        assert store.list_completed() == []

    def test_list_completed(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.mark_done(CheckpointStore.PREPROCESSING)
        store.mark_done(CheckpointStore.BRIDGE)
        completed = store.list_completed()
        assert CheckpointStore.PREPROCESSING in completed
        assert CheckpointStore.BRIDGE in completed
        assert CheckpointStore.CONTRACT_VALIDATED not in completed

    def test_resume_summary_string(self, tmp_path):
        store = CheckpointStore(tmp_path)
        store.mark_done(CheckpointStore.PREPROCESSING)
        summary = store.resume_summary()
        assert "✓" in summary
        assert "○" in summary
