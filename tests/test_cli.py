"""
CLI tests for run_pipeline.py

Tests argument parsing and --dry-run behavior.
Does NOT run the full pipeline.
"""
import sys
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_PIPELINE = REPO_ROOT / "run_pipeline.py"


class TestCLI:
    def test_help_exits_zero(self):
        """--help must exit 0 and print usage."""
        result = subprocess.run(
            [sys.executable, str(RUN_PIPELINE), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout

    def test_conflicting_skip_flags_error(self):
        """--skip-preprocessing and --skip-matching together must fail."""
        result = subprocess.run(
            [sys.executable, str(RUN_PIPELINE),
             "--skip-preprocessing", "--skip-matching"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0

    def test_dry_run_on_fixture(self, tmp_path):
        """--dry-run on small fixture must validate and exit 0."""
        fixture_dir = REPO_ROOT / "tests" / "fixtures" / "small_dataset"
        result = subprocess.run(
            [sys.executable, str(RUN_PIPELINE),
             "--input-dir", str(fixture_dir),
             "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should succeed (fixture passes validation)
        assert result.returncode == 0, (
            f"--dry-run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_dry_run_on_missing_data_fails(self, tmp_path):
        """--dry-run on a directory with no data must exit non-zero."""
        result = subprocess.run(
            [sys.executable, str(RUN_PIPELINE),
             "--input-dir", str(tmp_path),
             "--dry-run"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
