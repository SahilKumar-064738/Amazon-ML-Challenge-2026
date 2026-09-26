"""
Test raw dataset schema validation against the small fixture dataset.
"""
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "small_dataset"

from pipeline_validation import validate_raw_dataset


class TestRawDatasetValidation:
    def test_fixture_dataset_passes(self):
        """The small fixture dataset should pass raw validation."""
        result = validate_raw_dataset(FIXTURE_DIR)
        assert result["status"] == "ok"
        assert "train" in result["splits"]
        assert "test" in result["splits"]

    def test_missing_directory_raises(self):
        import pytest
        with pytest.raises(ValueError, match="not found"):
            validate_raw_dataset(Path("/nonexistent/path/to/data"))

    def test_row_counts_are_positive(self):
        result = validate_raw_dataset(FIXTURE_DIR)
        for split, files in result["splits"].items():
            for filename, count in files.items():
                assert count > 0, f"{split}/{filename} has 0 rows"

    def test_train_has_ground_truth(self):
        result = validate_raw_dataset(FIXTURE_DIR, splits=["train"])
        train_files = result["splits"]["train"]
        gt_files = [k for k in train_files if "ground_truth" in k]
        assert len(gt_files) == 1, "Expected exactly one ground truth file"
