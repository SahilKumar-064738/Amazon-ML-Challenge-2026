"""
Integration test: runs preprocessing on the small fixture dataset and
validates all outputs are correct.

Does NOT require the full 2.5 GB dataset.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "small_dataset"

from preprocessing.src.preprocess import run as preprocess_run
from preprocessing.src.config import REQUIRED_OUTPUT_COLUMNS, GT_OUTPUT_COLUMNS


class TestPreprocessingIntegration:
    @pytest.fixture(scope="class")
    def preprocess_outputs(self, tmp_path_factory):
        """Run preprocessing once; share outputs across all tests in this class."""
        tmpdir = tmp_path_factory.mktemp("preprocess_integ")
        preprocessed_dir = tmpdir / "processed"
        metadata_dir     = tmpdir / "metadata"

        preprocess_run(
            input_dir=FIXTURE_DIR,
            output_processed_dir=preprocessed_dir,
            output_metadata_dir=metadata_dir,
            splits=["train", "test"],
        )
        return {"processed": preprocessed_dir, "metadata": metadata_dir}

    def test_all_output_files_exist(self, preprocess_outputs):
        pd_dir = preprocess_outputs["processed"]
        for split in ("train", "test"):
            for src in ("source1_clean.tsv", "source2_clean.tsv", "source3_clean.tsv"):
                assert (pd_dir / split / src).exists(), f"Missing: {split}/{src}"
        assert (pd_dir / "train" / "ground_truth_clean.tsv").exists()

    def test_output_schema_correct(self, preprocess_outputs):
        pd_dir = preprocess_outputs["processed"]
        df = pd.read_csv(pd_dir / "train" / "source1_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False, nrows=1)
        assert list(df.columns) == REQUIRED_OUTPUT_COLUMNS

    def test_row_count_preserved(self, preprocess_outputs):
        """Preprocessing must not drop or add rows."""
        pd_dir = preprocess_outputs["processed"]
        # Fixture train/source1 has 5 rows
        df = pd.read_csv(pd_dir / "train" / "source1_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        assert len(df) == 5, f"Expected 5 rows, got {len(df)}"

    def test_entity_ids_preserved(self, preprocess_outputs):
        pd_dir = preprocess_outputs["processed"]
        df = pd.read_csv(pd_dir / "train" / "source1_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        ids = set(df["entity_id"])
        assert ids == {"S1-001", "S1-002", "S1-003", "S1-004", "S1-005"}

    def test_no_duplicate_entity_ids(self, preprocess_outputs):
        pd_dir = preprocess_outputs["processed"]
        for split in ("train", "test"):
            for src in ("source1_clean.tsv", "source2_clean.tsv", "source3_clean.tsv"):
                df = pd.read_csv(pd_dir / split / src,
                                 sep="\t", dtype=str, keep_default_na=False)
                dupes = df["entity_id"].duplicated().sum()
                assert dupes == 0, f"{split}/{src} has {dupes} duplicate entity_ids"

    def test_ground_truth_schema(self, preprocess_outputs):
        pd_dir = preprocess_outputs["processed"]
        df = pd.read_csv(pd_dir / "train" / "ground_truth_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False, nrows=1)
        assert list(df.columns) == GT_OUTPUT_COLUMNS

    def test_ground_truth_row_count(self, preprocess_outputs):
        """Ground truth must have exactly 5 rows (one per S1 entity)."""
        pd_dir = preprocess_outputs["processed"]
        df = pd.read_csv(pd_dir / "train" / "ground_truth_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        assert len(df) == 5, f"Expected 5 GT rows, got {len(df)}"

    def test_normalization_applied(self, preprocess_outputs):
        """Normalized columns should differ from raw (e.g., lowercase applied)."""
        pd_dir = preprocess_outputs["processed"]
        df = pd.read_csv(pd_dir / "train" / "source1_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        # Normalized names should be lowercase
        for name in df["business_name_normalized"]:
            if name:  # skip missing
                assert name == name.lower(), f"Not lowercased: {name}"

    def test_country_normalized(self, preprocess_outputs):
        """country_normalized should be lowercase."""
        pd_dir = preprocess_outputs["processed"]
        df = pd.read_csv(pd_dir / "train" / "source1_clean.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        for country in df["country_normalized"]:
            if country:
                assert country == country.lower(), f"Not lowercased: {country}"

    def test_preprocessing_report_written(self, preprocess_outputs):
        """Preprocessing report JSON must be written."""
        md_dir = preprocess_outputs["metadata"]
        assert (md_dir / "preprocessing_report.json").exists()

    def test_no_test_ground_truth(self, preprocess_outputs):
        """Test split must NOT have a ground_truth_clean.tsv."""
        pd_dir = preprocess_outputs["processed"]
        assert not (pd_dir / "test" / "ground_truth_clean.tsv").exists()
