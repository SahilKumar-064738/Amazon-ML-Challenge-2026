"""
Test the preprocessing -> MLnAWS contract (bridge layer).

Runs the full preprocessing + bridge on the small fixture dataset and
verifies that the bridge output satisfies the MLnAWS input contract.
"""
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "small_dataset"


def _run_preprocessing_on_fixture(tmpdir: Path) -> Path:
    """Run MLnNor preprocessing on the fixture dataset; return preprocessed dir."""
    from preprocessing.src.preprocess import run as preprocess_run

    preprocessed_dir = tmpdir / "preprocessed"
    metadata_dir     = tmpdir / "metadata"

    preprocess_run(
        input_dir=FIXTURE_DIR,
        output_processed_dir=preprocessed_dir,
        output_metadata_dir=metadata_dir,
        splits=["train", "test"],
    )
    return preprocessed_dir


class TestBridgeContract:
    @pytest.fixture(scope="class")
    def bridge_outputs(self, tmp_path_factory):
        """Run full preprocessing + bridge once; share outputs across tests."""
        tmpdir = tmp_path_factory.mktemp("bridge_test")
        preprocessed_dir = _run_preprocessing_on_fixture(tmpdir)

        from preprocessing.bridge import run_bridge
        real_dir = tmpdir / "real"
        run_bridge(
            preprocessed_dir=preprocessed_dir,
            real_dir=real_dir,
            splits=["train", "test"],
        )
        return real_dir

    def test_all_required_files_exist(self, bridge_outputs):
        """All 9 expected files must exist in dataset/real/."""
        expected_files = [
            "clean_s1_train.tsv",
            "clean_s2s3_train.tsv",
            "feature_s1_train.tsv",
            "feature_s2s3_train.tsv",
            "ground_truth_train.tsv",
            "clean_s1_test.tsv",
            "clean_s2s3_test.tsv",
            "feature_s1_test.tsv",
            "feature_s2s3_test.tsv",
        ]
        for fname in expected_files:
            path = bridge_outputs / fname
            assert path.exists(), f"Missing: {fname}"
            assert path.stat().st_size > 0, f"Empty: {fname}"

    def test_clean_file_schema(self, bridge_outputs):
        """Clean files must have entity_id and clean_text columns."""
        for fname in ["clean_s1_train.tsv", "clean_s2s3_train.tsv",
                      "clean_s1_test.tsv", "clean_s2s3_test.tsv"]:
            df = pd.read_csv(bridge_outputs / fname, sep="\t", dtype=str,
                             keep_default_na=False, nrows=3)
            assert list(df.columns) == ["entity_id", "clean_text"], \
                f"{fname} has wrong columns: {list(df.columns)}"

    def test_feature_file_schema(self, bridge_outputs):
        """Feature files must have entity_id, clean_name, clean_address, clean_country."""
        expected_cols = ["entity_id", "clean_name", "clean_address", "clean_country"]
        for fname in ["feature_s1_train.tsv", "feature_s2s3_train.tsv",
                      "feature_s1_test.tsv", "feature_s2s3_test.tsv"]:
            df = pd.read_csv(bridge_outputs / fname, sep="\t", dtype=str,
                             keep_default_na=False, nrows=3)
            assert list(df.columns) == expected_cols, \
                f"{fname} has wrong columns: {list(df.columns)}"

    def test_ground_truth_schema(self, bridge_outputs):
        """Ground truth must have source1_entity_id and matching_entity_ids (not matched)."""
        df = pd.read_csv(bridge_outputs / "ground_truth_train.tsv",
                         sep="\t", dtype=str, keep_default_na=False, nrows=3)
        assert "matching_entity_ids" in df.columns, \
            "ground_truth must use 'matching_entity_ids' column"
        assert "matched_entity_ids" not in df.columns, \
            "ground_truth must NOT have 'matched_entity_ids' column"

    def test_row_count_preserved(self, bridge_outputs):
        """S1 clean row count must equal S1 feature row count (same source data)."""
        for split in ("train", "test"):
            clean_df = pd.read_csv(bridge_outputs / f"clean_s1_{split}.tsv",
                                   sep="\t", dtype=str, keep_default_na=False)
            feat_df  = pd.read_csv(bridge_outputs / f"feature_s1_{split}.tsv",
                                   sep="\t", dtype=str, keep_default_na=False)
            assert len(clean_df) == len(feat_df), \
                f"S1 {split}: clean has {len(clean_df)} rows, feature has {len(feat_df)}"

    def test_s2s3_combined_is_sum_of_parts(self, bridge_outputs):
        """S2+S3 combined should have S2_rows + S3_rows total."""
        # We know fixture has 7 S2 train + 3 S3 train = 10 rows
        combined = pd.read_csv(bridge_outputs / "clean_s2s3_train.tsv",
                               sep="\t", dtype=str, keep_default_na=False)
        assert len(combined) == 10, f"Expected 10 S2+S3 train rows, got {len(combined)}"

    def test_clean_text_not_empty_for_valid_records(self, bridge_outputs):
        """Non-missing names/addresses should produce non-empty clean_text."""
        df = pd.read_csv(bridge_outputs / "clean_s1_train.tsv",
                         sep="\t", dtype=str, keep_default_na=False)
        empty_text = (df["clean_text"].str.strip() == "").sum()
        # Fixture has no missing names so all should have clean_text
        assert empty_text == 0, f"{empty_text} rows have empty clean_text"

    def test_entity_ids_preserved(self, bridge_outputs):
        """Entity IDs from preprocessing must appear unchanged in bridge output."""
        feat = pd.read_csv(bridge_outputs / "feature_s1_train.tsv",
                           sep="\t", dtype=str, keep_default_na=False)
        ids = set(feat["entity_id"])
        assert "S1-001" in ids
        assert "S1-005" in ids
        assert len(ids) == 5  # fixture has 5 S1 train rows

    def test_contract_validation_passes(self, bridge_outputs):
        """validate_matching_inputs() should pass on bridge outputs."""
        from pipeline_validation import validate_matching_inputs
        result = validate_matching_inputs(bridge_outputs)
        assert result["status"] == "ok"
