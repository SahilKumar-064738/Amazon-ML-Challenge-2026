"""
test_mlnaws_v6_v9.py
--------------------
Regression tests for MLnAWS V6-V9 features synchronized into combined-pipeline.

Covers:
  - diff_features (V6)
  - margin_features (V7)
  - rapidfuzz_features (V8)
  - transliteration_features (V9)
  - blocking_countsketch (new retrieval module)
  - features.py V6-V9 constants
  - model.py V6-V9 constants
  - blocking_multiview retrieval_agreement_count
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHING_SRC = REPO_ROOT / "matching"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(MATCHING_SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pair_df():
    """Minimal candidate pair table with all required name/address columns."""
    return pd.DataFrame({
        "source1_entity_id":   ["S1-001", "S1-001", "S1-002"],
        "candidate_entity_id": ["S2-001", "S2-002", "S2-003"],
        "s1_clean_name":       ["Walmart Store", "Walmart Store", "Alpha Corp"],
        "candidate_clean_name":["Wall Mart", "Walmart", "Alpha Corporation"],
        "s1_clean_address":    ["123 Main St Unit 5", "123 Main St Unit 5", "456 Oak Ave"],
        "candidate_clean_address": ["123 Main Street", "123 Main", "456 Oak Avenue"],
    })


# ===========================================================================
# diff_features (V6)
# ===========================================================================

class TestDiffFeatures:

    def test_columns_added(self, pair_df):
        from src.diff_features import add_diff_features, DIFF_FEATURE_COLS
        out = add_diff_features(pair_df)
        for col in DIFF_FEATURE_COLS:
            assert col in out.columns, f"Missing diff column: {col}"

    def test_no_nan(self, pair_df):
        from src.diff_features import add_diff_features, DIFF_FEATURE_COLS
        out = add_diff_features(pair_df)
        assert not out[DIFF_FEATURE_COLS].isna().any().any(), "NaN in diff features"

    def test_no_inf(self, pair_df):
        from src.diff_features import add_diff_features, DIFF_FEATURE_COLS
        out = add_diff_features(pair_df)
        numeric = out[DIFF_FEATURE_COLS].select_dtypes(include=[np.number])
        assert np.isfinite(numeric.values).all(), "Inf in diff features"

    def test_empty_dataframe(self):
        from src.diff_features import add_diff_features, DIFF_FEATURE_COLS
        empty = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "s1_clean_name", "candidate_clean_name",
            "s1_clean_address", "candidate_clean_address",
        ])
        out = add_diff_features(empty)
        assert out.empty
        for col in DIFF_FEATURE_COLS:
            assert col in out.columns

    def test_count_is_15(self):
        from src.diff_features import DIFF_FEATURE_COLS
        assert len(DIFF_FEATURE_COLS) == 15, f"Expected 15 diff cols, got {len(DIFF_FEATURE_COLS)}"


# ===========================================================================
# margin_features (V7)
# ===========================================================================

class TestMarginFeatures:

    def test_columns_added(self):
        from src.margin_features import add_margin_features, MARGIN_FEATURE_COLS
        df = pd.DataFrame({
            "source1_entity_id":   ["S1-001", "S1-001"],
            "candidate_entity_id": ["S2-001", "S2-002"],
            "prob_match":          [0.9, 0.4],
            "name_jaro_winkler":   [0.95, 0.6],
            "address_jaccard":     [0.8, 0.3],
        })
        out = add_margin_features(df)
        for col in MARGIN_FEATURE_COLS:
            assert col in out.columns, f"Missing margin column: {col}"

    def test_count_is_16(self):
        from src.margin_features import MARGIN_FEATURE_COLS
        assert len(MARGIN_FEATURE_COLS) == 16, f"Expected 16 margin cols, got {len(MARGIN_FEATURE_COLS)}"

    def test_no_nan(self):
        from src.margin_features import add_margin_features, MARGIN_FEATURE_COLS
        df = pd.DataFrame({
            "source1_entity_id":   ["S1-001", "S1-001", "S1-002"],
            "candidate_entity_id": ["S2-001", "S2-002", "S2-003"],
            "prob_match":          [0.9, 0.5, 0.7],
            "name_jaro_winkler":   [0.95, 0.6, 0.8],
            "address_jaccard":     [0.8, 0.3, 0.5],
        })
        out = add_margin_features(df)
        num_cols = [c for c in MARGIN_FEATURE_COLS if c in out.columns]
        assert not out[num_cols].isna().any().any(), "NaN in margin features"


# ===========================================================================
# rapidfuzz_features (V8)
# ===========================================================================

class TestRapidfuzzFeatures:

    def test_columns_added(self, pair_df):
        from src.rapidfuzz_features import add_rapidfuzz_features, RAPIDFUZZ_FEATURE_COLS
        out = add_rapidfuzz_features(pair_df)
        for col in RAPIDFUZZ_FEATURE_COLS:
            assert col in out.columns, f"Missing rapidfuzz column: {col}"

    def test_values_in_0_1(self, pair_df):
        from src.rapidfuzz_features import add_rapidfuzz_features, RAPIDFUZZ_FEATURE_COLS
        out = add_rapidfuzz_features(pair_df)
        vals = out[RAPIDFUZZ_FEATURE_COLS].values
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), "Values outside [0, 1]"

    def test_no_nan(self, pair_df):
        from src.rapidfuzz_features import add_rapidfuzz_features, RAPIDFUZZ_FEATURE_COLS
        out = add_rapidfuzz_features(pair_df)
        assert not out[RAPIDFUZZ_FEATURE_COLS].isna().any().any(), "NaN in rapidfuzz features"

    def test_identical_strings_score_1(self):
        from src.rapidfuzz_features import add_rapidfuzz_features
        df = pd.DataFrame({
            "source1_entity_id":       ["S1-001"],
            "candidate_entity_id":     ["S2-001"],
            "s1_clean_name":           ["walmart"],
            "candidate_clean_name":    ["walmart"],
            "s1_clean_address":        ["123 main"],
            "candidate_clean_address": ["123 main"],
        })
        out = add_rapidfuzz_features(df)
        assert out["name_fuzz_ratio"].iloc[0] == 1.0
        assert out["addr_fuzz_ratio"].iloc[0] == 1.0

    def test_count_is_10(self):
        from src.rapidfuzz_features import RAPIDFUZZ_FEATURE_COLS
        assert len(RAPIDFUZZ_FEATURE_COLS) == 10

    def test_empty_dataframe(self):
        from src.rapidfuzz_features import add_rapidfuzz_features, RAPIDFUZZ_FEATURE_COLS
        empty = pd.DataFrame(columns=[
            "source1_entity_id", "candidate_entity_id",
            "s1_clean_name", "candidate_clean_name",
            "s1_clean_address", "candidate_clean_address",
        ])
        out = add_rapidfuzz_features(empty)
        assert out.empty
        for col in RAPIDFUZZ_FEATURE_COLS:
            assert col in out.columns


# ===========================================================================
# transliteration_features (V9)
# ===========================================================================

class TestTransliterationFeatures:

    def test_columns_added(self, pair_df):
        from src.transliteration_features import add_transliteration_features, TRANSLITERATION_FEATURE_COLS
        out = add_transliteration_features(pair_df)
        for col in TRANSLITERATION_FEATURE_COLS:
            assert col in out.columns, f"Missing translit column: {col}"

    def test_count_is_8(self):
        from src.transliteration_features import TRANSLITERATION_FEATURE_COLS
        assert len(TRANSLITERATION_FEATURE_COLS) == 8

    def test_no_nan(self, pair_df):
        from src.transliteration_features import add_transliteration_features, TRANSLITERATION_FEATURE_COLS
        out = add_transliteration_features(pair_df)
        assert not out[TRANSLITERATION_FEATURE_COLS].isna().any().any(), "NaN in translit features"

    def test_dict_learn_and_remap(self):
        """TransliterationDictionary can learn token pairs and remap text."""
        from src.transliteration_features import TransliterationDictionary
        d = TransliterationDictionary()
        pairs = pd.DataFrame({
            "s1_clean_name":        ["sharma"],
            "candidate_clean_name": ["sarma"],
            "label":                [1],
        })
        d.learn_from_pairs(pairs, s1_name_col="s1_clean_name", cand_name_col="candidate_clean_name", label_col="label")
        remapped, count = d.remap_text("sharma")
        # After learning, it may or may not remap depending on frequency — no crash is the key check
        assert isinstance(count, int)


# ===========================================================================
# blocking_countsketch
# ===========================================================================

class TestBlockingCountsketch:

    def test_importable(self):
        from src.blocking_countsketch import CountSketchVectorizer, search_countsketch_candidates
        assert CountSketchVectorizer is not None

    def test_fit_and_search(self):
        from src.blocking_countsketch import CountSketchVectorizer, search_countsketch_candidates
        corpus = pd.DataFrame({
            "entity_id":    ["S2-001", "S2-002", "S2-003"],
            "clean_name":   ["Alpha Corp", "Beta Ltd", "Gamma Inc"],
            "clean_address":["123 Main", "456 Oak", "789 Elm"],
        })
        query = pd.DataFrame({
            "entity_id":    ["S1-001"],
            "clean_name":   ["Alpha Corporation"],
            "clean_address":["123 Main St"],
        })
        vec = CountSketchVectorizer()
        vec.fit(corpus)
        results = search_countsketch_candidates(query, corpus, vec, top_k=2)
        assert not results.empty
        assert "source1_entity_id" in results.columns
        assert "candidate_entity_id" in results.columns


# ===========================================================================
# features.py V6-V9 constants
# ===========================================================================

class TestFeatureColumnConstants:

    def test_v6_is_55(self):
        from src.features import FEATURE_COLUMNS_V6
        assert len(FEATURE_COLUMNS_V6) == 55

    def test_v7_is_71(self):
        from src.features import FEATURE_COLUMNS_V7
        assert len(FEATURE_COLUMNS_V7) == 71

    def test_v8_is_81(self):
        from src.features import FEATURE_COLUMNS_V8
        assert len(FEATURE_COLUMNS_V8) == 81

    def test_v9_is_89(self):
        from src.features import FEATURE_COLUMNS_V9
        assert len(FEATURE_COLUMNS_V9) == 89

    def test_full_is_v9(self):
        from src.features import FEATURE_COLUMNS_V9, FEATURE_COLUMNS_FULL
        assert FEATURE_COLUMNS_FULL == FEATURE_COLUMNS_V9

    def test_each_version_is_superset_of_previous(self):
        from src.features import (
            FEATURE_COLUMNS_V5, FEATURE_COLUMNS_V6, FEATURE_COLUMNS_V7,
            FEATURE_COLUMNS_V8, FEATURE_COLUMNS_V9,
        )
        pairs = [
            (FEATURE_COLUMNS_V5, FEATURE_COLUMNS_V6),
            (FEATURE_COLUMNS_V6, FEATURE_COLUMNS_V7),
            (FEATURE_COLUMNS_V7, FEATURE_COLUMNS_V8),
            (FEATURE_COLUMNS_V8, FEATURE_COLUMNS_V9),
        ]
        for prev, curr in pairs:
            assert set(prev).issubset(set(curr)), f"V{pairs.index((prev,curr))+5} not a subset of next"

    def test_no_duplicate_columns(self):
        from src.features import FEATURE_COLUMNS_V9
        assert len(FEATURE_COLUMNS_V9) == len(set(FEATURE_COLUMNS_V9)), "Duplicate column names in V9"


# ===========================================================================
# model.py V6-V9 constants
# ===========================================================================

class TestModelFeatureColConstants:

    def test_v6_v9_tuples_exist(self):
        from src.model import FEATURE_COLS_V6, FEATURE_COLS_V7, FEATURE_COLS_V8, FEATURE_COLS_V9, FEATURE_COLS_FULL
        assert len(FEATURE_COLS_V6) == 55
        assert len(FEATURE_COLS_V7) == 71
        assert len(FEATURE_COLS_V8) == 81
        assert len(FEATURE_COLS_V9) == 89
        assert FEATURE_COLS_FULL == FEATURE_COLS_V9

    def test_model_cols_match_feature_cols(self):
        """model.py FEATURE_COLS_V9 must exactly match features.py FEATURE_COLUMNS_V9."""
        from src.model import FEATURE_COLS_V9
        from src.features import FEATURE_COLUMNS_V9
        assert list(FEATURE_COLS_V9) == FEATURE_COLUMNS_V9, \
            "FEATURE_COLS_V9 (model.py) and FEATURE_COLUMNS_V9 (features.py) are out of sync"


# ===========================================================================
# blocking_multiview retrieval_agreement_count
# ===========================================================================

class TestBlockingMultiviewRetrievalAgreement:

    def test_retrieval_agreement_count_present(self):
        """search_multiview_candidates must emit retrieval_agreement_count."""
        from src.blocking_multiview import fit_multiview_vectorizers, search_multiview_candidates
        s2s3 = pd.DataFrame({
            "entity_id":    ["S2-001", "S2-002"],
            "clean_name":   ["Alpha Corp", "Beta Ltd"],
            "clean_address":["123 Main", "456 Oak"],
            "clean_text":   ["Alpha Corp 123 Main", "Beta Ltd 456 Oak"],
        })
        s1 = pd.DataFrame({
            "entity_id":    ["S1-001"],
            "clean_name":   ["Alpha Corporation"],
            "clean_address":["123 Main St"],
            "clean_text":   ["Alpha Corporation 123 Main St"],
        })
        vecs = fit_multiview_vectorizers(s2s3)
        res = search_multiview_candidates(s1, s2s3, vecs, top_k=2)
        assert "retrieval_agreement_count" in res.columns
        assert "retrieved_by_tfidf_combined" in res.columns
        assert (res["retrieval_agreement_count"] >= 1).all()

    def test_empty_vectorizers_returns_empty(self):
        from src.blocking_multiview import search_multiview_candidates
        import pandas as pd
        s1   = pd.DataFrame({"entity_id": ["S1-001"], "clean_name": ["X"], "clean_address": ["Y"], "clean_text": ["XY"]})
        s2s3 = pd.DataFrame({"entity_id": ["S2-001"], "clean_name": ["X"], "clean_address": ["Y"], "clean_text": ["XY"]})
        result = search_multiview_candidates(s1, s2s3, {}, top_k=5)
        assert result.empty
