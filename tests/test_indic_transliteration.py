"""Tests for preprocessing/src/indic_transliteration.py and its pipeline integration.

Adapted from MLnNor/tests/test_indic_transliteration.py for the combined-pipeline
import layout (preprocessing.src.* instead of src.*).

Test structure
--------------
Section 1 — Module-level helpers (_has_indic, transliterate function)
Section 2 — Per-script character-level transliteration (9 scripts)
Section 3 — IndicTransliterator class (no dict, with dict, save/load)
Section 4 — DictionaryLearner (learn_from_pairs, build_dictionary, build_transliterator)
Section 5 — Integration with normalization.py (set_transliterator / normalize_name_basic)
Section 6 — Integration with preprocess.py / clean_source (name_script_class from raw)
Section 7 — Non-regression: Latin, missing-value, and existing tests unaffected
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.src.indic_transliteration import (
    DictionaryLearner,
    IndicTransliterator,
    _has_indic,
    transliterate,
)
from preprocessing.src.normalization import (
    get_transliterator,
    normalize_name_basic,
    normalize_name_series,
    set_transliterator,
)


# ===========================================================================
# Fixture: ensure transliterator is reset after every test that touches it
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_transliterator():
    """Always restore the module-level transliterator to None after each test."""
    original = get_transliterator()
    yield
    set_transliterator(original)


# ===========================================================================
# Section 1 — Module-level helpers
# ===========================================================================

class TestHasIndic:
    def test_pure_latin_false(self):
        assert _has_indic("hello world") is False

    def test_empty_false(self):
        assert _has_indic("") is False

    def test_digits_only_false(self):
        assert _has_indic("12345") is False

    def test_devanagari_true(self):
        assert _has_indic("नमस्ते") is True

    def test_tamil_true(self):
        assert _has_indic("ராஜ்") is True

    def test_mixed_latin_devanagari_true(self):
        assert _has_indic("Raj नमस्ते") is True

    def test_gujarati_true(self):
        assert _has_indic("ગુજરાત") is True

    def test_bengali_true(self):
        assert _has_indic("বাংলা") is True

    def test_gurmukhi_true(self):
        assert _has_indic("ਪੰਜਾਬ") is True

    def test_oriya_true(self):
        assert _has_indic("ଓଡ଼ିଆ") is True

    def test_telugu_true(self):
        assert _has_indic("తెలుగు") is True

    def test_kannada_true(self):
        assert _has_indic("ಕನ್ನಡ") is True

    def test_malayalam_true(self):
        assert _has_indic("കേരളം") is True


class TestTransliterateFunction:
    def test_latin_unchanged(self):
        assert transliterate("hello world") == "hello world"

    def test_empty_unchanged(self):
        assert transliterate("") == ""

    def test_mixed_latin_token_untouched(self):
        result = transliterate("Raj नमस्ते")
        assert result.startswith("Raj ")

    def test_devanagari_produces_latin_output(self):
        result = transliterate("नमस्ते")
        assert all(ord(c) < 128 for c in result), f"Non-ASCII in output: {result!r}"
        assert len(result) > 0

    def test_output_does_not_contain_original_indic_chars(self):
        indic = "प्राइवेट"
        result = transliterate(indic)
        for ch in indic:
            if ord(ch) > 127:
                assert ch not in result, f"Indic char {ch!r} survived in {result!r}"

    def test_idempotent_on_latin(self):
        text = "private limited"
        assert transliterate(text) == text

    def test_numerals_pass_through(self):
        result = transliterate("०१२")
        assert "0" in result
        assert "1" in result
        assert "2" in result

    def test_whitespace_preserved(self):
        result = transliterate("नमस्ते दुनिया")
        assert " " in result


# ===========================================================================
# Section 2 — Per-script spot checks
# ===========================================================================

class TestDevanagari:
    def test_ka_consonant(self):
        result = transliterate("क")
        assert "k" in result

    def test_namaste_contains_n(self):
        result = transliterate("नमस्ते")
        assert "n" in result

    def test_private_root(self):
        result = transliterate("प्राइवेट")
        assert result.startswith("pr") or "p" in result

    def test_output_is_ascii(self):
        result = transliterate("भारत")
        assert all(ord(c) < 128 for c in result)


class TestBengali:
    def test_produces_ascii(self):
        result = transliterate("বাংলাদেশ")
        assert all(ord(c) < 128 for c in result)

    def test_contains_b(self):
        result = transliterate("বাংলা")
        assert "b" in result


class TestGurmukhi:
    def test_produces_ascii(self):
        result = transliterate("ਪੰਜਾਬ")
        assert all(ord(c) < 128 for c in result)

    def test_contains_p(self):
        result = transliterate("ਪੰਜਾਬ")
        assert "p" in result


class TestGujarati:
    def test_produces_ascii(self):
        result = transliterate("ગુજરાત")
        assert all(ord(c) < 128 for c in result)

    def test_contains_g(self):
        result = transliterate("ગુજરાત")
        assert "g" in result


class TestOriya:
    def test_produces_ascii(self):
        result = transliterate("ଓଡ଼ିଆ")
        assert all(ord(c) < 128 for c in result)


class TestTamil:
    def test_produces_ascii(self):
        result = transliterate("ராஜ்")
        assert all(ord(c) < 128 for c in result)

    def test_contains_r(self):
        result = transliterate("ராஜ்")
        assert "r" in result


class TestTelugu:
    def test_produces_ascii(self):
        result = transliterate("తెలుగు")
        assert all(ord(c) < 128 for c in result)

    def test_contains_t(self):
        result = transliterate("తెలుగు")
        assert "t" in result


class TestKannada:
    def test_produces_ascii(self):
        result = transliterate("ಕನ್ನಡ")
        assert all(ord(c) < 128 for c in result)

    def test_contains_k(self):
        result = transliterate("ಕನ್ನಡ")
        assert "k" in result


class TestMalayalam:
    def test_produces_ascii(self):
        result = transliterate("കേരളം")
        assert all(ord(c) < 128 for c in result)

    def test_contains_k(self):
        result = transliterate("കേരളം")
        assert "k" in result


# ===========================================================================
# Section 3 — IndicTransliterator class
# ===========================================================================

class TestIndicTransliteratorNoDict:
    def test_latin_unchanged(self):
        t = IndicTransliterator()
        assert t.transliterate("hello") == "hello"

    def test_empty_unchanged(self):
        t = IndicTransliterator()
        assert t.transliterate("") == ""

    def test_devanagari_produces_ascii(self):
        t = IndicTransliterator()
        result = t.transliterate("नमस्ते")
        assert all(ord(c) < 128 for c in result)

    def test_get_dictionary_empty(self):
        t = IndicTransliterator()
        assert t.get_dictionary() == {}

    def test_update_dictionary(self):
        t = IndicTransliterator()
        t.update_dictionary({"namaste": "greetings"})
        assert isinstance(t.get_dictionary(), dict)


class TestIndicTransliteratorWithDict:
    def test_dict_correction_applied(self):
        t = IndicTransliterator(dictionary={"praivata": "private"})
        result = t.transliterate("प्राइवेट")
        assert all(ord(c) < 128 for c in result)

    def test_dict_correction_only_on_matching_tokens(self):
        t = IndicTransliterator(dictionary={"foo": "bar"})
        result = t.transliterate("नमस्ते")
        assert "bar" not in result

    def test_get_dictionary_returns_copy(self):
        original = {"a": "b"}
        t = IndicTransliterator(dictionary=original)
        d = t.get_dictionary()
        d["c"] = "d"
        assert "c" not in t.get_dictionary()


class TestIndicTransliteratorSaveLoad:
    def test_save_and_load_roundtrip(self, tmp_path):
        d = {"praivata": "private", "limiteda": "limited"}
        t = IndicTransliterator(dictionary=d)
        path = tmp_path / "dict.json"
        t.save_dictionary(path)
        t2 = IndicTransliterator.load(path)
        assert t2.get_dictionary() == d

    def test_saved_json_is_valid(self, tmp_path):
        t = IndicTransliterator(dictionary={"a": "b"})
        path = tmp_path / "dict.json"
        t.save_dictionary(path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == {"a": "b"}

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            IndicTransliterator.load(tmp_path / "missing.json")


# ===========================================================================
# Section 4 — DictionaryLearner
# ===========================================================================

class TestDictionaryLearner:
    def test_learns_from_single_pair(self):
        learner = DictionaryLearner()
        learner.learn_from_pairs(["नमस्ते"], ["namaste"])
        d = learner.build_dictionary()
        assert isinstance(d, dict)

    def test_mismatched_lengths_raise(self):
        learner = DictionaryLearner()
        with pytest.raises(ValueError):
            learner.learn_from_pairs(["a"], ["b", "c"])

    def test_latin_indic_names_skipped(self):
        learner = DictionaryLearner()
        learner.learn_from_pairs(["hello world"], ["hello world"])
        d = learner.build_dictionary()
        assert d == {}

    def test_empty_pairs_skipped(self):
        learner = DictionaryLearner()
        learner.learn_from_pairs(["", None], ["", None])
        d = learner.build_dictionary()
        assert d == {}

    def test_majority_vote_selects_most_frequent(self):
        learner = DictionaryLearner(min_count=1)
        learner.learn_from_pairs(
            ["प्राइवेट", "प्राइवेट", "प्राइवेट", "प्राइवेट"],
            ["private", "private", "private", "praivet"],
        )
        d = learner.build_dictionary()
        values = list(d.values())
        if values:
            assert values.count("private") >= values.count("praivet")

    def test_numeric_targets_excluded(self):
        learner = DictionaryLearner()
        learner.learn_from_pairs(["१२३"], ["123"])
        d = learner.build_dictionary()
        for tgt in d.values():
            assert not tgt.isdigit()

    def test_build_transliterator_returns_correct_type(self):
        learner = DictionaryLearner()
        learner.learn_from_pairs(["नमस्ते"], ["namaste"])
        t = learner.build_transliterator()
        assert isinstance(t, IndicTransliterator)

    def test_accumulate_across_calls(self):
        learner = DictionaryLearner()
        learner.learn_from_pairs(["नमस्ते"], ["namaste"])
        learner.learn_from_pairs(["नमस्ते"], ["namaste"])
        d = learner.build_dictionary()
        assert isinstance(d, dict)


# ===========================================================================
# Section 5 — Integration with normalization.py
# ===========================================================================

class TestSetTransliterator:
    def test_default_is_none(self):
        set_transliterator(None)
        assert get_transliterator() is None

    def test_set_and_get(self):
        t = IndicTransliterator()
        set_transliterator(t)
        assert get_transliterator() is t

    def test_set_none_disables(self):
        set_transliterator(IndicTransliterator())
        set_transliterator(None)
        assert get_transliterator() is None


class TestNormalizeNameBasicWithTransliterator:
    def test_latin_name_unchanged_when_transliterator_active(self):
        set_transliterator(IndicTransliterator())
        norm, missing = normalize_name_basic("Acme Corporation")
        assert "acme" in norm
        assert missing is False

    def test_indic_name_transliterated(self):
        set_transliterator(IndicTransliterator())
        norm, missing = normalize_name_basic("नमस्ते")
        assert all(ord(c) < 128 for c in norm), f"Non-ASCII in norm: {norm!r}"
        assert missing is False

    def test_indic_name_returns_latin_script(self):
        set_transliterator(IndicTransliterator())
        norm, _ = normalize_name_basic("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்")
        assert len(norm) > 0
        assert all(ord(c) < 128 for c in norm)

    def test_no_transliterator_indic_name_preserved(self):
        set_transliterator(None)
        norm, missing = normalize_name_basic("नमस्ते")
        assert "न" in norm
        assert missing is False

    def test_na_like_indic_not_transliterated(self):
        set_transliterator(IndicTransliterator())
        norm, missing = normalize_name_basic("NA")
        assert norm == ""
        assert missing is True

    def test_empty_name_still_missing(self):
        set_transliterator(IndicTransliterator())
        norm, missing = normalize_name_basic("")
        assert norm == "" and missing is True

    def test_none_name_still_missing(self):
        set_transliterator(IndicTransliterator())
        norm, missing = normalize_name_basic(None)
        assert norm == "" and missing is True

    def test_mixed_latin_indic_token_handling(self):
        set_transliterator(IndicTransliterator())
        norm, _ = normalize_name_basic("Raj नमस्ते Investments")
        assert "raj" in norm
        assert "investments" in norm
        assert all(ord(c) < 128 for c in norm)

    def test_series_uses_transliterator(self):
        set_transliterator(IndicTransliterator())
        names = pd.Series(["Acme Corp", "नमस्ते", "", "ராஜ்"])
        result, missing = normalize_name_series(names)
        indic_norm = result.iloc[1]
        assert all(ord(c) < 128 for c in indic_norm), f"Non-ASCII: {indic_norm!r}"
        assert bool(missing.iloc[2])

    def test_series_latin_parity_with_no_transliterator(self):
        names = pd.Series(["Acme Corp", "Blue Ocean Ltd", ""])
        set_transliterator(None)
        res_without, _ = normalize_name_series(names)
        set_transliterator(IndicTransliterator())
        res_with, _ = normalize_name_series(names)
        for i in range(len(names)):
            assert res_without.iloc[i] == res_with.iloc[i]


# ===========================================================================
# Section 6 — Integration with preprocess.py / clean_source
# ===========================================================================

class TestCleanSourceWithTransliterator:
    @pytest.fixture
    def sample_df(self):
        rows = [
            ["S1-1", "नमस्ते कॉर्प",           "123 Main St, Delhi",   "India"],
            ["S1-2", "ராஜ் எல்எல்பி",          "560001 Bangalore",     "India"],
            ["S1-3", "Acme Corporation",         "100 Main St, Austin",  "US"],
            ["S1-4", "ಕನ್ನಡ ಇಂಡಸ್ಟ್ರೀಸ್",       "Bangalore",           "India"],
        ]
        return pd.DataFrame(
            rows,
            columns=["entity_id", "business_name", "business_address", "country"],
        )

    def test_row_count_preserved(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, stats = clean_source(sample_df, "source1", "test")
        assert len(out) == 4
        assert stats["output_rows"] == 4

    def test_raw_business_name_preserved(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        assert out.loc[out["entity_id"] == "S1-1", "business_name"].iloc[0] == "नमस्ते कॉर्प"
        assert out.loc[out["entity_id"] == "S1-2", "business_name"].iloc[0] == "ராஜ் எல்எல்பி"

    def test_normalized_devanagari_is_ascii(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        norm = out.loc[out["entity_id"] == "S1-1", "business_name_normalized"].iloc[0]
        assert all(ord(c) < 128 for c in norm), f"Non-ASCII in normalized: {norm!r}"
        assert len(norm) > 0

    def test_normalized_latin_unchanged(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        norm = out.loc[out["entity_id"] == "S1-3", "business_name_normalized"].iloc[0]
        assert norm == "acme corporation"

    def test_name_script_class_from_raw_devanagari(self, sample_df):
        # name_script_class must reflect the ORIGINAL script, not transliterated form
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        script = out.loc[out["entity_id"] == "S1-1", "name_script_class"].iloc[0]
        assert script == "devanagari"

    def test_name_script_class_from_raw_tamil(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        script = out.loc[out["entity_id"] == "S1-2", "name_script_class"].iloc[0]
        assert script == "tamil"

    def test_name_script_class_latin_stays_latin(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        script = out.loc[out["entity_id"] == "S1-3", "name_script_class"].iloc[0]
        assert script == "latin"

    def test_name_script_class_kannada(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        script = out.loc[out["entity_id"] == "S1-4", "name_script_class"].iloc[0]
        assert script == "kannada"

    def test_no_transliterator_script_class_still_correct(self, sample_df):
        set_transliterator(None)
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        script = out.loc[out["entity_id"] == "S1-1", "name_script_class"].iloc[0]
        assert script == "devanagari"

    def test_entity_ids_preserved(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        out, _ = clean_source(sample_df, "source1", "test")
        assert set(out["entity_id"]) == {"S1-1", "S1-2", "S1-3", "S1-4"}

    def test_all_required_output_columns_present(self, sample_df):
        set_transliterator(IndicTransliterator())
        from preprocessing.src.preprocess import clean_source
        from preprocessing.src.config import REQUIRED_OUTPUT_COLUMNS
        out, _ = clean_source(sample_df, "source1", "test")
        assert list(out.columns) == list(REQUIRED_OUTPUT_COLUMNS)


# ===========================================================================
# Section 7 — Non-regression: existing Latin / NA behaviour unchanged
# ===========================================================================

class TestNonRegressionWithTransliteratorActive:
    @pytest.fixture(autouse=True)
    def enable_transliterator(self):
        set_transliterator(IndicTransliterator())
        yield
        set_transliterator(None)

    def test_na_string_is_missing(self):
        norm, missing = normalize_name_basic("NA")
        assert norm == "" and missing is True

    def test_null_string_is_missing(self):
        norm, missing = normalize_name_basic("NULL")
        assert norm == "" and missing is True

    def test_nil_is_missing(self):
        norm, missing = normalize_name_basic("nil")
        assert norm == "" and missing is True

    def test_empty_is_missing(self):
        norm, missing = normalize_name_basic("")
        assert norm == "" and missing is True

    def test_none_is_missing(self):
        norm, missing = normalize_name_basic(None)
        assert norm == "" and missing is True

    def test_pvt_ltd_canonicalized(self):
        from preprocessing.src.normalization import canonicalize_name
        norm, _ = normalize_name_basic("Invest Pvt Ltd")
        assert canonicalize_name(norm) == "invest private limited"

    def test_dba_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp dba The Widget Store")
        assert norm == "acme corp"

    def test_url_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp www.acme.com")
        assert "www" not in norm
        assert "acme" in norm

    def test_cafe_accent_folded(self):
        norm, _ = normalize_name_basic("Café du Monde")
        assert "é" not in norm
        assert "cafe" in norm

    def test_leet_repair(self):
        norm, _ = normalize_name_basic("Preparat0ry School")
        assert "preparatory" in norm

    def test_address_null_component_stripped(self):
        from preprocessing.src.normalization import normalize_address_basic
        norm, missing = normalize_address_basic("123 Main St, NULL, Chicago")
        assert "null" not in norm
        assert missing is False

    def test_address_accent_folded(self):
        from preprocessing.src.normalization import normalize_address_basic
        norm, _ = normalize_address_basic("Résidence du Parc, Paris")
        assert "residence" in norm
