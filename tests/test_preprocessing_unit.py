"""
Unit tests for the preprocessing stage (MLnNor functionality).

These tests run on in-memory data — no raw files needed.
"""
import sys
from pathlib import Path
import pandas as pd
import pytest

# Make sure the repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.src.normalization import (
    normalize_name_basic,
    canonicalize_name,
    normalize_address_basic,
    canonicalize_address,
    normalize_country,
    extract_landmark,
    extract_address_numbers,
    extract_address_postal_code,
    make_address_sorted_tokens,
    classify_name_script,
)
from preprocessing.src.validation import (
    FatalValidationError,
    validate_schema,
    validate_entity_ids,
    validate_row_count_preserved,
    validate_id_set_preserved,
    validate_ground_truth_schema,
)
from preprocessing.src.diagnostics import (
    name_canonical_collisions,
    exact_duplicate_rows,
)
from preprocessing.src.config import SOURCE_COLUMNS, GT_COLUMNS


# ---------------------------------------------------------------------------
# Normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeName:
    def test_basic_lowercase_and_strip(self):
        result, is_missing = normalize_name_basic("  McDonald's  ")
        assert "mcdonald" in result
        assert is_missing is False

    def test_empty_string_is_missing(self):
        result, is_missing = normalize_name_basic("")
        assert result == ""
        assert is_missing is True

    def test_na_literal_is_missing(self):
        result, is_missing = normalize_name_basic("N/A")
        assert is_missing is True

    def test_null_literal_is_missing(self):
        result, is_missing = normalize_name_basic("NULL")
        assert is_missing is True

    def test_unicode_nfkc(self):
        # Full-width digits should normalize to ASCII
        result, is_missing = normalize_name_basic("\uff11\uff12\uff13")
        assert "123" in result
        assert is_missing is False


class TestCanonicalizeName:
    def test_legal_suffix_pvt_to_private(self):
        result = canonicalize_name("abc pvt")
        assert result == "abc private"

    def test_legal_suffix_ltd_to_limited(self):
        result = canonicalize_name("xyz ltd")
        assert result == "xyz limited"

    def test_ampersand_to_and(self):
        result = canonicalize_name("a & b")
        assert result == "a and b"

    def test_empty_input(self):
        assert canonicalize_name("") == ""


class TestNormalizeAddress:
    def test_basic(self):
        result, is_missing = normalize_address_basic("123 Main St, New York")
        assert "123 main" in result
        assert is_missing is False

    def test_null_component_removed(self):
        result, is_missing = normalize_address_basic("123 Main St, NULL, New York")
        assert "null" not in result.lower()
        assert is_missing is False

    def test_all_null_components_is_missing(self):
        result, is_missing = normalize_address_basic("NULL, NULL")
        assert is_missing is True


class TestNormalizeCountry:
    def test_lowercase(self):
        assert normalize_country("United States") == "united states"

    def test_whitespace_collapse(self):
        assert normalize_country("  united   states  ") == "united states"

    def test_empty(self):
        assert normalize_country("") == ""


class TestExtractLandmark:
    def test_near(self):
        result = extract_landmark("Near Connaught Place, New Delhi")
        assert result.lower().startswith("near")

    def test_no_landmark(self):
        assert extract_landmark("123 Main Street") == ""

    def test_word_boundary_safety(self):
        # "near" inside "nearby" must NOT match
        assert extract_landmark("123 Nearby Road") == ""


class TestExtractAddressNumbers:
    def test_single_number(self):
        assert extract_address_numbers("123 Main Street") == "123"

    def test_multiple_numbers(self):
        assert extract_address_numbers("Plot 42 Sector 17") == "42 17"

    def test_no_numbers(self):
        assert extract_address_numbers("main street downtown") == ""


class TestExtractPostalCode:
    def test_6_digit_pin(self):
        assert extract_address_postal_code("Chandigarh 160017") == "160017"

    def test_5_digit_zip(self):
        assert extract_address_postal_code("Seattle WA 98101") == "98101"

    def test_no_postal(self):
        assert extract_address_postal_code("Main Street") == ""


class TestClassifyScript:
    def test_latin(self):
        assert classify_name_script("hello world") == "latin"

    def test_empty(self):
        assert classify_name_script("") == "empty"

    def test_devanagari(self):
        # Hindi text
        assert classify_name_script("नमस्ते") == "devanagari"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_correct_schema_passes(self):
        df = pd.DataFrame(columns=SOURCE_COLUMNS)
        validate_schema(df, SOURCE_COLUMNS, "test")  # should not raise

    def test_wrong_schema_raises(self):
        df = pd.DataFrame(columns=["wrong_col"])
        with pytest.raises(FatalValidationError):
            validate_schema(df, SOURCE_COLUMNS, "test")


class TestEntityIdValidation:
    def test_valid_s1_ids_pass(self):
        df = pd.DataFrame({"entity_id": ["S1-001", "S1-002"]})
        validate_entity_ids(df, "source1", "test")  # should not raise

    def test_wrong_prefix_raises(self):
        df = pd.DataFrame({"entity_id": ["S2-001", "S2-002"]})
        with pytest.raises(FatalValidationError):
            validate_entity_ids(df, "source1", "test")

    def test_duplicate_ids_raises(self):
        df = pd.DataFrame({"entity_id": ["S1-001", "S1-001"]})
        with pytest.raises(FatalValidationError):
            validate_entity_ids(df, "source1", "test")


class TestRowCountInvariant:
    def test_equal_counts_pass(self):
        validate_row_count_preserved(100, 100, "test")

    def test_unequal_counts_raise(self):
        with pytest.raises(FatalValidationError):
            validate_row_count_preserved(100, 99, "test")


class TestIdSetInvariant:
    def test_equal_sets_pass(self):
        validate_id_set_preserved({"a", "b"}, {"a", "b"}, "test")

    def test_unequal_sets_raise(self):
        with pytest.raises(FatalValidationError):
            validate_id_set_preserved({"a", "b"}, {"a", "c"}, "test")


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_no_collisions(self):
        df = pd.DataFrame({
            "business_name_canonical": ["alpha", "beta", "gamma"],
            "country_normalized": ["us", "us", "us"],
        })
        result = name_canonical_collisions(df)
        assert result["group_count"] == 0

    def test_with_collisions(self):
        df = pd.DataFrame({
            "business_name_canonical": ["alpha", "alpha", "beta"],
            "country_normalized": ["us", "us", "us"],
            "entity_id": ["S1-001", "S1-002", "S1-003"],
        })
        result = name_canonical_collisions(df)
        assert result["group_count"] >= 1

    def test_exact_duplicates(self):
        df = pd.DataFrame({
            "entity_id": ["S1-001", "S1-002"],
            "business_name": ["same", "same"],
            "business_address": ["addr", "addr"],
            "country": ["US", "US"],
        })
        result = exact_duplicate_rows(df, ["business_name", "business_address", "country"])
        assert result["exact_duplicate_row_count"] == 2
