"""Extended normalization unit tests — P2 production upgrade.

Tests for new functionality merged from MLnNor latest commit:
  - Accent folding (Latin-gated)
  - URL / email stripping
  - Leet-speak repair
  - DBA / T/A / AKA stripping
  - Extended NA_LIKE_TOKENS (nil, n.a., unknown, etc.)
  - Extended LEGAL_SUFFIX_MAP (lp, lc, pa, eurl, snc, gie)
  - Extended ADDRESS_ABBREV_MAP (US + French + India additions)
  - Pipeline-level regression with new steps active

Adapted from MLnNor/tests/test_normalization.py (P2 section) for the
combined-pipeline import layout (preprocessing.src.* instead of src.*).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from preprocessing.src.normalization import (
    canonicalize_address,
    canonicalize_name,
    fold_accents,
    is_na_like,
    normalize_address_basic,
    normalize_address_series,
    normalize_name_basic,
    normalize_name_series,
)


# ===========================================================================
# Accent folding — fold_accents() helper
# ===========================================================================

class TestFoldAccents:
    def test_cafe(self):
        assert fold_accents("café") == "cafe"

    def test_naive(self):
        assert fold_accents("naïve") == "naive"

    def test_resume(self):
        assert fold_accents("résumé") == "resume"

    def test_ecole_francaise(self):
        assert fold_accents("École Française") == "Ecole Francaise"

    def test_pure_ascii_unchanged(self):
        assert fold_accents("hello world") == "hello world"

    def test_empty_string(self):
        assert fold_accents("") == ""

    def test_devanagari_untouched(self):
        text = "नमस्ते"
        assert fold_accents(text) == text

    def test_tamil_untouched(self):
        text = "ராஜ்"
        assert fold_accents(text) == text

    def test_gujarati_untouched(self):
        text = "ગુજરાત"
        assert fold_accents(text) == text

    def test_mixed_latin_non_latin_untouched(self):
        # If ANY non-Latin letter present, whole string left alone.
        text = "Café नमस्ते"
        assert fold_accents(text) == text


# ===========================================================================
# Accent folding — integrated into normalize_name_basic
# ===========================================================================

class TestAccentFoldingInNames:
    def test_cafe_variants_normalise_identically(self):
        n1, _ = normalize_name_basic("Café du Monde")
        n2, _ = normalize_name_basic("Cafe du Monde")
        assert n1 == n2

    def test_ecole_francaise_folded(self):
        norm, missing = normalize_name_basic("École Française")
        assert norm == "ecole francaise"
        assert missing is False

    def test_resume_in_name(self):
        norm, _ = normalize_name_basic("Résumé Writers LLC")
        assert "resume" in norm
        assert "é" not in norm

    def test_accented_name_not_missing(self):
        _, missing = normalize_name_basic("Société Générale")
        assert missing is False

    def test_devanagari_combining_marks_preserved(self):
        norm, missing = normalize_name_basic("नमस्ते")
        assert norm == "नमस्ते"
        assert missing is False

    def test_series_accent_parity(self):
        names = pd.Series(["Café du Monde", "Cafe du Monde", "École", "नमस्ते"])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ===========================================================================
# Accent folding — integrated into normalize_address_basic
# ===========================================================================

class TestAccentFoldingInAddresses:
    def test_residence_accent_folded(self):
        norm, _ = normalize_address_basic("Résidence du Parc, Paris")
        assert "residence" in norm
        assert "é" not in norm

    def test_accented_and_unaccented_address_canonicalise_identically(self):
        n1, _ = normalize_address_basic("Résidence du Parc, Paris")
        n2, _ = normalize_address_basic("Residence du Parc, Paris")
        assert n1 == n2

    def test_devanagari_address_untouched(self):
        text = "नमस्ते, दिल्ली"
        norm, missing = normalize_address_basic(text)
        assert "ते" in norm
        assert missing is False

    def test_series_address_accent_parity(self):
        addrs = pd.Series(["Résidence du Parc, Paris", "Residence du Parc, Paris", ""])
        result, _ = normalize_address_series(addrs)
        for i, a in enumerate(addrs):
            expected, _ = normalize_address_basic(a)
            assert result.iloc[i] == expected


# ===========================================================================
# URL / email stripping
# ===========================================================================

class TestURLEmailStripping:
    def test_email_stripped_from_name(self):
        norm, _ = normalize_name_basic("Acme Corp info@acme.com")
        assert "@" not in norm
        assert "acme" in norm

    def test_email_only_name_becomes_missing(self):
        norm, missing = normalize_name_basic("info@acme.com")
        assert missing is True

    def test_http_url_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp http://www.acme.com")
        assert "http" not in norm
        assert "acme" in norm

    def test_https_url_stripped(self):
        norm, _ = normalize_name_basic("Visit us at https://shop.acme.in/sale")
        assert "https" not in norm
        assert "visit" in norm

    def test_www_domain_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp www.acme.com")
        assert "www" not in norm
        assert "acme" in norm

    def test_bare_dotcom_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp acme.com")
        assert ".com" not in norm

    def test_bare_dotin_stripped(self):
        norm, _ = normalize_name_basic("Flipkart flipkart.in")
        assert ".in" not in norm
        assert "flipkart" in norm

    def test_legal_suffix_dot_not_stripped(self):
        # "S.A.R.L" is a legal suffix, not a domain — must survive
        norm, _ = normalize_name_basic("Meridian S.A.R.L")
        assert "s.a.r.l" in norm

    def test_address_dotcom_not_stripped(self):
        # URL stripping is NOT applied to addresses
        norm, _ = normalize_address_basic("123 Commerce.com Drive, Austin")
        assert "commerce.com" in norm

    def test_name_without_url_unchanged(self):
        norm, _ = normalize_name_basic("Acme Corporation")
        assert norm == "acme corporation"

    def test_series_url_parity(self):
        names = pd.Series(["Acme Corp www.acme.com", "info@acme.com", "Just A Name"])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ===========================================================================
# Leet-speak repair
# ===========================================================================

class TestLeetRepair:
    def test_preparatory_0_to_o(self):
        norm, _ = normalize_name_basic("Preparat0ry School")
        assert "preparatory" in norm

    def test_4cme_to_acme(self):
        norm, _ = normalize_name_basic("4cme Corp")
        assert "acme" in norm

    def test_3lite_to_elite(self):
        norm, _ = normalize_name_basic("3lite Networks")
        assert "elite" in norm

    def test_5ecure_to_secure(self):
        norm, _ = normalize_name_basic("5ecure Systems")
        assert "secure" in norm

    def test_7rading_to_trading(self):
        norm, _ = normalize_name_basic("7rading Co")
        assert "trading" in norm

    def test_1nvestments_to_investments(self):
        norm, _ = normalize_name_basic("1nvestments Ltd")
        assert "investments" in norm

    def test_pure_digit_token_untouched(self):
        norm, _ = normalize_name_basic("Studio 7")
        assert "7" in norm

    def test_building_number_untouched(self):
        norm, _ = normalize_name_basic("Section 5 Investments")
        assert "5" in norm

    def test_year_in_name_untouched(self):
        norm, _ = normalize_name_basic("Est. 1975 Bakeries")
        assert "1975" in norm

    def test_no_leet_digits_name_unchanged(self):
        norm, _ = normalize_name_basic("Blue Ocean Ltd")
        assert norm == "blue ocean ltd"

    def test_series_leet_parity(self):
        names = pd.Series(["Preparat0ry School", "4cme Corp", "Blue Ocean Ltd"])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ===========================================================================
# DBA / T/A / AKA stripping
# ===========================================================================

class TestDBAStripping:
    def test_dba_lowercase(self):
        norm, _ = normalize_name_basic("Acme Corp dba The Widget Store")
        assert norm == "acme corp"
        assert canonicalize_name(norm) == "acme corporation"

    def test_dba_uppercase(self):
        norm, _ = normalize_name_basic("Acme Corp DBA The Widget Store")
        assert norm == "acme corp"

    def test_d_slash_b_slash_a(self):
        norm, _ = normalize_name_basic("Acme Corp d/b/a The Widget Store")
        assert norm == "acme corp"

    def test_doing_business_as(self):
        norm, _ = normalize_name_basic("Smith Plumbing doing business as Smith & Sons")
        assert "smith plumbing" in norm
        assert "sons" not in norm

    def test_trading_as(self):
        norm, _ = normalize_name_basic("Global Ventures trading as GV Express")
        assert "global ventures" in norm
        assert "express" not in norm

    def test_t_slash_a(self):
        norm, _ = normalize_name_basic("Jones Bakery t/a The Cake Shop")
        assert "jones bakery" in norm
        assert "cake" not in norm

    def test_also_known_as(self):
        norm, _ = normalize_name_basic("First National Bank also known as FNB")
        assert "first national bank" in norm
        assert "fnb" not in norm

    def test_aka(self):
        norm, _ = normalize_name_basic("First National Bank aka FNB")
        assert "first national bank" in norm
        assert "fnb" not in norm

    def test_name_without_dba_unchanged(self):
        norm, _ = normalize_name_basic("Acme Corporation")
        assert norm == "acme corporation"

    def test_series_dba_parity(self):
        names = pd.Series([
            "Acme Corp dba The Widget Store",
            "Jones Bakery t/a The Cake Shop",
            "No Alias Here Inc",
        ])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ===========================================================================
# Extended NA_LIKE_TOKENS
# ===========================================================================

class TestExtendedNALike:
    def test_n_dot_a_dot_is_missing(self):
        _, missing = normalize_name_basic("n.a.")
        assert missing is True

    def test_nil_is_missing(self):
        _, missing = normalize_name_basic("nil")
        assert missing is True

    def test_nil_mixed_case(self):
        _, missing = normalize_name_basic("NIL")
        assert missing is True

    def test_not_available_is_missing(self):
        _, missing = normalize_name_basic("not available")
        assert missing is True

    def test_not_applicable_is_missing(self):
        _, missing = normalize_name_basic("not applicable")
        assert missing is True

    def test_unknown_is_missing(self):
        _, missing = normalize_name_basic("unknown")
        assert missing is True

    def test_lone_hyphen_is_missing(self):
        _, missing = normalize_name_basic("-")
        assert missing is True

    def test_en_dash_is_missing(self):
        _, missing = normalize_name_basic("\u2013")
        assert missing is True

    def test_em_dash_is_missing(self):
        _, missing = normalize_name_basic("\u2014")
        assert missing is True

    def test_legitimate_name_with_hyphen_not_missing(self):
        _, missing = normalize_name_basic("Coca-Cola Ltd")
        assert missing is False

    def test_is_na_like_helper_nil(self):
        assert is_na_like("nil")
        assert is_na_like("NIL")
        assert is_na_like("not available")
        assert is_na_like("unknown")
        assert is_na_like("n.a.")
        assert not is_na_like("National Bank")


# ===========================================================================
# Extended LEGAL_SUFFIX_MAP — US additions
# ===========================================================================

class TestExtendedLegalSuffixUS:
    def test_lp_canonicalized(self):
        n, _ = normalize_name_basic("Meridian LP")
        assert canonicalize_name(n) == "meridian lp"

    def test_l_dot_p_canonicalized(self):
        n, _ = normalize_name_basic("Meridian L.P.")
        assert canonicalize_name(n) == "meridian lp"

    def test_lc_canonicalized(self):
        n, _ = normalize_name_basic("Meridian LC")
        assert canonicalize_name(n) == "meridian lc"

    def test_pa_canonicalized(self):
        n, _ = normalize_name_basic("Smith Dental PA")
        assert canonicalize_name(n) == "smith dental pa"

    def test_p_dot_a_canonicalized(self):
        n, _ = normalize_name_basic("Smith Dental P.A.")
        assert canonicalize_name(n) == "smith dental pa"

    def test_lp_token_boundary_safe(self):
        n, _ = normalize_name_basic("Tulip Help Group")
        canon = canonicalize_name(n)
        assert "tulip" in canon
        assert "help" in canon

    def test_pa_token_boundary_safe(self):
        n, _ = normalize_name_basic("Pasta Palace Inc")
        canon = canonicalize_name(n)
        assert "pasta" in canon


# ===========================================================================
# Extended LEGAL_SUFFIX_MAP — French additions
# ===========================================================================

class TestExtendedLegalSuffixFrance:
    def test_eurl_canonicalized(self):
        n, _ = normalize_name_basic("Dupont EURL")
        assert canonicalize_name(n) == "dupont eurl"

    def test_e_dot_u_dot_r_dot_l_canonicalized(self):
        n, _ = normalize_name_basic("Dupont E.U.R.L")
        assert canonicalize_name(n) == "dupont eurl"

    def test_snc_canonicalized(self):
        n, _ = normalize_name_basic("Dupont SNC")
        assert canonicalize_name(n) == "dupont snc"

    def test_snc_dot_separated(self):
        n, _ = normalize_name_basic("Dupont S.N.C")
        assert canonicalize_name(n) == "dupont snc"

    def test_gie_canonicalized(self):
        n, _ = normalize_name_basic("Dupont GIE")
        assert canonicalize_name(n) == "dupont gie"

    def test_gie_dot_separated(self):
        n, _ = normalize_name_basic("Dupont G.I.E")
        assert canonicalize_name(n) == "dupont gie"

    def test_existing_sarl_still_works(self):
        n, _ = normalize_name_basic("Dupont SARL")
        assert canonicalize_name(n) == "dupont sarl"

    def test_existing_sas_still_works(self):
        n, _ = normalize_name_basic("Dupont SAS")
        assert canonicalize_name(n) == "dupont sas"


# ===========================================================================
# Extended ADDRESS_ABBREV_MAP — US additions
# ===========================================================================

class TestExtendedAddressAbbrevUS:
    def test_expy_to_expressway(self):
        n, _ = normalize_address_basic("100 Northern Expy, Atlanta, GA")
        assert "expressway" in canonicalize_address(n)

    def test_fwy_to_freeway(self):
        n, _ = normalize_address_basic("500 Harbor Fwy, Los Angeles, CA")
        assert "freeway" in canonicalize_address(n)

    def test_tpke_to_turnpike(self):
        n, _ = normalize_address_basic("1 Garden State Tpke, NJ")
        assert "turnpike" in canonicalize_address(n)

    def test_tpk_to_turnpike(self):
        n, _ = normalize_address_basic("1 Garden State Tpk, NJ")
        assert "turnpike" in canonicalize_address(n)

    def test_xing_to_crossing(self):
        n, _ = normalize_address_basic("10 River Xing, Portland, OR")
        assert "crossing" in canonicalize_address(n)


# ===========================================================================
# Extended ADDRESS_ABBREV_MAP — French additions
# ===========================================================================

class TestExtendedAddressAbbrevFrance:
    def test_bd_to_boulevard(self):
        n, _ = normalize_address_basic("12 Bd Haussmann, Paris")
        assert "boulevard" in canonicalize_address(n)

    def test_imp_to_impasse(self):
        n, _ = normalize_address_basic("3 Imp des Lilas, Lyon")
        assert "impasse" in canonicalize_address(n)

    def test_res_to_residence(self):
        n, _ = normalize_address_basic("5 Res du Parc, Marseille")
        assert "residence" in canonicalize_address(n)

    def test_res_token_boundary_safe(self):
        n, _ = normalize_address_basic("Restaurant Row, Nice")
        canon = canonicalize_address(n)
        assert "restaurant" in canon


# ===========================================================================
# Extended ADDRESS_ABBREV_MAP — India additions
# ===========================================================================

class TestExtendedAddressAbbrevIndia:
    def test_sec_to_sector(self):
        n, _ = normalize_address_basic("Plot 12, Sec 17, Chandigarh")
        assert "sector" in canonicalize_address(n)

    def test_sec_token_boundary_safe(self):
        n, _ = normalize_address_basic("Second Floor, Sector 21, Noida")
        canon = canonicalize_address(n)
        assert "second" in canon
        assert "sector" in canon

    def test_sector_sec_pair_canonicalise_identically(self):
        n1, _ = normalize_address_basic("Plot 12, Sector 17, Chandigarh")
        n2, _ = normalize_address_basic("Plot 12, Sec 17, Chandigarh")
        assert canonicalize_address(n1) == canonicalize_address(n2)


# ===========================================================================
# Pipeline-level regression: clean_source with all new steps active
# ===========================================================================

class TestCleanSourceRegression:
    def test_accented_french_name_and_address(self):
        from preprocessing.src.preprocess import clean_source
        rows = [["S1-10", "Société Café SARL", "12 Bd Haussmann, Paris 75009", "France"]]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, stats = clean_source(df, "source1", "test")
        assert out.loc[0, "business_name"] == "Société Café SARL"
        assert out.loc[0, "business_name_normalized"] == "societe cafe sarl"
        assert "boulevard" in out.loc[0, "business_address_canonical"]

    def test_dba_and_leet_name(self):
        from preprocessing.src.preprocess import clean_source
        rows = [["S1-20", "Preparat0ry Academy dba Prep Co", "500 Harbor Fwy, LA, CA", "US"]]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, _ = clean_source(df, "source1", "test")
        norm = out.loc[0, "business_name_normalized"]
        assert "preparatory" in norm
        assert "dba" not in norm
        assert "academy" in norm
        assert "freeway" in out.loc[0, "business_address_canonical"]

    def test_url_in_name_stripped(self):
        from preprocessing.src.preprocess import clean_source
        rows = [["S1-30", "Acme Corp www.acme.com", "100 Main St, Austin, TX", "US"]]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, _ = clean_source(df, "source1", "test")
        assert "www" not in out.loc[0, "business_name_normalized"]
        assert "acme" in out.loc[0, "business_name_normalized"]

    def test_extended_na_tokens(self):
        from preprocessing.src.preprocess import clean_source
        rows = [
            ["S1-40", "nil",           "123 Main St", "US"],
            ["S1-41", "not available", "123 Main St", "US"],
            ["S1-42", "unknown",       "123 Main St", "US"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, stats = clean_source(df, "source1", "test")
        assert stats["name_missing_count"] == 3

    def test_row_count_preserved_after_all_new_steps(self):
        from preprocessing.src.preprocess import clean_source
        rows = [
            ["S1-50", "Café EURL",            "12 Allée des Roses, Paris", "France"],
            ["S1-51", "4cme Corp",             "Sec 17, Chandigarh",       "India"],
            ["S1-52", "Global dba Local Inc",  "500 Tpke, NJ",             "US"],
            ["S1-53", "nil",                   "",                          "US"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, stats = clean_source(df, "source1", "test")
        assert len(out) == 4
        assert stats["output_rows"] == 4
        assert set(out["entity_id"]) == {"S1-50", "S1-51", "S1-52", "S1-53"}
