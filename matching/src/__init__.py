"""Business Entity Resolution — matching/ML pipeline package.

Re-exports versioned feature column lists and new feature-family modules
added in MLnAWS V6-V9.

Feature set versioning:
  FEATURE_COLUMNS_BASELINE  -- 10 cols  (Phase 2 baseline)
  FEATURE_COLUMNS           -- 13 cols  (+ retrieval-agreement)
  FEATURE_COLUMNS_V2        -- 15 cols  (+ length)
  FEATURE_COLUMNS_V3        -- 24 cols  (+ address components)
  FEATURE_COLUMNS_V4        -- 32 cols  (+ frequency features)
  FEATURE_COLUMNS_V5        -- 40 cols  (+ RRF features)
  FEATURE_COLUMNS_V6        -- 55 cols  (+ token/numeric diff features)
  FEATURE_COLUMNS_V7        -- 71 cols  (+ margin/ambiguity features)
  FEATURE_COLUMNS_V8        -- 81 cols  (+ RapidFuzz similarity family)
  FEATURE_COLUMNS_V9        -- 89 cols  (+ Indic transliteration)
  FEATURE_COLUMNS_FULL      -- alias for V9
"""

from src.features import (
    FEATURE_COLUMNS_BASELINE,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V2,
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V4,
    FEATURE_COLUMNS_V5,
    FEATURE_COLUMNS_V6,
    FEATURE_COLUMNS_V7,
    FEATURE_COLUMNS_V8,
    FEATURE_COLUMNS_V9,
    FEATURE_COLUMNS_FULL,
    ADDRESS_COMPONENT_FEATURE_COLS,
    FREQUENCY_FEATURE_COLS,
    RRF_FEATURE_COLS,
    LENGTH_FEATURE_COLS,
    RETRIEVAL_AGREEMENT_FEATURE_COLS,
    DIFF_FEATURE_COLS,
    MARGIN_FEATURE_COLS,
    RAPIDFUZZ_FEATURE_COLS,
    TRANSLITERATION_FEATURE_COLS,
)

from src.model import (
    FEATURE_COLS,
    FEATURE_COLS_V2,
    FEATURE_COLS_V3,
    FEATURE_COLS_V4,
    FEATURE_COLS_V5,
    FEATURE_COLS_V6,
    FEATURE_COLS_V7,
    FEATURE_COLS_V8,
    FEATURE_COLS_V9,
    FEATURE_COLS_FULL,
)

from src.diff_features import add_diff_features
from src.margin_features import add_margin_features
from src.rapidfuzz_features import add_rapidfuzz_features
from src.transliteration_features import (
    add_transliteration_features,
    TransliterationDictionary,
    get_default_transliteration_dict,
)
