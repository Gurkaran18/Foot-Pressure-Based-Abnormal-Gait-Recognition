"""T8: No synthetic demographic data test (spec §11)."""
import os
from pathlib import Path
import pytest
import numpy as np

from experiments.external_transport.etx import demographics, paths
from experiments.external_transport.tests.conftest import requires_gaitpdb


@requires_gaitpdb
def test_t8_demographics_matches_html_and_no_synthesis():
    root_str = paths.gaitpdb_root()
    if not root_str or not Path(root_str).exists():
        pytest.skip("GAITPDB_ROOT not set or not found")

    root = Path(root_str)
    df = demographics.load_demographics(root)

    # All records match HTML cross-check
    assert df["html_match"].all(), "Some records failed HTML cross-check"

    # Ju height conversion verified: all heights in meters (between 1.0 and 2.3)
    valid_hts = df["height_m"].dropna()
    assert (valid_hts >= 1.2).all() and (valid_hts <= 2.3).all(), "Heights must be in meters"

    # Missing values remain NaN (never imputed)
    assert df["height_m"].isna().sum() > 0, "Missing heights should remain NaN"
    assert df["tuag_s"].isna().sum() > 0, "Missing TUAG should remain NaN"

    # Gender is valid M/F only
    assert set(df["sex"].dropna().unique()).issubset({"M", "F"})

    # UPDRS and H&Y are explicitly marked desc_only
    assert "hy_desc_only" in df.columns
    assert "updrs_desc_only" in df.columns


def test_t8_static_no_random_synthesis():
    """Ensure demographics.py has no random generation or imputation logic."""
    demo_src = Path(demographics.__file__).read_text(encoding="utf-8")
    assert "np.random" not in demo_src, "Demographics module must not synthesize random data"
    assert "import random" not in demo_src, "Demographics module must not synthesize random data"
    assert "load_subject_info" not in demo_src, "Must not call main.py load_subject_info"


def test_t8_desc_columns_not_in_model_features():
    """Ensure descriptive clinical columns never reach model feature sets."""
    cfg = paths.load_config()
    for col in ["hy_desc_only", "updrs_desc_only", "HoehnYahr", "UPDRS", "UPDRSM"]:
        assert col not in cfg["features"]["t1_core"]
        assert col not in cfg["features"]["t1_fallback"]
