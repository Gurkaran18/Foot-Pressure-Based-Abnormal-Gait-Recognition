"""Tests for etx/metrics.py — AUC, DeLong, Brier, calibration, threshold_metrics, AAUC."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from experiments.external_transport.etx.metrics import (
    auc, delong, brier, calibration, threshold_metrics, aauc
)


def _make_data(n=200, seed=1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.clip(rng.uniform(0, 1, n) * 0.4 + y * 0.3, 0, 1)
    return y, p


# --- AUC ---

def test_auc_matches_sklearn():
    y, p = _make_data()
    assert auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-10)


def test_auc_nan_on_single_class():
    y = np.ones(10, int)
    p = np.random.rand(10)
    assert np.isnan(auc(y, p))


# --- DeLong ---

def test_delong_auc_matches_sklearn():
    y, p = _make_data()
    result = delong(y, p)
    assert result["auc"] == pytest.approx(roc_auc_score(y, p), abs=1e-9)


def test_delong_variance_positive():
    y, p = _make_data()
    result = delong(y, p)
    assert result["variance"] > 0
    assert result["ci_lower"] < result["auc"] < result["ci_upper"]


def test_delong_paired_same_model():
    y, p = _make_data()
    result = delong(y, p, p)  # identical models → delta=0
    assert result["delta_auc"] == pytest.approx(0.0, abs=1e-10)
    assert result["p_value"] == pytest.approx(1.0, abs=0.01)


def test_delong_paired_different_models():
    y, p1 = _make_data(seed=1)
    _, p2 = _make_data(seed=99)
    result = delong(y, p1, p2)
    assert "z" in result and "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


# --- Brier ---

def test_brier_perfect():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier(y, p) == pytest.approx(0.0, abs=1e-10)


def test_brier_worst():
    y = np.array([0, 1, 0, 1])
    p = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier(y, p) == pytest.approx(1.0, abs=1e-10)


def test_brier_uniform():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.5)
    assert brier(y, p) == pytest.approx(0.25, abs=1e-10)


# --- Calibration ---

def test_calibration_well_calibrated():
    rng = np.random.default_rng(42)
    p = rng.uniform(0.1, 0.9, 500)
    y = rng.binomial(1, p)
    cal = calibration(y, p)
    assert abs(cal["intercept"]) < 0.5, "Intercept far from 0 on well-calibrated data"
    assert abs(cal["slope"] - 1.0) < 0.5, "Slope far from 1 on well-calibrated data"


def test_calibration_returns_slope_one_on_perfect():
    """When predicted probs match true probs exactly, slope ≈ 1."""
    rng = np.random.default_rng(7)
    p = rng.uniform(0.2, 0.8, 300)
    y = rng.binomial(1, p)
    cal = calibration(y, p)
    assert cal["slope"] == pytest.approx(1.0, abs=0.4)


# --- Threshold metrics ---

def test_threshold_perfect_sensitivity():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8])
    m = threshold_metrics(y, p, thr=0.5)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)


def test_threshold_all_predicted_positive():
    y = np.array([0, 1, 0, 1])
    p = np.array([1.0, 1.0, 1.0, 1.0])
    m = threshold_metrics(y, p, thr=0.5)
    assert m["sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(0.0)


def test_threshold_counts():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.4, 0.6, 0.4, 0.6])
    m = threshold_metrics(y, p, thr=0.5)
    assert m["tp"] == 1 and m["fp"] == 1 and m["tn"] == 1 and m["fn"] == 1


# --- AAUC ---

def test_aauc_covariate_only_is_half():
    """AAUC ≈ 0.5 when the score depends only on the covariate (spec test)."""
    rng = np.random.default_rng(99)
    n = 200
    cov = rng.uniform(0, 1, n)
    y = rng.integers(0, 2, n)
    # Score = covariate + noise; diagnosis independent of score after adjusting for cov
    score = cov + rng.normal(0, 0.01, n)
    result = aauc(y, score, cov)
    # After full adjustment for covariate, AAUC should be near 0.5
    assert abs(result["aauc"] - 0.5) < 0.15, \
        f"AAUC not near 0.5 when score = covariate: {result['aauc']:.3f}"


def test_aauc_returns_finite():
    y, p = _make_data()
    cov = np.random.default_rng(5).uniform(0.5, 1.5, len(y))
    result = aauc(y, p, cov)
    assert np.isfinite(result["aauc"])
    assert result["ci_lower"] < result["aauc"] < result["ci_upper"]
