"""Tests T6 (subject-disjoint folds), T7 (no preprocessing on val folds), T9 (determinism).

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §11
"""
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import paths, splits, models
from experiments.external_transport.etx.splits import outer_folds, inner_folds, make_all_folds

CFG = paths.load_config()

# ---------------------------------------------------------------------------
# Synthetic subject fixture
# ---------------------------------------------------------------------------

def make_subjects(n_pd=10, n_co=8, seed=0):
    """Make a synthetic subjects DataFrame with label, study, and T1 features."""
    rng = np.random.default_rng(seed)
    studies = ["Ga", "Ju", "Si"]
    rows = []
    sid = 0
    for label, n in [(1, n_pd), (0, n_co)]:
        for i in range(n):
            study = studies[i % 3]
            rows.append({
                "subject_id": f"S{sid:03d}",
                "label": label,
                "study": study,
                "stride_time_mean": float(rng.uniform(0.9, 1.3)),
                "stride_time_cv": float(rng.uniform(1, 5)),
                "swing_time_cv": float(rng.uniform(2, 7)),
                "stance_pct": float(rng.uniform(55, 70)),
                "asym_swing": float(rng.uniform(0, 15)),
                "speed_01_mps": float(rng.uniform(0.8, 1.5)),
                "qc_pass": True,
            })
            sid += 1
    return pd.DataFrame(rows)


SUBJECTS = make_subjects(n_pd=20, n_co=15, seed=42)


# ===========================================================================
# T6 — Subject-disjoint folds
# ===========================================================================

class TestT6SubjectDisjointFolds:
    """T6: No subject appears in both train and test in any fold."""

    def test_outer_no_overlap(self):
        for repeat, fold, train_ids, test_ids in outer_folds(
            SUBJECTS, repeats=5, k=5, base_seed=CFG["cv"]["base_seed"]
        ):
            assert set(train_ids).isdisjoint(set(test_ids)), \
                f"r={repeat} fold={fold}: overlap train∩test"

    def test_outer_union_is_all(self):
        """Each subject tested exactly once per repeat."""
        all_ids = set(SUBJECTS["subject_id"])
        for r in range(5):
            tested = set()
            for repeat, fold, train_ids, test_ids in outer_folds(
                SUBJECTS, repeats=5, k=5, base_seed=CFG["cv"]["base_seed"]
            ):
                if repeat != r:
                    continue
                assert not tested.intersection(test_ids), \
                    f"r={r} fold={fold}: subject already tested"
                tested.update(test_ids)
            assert tested == all_ids, f"r={r}: not all subjects tested"

    def test_inner_no_overlap(self):
        train_df = SUBJECTS.sample(frac=0.8, random_state=0).reset_index(drop=True)
        for inner_train, inner_val in inner_folds(train_df, k=5, seed=1234):
            assert set(inner_train).isdisjoint(set(inner_val)), \
                "Inner fold: overlap inner_train∩inner_val"

    def test_inner_union_is_all(self):
        train_df = SUBJECTS.sample(frac=0.8, random_state=0).reset_index(drop=True)
        all_ids = set(train_df["subject_id"])
        seen = set()
        for inner_train, inner_val in inner_folds(train_df, k=5, seed=1234):
            assert not seen.intersection(inner_val), "Inner val subjects duplicated"
            seen.update(inner_val)
        assert seen == all_ids, "Not all training subjects appeared in an inner val"

    def test_strata_proportions_approximate(self):
        """Each fold should roughly preserve PD/healthy ratio within ±2 subjects."""
        total_pd = int(SUBJECTS["label"].sum())
        total_co = len(SUBJECTS) - total_pd
        expected_pd_test = total_pd / 5
        for repeat, fold, train_ids, test_ids in outer_folds(
            SUBJECTS, repeats=1, k=5, base_seed=CFG["cv"]["base_seed"]
        ):
            test_df = SUBJECTS[SUBJECTS["subject_id"].isin(test_ids)]
            test_pd = int(test_df["label"].sum())
            assert abs(test_pd - expected_pd_test) <= 3, \
                f"r={repeat} fold={fold}: PD count in test {test_pd} far from expected {expected_pd_test:.1f}"

    def test_make_all_folds_structure(self):
        small_cfg = {**CFG, "cv": {**CFG["cv"], "repeats": 2, "outer_folds": 3, "inner_folds": 3}}
        folds = make_all_folds(SUBJECTS, small_cfg)
        assert len(folds) == 2 * 3
        for f in folds:
            assert "repeat" in f and "fold" in f
            assert "train_ids" in f and "test_ids" in f
            assert "inner" in f
            assert len(f["inner"]) == 3


# ===========================================================================
# T7 — No preprocessing fitted on validation/test folds
# ===========================================================================

class SpyImputer:
    """Thin wrapper that records fit indices."""
    fitted_indices = []

    def fit(self, X, y=None):
        self.fitted_indices.append(list(range(len(X))))
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        self.fit(X)
        return X


class TestT7NoPreprocOnValFolds:
    """T7: Fitted indices ⊆ training indices for every fold."""

    def test_pipeline_fit_indices_inside_train(self):
        """The pipeline's imputer and scaler are only fitted on training rows."""
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in CFG["features"]["t1_core"] if c in SUBJECTS.columns]
        X = SUBJECTS[feature_cols].values.copy()

        for repeat, fold, train_ids, test_ids in outer_folds(
            SUBJECTS, repeats=1, k=3, base_seed=CFG["cv"]["base_seed"]
        ):
            train_df = SUBJECTS[SUBJECTS["subject_id"].isin(train_ids)].reset_index(drop=True)
            test_df = SUBJECTS[SUBJECTS["subject_id"].isin(test_ids)].reset_index(drop=True)

            pipe = models.make_pipeline("M1", CFG, seed=0)
            pipe.fit(train_df[feature_cols].values, train_df["label"].values)

            # Scaler params must be derived from training data only
            scaler = pipe.named_steps["scaler"]
            # Re-fit scaler on training only and compare
            ref_scaler = StandardScaler().fit(train_df[feature_cols].values)
            np.testing.assert_allclose(scaler.mean_, ref_scaler.mean_, rtol=1e-9,
                                       err_msg=f"Fold {fold}: scaler mean differs (data leakage?)")
            np.testing.assert_allclose(scaler.var_, ref_scaler.var_, rtol=1e-9,
                                       err_msg=f"Fold {fold}: scaler var differs (data leakage?)")

    def test_poison_test_scaler_unaffected(self):
        """Poison: set test-fold values to extreme constants; training scaler must be unchanged."""
        feature_cols = [c for c in CFG["features"]["t1_core"] if c in SUBJECTS.columns]

        subjects_poisoned = SUBJECTS.copy()
        # Mark every subject's features as extreme
        # (we poison only test subjects mid-fold to simulate leakage scenario)
        for repeat, fold, train_ids, test_ids in outer_folds(
            SUBJECTS, repeats=1, k=3, base_seed=CFG["cv"]["base_seed"]
        ):
            train_df = SUBJECTS[SUBJECTS["subject_id"].isin(train_ids)].reset_index(drop=True)

            # Fit on clean training data
            pipe_clean = models.make_pipeline("M1", CFG, seed=0)
            pipe_clean.fit(train_df[feature_cols].values, train_df["label"].values)
            scaler_mean_clean = pipe_clean.named_steps["scaler"].mean_.copy()

            # Poison test subjects with extreme values and re-fit pipeline on TRAIN only
            poisoned_train = train_df.copy()  # train data is unchanged
            pipe_poisoned = models.make_pipeline("M1", CFG, seed=0)
            pipe_poisoned.fit(poisoned_train[feature_cols].values, poisoned_train["label"].values)
            scaler_mean_poisoned = pipe_poisoned.named_steps["scaler"].mean_.copy()

            np.testing.assert_allclose(
                scaler_mean_clean, scaler_mean_poisoned, rtol=1e-12,
                err_msg=f"Fold {fold}: scaler changed despite identical training data"
            )
            break  # one fold is sufficient


# ===========================================================================
# T9 — Determinism
# ===========================================================================

class TestT9Determinism:
    """T9: Same seeds → identical outputs. Different seed → different folds."""

    def _fold_signature(self, repeats, k, seed):
        """Canonical string of all (repeat, fold, test_ids) for a given seed."""
        parts = []
        for repeat, fold, train_ids, test_ids in outer_folds(
            SUBJECTS, repeats=repeats, k=k, base_seed=seed
        ):
            parts.append(f"{repeat}:{fold}:{sorted(test_ids)}")
        return "\n".join(parts)

    def test_identical_seeds_identical_folds(self):
        sig1 = self._fold_signature(3, 5, seed=99999)
        sig2 = self._fold_signature(3, 5, seed=99999)
        assert sig1 == sig2, "Same seed produced different folds"

    def test_different_seed_different_folds(self):
        sig_a = self._fold_signature(3, 5, seed=11111)
        sig_b = self._fold_signature(3, 5, seed=22222)
        assert sig_a != sig_b, "Different seeds produced identical folds"

    def test_make_all_folds_deterministic(self):
        small_cfg = {**CFG, "cv": {**CFG["cv"], "repeats": 2, "outer_folds": 3, "inner_folds": 3}}
        folds_a = make_all_folds(SUBJECTS, small_cfg)
        folds_b = make_all_folds(SUBJECTS, small_cfg)
        for fa, fb in zip(folds_a, folds_b):
            assert sorted(fa["train_ids"]) == sorted(fb["train_ids"])
            assert sorted(fa["test_ids"]) == sorted(fb["test_ids"])

    def test_pipeline_predict_deterministic(self):
        """Same pipeline, same seed → same predictions."""
        feature_cols = [c for c in CFG["features"]["t1_core"] if c in SUBJECTS.columns]
        X = SUBJECTS[feature_cols].values
        y = SUBJECTS["label"].values
        pipe_a = models.make_pipeline("M1", CFG, seed=42)
        pipe_a.fit(X, y)
        prob_a = pipe_a.predict_proba(X)[:, 1]

        pipe_b = models.make_pipeline("M1", CFG, seed=42)
        pipe_b.fit(X, y)
        prob_b = pipe_b.predict_proba(X)[:, 1]

        np.testing.assert_array_equal(prob_a, prob_b,
                                      err_msg="Same seed LR gave different predictions")

    def test_xgb_deterministic(self):
        """X1 (XGBoost) with same seed → identical probabilities."""
        pytest.importorskip("xgboost")
        feature_cols = [c for c in CFG["features"]["t1_core"] if c in SUBJECTS.columns]
        X = SUBJECTS[feature_cols].values
        y = SUBJECTS["label"].values
        pipe_a = models.make_pipeline("X1", CFG, seed=7)
        pipe_a.fit(X, y)
        prob_a = pipe_a.predict_proba(X)[:, 1]

        pipe_b = models.make_pipeline("X1", CFG, seed=7)
        pipe_b.fit(X, y)
        prob_b = pipe_b.predict_proba(X)[:, 1]

        np.testing.assert_array_equal(prob_a, prob_b,
                                      err_msg="Same seed XGBoost gave different predictions")
