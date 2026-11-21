"""Repeat summaries, bootstrap-of-CV (grouped duplicates), LeDell CI, external bootstrap, coef stability.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §9 (R5)
"""
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from scipy import stats

from .metrics import auc as _auc


# ---------------------------------------------------------------------------
# Repeat summaries
# ---------------------------------------------------------------------------

def summarize_repeats(values: np.ndarray) -> dict:
    """Summarise an array of per-repeat AUC (or any metric) values.

    Returns mean, std, and 2.5–97.5 percentile range (partition variability).
    """
    values = np.asarray(values, float)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "p2_5": float(np.percentile(values, 2.5)),
        "p97_5": float(np.percentile(values, 97.5)),
        "n_repeats": int(len(values)),
    }


# ---------------------------------------------------------------------------
# Bootstrap-of-CV with grouped duplicates  (R5)
# ---------------------------------------------------------------------------

def bootstrap_cv(
    run_cv: Callable,
    subjects_df: pd.DataFrame,
    B: int,
    seed: int,
    k: int = 5,
) -> dict:
    """Bootstrap-of-CV with grouped (subject-level) resampling (spec R5).

    Assign folds on UNIQUE subjects, then expand duplicates into the same fold;
    fail loudly on errors (no bare except).

    Parameters
    ----------
    run_cv : callable(boot_df, fold_splits) -> float
        Function running CV on boot_df using fold_splits (list of (train_idx, test_idx))
        and returning metric (e.g. AUC).
    subjects_df : pd.DataFrame
        One row per subject with columns 'subject_id' and 'label'.
    B : int
        Number of bootstrap draws.
    seed : int
        RNG seed.
    k : int
        Number of folds (default 5).

    Returns
    -------
    dict with keys: mean, se, ci_lower, ci_upper, B, seed
    """
    rng = np.random.default_rng(seed)
    unique_sids = np.array(subjects_df["subject_id"].unique())
    n_unique = len(unique_sids)

    subj_meta = subjects_df.drop_duplicates("subject_id").set_index("subject_id")

    boot_values = []

    for b in range(B):
        draw_seed = seed + b + 1
        # 1. Resample subject IDs with replacement
        sampled_sids = rng.choice(unique_sids, size=n_unique, replace=True)
        unique_sampled, counts = np.unique(sampled_sids, return_counts=True)

        # 2. Assign folds on UNIQUE subjects drawn
        u_labels = subj_meta.loc[unique_sampled, "label"].values
        min_class_count = min(np.sum(u_labels == 0), np.sum(u_labels == 1))
        n_splits = min(k, max(2, min_class_count))

        if min_class_count >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=draw_seed)
            split_gen = splitter.split(unique_sampled, u_labels)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=draw_seed)
            split_gen = splitter.split(unique_sampled)

        fold_map = {}
        for fold_idx, (_, test_indices) in enumerate(split_gen):
            for s in unique_sampled[test_indices]:
                fold_map[s] = fold_idx

        # 3. Expand duplicates into the same fold
        rows = []
        for sid, cnt in zip(unique_sampled, counts):
            subj_row = subj_meta.loc[[sid]].reset_index()
            subj_row["fold"] = fold_map[sid]
            for _ in range(cnt):
                rows.append(subj_row)

        boot_df = pd.concat(rows, ignore_index=True)

        # Build fold_splits from the expanded rows
        fold_splits = []
        for f in range(n_splits):
            train_idx = np.where(boot_df["fold"] != f)[0]
            test_idx = np.where(boot_df["fold"] == f)[0]
            fold_splits.append((train_idx, test_idx))

        # 4. Run CV evaluation (fail loudly, no bare except)
        val = run_cv(boot_df, fold_splits)
        boot_values.append(val)

    boot_values = np.array(boot_values, dtype=float)
    mean_val = float(np.mean(boot_values))
    se_val = float(np.std(boot_values, ddof=1))
    return {
        "mean": mean_val,
        "se": se_val,
        "ci_lower": float(np.percentile(boot_values, 2.5)),
        "ci_upper": float(np.percentile(boot_values, 97.5)),
        "B": B,
        "seed": seed,
        "n_valid_draws": int(len(boot_values)),
    }


# ---------------------------------------------------------------------------
# LeDell influence-curve CI  (LeDell, Petersen & van der Laan 2015)
# ---------------------------------------------------------------------------

def ledell_ci(
    y: np.ndarray,
    p: np.ndarray,
    folds: np.ndarray = None,
    alpha: float = 0.05,
) -> dict:
    """LeDell influence-curve CI for cross-validated AUC.

    Asymptotically exact variance via the influence function:
    For positive case i: D_i = (n / n1) * (psi_pos[i] - theta)
    For negative case j: D_j = (n / n0) * (theta - psi_neg[j])
    Var(theta) = (1 / n^2) * sum(D_k^2).

    Parameters
    ----------
    y : array of 0/1 labels (n,)
    p : array of predicted probabilities (n,)
    folds : optional array of integer fold assignments
    alpha : significance level (default 0.05 → 95% CI)

    Returns
    -------
    dict: {auc, se, ci_lower, ci_upper, method}
    """
    y, p = np.asarray(y, int), np.asarray(p, float)
    n = len(y)
    theta = _auc(y, p)

    pos_mask = y == 1
    neg_mask = y == 0
    n1 = int(pos_mask.sum())
    n0 = int(neg_mask.sum())

    if n1 == 0 or n0 == 0:
        return {"auc": float("nan"), "se": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}

    p_pos = p[pos_mask]
    p_neg = p[neg_mask]

    psi_pos = np.mean(
        (p_neg[np.newaxis, :] < p_pos[:, np.newaxis]).astype(float)
        + 0.5 * (p_neg[np.newaxis, :] == p_pos[:, np.newaxis]).astype(float),
        axis=1,
    )
    psi_neg = np.mean(
        (p_pos[np.newaxis, :] > p_neg[:, np.newaxis]).astype(float)
        + 0.5 * (p_pos[np.newaxis, :] == p_neg[:, np.newaxis]).astype(float),
        axis=1,
    )

    # Influence function values
    D = np.zeros(n)
    D[pos_mask] = (float(n) / n1) * (psi_pos - theta)
    D[neg_mask] = (float(n) / n0) * (theta - psi_neg)

    # Variance of theta = (1 / n^2) * sum(D^2) = (1 / n) * Var(D)
    var_theta = float(np.sum(D ** 2) / (n ** 2))
    se = float(np.sqrt(var_theta))

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return {
        "auc": float(theta),
        "se": se,
        "ci_lower": float(theta - z * se),
        "ci_upper": float(theta + z * se),
        "method": "ledell_ic",
    }


# ---------------------------------------------------------------------------
# External bootstrap (for Phase C, WearGait-PD)
# ---------------------------------------------------------------------------

def external_bootstrap(
    y: np.ndarray,
    p: np.ndarray,
    B: int,
    seed: int,
) -> dict:
    """Non-parametric bootstrap CI for a single-set evaluation (e.g. external test set).

    Returns {mean, se, ci_lower, ci_upper, B, seed}.
    """
    y, p = np.asarray(y, int), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) == 2:
            boot_aucs.append(_auc(y_b, p_b))
    boot_aucs = np.array(boot_aucs)
    return {
        "mean": float(np.mean(boot_aucs)),
        "se": float(np.std(boot_aucs, ddof=1)),
        "ci_lower": float(np.percentile(boot_aucs, 2.5)),
        "ci_upper": float(np.percentile(boot_aucs, 97.5)),
        "B": B,
        "seed": seed,
        "n_valid_draws": int(len(boot_aucs)),
    }
