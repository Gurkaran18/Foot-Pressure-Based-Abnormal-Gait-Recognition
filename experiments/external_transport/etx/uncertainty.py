"""Repeat summaries, bootstrap-of-CV (grouped duplicates), LeDell CI, external bootstrap, coef stability.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §9 (R5)
"""
import hashlib
from typing import Callable

import numpy as np
import pandas as pd

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
) -> dict:
    """Bootstrap-of-CV with grouped (subject-level) resampling.

    Parameters
    ----------
    run_cv : callable(df) -> float
        A function that takes a subject DataFrame and returns an AUC (or metric).
        Internally it must do subject-level CV (not window-level).
    subjects_df : pd.DataFrame
        One row per subject with at least 'subject_id'.
    B : int
        Number of bootstrap draws.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys: mean, se, ci_lower, ci_upper, B, seed
    """
    rng = np.random.default_rng(seed)
    ids = subjects_df["subject_id"].unique()
    boot_values = []

    for b in range(B):
        sampled_ids = rng.choice(ids, size=len(ids), replace=True)
        # Build a resampled DataFrame; subjects that appear k times get k copies
        # with the same fold assignment (grouped duplicates — same fold as original)
        counts = {}
        for sid in sampled_ids:
            counts[sid] = counts.get(sid, 0) + 1
        rows = []
        for sid, cnt in counts.items():
            row = subjects_df[subjects_df["subject_id"] == sid].copy()
            for _ in range(cnt):
                rows.append(row)
        boot_df = pd.concat(rows, ignore_index=True)
        try:
            val = run_cv(boot_df)
            boot_values.append(val)
        except Exception:
            pass  # skip failed draws

    boot_values = np.array(boot_values)
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
    folds: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """LeDell influence-curve CI for cross-validated AUC.

    Parameters
    ----------
    y : array of 0/1 labels (n,)
    p : array of predicted probabilities (n,)
    folds : array of integer fold assignments (n,)  — same fold = same subject's data
    alpha : significance level (default 0.05 → 95% CI)

    Returns
    -------
    dict: {auc, se, ci_lower, ci_upper}
    """
    from scipy import stats as _stats
    y, p, folds = np.asarray(y, int), np.asarray(p, float), np.asarray(folds, int)
    theta = _auc(y, p)

    pos_mask = y == 1
    neg_mask = y == 0
    n1 = int(pos_mask.sum())
    n0 = int(neg_mask.sum())

    # Influence function for AUC
    # IC_i (positive) = (1/n0) sum_j 1{X_i > Y_j} - theta  (Delong's V10 term)
    # IC_j (negative) = (1/n1) sum_i 1{X_i > Y_j} - theta  (Delong's V01 term)
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

    ic = np.zeros(len(y))
    ic[pos_mask] = (psi_pos - theta) / n1
    ic[neg_mask] = (psi_neg - theta) / n0

    # Variance = sum over folds of (fold IC)^2 (grouped)
    var_ic = 0.0
    unique_folds = np.unique(folds)
    for f in unique_folds:
        fold_ic = ic[folds == f].sum()
        var_ic += fold_ic ** 2
    se = float(np.sqrt(var_ic))

    z = float(_stats.norm.ppf(1 - alpha / 2))
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
