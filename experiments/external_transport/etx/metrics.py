"""AUC, fast DeLong (Sun & Xu 2014), Brier, calibration intercept/slope, sens/spec/BAcc, AAUC.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §4, §9
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


# ---------------------------------------------------------------------------
# AUC
# ---------------------------------------------------------------------------

def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve (wraps sklearn for consistency)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


# ---------------------------------------------------------------------------
# Fast DeLong variance and paired test  (Sun & Xu 2014, Algorithm 2)
# ---------------------------------------------------------------------------

def _delong_components(y: np.ndarray, p: np.ndarray):
    """Return (theta, V10, V01) for the DeLong estimator."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)
    if n1 == 0 or n0 == 0:
        raise ValueError("Need both positive and negative cases for DeLong.")

    p_pos = p[pos_idx]  # shape (n1,)
    p_neg = p[neg_idx]  # shape (n0,)

    # Psi_{X_i} = (rank of X_i among Y)  (placement values)
    # V10[i] = (1/n0) * sum_j 1{Y_j < X_i} + 0.5 * 1{Y_j == X_i}
    psi_pos = np.mean(
        (p_neg[np.newaxis, :] < p_pos[:, np.newaxis]).astype(float)
        + 0.5 * (p_neg[np.newaxis, :] == p_pos[:, np.newaxis]).astype(float),
        axis=1,
    )  # (n1,)

    psi_neg = np.mean(
        (p_pos[np.newaxis, :] > p_neg[:, np.newaxis]).astype(float)
        + 0.5 * (p_pos[np.newaxis, :] == p_neg[:, np.newaxis]).astype(float),
        axis=1,
    )  # (n0,)

    theta = float(np.mean(psi_pos))  # = AUC
    V10 = float(np.var(psi_pos, ddof=1) / n1)
    V01 = float(np.var(psi_neg, ddof=1) / n0)
    return theta, V10, V01


def delong(
    y: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray = None,
) -> dict:
    """DeLong variance and optional paired test.

    If p2 is None: returns {auc, variance, se, ci_lower, ci_upper}.
    If p2 is given: also returns {delta_auc, z, p_value} for the paired test.
    CI is 95% normal interval.
    """
    y = np.asarray(y, int)
    p1 = np.asarray(p1, float)
    theta1, V10_1, V01_1 = _delong_components(y, p1)
    var1 = V10_1 + V01_1
    se1 = float(np.sqrt(var1))
    result = {
        "auc": theta1,
        "variance": var1,
        "se": se1,
        "ci_lower": float(theta1 - 1.96 * se1),
        "ci_upper": float(theta1 + 1.96 * se1),
    }

    if p2 is not None:
        p2 = np.asarray(p2, float)
        theta2, V10_2, V01_2 = _delong_components(y, p2)

        # Covariance between the two estimators requires the joint placement values.
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        p1_pos, p1_neg = p1[pos_idx], p1[neg_idx]
        p2_pos, p2_neg = p2[pos_idx], p2[neg_idx]

        psi1_pos = np.mean(
            (p1_neg[np.newaxis, :] < p1_pos[:, np.newaxis]).astype(float)
            + 0.5 * (p1_neg[np.newaxis, :] == p1_pos[:, np.newaxis]).astype(float),
            axis=1,
        )
        psi2_pos = np.mean(
            (p2_neg[np.newaxis, :] < p2_pos[:, np.newaxis]).astype(float)
            + 0.5 * (p2_neg[np.newaxis, :] == p2_pos[:, np.newaxis]).astype(float),
            axis=1,
        )
        psi1_neg = np.mean(
            (p1_pos[np.newaxis, :] > p1_neg[:, np.newaxis]).astype(float)
            + 0.5 * (p1_pos[np.newaxis, :] == p1_neg[:, np.newaxis]).astype(float),
            axis=1,
        )
        psi2_neg = np.mean(
            (p2_pos[np.newaxis, :] > p2_neg[:, np.newaxis]).astype(float)
            + 0.5 * (p2_pos[np.newaxis, :] == p2_neg[:, np.newaxis]).astype(float),
            axis=1,
        )

        n1, n0 = len(pos_idx), len(neg_idx)
        cov10 = float(np.cov(psi1_pos, psi2_pos, ddof=1)[0, 1]) / n1
        cov01 = float(np.cov(psi1_neg, psi2_neg, ddof=1)[0, 1]) / n0
        var_diff = var1 + (V10_2 + V01_2) - 2 * (cov10 + cov01)
        delta = theta1 - theta2
        z = float(delta / np.sqrt(max(var_diff, 1e-16)))
        result.update({
            "auc2": theta2,
            "delta_auc": delta,
            "var_diff": var_diff,
            "z": z,
            "p_value": float(2 * stats.norm.sf(abs(z))),
        })

    return result


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between probabilities and 0/1 labels."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


# ---------------------------------------------------------------------------
# Calibration intercept and slope (logistic recalibration)
# ---------------------------------------------------------------------------

def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    """Logistic recalibration: fit intercept and slope with model logit as offset.

    Returns {intercept, slope}.
    A well-calibrated model has intercept ≈ 0 and slope ≈ 1.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, int)
    p = np.asarray(p, float)
    # Clip to avoid log(0)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))

    # Intercept-only model (slope = 1 forced as offset): calibration-in-the-large
    # Then full model: logit as single feature
    X = logit.reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    lr.fit(X, y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    return {"intercept": intercept, "slope": slope}


# ---------------------------------------------------------------------------
# Threshold metrics: sensitivity, specificity, balanced accuracy
# ---------------------------------------------------------------------------

def threshold_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """Sensitivity, specificity and balanced accuracy at a decision threshold."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    bacc = float(np.nanmean([sens, spec]))
    return {
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": bacc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": thr,
    }


# ---------------------------------------------------------------------------
# AAUC: adjusted AUC (Janes & Pepe)
# ---------------------------------------------------------------------------

def aauc(
    y: np.ndarray,
    score: np.ndarray,
    covariate: np.ndarray,
    strata: np.ndarray = None,
) -> dict:
    """Adjusted AUC using placement values (Janes & Pepe, Biostatistics 2008).

    Computes AUC of the score adjusted for a continuous covariate (e.g. speed).
    Within each stratum (e.g. study), placement values of the covariate are
    computed on the control group and subtracted from the score's placement values.
    Returns {aauc, variance, se, ci_lower, ci_upper}.
    """
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    covariate = np.asarray(covariate, float)

    if strata is None:
        strata = np.ones(len(y), dtype=int)
    else:
        strata = np.asarray(strata)

    pos_mask = y == 1
    neg_mask = y == 0

    # Placement values of `score` among controls
    p_score_neg = score[neg_mask]
    placement_score = np.mean(
        (p_score_neg[np.newaxis, :] < score[:, np.newaxis]).astype(float)
        + 0.5 * (p_score_neg[np.newaxis, :] == score[:, np.newaxis]).astype(float),
        axis=1,
    )

    # Placement values of `covariate` among controls (within strata)
    placement_cov = np.zeros(len(y))
    for s in np.unique(strata):
        mask_s = strata == s
        ctrl_s = covariate[mask_s & neg_mask]
        for i in np.where(mask_s)[0]:
            cov_i = covariate[i]
            placement_cov[i] = float(
                np.mean((ctrl_s < cov_i).astype(float) + 0.5 * (ctrl_s == cov_i).astype(float))
            )

    # Adjusted score = placement of score - placement of covariate
    adjusted = placement_score - placement_cov

    # AUC of adjusted score
    if len(np.unique(y)) < 2:
        return {"aauc": float("nan"), "variance": float("nan"), "se": float("nan"),
                "ci_lower": float("nan"), "ci_upper": float("nan")}

    theta_adj, V10, V01 = _delong_components(y, adjusted)
    var_adj = V10 + V01
    se_adj = float(np.sqrt(var_adj))
    return {
        "aauc": theta_adj,
        "variance": var_adj,
        "se": se_adj,
        "ci_lower": float(theta_adj - 1.96 * se_adj),
        "ci_upper": float(theta_adj + 1.96 * se_adj),
    }
