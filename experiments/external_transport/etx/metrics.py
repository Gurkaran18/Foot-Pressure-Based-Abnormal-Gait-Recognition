"""AUC, fast DeLong (Sun & Xu 2014), Brier, calibration intercept/slope, sens/spec/BAcc, AAUC.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §4, §9
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats, optimize


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
    """Logistic recalibration:
    - Intercept: fitted with model logit as an offset (slope fixed to 1).
      Solves sum(y - 1 / (1 + exp(-(alpha + logit)))) = 0.
    - Slope: fitted separately via unconstrained logistic regression y ~ logit.

    Returns {intercept, slope}.
    A well-calibrated model has intercept ≈ 0 and slope ≈ 1.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p))

    # 1. Calibration-in-the-large: intercept with logit as offset (slope fixed to 1)
    def _score(alpha):
        p_adj = 1.0 / (1.0 + np.exp(-(alpha + logit)))
        return np.sum(y - p_adj)

    # Bracket search for root
    lo, hi = -10.0, 10.0
    while _score(lo) < 0:
        lo -= 10.0
    while _score(hi) > 0:
        hi += 10.0

    try:
        intercept = float(optimize.brentq(_score, lo, hi))
    except Exception:
        # Fallback to optimization if root finding fails
        def _nll(alpha):
            p_adj = np.clip(1.0 / (1.0 + np.exp(-(alpha + logit))), 1e-15, 1.0 - 1e-15)
            return -np.sum(y * np.log(p_adj) + (1 - y) * np.log(1 - p_adj))
        res = optimize.minimize_scalar(_nll, bounds=(lo, hi), method="bounded")
        intercept = float(res.x)

    # 2. Calibration slope from separate unconstrained fit y ~ logit
    X = logit.reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    lr.fit(X, y)
    slope = float(lr.coef_[0, 0])

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
# AAUC: Janes & Pepe Covariate-Adjusted AUC
# ---------------------------------------------------------------------------

def _compute_aauc_point(y: np.ndarray, score: np.ndarray, covariate: np.ndarray, strata: np.ndarray) -> float:
    """Within each study/stratum: model score ~ speed in CONTROLS (linear),
    standardise residuals, case placement value = share of control residuals
    below the case residual; AAUC = mean over cases.
    """
    case_placement_values = []

    unique_strata = np.unique(strata)
    for s in unique_strata:
        mask_s = strata == s
        ctrl_mask = mask_s & (y == 0)
        case_mask = mask_s & (y == 1)

        n_ctrl = int(np.sum(ctrl_mask))
        n_case = int(np.sum(case_mask))

        if n_ctrl == 0 or n_case == 0:
            continue

        cov_ctrl = covariate[ctrl_mask]
        sc_ctrl = score[ctrl_mask]

        # Model score ~ covariate in CONTROLS (linear regression)
        if len(np.unique(cov_ctrl)) > 1:
            slope, intercept, _, _, _ = stats.linregress(cov_ctrl, sc_ctrl)
        else:
            slope, intercept = 0.0, float(np.mean(sc_ctrl))

        # Control residuals and residual SD
        r_ctrl = sc_ctrl - (intercept + slope * cov_ctrl)
        sd_ctrl = float(np.std(r_ctrl, ddof=1)) if len(r_ctrl) > 1 else 1.0
        if sd_ctrl <= 1e-12 or np.isnan(sd_ctrl):
            sd_ctrl = 1.0
        e_ctrl = r_ctrl / sd_ctrl

        # Case residuals standardised by control SD
        cov_case = covariate[case_mask]
        sc_case = score[case_mask]
        r_case = sc_case - (intercept + slope * cov_case)
        e_case = r_case / sd_ctrl

        # Case placement value: share of control residuals below the case residual
        # Vectorised comparison: e_ctrl[newaxis, :] vs e_case[:, newaxis]
        u_case = np.mean(
            (e_ctrl[np.newaxis, :] < e_case[:, np.newaxis]).astype(float)
            + 0.5 * (e_ctrl[np.newaxis, :] == e_case[:, np.newaxis]).astype(float),
            axis=1,
        )
        case_placement_values.extend(u_case)

    if len(case_placement_values) == 0:
        return float("nan")

    return float(np.mean(case_placement_values))


def aauc(
    y: np.ndarray,
    score: np.ndarray,
    covariate: np.ndarray,
    strata: np.ndarray = None,
    B: int = 500,
    seed: int = 42,
) -> dict:
    """Janes & Pepe covariate-adjusted AUC with bootstrap CI.

    Within each study/stratum, model score ~ speed in CONTROLS (linear),
    standardise residuals, case placement value = share of control residuals
    below the case residual; AAUC = mean over cases.
    CI is estimated via stratified participant bootstrap.
    """
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    covariate = np.asarray(covariate, float)

    if strata is None:
        strata = np.ones(len(y), dtype=int)
    else:
        strata = np.asarray(strata)

    point_aauc = _compute_aauc_point(y, score, covariate, strata)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    n = len(y)
    boot_vals = []

    # Resample within (stratum, label) cells to ensure cases & controls in each stratum
    cells = []
    for s in np.unique(strata):
        for lab in (0, 1):
            idx = np.where((strata == s) & (y == lab))[0]
            if len(idx) > 0:
                cells.append(idx)

    for _ in range(B):
        boot_idx = []
        for cell_idx in cells:
            sampled_idx = rng.choice(cell_idx, size=len(cell_idx), replace=True)
            boot_idx.extend(sampled_idx)
        boot_idx = np.array(boot_idx)

        val = _compute_aauc_point(
            y[boot_idx], score[boot_idx], covariate[boot_idx], strata[boot_idx]
        )
        if not np.isnan(val):
            boot_vals.append(val)

    if len(boot_vals) > 0:
        ci_lower = float(np.percentile(boot_vals, 2.5))
        ci_upper = float(np.percentile(boot_vals, 97.5))
        se = float(np.std(boot_vals, ddof=1))
    else:
        ci_lower, ci_upper, se = float("nan"), float("nan"), float("nan")

    return {
        "aauc": point_aauc,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "B": B,
    }
