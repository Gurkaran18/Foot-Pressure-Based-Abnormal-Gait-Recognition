"""K-curve: S1/S2 stride sampling + K-stride feature tables (Lead) and evaluation (Teammate).

Owner: Lead (samplers, feature tables) / Teammate (evaluate). Spec §4, R4; decision D7.

S1_contiguous: pick a random start time = heel strike of a valid stride (either foot) such that each foot
               still has >= K valid strides from there; take the next K valid strides of each foot.
               Mimics one continuous stretch of walking.
S2_random:     K valid strides per foot drawn at random without replacement (sensitivity scheme).
K = "all":     every valid stride (the full-walk reference).
A subject is eligible for (scheme, K) only if both feet have enough valid strides; ineligible subjects
are reported in the eligibility counts, never padded.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import walk_features_v2
from .guards import safe_write
from . import models, paths, splits


def _valid(st: pd.DataFrame) -> pd.DataFrame:
    return st[st["valid"]].sort_values("hs_s").reset_index(drop=True)


def draw_seed(base_seed: int, *parts) -> int:
    """Deterministic 32-bit seed from the base seed and labels (scheme, K, draw, ...)."""
    h = hashlib.sha256(":".join(map(str, (base_seed, *parts))).encode()).hexdigest()
    return int(h[:8], 16)


def sample_strides(strides_L, strides_R, K, scheme: str, rng):
    """Lead. Return (L_subset, R_subset) of valid strides, or None if ineligible."""
    L, R = _valid(strides_L), _valid(strides_R)
    if K == "all":
        return L, R
    K = int(K)
    if min(len(L), len(R)) < K:
        return None
    if scheme == "S2_random":
        return (L.iloc[np.sort(rng.choice(len(L), K, replace=False))],
                R.iloc[np.sort(rng.choice(len(R), K, replace=False))])
    if scheme != "S1_contiguous":
        raise ValueError(f"unknown scheme {scheme!r}")
    lh, rh = L["hs_s"].to_numpy(), R["hs_s"].to_numpy()          # sorted (from _valid)
    starts = np.sort(np.r_[lh, rh])
    n_after_L = len(lh) - np.searchsorted(lh, starts, side="left")   # valid strides with hs >= t
    n_after_R = len(rh) - np.searchsorted(rh, starts, side="left")
    ok = starts[(n_after_L >= K) & (n_after_R >= K)]
    if not len(ok):
        return None
    t0 = ok[rng.integers(len(ok))]
    return L[L["hs_s"] >= t0].iloc[:K], R[R["hs_s"] >= t0].iloc[:K]


def feature_tables(strides: pd.DataFrame, subjects: pd.DataFrame, cfg: dict) -> tuple:
    """Lead. K-stride feature tables for every (scheme, K, draw) and the eligibility counts.

    strides:  events/strides.csv (recording_id, foot, ... valid)
    subjects: subject_v2.csv (subject_id, label, study, source_recordings)
    Returns (features_df, eligibility_df). One row per subject per (scheme, K, draw).
    """
    kc, base = cfg["kcurve"], cfg["cv"]["base_seed"]
    by_rec = {rid: g for rid, g in strides.groupby("recording_id")}
    rows, elig = [], []
    for scheme in kc["schemes"]:
        for K in kc["levels"]:
            draws = 1 if K == "all" else kc["draws"]
            for d in range(draws):
                rng = np.random.default_rng(draw_seed(base, "kcurve", scheme, K, d))
                n_ok = {0: 0, 1: 0}
                for s in subjects.sort_values("subject_id").itertuples(index=False):
                    g = by_rec[s.source_recordings]
                    sub = sample_strides(g[g.foot == "L"], g[g.foot == "R"], K, scheme, rng)
                    if sub is None:
                        continue
                    n_ok[int(s.label)] += 1
                    rows.append({"scheme": scheme, "K": str(K), "draw": d, "subject_id": s.subject_id,
                                 "label": s.label, "study": s.study,
                                 **walk_features_v2(sub[0], sub[1], cfg["features"])})
                elig.append({"scheme": scheme, "K": str(K), "draw": d,
                             "n_eligible_Co": n_ok[0], "n_eligible_PD": n_ok[1]})
    return pd.DataFrame(rows), pd.DataFrame(elig)


def _compute_icc21(matrix: np.ndarray) -> float:
    """Compute ICC(2,1) two-way random single-rater absolute agreement."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return np.nan
    mean_total = np.mean(matrix)
    mean_row = np.mean(matrix, axis=1)
    mean_col = np.mean(matrix, axis=0)

    ssr = k * np.sum((mean_row - mean_total) ** 2)
    ssc = n * np.sum((mean_col - mean_total) ** 2)
    sst = np.sum((matrix - mean_total) ** 2)
    sse = sst - ssr - ssc

    msr = ssr / (n - 1)
    msc = ssc / (k - 1)
    mse = max(0.0, sse / ((n - 1) * (k - 1)))

    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denom <= 0:
        return 0.0
    return float(np.clip((msr - mse) / denom, -1.0, 1.0))


def evaluate(feature_tables_df: pd.DataFrame, cfg: dict, full_subjects_df=None, out_dir=None):
    """Teammate. CV AUC per (scheme, K, draw, regime); ICC(2,1), within-SD, bias, univariate AUC.
    
    Evaluates both M1-fb and M1.
    """
    kc = cfg["kcurve"]
    base_seed = cfg["cv"]["base_seed"]

    # Discover full-walk reference features
    if full_subjects_df is None:
        full_df = feature_tables_df[feature_tables_df["K"] == "all"].copy()
        if len(full_df) == 0:
            # Pre-freeze fix 5: no "latest folder" guess; pass full_subjects_df explicitly.
            raise ValueError("No K='all' rows and no full_subjects_df given for K-curve evaluation.")
    else:
        full_df = full_subjects_df.copy()

    # Models to evaluate
    model_configs = {
        "M1-fb": cfg["features"]["t1_fallback"],
        "M1": cfg["features"]["t1_core"],
    }

    results_rows = []
    schemes = sorted(feature_tables_df["scheme"].unique())
    k_levels = [lvl for lvl in kc["levels"] if str(lvl) in feature_tables_df["K"].values]

    # Full-walk reference univariate AUCs
    full_uauc = {}
    for feat in cfg["features"]["t1_core"]:
        if feat in full_df.columns:
            valid_m = full_df[[feat, "label"]].dropna()
            if len(valid_m["label"].unique()) == 2:
                uauc = roc_auc_score(valid_m["label"].values, valid_m[feat].values)
                full_uauc[feat] = max(uauc, 1.0 - uauc)

    for scheme in schemes:
        for K_val in k_levels:
            k_str = str(K_val)
            sub_k = feature_tables_df[(feature_tables_df["scheme"] == scheme) & (feature_tables_df["K"] == k_str)]
            draws = sorted(sub_k["draw"].unique())

            for d in draws:
                draw_df = sub_k[sub_k["draw"] == d].reset_index(drop=True)
                if len(draw_df) < 10:
                    continue
                n_pd = int((draw_df["label"] == 1).sum())
                n_co = int((draw_df["label"] == 0).sum())

                # Generate 5 folds
                fold_seed = draw_seed(base_seed, "kcv", scheme, k_str, d)
                fold_splits = list(splits.inner_folds(draw_df, k=cfg["cv"]["outer_folds"], seed=fold_seed))

                for model_id, feat_cols in model_configs.items():
                    # 1. Regime: train-full / test-K
                    y_true_full, p_pred_full = [], []
                    for tr_ids, te_ids in fold_splits:
                        # Train on full-walk features for tr_ids
                        tr_full = full_df[full_df["subject_id"].isin(tr_ids)].dropna(subset=feat_cols)
                        te_k = draw_df[draw_df["subject_id"].isin(te_ids)].dropna(subset=feat_cols)
                        if len(tr_full["label"].unique()) < 2 or len(te_k) == 0:
                            continue
                        pipe = models.make_pipeline(model_id, cfg, seed=fold_seed)
                        pipe.fit(tr_full[feat_cols].values, tr_full["label"].values)
                        probs = pipe.predict_proba(te_k[feat_cols].values)[:, 1]
                        y_true_full.extend(te_k["label"].values)
                        p_pred_full.extend(probs)

                    auc_full = roc_auc_score(y_true_full, p_pred_full) if len(np.unique(y_true_full)) == 2 else np.nan
                    results_rows.append({
                        "scheme": scheme, "K": k_str, "draw": d,
                        "regime": "train-full/test-K", "model_id": model_id,
                        "n_eligible_PD": n_pd, "n_eligible_Co": n_co,
                        "auc": float(auc_full),
                    })

                    # 2. Regime: train-K / test-K
                    y_true_k, p_pred_k = [], []
                    for tr_ids, te_ids in fold_splits:
                        tr_k = draw_df[draw_df["subject_id"].isin(tr_ids)].dropna(subset=feat_cols)
                        te_k = draw_df[draw_df["subject_id"].isin(te_ids)].dropna(subset=feat_cols)
                        if len(tr_k["label"].unique()) < 2 or len(te_k) == 0:
                            continue
                        pipe = models.make_pipeline(model_id, cfg, seed=fold_seed)
                        pipe.fit(tr_k[feat_cols].values, tr_k["label"].values)
                        probs = pipe.predict_proba(te_k[feat_cols].values)[:, 1]
                        y_true_k.extend(te_k["label"].values)
                        p_pred_k.extend(probs)

                    auc_k = roc_auc_score(y_true_k, p_pred_k) if len(np.unique(y_true_k)) == 2 else np.nan
                    results_rows.append({
                        "scheme": scheme, "K": k_str, "draw": d,
                        "regime": "train-K/test-K", "model_id": model_id,
                        "n_eligible_PD": n_pd, "n_eligible_Co": n_co,
                        "auc": float(auc_k),
                    })

    results_df = pd.DataFrame(results_rows)

    # Stability Statistics per feature
    stability_rows = []
    full_by_subj = full_df.set_index("subject_id")
    for scheme in schemes:
        for K_val in [lvl for lvl in k_levels if str(lvl) != "all"]:
            k_str = str(K_val)
            sub_k = feature_tables_df[(feature_tables_df["scheme"] == scheme) & (feature_tables_df["K"] == k_str)]
            draws = sorted(sub_k["draw"].unique())

            for feat in cfg["features"]["t1_core"]:
                if feat not in sub_k.columns or feat not in full_by_subj.columns:
                    continue

                # Pivot subjects x draws
                piv = sub_k.pivot(index="subject_id", columns="draw", values=feat).dropna()
                if len(piv) < 5 or piv.shape[1] < 2:
                    continue

                mat = piv.values
                icc = _compute_icc21(mat)
                within_sd = float(np.mean(np.std(mat, axis=1, ddof=1)))

                # Bias vs full
                common_subj = piv.index.intersection(full_by_subj.index)
                feat_k_mean = float(sub_k[sub_k["subject_id"].isin(common_subj)][feat].mean())
                feat_full_mean = float(full_by_subj.loc[common_subj, feat].mean())
                bias = feat_k_mean - feat_full_mean

                # Univariate AUC change
                uaucs = []
                for d in draws:
                    d_df = sub_k[sub_k["draw"] == d][[feat, "label"]].dropna()
                    if len(d_df["label"].unique()) == 2:
                        ua = roc_auc_score(d_df["label"].values, d_df[feat].values)
                        uaucs.append(max(ua, 1.0 - ua))
                mean_uauc = float(np.mean(uaucs)) if uaucs else np.nan
                d_uauc = mean_uauc - full_uauc.get(feat, np.nan)

                stability_rows.append({
                    "scheme": scheme,
                    "K": k_str,
                    "feature": feat,
                    "icc21": round(icc, 4),
                    "within_sd": round(within_sd, 5),
                    "bias_vs_full": round(bias, 5),
                    "univariate_auc": round(mean_uauc, 4),
                    "delta_univariate_auc": round(d_uauc, 4),
                })

    stability_df = pd.DataFrame(stability_rows)

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        safe_write(out_p / "results.csv", results_df.to_csv(index=False))
        safe_write(out_p / "stability.csv", stability_df.to_csv(index=False))
        print(f"[E8] K-curve results written -> {out_p / 'results.csv'}")
        print(f"[E8] Stability written -> {out_p / 'stability.csv'}")

    return results_df, stability_df
