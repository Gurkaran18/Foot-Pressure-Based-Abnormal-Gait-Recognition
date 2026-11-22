"""Pseudo-SelfPace walks (4 segments x 4-5 consecutive valid strides) built from gaitpdb walks.

Owner: Lead (walk construction, feature tables) / Teammate (evaluate). Spec §4, R1/R4; decision D7.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import walk_features_v2
from .guards import safe_write
from .kcurve import draw_seed
from . import models, paths, splits


def eligible_strides(st: pd.DataFrame) -> pd.DataFrame:
    """Valid strides whose neighbours (stride_idx +/- 1) are also valid."""
    st = st.sort_values("stride_idx")
    valid_idx = set(st.loc[st["valid"], "stride_idx"])
    keep = st["valid"] & st["stride_idx"].map(lambda i: (i - 1) in valid_idx and (i + 1) in valid_idx)
    return st[keep]


def _runs(idx) -> list:
    """Split sorted stride indices into runs of consecutive integers."""
    idx = list(idx)
    if not idx:
        return []
    runs, cur = [], [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1:
            cur.append(i)
        else:
            runs.append(cur)
            cur = [i]
    runs.append(cur)
    return runs


def _segments_one_foot(st: pd.DataFrame, cfg: dict, rng):
    el = eligible_strides(st)
    runs = _runs(sorted(el["stride_idx"]))
    lengths = rng.choice(cfg["strides_per_segment"], cfg["segments"])
    # greedy random placement: shuffle runs, carve segments from free space
    free = [list(r) for r in runs]
    chosen = []
    for L in lengths:
        cand = [(ri, s) for ri, r in enumerate(free) for s in range(len(r) - L + 1)]
        if not cand:
            return None
        ri, s = cand[rng.integers(len(cand))]
        r = free.pop(ri)
        chosen.append(r[s:s + L])
        left, right = r[:s], r[s + L:]       # keep a 1-stride gap so segments are never adjacent
        if len(left) > 1:
            free.append(left[:-1])
        if len(right) > 1:
            free.append(right[1:])
    idx = [i for seg in chosen for i in seg]
    out = el[el["stride_idx"].isin(idx)].copy()
    seg_of = {i: k for k, seg in enumerate(sorted(chosen)) for i in seg}
    out["segment"] = out["stride_idx"].map(seg_of)
    return out


def emulate_walk(strides_L, strides_R, cfg: dict, rng):
    """Lead. (L, R) emulated stride sets with a `segment` column, or None if not possible."""
    L = _segments_one_foot(strides_L, cfg, rng)
    R = _segments_one_foot(strides_R, cfg, rng)
    return None if L is None or R is None else (L, R)


def feature_tables(strides: pd.DataFrame, subjects: pd.DataFrame, cfg: dict) -> tuple:
    """Lead. Emulated-walk feature table (one row per subject per draw) + eligibility counts."""
    em, base = cfg["emulation"], cfg["cv"]["base_seed"]
    by_rec = {rid: g for rid, g in strides.groupby("recording_id")}
    rows, elig = [], []
    for d in range(em["draws"]):
        rng = np.random.default_rng(draw_seed(base, "emulation", d))
        n_ok = {0: 0, 1: 0}
        for s in subjects.sort_values("subject_id").itertuples(index=False):
            g = by_rec[s.source_recordings]
            w = emulate_walk(g[g.foot == "L"], g[g.foot == "R"], em, rng)
            if w is None:
                continue
            n_ok[int(s.label)] += 1
            rows.append({"draw": d, "subject_id": s.subject_id, "label": s.label, "study": s.study,
                         **walk_features_v2(w[0], w[1], cfg["features"])})
        elig.append({"draw": d, "n_eligible_Co": n_ok[0], "n_eligible_PD": n_ok[1]})
    return pd.DataFrame(rows), pd.DataFrame(elig)


def evaluate(feature_tables_df: pd.DataFrame, cfg: dict, full_subjects_df=None, out_dir=None):
    """Teammate. Emulated internal AUC (train-full/test-K and train-K/test-K).
    
    Scores both M1-fb (primary) and M1 (secondary).
    Evaluates D15 rule for M1-fb.
    """
    em = cfg["emulation"]
    base_seed = cfg["cv"]["base_seed"]

    if full_subjects_df is None:
        candidates = sorted(paths.OUTPUTS_DIR.glob("*/features/subject_v2.csv"))
        if candidates:
            full_df = pd.read_csv(candidates[-1])
        else:
            raise ValueError("Full-walk subject table not found for emulation evaluation.")
    else:
        full_df = full_subjects_df.copy()

    model_configs = {
        "M1-fb": cfg["features"]["t1_fallback"],
        "M1": cfg["features"]["t1_core"],
    }

    results_rows = []
    draws = sorted(feature_tables_df["draw"].unique())

    for d in draws:
        draw_df = feature_tables_df[feature_tables_df["draw"] == d].reset_index(drop=True)
        if len(draw_df) < 10:
            continue
        n_pd = int((draw_df["label"] == 1).sum())
        n_co = int((draw_df["label"] == 0).sum())

        fold_seed = draw_seed(base_seed, "emcv", d)
        fold_splits = list(splits.inner_folds(draw_df, k=cfg["cv"]["outer_folds"], seed=fold_seed))

        for model_id, feat_cols in model_configs.items():
            # 1. Regime: train-full / test-K
            y_true_full, p_pred_full = [], []
            for tr_ids, te_ids in fold_splits:
                tr_full = full_df[full_df["subject_id"].isin(tr_ids)].dropna(subset=feat_cols)
                te_em = draw_df[draw_df["subject_id"].isin(te_ids)].dropna(subset=feat_cols)
                if len(tr_full["label"].unique()) < 2 or len(te_em) == 0:
                    continue
                pipe = models.make_pipeline(model_id, cfg, seed=fold_seed)
                pipe.fit(tr_full[feat_cols].values, tr_full["label"].values)
                probs = pipe.predict_proba(te_em[feat_cols].values)[:, 1]
                y_true_full.extend(te_em["label"].values)
                p_pred_full.extend(probs)

            auc_full = roc_auc_score(y_true_full, p_pred_full) if len(np.unique(y_true_full)) == 2 else np.nan
            results_rows.append({
                "draw": d,
                "model_id": model_id,
                "regime": "train-full/test-K",
                "n_eligible_PD": n_pd,
                "n_eligible_Co": n_co,
                "auc": float(auc_full),
            })

            # 2. Regime: train-K / test-K
            y_true_k, p_pred_k = [], []
            for tr_ids, te_ids in fold_splits:
                tr_em = draw_df[draw_df["subject_id"].isin(tr_ids)].dropna(subset=feat_cols)
                te_em = draw_df[draw_df["subject_id"].isin(te_ids)].dropna(subset=feat_cols)
                if len(tr_em["label"].unique()) < 2 or len(te_em) == 0:
                    continue
                pipe = models.make_pipeline(model_id, cfg, seed=fold_seed)
                pipe.fit(tr_em[feat_cols].values, tr_em["label"].values)
                probs = pipe.predict_proba(te_em[feat_cols].values)[:, 1]
                y_true_k.extend(te_em["label"].values)
                p_pred_k.extend(probs)

            auc_k = roc_auc_score(y_true_k, p_pred_k) if len(np.unique(y_true_k)) == 2 else np.nan
            results_rows.append({
                "draw": d,
                "model_id": model_id,
                "regime": "train-K/test-K",
                "n_eligible_PD": n_pd,
                "n_eligible_Co": n_co,
                "auc": float(auc_k),
            })

    results_df = pd.DataFrame(results_rows)

    # Summarize performance
    summary = {}
    for (model_id, regime), grp in results_df.groupby(["model_id", "regime"]):
        aucs = grp["auc"].dropna().values
        summary[f"{model_id}_{regime}"] = {
            "model_id": model_id,
            "regime": regime,
            "n_draws": int(len(aucs)),
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs, ddof=1)),
            "median": float(np.median(aucs)),
            "p2_5": float(np.percentile(aucs, 2.5)),
            "p97_5": float(np.percentile(aucs, 97.5)),
        }

    # D15 Rule decision for M1-fb
    m1_fb_full_med = summary.get("M1-fb_train-full/test-K", {}).get("median", np.nan)
    m1_fb_k_med = summary.get("M1-fb_train-K/test-K", {}).get("median", np.nan)
    d15_diff = m1_fb_k_med - m1_fb_full_med
    d15_decision = "k_matched" if (not np.isnan(d15_diff) and d15_diff >= 0.02) else "full_walk"

    summary["d15_decision_m1_fb"] = {
        "rule": "train-K/test-K median AUC >= train-full/test-K median AUC + 0.02",
        "train_full_median_auc": m1_fb_full_med,
        "train_k_median_auc": m1_fb_k_med,
        "difference": float(d15_diff),
        "threshold": 0.02,
        "selected_training_regime": d15_decision,
    }

    # H3 internal reference: M1-fb emulated AUC
    # Uses the regime chosen by D15
    h3_regime_key = f"M1-fb_{'train-K/test-K' if d15_decision == 'k_matched' else 'train-full/test-K'}"
    summary["h3_internal_reference_m1_fb"] = {
        "regime": h3_regime_key,
        "auc_mean": summary[h3_regime_key]["mean"],
        "auc_median": summary[h3_regime_key]["median"],
        "auc_std": summary[h3_regime_key]["std"],
        "auc_ci_lower": summary[h3_regime_key]["p2_5"],
        "auc_ci_upper": summary[h3_regime_key]["p97_5"],
    }

    if out_dir is not None:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        safe_write(out_p / "results.csv", results_df.to_csv(index=False))
        safe_write(out_p / "summary.json", json.dumps(summary, indent=2))
        print(f"[E9] Emulation results written -> {out_p / 'results.csv'}")
        print(f"[E9] Emulation summary written -> {out_p / 'summary.json'}")

    return results_df, summary
