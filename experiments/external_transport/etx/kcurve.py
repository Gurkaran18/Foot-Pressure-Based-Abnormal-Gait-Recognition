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

import numpy as np
import pandas as pd

from .features import walk_features_v2


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


def evaluate(feature_tables, cfg):
    """Teammate. CV AUC per (scheme, K, draw, regime); ICC(2,1), within-SD, bias, univariate AUC."""
    raise NotImplementedError
