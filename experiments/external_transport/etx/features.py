"""T1 features: v2 (spec §6, primary) and v1 (legacy-equivalent, Block 4 equivalence only).

Owner: Lead (data/*). Test T5.

v2 definitions (valid strides only; T = stride, S = stance, W = swing, seconds):
  stride_time_mean      mean T over both feet pooled
  stride_time_cv        (CV(T_L) + CV(T_R)) / 2,  CV = 100 * sd(ddof=1) / mean   (decision D17)
  swing_time_cv         (CV(W_L) + CV(W_R)) / 2
  stance_pct            100 * mean(S / T) over both feet pooled
  asym_swing            100 * |ln(mean W_R / mean W_L)|
  double_support_v2_pct per valid stride: overlap of its contact [hs, to] with the other foot's valid
                        contact intervals inside [hs, next_hs], / T; averaged per foot, then over feet
                        (conditional feature, outside the primary model; decision D1)
"""
import numpy as np
import pandas as pd

T1_CORE = ["stride_time_mean", "stride_time_cv", "swing_time_cv", "stance_pct", "asym_swing"]
WALK_V2_FEATURES = T1_CORE + ["double_support_v2_pct"]


def cv(x, ddof: int = 1) -> float:
    x = np.asarray(x, float)
    if x.size < 2:
        return np.nan
    return 100.0 * x.std(ddof=ddof) / x.mean()


def _ds_one_side(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean over strides of `a` of the fraction of the stride where `b`'s valid contacts overlap a's contact."""
    if a.empty or b.empty:
        return np.nan
    b_hs, b_to = b["hs_s"].to_numpy(), b["to_s"].to_numpy()
    fr = []
    for hs, to, nhs, T in a[["hs_s", "to_s", "next_hs_s", "stride_s"]].itertuples(index=False):
        lo = np.maximum(b_hs, hs)
        hi = np.minimum(np.minimum(b_to, to), nhs)
        fr.append(np.clip(hi - lo, 0, None).sum() / T)
    return float(np.mean(fr))


def walk_features_v2(strides_L: pd.DataFrame, strides_R: pd.DataFrame, cfg: dict) -> dict:
    """T1-core + double_support_v2_pct + n_valid_L/R. cfg = `features` block of config/a2.yaml."""
    L = strides_L[strides_L["valid"]] if len(strides_L) else strides_L
    R = strides_R[strides_R["valid"]] if len(strides_R) else strides_R
    ddof = cfg.get("cv_ddof", 1)
    out = {"n_valid_L": int(len(L)), "n_valid_R": int(len(R))}
    if len(L) < 2 or len(R) < 2:
        out.update({k: np.nan for k in WALK_V2_FEATURES})
        return out
    both = pd.concat([L, R])
    out["stride_time_mean"] = float(both["stride_s"].mean())
    out["stride_time_cv"] = (cv(L["stride_s"], ddof) + cv(R["stride_s"], ddof)) / 2
    out["swing_time_cv"] = (cv(L["swing_s"], ddof) + cv(R["swing_s"], ddof)) / 2
    out["stance_pct"] = float(100.0 * (both["stance_s"] / both["stride_s"]).mean())
    out["asym_swing"] = float(100.0 * abs(np.log(R["swing_s"].mean() / L["swing_s"].mean())))
    out["double_support_v2_pct"] = 100.0 * float(np.nanmean([_ds_one_side(L, R), _ds_one_side(R, L)]))
    return out
