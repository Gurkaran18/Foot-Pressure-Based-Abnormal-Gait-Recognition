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


V1_FEATURES = ["stride_time_mean", "stride_time_cv", "swing_time_cv", "stance_pct", "swing_pct",
               "double_support_pct", "cadence", "asym_stride", "asym_swing", "asym_peak", "peak_force_L",
               "peak_force_R", "peak_force_norm"]


def walk_features_v1(strides_L, strides_R, fl, fr, fs: float, signal_cfg: dict) -> dict:
    """Exact reproduction of gait_common.gait_features over the whole recording (legacy v1).

    For the Block 4 equivalence check ONLY. Must be called with strides detected using
    debounce_impl='legacy'. Reproduces legacy choices on purpose: ddof=0 CV, MIN_STRIDES=3,
    whole-span double-support mask recomputed on the stride span (issue I4), mean(L+R) body proxy (I5).
    """
    from .signal import contact_mask
    legacy_cfg = {**signal_cfg, "debounce_impl": "legacy"}
    L = strides_L[strides_L["valid"]]
    R = strides_R[strides_R["valid"]]
    feats = dict.fromkeys(V1_FEATURES, np.nan)
    feats["n_strides"] = int(len(L) + len(R))
    if min(len(L), len(R)) < 3:
        return feats

    def cv0(x):
        x = np.asarray(x, float)
        return float(100 * np.std(x) / np.mean(x)) if len(x) >= 2 and np.mean(x) > 0 else np.nan

    def asym(a, b):
        return float(100 * abs(a - b) / (0.5 * (a + b))) if (a + b) > 0 else np.nan

    stride = np.r_[L["stride_s"], R["stride_s"]]
    stance_pct = 100 * np.r_[L["stance_s"] / L["stride_s"], R["stance_s"] / R["stride_s"]]
    swing_L, swing_R = L["swing_s"].mean(), R["swing_s"].mean()
    peak_L, peak_R = L["peak_n"].mean(), R["peak_n"].mean()
    a = int(min(L["hs_idx"].min(), R["hs_idx"].min()))
    b = int(max(L["next_hs_idx"].max(), R["next_hs_idx"].max()))
    both = contact_mask(fl[a:b], fs, legacy_cfg) & contact_mask(fr[a:b], fs, legacy_cfg)
    body = float(np.mean(fl[a:b] + fr[a:b]))
    feats.update({
        "stride_time_mean": float(stride.mean()),
        "stride_time_cv": float(np.mean([cv0(L["stride_s"]), cv0(R["stride_s"])])),
        "swing_time_cv": float(np.mean([cv0(L["swing_s"]), cv0(R["swing_s"])])),
        "stance_pct": float(stance_pct.mean()),
        "swing_pct": float(100 - stance_pct.mean()),
        "double_support_pct": float(100 * both.mean()),
        "cadence": float(120.0 / stride.mean()),
        "asym_stride": asym(L["stride_s"].mean(), R["stride_s"].mean()),
        "asym_swing": float(100 * abs(np.log(swing_R / swing_L))) if swing_L > 0 and swing_R > 0 else np.nan,
        "asym_peak": asym(peak_L, peak_R),
        "peak_force_L": float(peak_L),
        "peak_force_R": float(peak_R),
        "peak_force_norm": float(0.5 * (peak_L + peak_R) / body) if body > 0 else np.nan,
    })
    return feats


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
