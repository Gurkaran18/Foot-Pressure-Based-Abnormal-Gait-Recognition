"""HS/TO detection (logic identical to gait_common.detect_strides, in seconds). Output: events/strides.parquet.

Owner: Lead (data/*). Spec §4, §5; test T4 (+ legacy equivalence in Block 4).

One row per stride: heel strike -> exactly one toe-off -> next heel strike. Validity:
  1. stride duration within cfg stride_range_s            (else invalid_reason = 'stride_range')
  2. if >= 3 strides pass (1): |stride - median| <= outlier_mads * 1.4826 * MAD   (else 'outlier_mad')
Sample indices (hs_idx, ...) are kept so the v1 legacy-equivalence test can compare them exactly.
"""
import numpy as np
import pandas as pd

from .signal import contact_mask

STRIDE_COLUMNS = ["hs_idx", "to_idx", "next_hs_idx", "hs_s", "to_s", "next_hs_s", "stride_s",
                  "stance_s", "swing_s", "peak_n", "valid", "invalid_reason"]


def detect_strides(force, fs: float, cfg: dict) -> pd.DataFrame:
    """Stride table for one foot. cfg = the `signal` block of config/a2.yaml."""
    force = np.asarray(force, float)
    mask = contact_mask(force, fs, cfg)
    d = np.diff(mask.astype(int))
    hs = np.flatnonzero(d == 1) + 1
    to = np.flatnonzero(d == -1) + 1

    rows = []
    for a, b in zip(hs[:-1], hs[1:]):
        t = to[(to > a) & (to < b)]
        if len(t) != 1:                      # zero or several toe-offs: not a clean cycle
            continue
        t = int(t[0])
        rows.append((int(a), t, int(b), a / fs, t / fs, b / fs, (b - a) / fs, (t - a) / fs,
                     (b - t) / fs, float(force[a:t].max())))
    st = pd.DataFrame(rows, columns=STRIDE_COLUMNS[:10])
    if st.empty:
        st["valid"] = pd.Series(dtype=bool)
        st["invalid_reason"] = pd.Series(dtype=object)
        return st

    in_range = st["stride_s"].between(*cfg["stride_range_s"])
    ok = in_range.copy()
    if ok.sum() >= 3:
        med = st.loc[ok, "stride_s"].median()
        mad = 1.4826 * np.median(np.abs(st.loc[ok, "stride_s"] - med)) + 1e-6
        ok &= (st["stride_s"] - med).abs() <= cfg["outlier_mads"] * mad
    st["valid"] = ok
    st["invalid_reason"] = np.where(ok, "", np.where(~in_range, "stride_range", "outlier_mad"))
    return st


def trial_strides(trial: dict, cfg: dict):
    """(left strides, right strides) for a trial dict from signal.load_trial."""
    return (detect_strides(trial["fl"], trial["fs"], cfg),
            detect_strides(trial["fr"], trial["fs"], cfg))
