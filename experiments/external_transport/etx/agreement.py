"""R6 insole-walkway agreement (A1.1 Methods Revisions R6; tolerances D8). Label-free.

Owner: Lead (data/*).

Inputs per blinded SelfPace trial: insole total force (L/R) and the walkway contact columns
`L/R Foot Contact` (1 while that foot is on the ProtoKinetics walkway, 0 otherwise; same file timeline).
  1. one clock offset per trial, shared by both feet: maximise the agreement of the binary contact signals
     (insole contact mask vs walkway contact) over the on-mat samples, lags within +-max_lag_s.
     Per-foot offsets are diagnostic only; their difference = inter-insole desynchronisation.
  2. walkway strides: consecutive initial contacts (IC) of one foot within walkway_max_stride_s,
     with the final contact (FC) in between. Contacts touching the recording edges are dropped.
  3. pairing: each walkway IC -> nearest insole heel strike of the same foot within +-pair_tol_s
     (after removing the offset). A paired stride needs both its IC and next IC matched to the HS and
     next HS of one *valid* insole stride (the strides that enter the features).
  4. offset-free per-stride differences (insole - walkway): stride, stance, swing; per-trial DS%
     (fraction of the stride with both feet in contact, same definition for both systems).
  5. bias + 95% limits: stride-level with the Bland & Altman (2007) repeated-measures correction,
     and participant-mean limits. Drift: slope of the HS residual vs time per trial.
Sign convention: insole_time = walkway_time + offset_s.
"""
import numpy as np
import pandas as pd

from .signal import contact_mask
from .weargait import fill_short_gaps


def runs(mask):
    """(starts, ends) of True runs; end is exclusive."""
    d = np.diff(np.r_[0, np.asarray(mask, bool).astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def insole_contact(force, fs, signal_cfg, wg_cfg):
    """Insole contact mask with the same detection as the stride events (short gaps filled, NaN -> 0)."""
    x = fill_short_gaps(force, int(round(wg_cfg.get("nan_interp_max_s", 0.0) * fs)))
    return contact_mask(np.nan_to_num(x), fs, signal_cfg)


def _shift(x, lag):
    """y[t] = x[t + lag] (no wrap-around; out-of-range = False)."""
    y = np.zeros_like(x)
    if lag >= 0:
        y[:len(x) - lag] = x[lag:]
    else:
        y[-lag:] = x[:len(x) + lag]
    return y


def lag_scores(ins, ww, onmat, max_lag):
    """Agreement (fraction of on-mat samples where shifted insole contact == walkway contact) per lag."""
    lags = np.arange(-max_lag, max_lag + 1)
    n = max(int(onmat.sum()), 1)
    return lags, np.array([((_shift(ins, int(L)) == ww) & onmat).sum() / n for L in lags])


def estimate_offset(ins_L, ins_R, ww_L, ww_R, onmat, fs, max_lag_s):
    """Shared offset (s) from L+R jointly, plus diagnostic per-foot offsets and agreement values."""
    m = int(round(max_lag_s * fs))
    lags, sL = lag_scores(ins_L, ww_L, onmat, m)
    _, sR = lag_scores(ins_R, ww_R, onmat, m)
    s = (sL + sR) / 2
    k = int(np.argmax(s))
    return {"offset_s": lags[k] / fs, "agree_shared": float(s[k]), "agree_at_zero": float(s[m]),
            "offset_L_s": lags[int(np.argmax(sL))] / fs, "offset_R_s": lags[int(np.argmax(sR))] / fs}


def walkway_strides(ww, fs, max_stride_s):
    """Walkway strides of one foot: ic_s, fc_s, next_ic_s (seconds on the file timeline)."""
    a, b = runs(ww)
    keep = (a > 0) & (b < len(ww))                      # drop contacts cut by the recording edges
    a, b = a[keep], b[keep]
    rows = []
    for k in range(len(a) - 1):
        if (a[k + 1] - a[k]) / fs <= max_stride_s:
            rows.append((a[k] / fs, b[k] / fs, a[k + 1] / fs))
    return pd.DataFrame(rows, columns=["ic_s", "fc_s", "next_ic_s"]), a / fs


def _nearest(x, targets):
    """Index of the nearest value in sorted `targets` for each x, and the absolute distance."""
    targets = np.asarray(targets, float)
    if not len(targets):
        return np.full(len(x), -1), np.full(len(x), np.inf)
    i = np.clip(np.searchsorted(targets, x), 1, len(targets) - 1) if len(targets) > 1 else np.zeros(len(x), int)
    if len(targets) > 1:
        left = np.abs(x - targets[i - 1]) <= np.abs(x - targets[i])
        i = np.where(left, i - 1, i)
    return i, np.abs(x - targets[i])


def pair_strides(ins_st, ww_st, ww_ic, offset_s, tol_s):
    """Paired strides (one foot) + IC match counts. ins_st = weargait stride table (all detected strides)."""
    hs_all = np.sort(np.r_[ins_st["hs_s"].to_numpy(), ins_st["next_hs_s"].to_numpy()]) - offset_s
    hs_all = np.unique(np.round(hs_all, 6))
    _, dist = _nearest(np.asarray(ww_ic), hs_all)
    n_matched = int((dist <= tol_s).sum())

    v = ins_st[ins_st["valid"]].sort_values("hs_s")
    hs, nhs = v["hs_s"].to_numpy() - offset_s, v["next_hs_s"].to_numpy() - offset_s
    rows = []
    if len(v) and len(ww_st):
        i, d = _nearest(ww_st["ic_s"].to_numpy(), hs)
        for (ic, fc, nic), j, dj in zip(ww_st.itertuples(index=False, name=None), i, d):
            if dj > tol_s or abs(nhs[j] - nic) > tol_s:
                continue
            r = v.iloc[j]
            ins_stride, ins_stance = r.next_hs_s - r.hs_s, r.to_s - r.hs_s
            rows.append({"t_s": ic, "hs_s": r.hs_s, "ic_s": ic,
                         "res_hs_ms": 1000 * ((r.hs_s - offset_s) - ic),
                         "res_to_ms": 1000 * ((r.to_s - offset_s) - fc),
                         "d_stride_ms": 1000 * (ins_stride - (nic - ic)),
                         "d_stance_ms": 1000 * (ins_stance - (fc - ic)),
                         "d_swing_ms": 1000 * ((ins_stride - ins_stance) - (nic - fc)),
                         "ww_stride_s": nic - ic, "ins_stride_s": ins_stride})
    return pd.DataFrame(rows), {"n_ww_ic": int(len(ww_ic)), "n_ww_ic_matched": n_matched,
                                "n_ww_strides": int(len(ww_st)), "n_ins_valid": int(len(v)),
                                "n_paired": len(rows)}


def ds_pct(own, other, start_idx, end_idx):
    """Mean over strides of the % of [start, end) where both feet are in contact."""
    both = np.asarray(own, bool) & np.asarray(other, bool)
    c = np.r_[0, np.cumsum(both)]
    fr = [(c[b] - c[a]) / (b - a) for a, b in zip(start_idx, end_idx) if b > a]
    return 100.0 * float(np.mean(fr)) if fr else np.nan


def rm_loa(diff, subject):
    """Bland & Altman (2007) limits for multiple observations per subject (true value varies).

    s2_total = s2_within + s2_between, with s2_between = (MSB - MSW) / D,
    D = (n^2 - sum m_i^2) / ((k - 1) n). Returns mean, sd_total, lower, upper, sd_plain, n, k.
    """
    df = pd.DataFrame({"d": np.asarray(diff, float), "s": np.asarray(subject)}).dropna()
    n, g = len(df), df.groupby("s")["d"]
    m, means, k = g.size(), g.mean(), g.ngroups
    grand = df["d"].mean()
    if k < 2 or n <= k:
        sd = float(df["d"].std(ddof=1)) if n > 1 else np.nan
        return {"mean": grand, "sd_total": sd, "lower": grand - 1.96 * sd, "upper": grand + 1.96 * sd,
                "sd_plain": sd, "n": n, "k": k}
    ssb = float((m * (means - grand) ** 2).sum())
    ssw = float(((df["d"] - df["s"].map(means)) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    D = (n ** 2 - float((m ** 2).sum())) / ((k - 1) * n)
    s2 = msw + max(msb - msw, 0.0) / D
    sd = float(np.sqrt(s2))
    return {"mean": float(grand), "sd_total": sd, "lower": float(grand - 1.96 * sd),
            "upper": float(grand + 1.96 * sd), "sd_plain": float(df["d"].std(ddof=1)), "n": n, "k": k}


def mean_loa(diff, subject):
    """Participant-mean limits: bias and SD of per-subject mean differences."""
    mm = pd.Series(np.asarray(diff, float)).groupby(np.asarray(subject)).mean().dropna()
    sd = float(mm.std(ddof=1)) if len(mm) > 1 else np.nan
    return {"mean": float(mm.mean()), "sd": sd, "lower": float(mm.mean() - 1.96 * sd),
            "upper": float(mm.mean() + 1.96 * sd), "k": int(len(mm))}


def drift_slope(t_s, res_ms):
    """OLS slope (ms per s) of a residual vs time; NaN if < 3 points or no time spread."""
    t, r = np.asarray(t_s, float), np.asarray(res_ms, float)
    if len(t) < 3 or np.ptp(t) == 0:
        return np.nan
    return float(np.polyfit(t, r, 1)[0])
