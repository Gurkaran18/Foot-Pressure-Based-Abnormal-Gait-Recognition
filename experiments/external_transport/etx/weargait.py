"""WearGait-PD loader + stride events + features, using the same v2 pipeline as gaitpdb.

Owner: Lead (data/*). Phase B decisions B-D1..B-D7 (config `weargait` block, approved 2026-11-22).

Reads only blinded files ($WEARGAIT_ROOT/blinded/WG###_SelfPace.csv) and only these columns:
Time ("0.01 sec"), GeneralEvent (Standing/Walk/Turn), LTotalForce, RTotalForce (Moticon insole total force).
`ClinicalEvent` (FoG/Stagger tags, label-revealing) is never read (B-D2). `L/R Foot Contact` and
`L/R Foot Pressure` are walkway-derived (0 off the mat) and are used only for the R6 agreement check.

Stride validity (one foot), in this order of reasons:
  0. gaps of <= nan_interp_max_s (1-2 samples: packet drops) are linearly interpolated first (B-D3 rev.)
  1. 'missing_data'  any remaining NaN within [hs - margin, next_hs + margin]               (B-D3)
  2. 'outside_walk'  stride not fully inside one GeneralEvent == "Walk" pass                 (B-D4)
  3. 'stride_range'  duration outside signal.stride_range_s                                  (as gaitpdb)
  4. 'outlier_mad'   3-MAD rule, computed over the strides that pass 1-3                      (as gaitpdb,
                     but on straight-walking strides only, like gaitpdb's straight walks)
Heel strikes / toe-offs come from events.detect_strides on the force with NaN set to 0 (identical
contact detection to gaitpdb); rule 1 removes any cycle a dropout could have created or distorted.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import events
from .signal import infer_fs, samples

READ_COLS = ["Time", "GeneralEvent", "LTotalForce", "RTotalForce"]
INSOLE_COLS = ["LTotalForce", "RTotalForce"]
CODE_RE = re.compile(r"^(WG\d{3})_")


def code_of(path) -> str:
    m = CODE_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"not a blinded WearGait-PD file: {path}")
    return m.group(1)


def has_insole(path) -> bool:
    return set(INSOLE_COLS) <= set(pd.read_csv(path, nrows=0).columns)


def pass_ids(general_event) -> np.ndarray:
    """0, 1, 2, ... for each contiguous 'Walk' run; -1 elsewhere (Standing / Turn / unlabeled)."""
    walk = np.asarray(general_event) == "Walk"
    start = walk & ~np.r_[False, walk[:-1]]
    ids = np.cumsum(start) - 1
    return np.where(walk, ids, -1)


def load_trial(path, never_read=("ClinicalEvent",)) -> dict:
    """Blinded SelfPace trial -> {code, t, fs, fl, fr (NaN kept), pass_id}. Raises if no insole columns."""
    if any(c in READ_COLS for c in never_read):
        raise RuntimeError("READ_COLS contains a never-read column")
    if not has_insole(path):
        raise LookupError(f"no insole columns in {Path(path).name} (B-D1: excluded)")
    d = pd.read_csv(path, usecols=READ_COLS, low_memory=False)
    t = d["Time"].astype(str).str.replace("sec", "", regex=False).str.strip().astype(float).to_numpy()
    return {"code": code_of(path), "t": t, "fs": infer_fs(t),
            "fl": d["LTotalForce"].to_numpy(float), "fr": d["RTotalForce"].to_numpy(float),
            "pass_id": pass_ids(d["GeneralEvent"].to_numpy())}


def fill_short_gaps(x, max_len: int) -> np.ndarray:
    """Linearly interpolate interior NaN runs of length <= max_len; longer or edge runs stay NaN."""
    x = np.array(x, float)
    nan = np.isnan(x)
    if max_len <= 0 or not nan.any():
        return x
    d = np.diff(np.r_[0, nan.astype(int), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a <= max_len and a > 0 and b < len(x):
            x[a:b] = np.interp(np.arange(a, b), [a - 1, b], [x[a - 1], x[b]])
    return x


def detect_strides(force, pass_id, fs: float, signal_cfg: dict, wg_cfg: dict) -> pd.DataFrame:
    """Stride table for one foot with WearGait-PD validity rules (see module docstring)."""
    max_gap = int(round(wg_cfg.get("nan_interp_max_s", 0.0) * fs))
    force = fill_short_gaps(force, max_gap)
    nan = np.isnan(force)
    st = events.detect_strides(np.where(nan, 0.0, force), fs, signal_cfg)
    if st.empty:
        st["pass_id"] = pd.Series(dtype=int)
        return st
    m = samples(wg_cfg["nan_margin_s"], fs)
    nan_cum = np.r_[0, np.cumsum(nan)]
    lo = np.clip(st["hs_idx"].to_numpy() - m, 0, len(force))
    hi = np.clip(st["next_hs_idx"].to_numpy() + m + 1, 0, len(force))
    missing = (nan_cum[hi] - nan_cum[lo]) > 0
    pid = np.asarray(pass_id)
    p_start = pid[st["hs_idx"].to_numpy()]
    inside = np.array([p >= 0 and (pid[a:b] == p).all()
                       for p, a, b in zip(p_start, st["hs_idx"], st["next_hs_idx"])], dtype=bool)
    in_range = st["stride_s"].between(*signal_cfg["stride_range_s"]).to_numpy()

    ok = ~missing & inside & in_range
    mad_ok = ok.copy()
    if ok.sum() >= 3:
        s = st["stride_s"].to_numpy()
        med = np.median(s[ok])
        mad = 1.4826 * np.median(np.abs(s[ok] - med)) + 1e-6
        mad_ok = ok & (np.abs(s - med) <= signal_cfg["outlier_mads"] * mad)
    st["valid"] = mad_ok
    st["invalid_reason"] = np.select([missing, ~inside, ~in_range, ~mad_ok],
                                     ["missing_data", "outside_walk", "stride_range", "outlier_mad"], "")
    st["pass_id"] = np.where(inside, p_start, -1)
    return st


def trial_strides(trial: dict, cfg: dict):
    """(left, right) stride tables for a trial from load_trial. cfg = full config."""
    return tuple(detect_strides(trial[k], trial["pass_id"], trial["fs"], cfg["signal"], cfg["weargait"])
                 for k in ("fl", "fr"))
