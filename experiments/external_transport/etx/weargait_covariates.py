"""Blinded WearGait-PD covariates: SelfPace walkway speed, age, sex keyed by WG code (no group).

Owner: Lead (data/*). Blinding step (CLAUDE.md §3 exception, approved 2026-11-22): the key in labels/
is read for `code` and `original_subject` only. Group-revealing differences between the two
group-split demographics files (column names, value formatting, PD-only columns) are removed by
harmonising to three columns with one format: speed_selfpace_mps (float, 3 dp), age_y (int), sex (M/F).

Speed (B-D9): computed for every trial from the raw walkway footprints in the blinded SelfPace file
(pressure-weighted centroid of each walkway contact; same-foot stride length / stride time, summed over
on-mat strides; cell size 0.0127 m = the 0.5-inch ProtoKinetics Zeno sensor pitch, a physical constant,
not fitted). The official PKMAS SelfPace velocity exists for only 94/123 subjects and its missingness is
group-related, so it is used only to check agreement, never as the covariate.
UPDRS / H&Y / medication / DBS columns are never read (CLAUDE.md §3: descriptive only, not model inputs).
"""
import re

import numpy as np
import pandas as pd

ID_RE = re.compile(r"^\s*(HC|NLS|WHC|WPD)\s*0*(\d+)", re.IGNORECASE)


def norm_id(x):
    """'NLS003' -> 'NLS03', 'hc5' -> 'HC05' (same normalisation as weargait_blind.parse_name)."""
    m = ID_RE.match(str(x))
    return f"{m.group(1).upper()}{int(m.group(2)):02d}" if m else None


def _col(df, pattern):
    hits = [c for c in df.columns if re.match(pattern, str(c).strip(), re.IGNORECASE)]
    if len(hits) != 1:
        raise KeyError(f"expected one column matching {pattern!r}, got {hits}")
    return hits[0]


def read_demographics(path) -> pd.DataFrame:
    """Group-split demographics file (title row, then header row) -> original_subject, age_y, sex."""
    d = pd.read_csv(path, header=1, low_memory=False, dtype=str)
    out = pd.DataFrame({"original_subject": d[_col(d, r"^Subject ID$")].map(norm_id),
                        "age_y": pd.to_numeric(d[_col(d, r"^Age( \(years\))?$")], errors="coerce"),
                        "sex": d[_col(d, r"^Sex$")].str.strip().str.lower().map({"male": "M", "female": "F"})})
    return out.dropna(subset=["original_subject"])


def read_speed(path, task="SelfPace") -> pd.DataFrame:
    """PKMAS walkway metrics -> original_subject, speed_selfpace_mps (mean over rows of the task)."""
    d = pd.read_csv(path, low_memory=False, dtype=str)
    d = d[d["Task"].fillna("").str.strip() == task]
    v = pd.to_numeric(d[_col(d, r"^Velocity \(cm\./sec\.\)$")], errors="coerce") / 100.0
    s = pd.DataFrame({"original_subject": d["Participant ID"].map(norm_id), "v": v}).dropna()
    return s.groupby("original_subject", as_index=False)["v"].mean().rename(columns={"v": "speed_selfpace_mps"})


def _contact_centroid(xs, ys, ps, feet, foot):
    X, Y, W = [], [], []
    for x, y, p, f in zip(xs, ys, ps, feet):
        if not isinstance(x, str) or not x:
            continue
        fa = np.array(f.split("|"))
        k = fa == foot
        X.append(np.array(x.split("|"), float)[k])
        Y.append(np.array(y.split("|"), float)[k])
        W.append(np.array(p.split("|"), float)[k])
    if not X:
        return None
    X, Y, W = map(np.concatenate, (X, Y, W))
    return (np.average(X, weights=W), np.average(Y, weights=W)) if W.sum() > 0 else None


WALKWAY_COLS = ["L Foot Contact", "R Foot Contact", "Walkway_X", "Walkway_Y", "WalkwayPressureLevel", "WalkwayFoot"]


def walkway_speed(path, cell_m=0.0127, max_stride_s=2.5, fs=100.0):
    """(speed m/s, n strides) from walkway footprints of one blinded SelfPace file; (nan, 0) if none."""
    d = pd.read_csv(path, usecols=WALKWAY_COLS, low_memory=False,
                    dtype={c: str for c in WALKWAY_COLS[2:]})
    dist, dur = [], []
    for s in "LR":
        m = d[f"{s} Foot Contact"].fillna(0).to_numpy() == 1
        e = np.diff(np.r_[0, m.astype(int), 0])
        a, b = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
        cen = [(i / fs, _contact_centroid(*(d[c].iloc[i:j] for c in WALKWAY_COLS[2:]), s)) for i, j in zip(a, b)]
        for (t0, c0), (t1, c1) in zip(cen[:-1], cen[1:]):
            if c0 is None or c1 is None or t1 - t0 > max_stride_s:
                continue
            dist.append(np.hypot(c1[0] - c0[0], c1[1] - c0[1]) * cell_m)
            dur.append(t1 - t0)
    return (float(np.sum(dist) / np.sum(dur)), len(dist)) if dur else (np.nan, 0)


def build(key: pd.DataFrame, demo_files, speed_file, walkway_speed_by_code=None) -> pd.DataFrame:
    """key: code, original_subject (no group). Returns code, speed_selfpace_mps, age_y, sex sorted by code.

    walkway_speed_by_code: {code: m/s} from walkway_speed(); if given it is the covariate and the PKMAS
    value is returned separately as `_speed_pkmas_check` (for the agreement check only; never saved)."""
    if "group" in key.columns:
        raise ValueError("pass only code/original_subject from the key")
    demo = pd.concat([read_demographics(f) for f in demo_files], ignore_index=True)
    demo = demo.drop_duplicates("original_subject")
    k = key[["code", "original_subject"]].drop_duplicates()
    out = k.merge(read_speed(speed_file), on="original_subject", how="left").merge(demo, on="original_subject", how="left")
    if walkway_speed_by_code is not None:
        out = out.rename(columns={"speed_selfpace_mps": "_speed_pkmas_check"})
        out["speed_selfpace_mps"] = out["code"].map(walkway_speed_by_code)
    out["age_y"] = out["age_y"].round().astype("Int64")
    out["speed_selfpace_mps"] = out["speed_selfpace_mps"].round(3)
    cols = ["code", "speed_selfpace_mps", "age_y", "sex"] + (["_speed_pkmas_check"] if walkway_speed_by_code is not None else [])
    return out[cols].sort_values("code").reset_index(drop=True)
