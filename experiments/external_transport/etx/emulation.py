"""Pseudo-SelfPace walks (4 segments x 6-7 consecutive valid strides) built from gaitpdb walks.

Owner: Lead (walk construction, feature tables) / Teammate (evaluate). Spec §4, R1/R4; decision D7.

WearGait-PD SelfPace = 4 passes over a 16-ft walkway. Each emulated walk, per foot:
  1. drop turn-adjacent strides: valid strides whose previous or next stride (by stride_idx) is
     invalid or missing (turns/pauses at the walkway ends are what invalid strides mostly are);
  2. find runs of consecutive remaining strides (consecutive stride_idx);
  3. place 4 non-overlapping segments, each of 6 or 7 consecutive strides (length drawn per segment),
     inside those runs at random positions.
Returns None if a foot cannot host 4 segments. Feet are sampled independently (documented limitation).
"""
import numpy as np
import pandas as pd

from .features import walk_features_v2
from .kcurve import draw_seed


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


def evaluate(feature_tables, cfg):
    """Teammate. Emulated internal AUC (train-full/test-K and train-K/test-K)."""
    raise NotImplementedError
