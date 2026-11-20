"""K-sampling and emulation tests (spec §11 additional tests)."""
import numpy as np
import pandas as pd
import pytest

from experiments.external_transport.etx import emulation, kcurve, paths

CFG = paths.load_config()


def foot_table(n, t0=0.0, invalid=()):
    st = 1.0 + 0.02 * np.sin(np.arange(n))
    hs = t0 + np.r_[0, np.cumsum(st)[:-1]]
    df = pd.DataFrame({"stride_idx": np.arange(n), "hs_s": hs, "to_s": hs + 0.6 * st, "next_hs_s": hs + st,
                       "stride_s": st, "stance_s": 0.6 * st, "swing_s": 0.4 * st, "valid": True})
    df.loc[list(invalid), "valid"] = False
    return df


L, R = foot_table(60, 0.0, invalid=(14, 15, 30, 45)), foot_table(60, 0.5, invalid=(14, 31, 46))


@pytest.mark.parametrize("K", [5, 10, 25])
def test_s1_contiguous_window(K):
    rng = np.random.default_rng(0)
    for _ in range(20):
        a, b = kcurve.sample_strides(L, R, K, "S1_contiguous", rng)
        assert len(a) == len(b) == K and a.valid.all() and b.valid.all()
        t0 = min(a.hs_s.min(), b.hs_s.min())
        for foot, sub in ((L, a), (R, b)):     # the K strides are the NEXT K valid strides after t0
            nxt = foot[foot.valid & (foot.hs_s >= t0)].sort_values("hs_s").iloc[:K]
            assert sub.stride_idx.tolist() == nxt.stride_idx.tolist()


def test_s2_random_and_all_and_ineligible():
    rng = np.random.default_rng(1)
    a, b = kcurve.sample_strides(L, R, 10, "S2_random", rng)
    assert len(a) == 10 and a.stride_idx.is_unique and set(a.stride_idx) <= set(L[L.valid].stride_idx)
    a, b = kcurve.sample_strides(L, R, "all", "S1_contiguous", rng)
    assert len(a) == L.valid.sum() and len(b) == R.valid.sum()
    assert kcurve.sample_strides(L.iloc[:8], R, 10, "S2_random", rng) is None


def test_emulation_segments():
    rng = np.random.default_rng(2)
    em = CFG["emulation"]
    for _ in range(30):
        a, b = emulation.emulate_walk(L, R, em, rng)
        for foot, sub in ((L, a), (R, b)):
            assert sub.segment.nunique() == em["segments"]
            bad = set(foot.loc[~foot.valid, "stride_idx"])
            for _, seg in sub.groupby("segment"):
                idx = sorted(seg.stride_idx)
                assert len(idx) in em["strides_per_segment"]
                assert idx == list(range(idx[0], idx[-1] + 1))                  # consecutive strides
                for i in idx:                                                   # no turn-adjacent strides
                    assert (i - 1) not in bad and (i + 1) not in bad and i not in bad
            assert sub.stride_idx.is_unique                                     # segments don't overlap
    assert emulation.emulate_walk(L.iloc[:20], R, em, rng) is None              # too short for 4 segments


def test_feature_tables_one_row_per_subject_per_draw_and_deterministic():
    cfg = {**CFG, "kcurve": {**CFG["kcurve"], "levels": [5, "all"], "draws": 3},
           "emulation": {**CFG["emulation"], "draws": 2}}
    rows = []
    for sid, lab in (("A", 0), ("B", 1)):
        for foot, t in (("L", L), ("R", R)):
            rows.append(t.assign(recording_id=f"{sid}_01", foot=foot))
    strides = pd.concat(rows)
    subjects = pd.DataFrame({"subject_id": ["A", "B"], "label": [0, 1], "study": "Ga",
                             "source_recordings": ["A_01", "B_01"]})
    f1, e1 = kcurve.feature_tables(strides, subjects, cfg)
    f2, _ = kcurve.feature_tables(strides, subjects, cfg)
    pd.testing.assert_frame_equal(f1, f2)                                       # deterministic
    assert not f1.duplicated(["scheme", "K", "draw", "subject_id"]).any()
    assert len(f1) == 2 * 2 * (3 + 1)          # 2 schemes x 2 subjects x (3 draws at K=5 + 1 at all)
    assert (e1.n_eligible_Co == 1).all() and (e1.n_eligible_PD == 1).all()
    g, _ = emulation.feature_tables(strides, subjects, cfg)
    assert len(g) == 4 and not g.duplicated(["draw", "subject_id"]).any()
    assert g.stride_time_cv.notna().all()
