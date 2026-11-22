"""R6 agreement: offset recovery, walkway strides, pairing differences, repeated-measures limits."""
import numpy as np
import pandas as pd

from experiments.external_transport.etx import agreement as ag

FS = 100.0


def contact(n, starts, stance):
    m = np.zeros(n, bool)
    for s in starts:
        m[s:s + stance] = True
    return m


def test_offset_recovered_shared_lr():
    n = 4000
    hsL = np.arange(200, 3800, 110)
    hsR = hsL + 55
    wwL, wwR = contact(n, hsL, 66), contact(n, hsR, 66)
    onmat = np.zeros(n, bool)
    onmat[150:3850] = True
    insL, insR = contact(n, hsL + 7, 66), contact(n, hsR + 7, 66)        # insole clock 70 ms later
    # (with unequal stance lengths the agreement has a plateau: the offset is only identified up to
    #  half the stance difference, which is why R6 judges offset-free durations, not latencies)
    off = ag.estimate_offset(insL, insR, wwL, wwR, onmat, FS, 5.0)
    assert abs(off["offset_s"] - 0.07) < 1e-9 and off["agree_shared"] == 1.0


def test_walkway_strides_and_pairing():
    n = 3000
    ics = np.r_[np.arange(100, 600, 110), np.arange(1500, 2000, 110)]     # two passes
    ww = contact(n, ics, 66)
    st, ic = ag.walkway_strides(ww, FS, 2.5)
    assert len(ic) == len(ics) and len(st) == len(ics) - 2                  # no stride across the pass gap
    assert np.allclose(st.next_ic_s - st.ic_s, 1.10)
    off = 0.05
    ins = pd.DataFrame({"hs_s": ics[:-1] / FS + off + 0.01, "to_s": ics[:-1] / FS + off + 0.01 + 0.62,
                        "next_hs_s": ics[1:] / FS + off + 0.01, "valid": True})
    ins = ins[np.diff(ics) < 250]
    p, cnt = ag.pair_strides(ins, st, ic, off, 0.15)
    assert cnt["n_paired"] == len(st) and cnt["n_ww_ic_matched"] == len(ics)
    assert np.allclose(p.d_stride_ms, 0, atol=1e-6)
    assert np.allclose(p.d_stance_ms, -40, atol=1e-6) and np.allclose(p.d_swing_ms, 40, atol=1e-6)
    assert np.allclose(p.res_hs_ms, 10, atol=1e-6)


def test_rm_loa_balanced_matches_formula():
    rng = np.random.default_rng(0)
    k, m = 30, 8
    subj = np.repeat(np.arange(k), m)
    d = rng.normal(0, 5, k)[subj] + rng.normal(0, 10, k * m)
    r = ag.rm_loa(d, subj)
    df = pd.DataFrame({"d": d, "s": subj})
    means = df.groupby("s").d.mean()
    msw = ((df.d - df.s.map(means)) ** 2).sum() / (k * m - k)
    msb = m * ((means - d.mean()) ** 2).sum() / (k - 1)
    assert np.isclose(r["sd_total"] ** 2, msw + (msb - msw) / m)           # balanced: D = m
    assert 9 < np.sqrt(msw) < 11 and r["sd_total"] > np.sqrt(msw)


def test_ds_and_drift():
    a = contact(500, [0, 100, 200, 300], 60)
    b = contact(500, [50, 150, 250, 350], 60)
    assert np.isclose(ag.ds_pct(a, b, [100, 200], [200, 300]), 20.0)   # overlaps 100-110 and 150-160
    t = np.linspace(0, 40, 30)
    assert np.isclose(ag.drift_slope(t, 0.5 * t + 3), 0.5)
