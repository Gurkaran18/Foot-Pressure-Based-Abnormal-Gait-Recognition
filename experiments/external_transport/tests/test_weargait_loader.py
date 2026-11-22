"""B2 loader: never reads ClinicalEvent, NaN strides invalid, only in-walk strides valid, no-insole flagged."""
import numpy as np
import pandas as pd
import pytest

from experiments.external_transport.etx import paths, weargait
from experiments.external_transport.scripts import b_02_weargait_features as b02

CFG = paths.load_config()
FS = 100.0


def synth_trial(tmp_path, code="WG007", insole=True, nan_at=None):
    """4 passes of 'Walk' (8 s) separated by 'Turn' (2 s) after 3 s 'Standing'; steady 1.1 s strides."""
    rng = np.random.default_rng(3)
    ev = ["Standing"] * 300
    for p in range(4):
        ev += ["Walk"] * 800 + (["Turn"] * 200 if p < 3 else ["Standing"] * 300)
    n = len(ev)
    t = np.arange(n) / FS

    def force(offset):
        f = np.full(n, 3.0)
        hs = 3.0 + offset
        while hs < t[-1]:
            st = 1.10 + rng.uniform(-0.03, 0.03)
            on = (t >= hs) & (t < hs + 0.6 * st)
            f[on] = 700.0
            hs += st
        return f
    fl, fr = force(0.0), force(0.55)
    fl[:250] = np.nan                                       # insoles start late, as in WearGait-PD
    if nan_at is not None:
        fl[nan_at:nan_at + 3] = np.nan
    d = pd.DataFrame({"Time": [f"{x:g} sec" for x in t], "GeneralEvent": ev,
                      "ClinicalEvent": ["FoG"] * n,          # label-revealing: must never be read
                      "L Foot Contact": 0, "R Foot Contact": 0})
    if insole:
        d["LTotalForce"], d["RTotalForce"] = fl, fr
    p = tmp_path / f"{code}_SelfPace.csv"
    d.to_csv(p, index=False)
    return p


def test_load_never_reads_clinical_event(tmp_path):
    tr = weargait.load_trial(synth_trial(tmp_path))
    assert "ClinicalEvent" not in weargait.READ_COLS and tr["code"] == "WG007"
    assert abs(tr["fs"] - 100.0) < 1e-9 and set(np.unique(tr["pass_id"])) == {-1, 0, 1, 2, 3}
    with pytest.raises(RuntimeError):
        weargait.load_trial(synth_trial(tmp_path), never_read=("GeneralEvent",))


def test_strides_only_inside_walk_and_nan_invalidates(tmp_path):
    tr = weargait.load_trial(synth_trial(tmp_path))
    L, R = weargait.trial_strides(tr, CFG)
    for st in (L, R):
        v = st[st.valid]
        assert len(v) >= 20 and (v.pass_id >= 0).all()
        assert (np.abs(v.stride_s - 1.10) < 0.05).all()
        ev = np.where(tr["pass_id"] >= 0, "W", "x")
        for a, b in zip(v.hs_idx, v.next_hs_idx):                # whole stride inside one pass
            assert (ev[a:b] == "W").all() and len(set(tr["pass_id"][a:b])) == 1
        assert (st.loc[~st.valid, "invalid_reason"] != "").all()
    assert "outside_walk" in set(L.invalid_reason)

    # a 30 ms dropout in the middle of pass 1 invalidates exactly the stride(s) touching it
    v = L[L.valid & (L.pass_id == 1)].iloc[2]
    k = int(v.hs_idx) + 20
    tr2 = weargait.load_trial(synth_trial(tmp_path, nan_at=k))
    L2, _ = weargait.trial_strides(tr2, CFG)
    hit = L2[(L2.hs_idx <= k + 3) & (L2.next_hs_idx >= k)]
    assert len(hit) >= 1 and (hit.invalid_reason == "missing_data").all()
    assert L2.valid.sum() >= L.valid.sum() - 3


def test_compute_flags_no_insole(tmp_path):
    files = [synth_trial(tmp_path, "WG001"), synth_trial(tmp_path, "WG002", insole=False)]
    strides, walks = b02.compute([str(f) for f in files], CFG)
    w = walks.set_index("code")
    assert bool(w.loc["WG001", "qc_pass"]) and w.loc["WG001", "n_passes"] == 4
    assert not w.loc["WG002", "has_insole"] and w.loc["WG002", "qc_flags"] == "no_insole"
    assert set(strides.code) == {"WG001"} and abs(w.loc["WG001", "stride_time_mean"] - 1.10) < 0.02


def test_short_gaps_filled_long_gaps_kept():
    x = np.array([0., 1., np.nan, 3., np.nan, np.nan, 6., np.nan, np.nan, np.nan, 10., np.nan])
    y = weargait.fill_short_gaps(x, 2)
    assert np.allclose(y[:7], [0, 1, 2, 3, 4, 5, 6])
    assert np.isnan(y[7:10]).all() and np.isnan(y[11])          # 3-sample gap and edge gap stay NaN


def test_single_sample_dropout_does_not_invalidate(tmp_path):
    tr = weargait.load_trial(synth_trial(tmp_path))
    L, _ = weargait.trial_strides(tr, CFG)
    k = int(L[L.valid & (L.pass_id == 1)].iloc[2].hs_idx) + 20
    tr["fl"] = tr["fl"].copy()
    tr["fl"][k] = np.nan
    L1, _ = weargait.trial_strides(tr, CFG)
    assert L1.valid.sum() == L.valid.sum()
