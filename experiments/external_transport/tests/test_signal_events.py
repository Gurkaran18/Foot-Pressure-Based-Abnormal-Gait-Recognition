"""T4: sampling-rate handling (parameters in seconds) + stride-event correctness.

Synthetic gait: a continuous force curve (trapezoid contacts, 30 ms ramps) sampled at 50/100/120/200 Hz.
Truth: stride 1.10 s, stance 0.66 s, swing 0.44 s.

Note on the spec's T4 glitch criterion: the 50 ms smoothing widens a short rectangular pulse by
~0.8 x 50 ms before thresholding (same as the legacy design), so the 80 ms-removed / 120 ms-kept
criterion is tested on the debounce stage, where the seconds -> samples conversion lives.
"""
import numpy as np
import pytest

from experiments.external_transport.etx import events, paths, signal
from experiments.external_transport.tests.conftest import requires_gaitpdb

CFG = paths.load_config()["signal"]
FS_LIST = [50, 100, 120, 200]
STRIDE, STANCE, RAMP, N_STRIDES, T0 = 1.10, 0.66, 0.03, 30, 0.5
# natural stride-to-stride variation (uniform +-30 ms, fixed seed: no true outliers); stance = 60 % of stride
TRUE_STRIDES = STRIDE + np.random.default_rng(7).uniform(-0.03, 0.03, N_STRIDES)
TRUE_HS = T0 + np.r_[0, np.cumsum(TRUE_STRIDES)[:-1]]


def gait_force(t):
    """Continuous total-force curve: trapezoid contacts of 800 N, baseline 5 N."""
    f = np.full_like(t, 5.0)
    for k in range(N_STRIDES):
        hs = TRUE_HS[k]
        to = hs + 0.60 * TRUE_STRIDES[k]
        up = np.clip((t - hs) / RAMP, 0, 1)
        down = np.clip((to - t) / RAMP, 0, 1)
        f = np.maximum(f, 5.0 + 795.0 * np.minimum(up, down) * ((t >= hs) & (t <= to)))
    return f


def detect_at(fs):
    t = np.arange(0, T0 + N_STRIDES * STRIDE + 1.0, 1 / fs)
    st = events.detect_strides(gait_force(t), fs, CFG)
    return st[st.valid]


def test_samples_conversion():
    assert signal.samples(0.05, 100) == 5 and signal.samples(0.10, 100) == 10   # legacy values at 100 Hz
    assert signal.samples(0.10, 200) == 20 and signal.samples(0.10, 50) == 5
    assert signal.samples(0.001, 50) == 1                                         # never zero
    assert [signal.odd_samples(0.05, fs) for fs in FS_LIST] == [3, 5, 7, 11]      # odd: no half-sample shift


@pytest.mark.parametrize("fs", FS_LIST)
def test_t4_durations_correct_at_every_fs(fs):
    st = detect_at(fs)
    assert len(st) == N_STRIDES - 1                         # every cycle found, none invalid
    assert np.allclose(st.stride_s.to_numpy(), TRUE_STRIDES[:-1], rtol=0, atol=1 / fs + 1e-9)
    # stance/swing carry the (fs-independent) smoothing+threshold bias; it must be the same at all fs
    assert abs(st.stance_s.mean() - detect_at(200).stance_s.mean()) <= 1 / fs + 1 / 200


@pytest.mark.parametrize("fs", [50, 100, 120])
def test_t4_event_times_agree_across_fs(fs):
    ref, st = detect_at(200), detect_at(fs)
    assert len(ref) == len(st)
    for col in ["hs_s", "to_s", "next_hs_s"]:
        assert np.max(np.abs(st[col].to_numpy() - ref[col].to_numpy())) <= 1 / fs + 1 / 200 + 0.005


@pytest.mark.parametrize("fs", FS_LIST)
def test_t4_debounce_glitch_removed_contact_kept_in_seconds(fs):
    t = np.arange(0, 3.0, 1 / fs)
    mask = np.zeros_like(t, bool)
    mask[(t >= 0.20) & (t < 0.90)] = True                  # normal stance
    mask[(t >= 1.30) & (t < 1.38)] = True                  # 80 ms glitch during swing -> removed
    mask[(t >= 2.00) & (t < 2.12)] = True                  # 120 ms true contact -> kept
    out = signal.drop_short_runs(mask, signal.samples(CFG["debounce_s"], fs))
    assert not out[(t >= 1.30) & (t < 1.38)].any()
    assert out[(t >= 2.00) & (t < 2.12)].all()
    assert out[(t >= 0.20) & (t < 0.90)].all()


def test_legacy_debounce_bug_reproduced_and_fixed():
    m = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0], bool)   # starts in swing; 1-sample blip at 7
    fixed = signal.drop_short_runs(m, 2)
    assert not fixed[7] and fixed[2:5].all() and fixed[10:13].all()      # blip removed
    legacy = signal.drop_short_runs(m, 2, legacy=True)
    assert legacy[7]                                                     # legacy keeps the blip (bug)
    gc = pytest.importorskip("gait_common")
    assert legacy.tolist() == gc._drop_short_runs(m, 2).tolist()


def test_invalid_reasons():
    fs = 100
    t = np.arange(0, 40, 1 / fs)
    f = gait_force(t)
    f[(t > T0 + 10 * STRIDE - 0.1) & (t < T0 + 10 * STRIDE + 2.2)] = 5.0   # a long pause -> stride > 3 s
    st = events.detect_strides(f, fs, CFG)
    assert (st.loc[~st.valid, "invalid_reason"] != "").all()
    assert (st.loc[st.valid, "invalid_reason"] == "").all()
    assert "stride_range" in set(st.invalid_reason)


@requires_gaitpdb
def test_matches_legacy_on_sample_of_real_files(real_manifest):
    gc = pytest.importorskip("gait_common")
    rec, _ = real_manifest
    root = paths.gaitpdb_root()
    for rel in rec.sort_values("recording_id").rel_path.iloc[::30]:     # ~11 files across studies
        trial = signal.load_trial(root / rel)
        assert trial["fs"] == pytest.approx(100.0)
        for force in (trial["fl"], trial["fr"]):
            new = events.detect_strides(force, trial["fs"], {**CFG, "debounce_impl": "legacy"})
            old = gc.detect_strides(force, 100)
            assert new[["hs_idx", "to_idx", "next_hs_idx"]].to_numpy().tolist() == \
                old[["hs", "to", "next_hs"]].to_numpy().tolist()
            assert new.valid.tolist() == old.valid.tolist()
            assert np.allclose(new.stride_s, old.stride, rtol=0, atol=1e-12)
