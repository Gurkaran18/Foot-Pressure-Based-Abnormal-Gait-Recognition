"""T5 (feature calculations vs closed form) and T3 (cache invalidation)."""
import copy

import numpy as np
import pandas as pd
import pytest

from experiments.external_transport.etx import cache, features, paths, qc

CFG = paths.load_config()


def stride_table(strides, swing, t0=0.0):
    """Valid stride table from stride durations and swing durations (stance = stride - swing)."""
    strides, swing = np.asarray(strides, float), np.asarray(swing, float)
    hs = t0 + np.r_[0, np.cumsum(strides)[:-1]]
    return pd.DataFrame({"hs_s": hs, "to_s": hs + strides - swing, "next_hs_s": hs + strides,
                         "stride_s": strides, "stance_s": strides - swing, "swing_s": swing,
                         "valid": True})


# ---- T5: spec's synthetic bilateral gait -------------------------------------------------
N = 40
T_L = np.tile([1.00, 1.10], N // 2)          # left strides alternate 1.00 / 1.10 s
T_R = np.full(N, 1.05)                       # right strides 1.05 s
W_L, W_R = np.full(N, 0.40), np.full(N, 0.44)


def test_t5_closed_form_values():
    f = features.walk_features_v2(stride_table(T_L, W_L), stride_table(T_R, W_R, 0.5), CFG["features"])
    cvL = 100 * np.std(T_L, ddof=1) / T_L.mean()
    assert f["n_valid_L"] == N and f["n_valid_R"] == N
    assert f["stride_time_mean"] == pytest.approx(np.r_[T_L, T_R].mean(), abs=1e-9)
    assert f["stride_time_cv"] == pytest.approx((cvL + 0.0) / 2, abs=1e-9)
    assert f["swing_time_cv"] == pytest.approx(0.0, abs=1e-9)
    stance_pct = 100 * np.mean(np.r_[(T_L - W_L) / T_L, (T_R - W_R) / T_R])
    assert f["stance_pct"] == pytest.approx(stance_pct, abs=1e-9)
    assert f["asym_swing"] == pytest.approx(100 * np.log(1.1), abs=1e-9)      # = 9.531
    assert round(f["asym_swing"], 3) == 9.531


def test_t5_cv_uses_sample_sd():
    x = [1.0, 1.1, 1.0, 1.1, 1.0]
    assert features.cv(x) == pytest.approx(100 * np.std(x, ddof=1) / np.mean(x))
    assert features.cv(x) > features.cv(x, ddof=0)                            # ddof=0 understates
    assert np.isnan(features.cv([1.0]))


def test_t5_double_support_hand_computed():
    # L: strides 1.0 s, contact [0, 0.6]; R: contact [0.5, 1.1] (each 1.0 s stride, offset 0.5 s)
    L = stride_table([1.0] * 4, [0.4] * 4, 0.0)
    R = stride_table([1.0] * 4, [0.4] * 4, 0.5)
    f = features.walk_features_v2(L, R, CFG["features"])
    # each L stride [hs, hs+1]: overlaps R contact [hs-0.5, hs+0.1] (0.1 s) and [hs+0.5, hs+0.6] (0.1 s)
    # -> 0.2 of the stride, except the first L stride (no earlier R contact) -> 0.1. Mean L = (0.1 + 0.6)/4
    # R strides: overlap with L contacts at both ends -> 0.2 each, except the last (no later L) -> 0.1
    expected = 100 * np.mean([(0.1 + 3 * 0.2) / 4, (3 * 0.2 + 0.1) / 4])
    assert f["double_support_v2_pct"] == pytest.approx(expected, abs=1e-9)


def test_invalid_strides_ignored_and_too_few():
    L = stride_table(T_L, W_L)
    L.loc[0, ["stride_s", "valid"]] = [9.9, False]
    f1 = features.walk_features_v2(L, stride_table(T_R, W_R, .5), CFG["features"])
    f2 = features.walk_features_v2(L[L.valid], stride_table(T_R, W_R, .5), CFG["features"])
    assert f1 == pytest.approx(f2, nan_ok=True)
    f3 = features.walk_features_v2(L.iloc[:1], stride_table(T_R, W_R, .5), CFG["features"])
    assert np.isnan(f3["stride_time_mean"]) and "missing_feature" in qc.qc_flags(f3, 10)


def test_subject_table_one_row_per_subject():
    w = pd.DataFrame({"subject_id": ["A", "A", "B"], "label": [1, 1, 0], "study": ["Ga"] * 3,
                      "include_primary": [True, False, True], "recording_id": ["A_01", "A_10", "B_01"],
                      "n_valid_L": 20, "n_valid_R": 20, "qc_pass": True, "qc_flags": "",
                      **{k: 1.0 for k in features.WALK_V2_FEATURES}})
    s = qc.subject_table(w)
    assert s.subject_id.tolist() == ["A", "B"] and s.source_recordings.tolist() == ["A_01", "B_01"]
    with pytest.raises(RuntimeError):
        qc.subject_table(w.assign(include_primary=True))


# ---- T3: cache invalidation --------------------------------------------------------------
RECS = [("GaCo01_01", "aa"), ("GaPt03_01", "bb")]


def test_t3_identical_inputs_identical_key():
    assert cache.cache_key(RECS, CFG) == cache.cache_key(list(reversed(RECS)), copy.deepcopy(CFG))


def test_t3_every_change_gives_new_key(tmp_path):
    base = cache.cache_key(RECS, CFG)
    assert cache.cache_key(RECS[:1] + [("GaPt04_01", "bb")], CFG) != base          # recording id
    assert cache.cache_key(RECS[:1] + [("GaPt03_01", "cc")], CFG) != base          # file content
    for block in ("signal", "conditions", "features"):
        for k, v in CFG[block].items():
            c = copy.deepcopy(CFG)
            c[block][k] = (v + 1 if isinstance(v, (int, float)) and not isinstance(v, bool)
                           else f"changed-{v}")
            assert cache.cache_key(RECS, c) != base, f"{block}.{k} did not change the key"
    # editing a feature module's source text changes the key
    mods = [paths.PKG_ROOT / "etx" / m for m in cache.MODULE_FILES]
    fake = tmp_path / "features.py"
    fake.write_text(mods[2].read_text() + "\n# edit\n")
    assert cache.cache_key(RECS, CFG, mods[:2] + [fake] + mods[3:]) != base


def test_t3_cache_never_touches_legacy_csv():
    src = (paths.PKG_ROOT / "etx" / "cache.py").read_text().split('"""')[2]      # code after docstring
    assert "gait_features_subject" not in src and "results_new" not in src
