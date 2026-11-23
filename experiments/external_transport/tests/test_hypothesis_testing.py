"""C2 (post-freeze) methods on fake data: key de-duplication, DeLong H1, AAUC H2 draws, H3 bootstrap, Holm.

No real labels: the key here is a fake file in tmp_path with the same columns as labels/weargait_key.csv.
"""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.external_transport.scripts import c_02_hypothesis_testing as c02


def _setup(tmp_path, n=60, signal=1.0, seed=3):
    rng = np.random.default_rng(seed)
    codes = [f"WG{i:03d}" for i in range(1, n + 3)]
    y = (np.arange(n + 2) % 2).astype(int)
    lp = signal * y + rng.normal(0, 1, n + 2)
    ext = pd.DataFrame({"subject_id": codes, "speed_01_mps": rng.normal(1.1 - 0.1 * y, 0.15),
                        "qc_pass": [True] * n + [False, False], "qc_flags": [""] * n + ["no_insole"] * 2})
    ext_p = tmp_path / "ext.csv"
    ext.to_csv(ext_p, index=False)
    sc = pd.DataFrame({"subject_id": codes[:n], "m1_fb_logit": lp[:n], "m1_logit": lp[:n] * 1.2})
    sc["m1_fb_score"] = 1 / (1 + np.exp(-sc.m1_fb_logit))
    sc["m1_score"] = 1 / (1 + np.exp(-sc.m1_logit))
    d = tmp_path / "frozen"
    d.mkdir()
    sc.to_csv(d / "external_scores.csv", index=False)
    (d / "frozen_manifest.json").write_text(json.dumps({"external_sha256": c02.sha256(ext_p), "train_sha256": "x"}))
    key = pd.DataFrame([{"code": c, "group": int(g), "task": t, "original_subject": "X"}
                        for c, g in zip(codes, y) for t in ("SelfPace", "SelfPace_mat", "SelfPace_matTURN")])
    key.to_csv(tmp_path / "key.csv", index=False)                        # 3 rows per code, like the real key
    e = tmp_path / "em"
    e.mkdir()
    ref = {"regime": "x_train-full/test-K", "auc_mean": 0.67, "auc_median": 0.67, "auc_std": 0.02}
    (e / "summary.json").write_text(json.dumps({"h3_internal_reference_m1_fb": ref, "h3_internal_reference_m1": ref}))
    pd.DataFrame({"model_id": ["M1-fb"] * 200 + ["M1"] * 200,
                  "auc": np.r_[rng.normal(0.67, 0.04, 200), rng.normal(0.79, 0.03, 200)]}).to_csv(
        e / "h3_internal_bootstrap.csv", index=False)
    return ["--scores-csv", str(d / "external_scores.csv"), "--key-csv", str(tmp_path / "key.csv"),
            "--external-table", str(ext_p), "--emulation-summary", str(e / "summary.json"),
            "--run-id", "t", "--B", "300", "--skip-tag-check"]


def test_c02_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(c02.paths, "outputs_root", lambda r: tmp_path / "out" / r)
    c02.main(_setup(tmp_path))
    r = json.loads((tmp_path / "out" / "t" / "hypothesis_testing" / "results.json").read_text())
    fb = r["M1-fb"]
    assert fb["n"] == 60 and fb["n_pd"] == 30                          # one row per code, not 3
    assert r["bd1_not_scored_group_split"]["n_not_scored"] == 2
    assert 0 < fb["H1"]["p_value"] < 0.01 and fb["H1"]["auc"] > 0.7    # DeLong one-sided
    assert fb["H2"]["speed_overlap"]["testable"] and fb["H2"]["B_valid"] > 250
    assert fb["H2"]["p_value"] not in (0.001, 0.5)
    assert fb["H3"]["gap"] == pytest.approx(0.67 - fb["H1"]["auc"])
    assert set(fb["holm"]) == {"H1", "H2", "H3"} and "holm" not in r["M1"]


def test_c02_refuses_missing_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(c02.paths, "outputs_root", lambda r: tmp_path / "out" / r)
    args = _setup(tmp_path)
    (tmp_path / "em" / "summary.json").write_text("{}")                 # no H3 reference -> no fallback numbers
    with pytest.raises(KeyError):
        c02.main(args)
    b = tmp_path / "b"
    b.mkdir()
    args = _setup(b)
    pd.read_csv(b / "ext.csv").assign(speed_01_mps=9.9).to_csv(b / "ext.csv", index=False)
    with pytest.raises(RuntimeError, match="SHA-256"):                  # table changed after the freeze
        c02.main(args)


def test_holm_and_overlap():
    h = c02.holm_bonferroni({"H1": 0.01, "H2": 0.04, "H3": 0.02})      # thresholds .0167, .025, .05
    assert h["H1"]["rejected"] and h["H3"]["rejected"] and h["H2"]["rejected"]
    h = c02.holm_bonferroni({"H1": 0.01, "H2": 0.04, "H3": 0.03})      # .03 > .025 stops the step-down
    assert h["H1"]["rejected"] and not h["H3"]["rejected"] and not h["H2"]["rejected"]
    h = c02.holm_bonferroni({"H1": 0.02, "H2": 0.001, "H3": 0.5})
    assert h["H2"]["rejected"] and h["H1"]["rejected"] and not h["H3"]["rejected"]
    ov = c02.speed_overlap(np.array([1, 1, 0, 0]), np.array([0.5, 0.6, 1.0, 1.2]))
    assert not ov["testable"]
