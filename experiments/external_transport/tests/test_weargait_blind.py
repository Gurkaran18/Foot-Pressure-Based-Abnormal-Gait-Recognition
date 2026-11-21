"""B1 blinding: file selection, random one-to-one codes, group-free names, leak check, path rules."""
import numpy as np
import pytest

from experiments.external_transport.etx import paths
from experiments.external_transport.etx import weargait_blind as wb

NAMES = ["HC1_SelfPace.csv", "HC01_SelfPace_mat.csv", "HC12_SelfPace_matTURN.csv", "HC12_TUG.csv",
         "NLS003_SelfPace.csv", "NLS3_HurriedPace_mat.csv", "HC5.mat", "NLS4 (control).mat",
         "CONTROLS - Demographic+Clinical - datasetV1.csv"]


def test_parse_and_select():
    assert wb.parse_name("HC1_SelfPace.csv") == {"subject": "HC01", "group": 0, "task": "SelfPace"}
    assert wb.parse_name("NLS003_SelfPace_matTURN.csv") == {"subject": "NLS03", "group": 1,
                                                           "task": "SelfPace_matTURN"}
    assert wb.parse_name("HC5.mat") is None and wb.parse_name("NLS4 (control).mat") is None
    sel = wb.select(NAMES)
    assert {s["task"] for s in sel} <= set(wb.DEFAULT_TASKS)
    assert sorted(s["name"] for s in sel) == ["HC01_SelfPace_mat.csv", "HC12_SelfPace_matTURN.csv",
                                              "HC1_SelfPace.csv", "NLS003_SelfPace.csv"]
    assert {s["subject"] for s in sel} == {"HC01", "HC12", "NLS03"}      # HC1 and HC01 are one person


def test_codes_one_to_one_and_group_free():
    subs = [f"HC{i:02d}" for i in range(1, 52)] + [f"NLS{i:02d}" for i in range(1, 73)]
    c = wb.assign_codes(subs, np.random.default_rng(0))
    assert len(c) == 123 and len(set(c.values())) == 123
    assert sorted(c.values()) == [f"WG{k:03d}" for k in range(1, 124)]
    hc_codes = sorted(int(c[s][2:]) for s in subs if s.startswith("HC"))
    assert hc_codes != list(range(1, 52))                                # controls are not WG001..WG051
    for s, code in c.items():
        name = wb.blinded_name(code, "SelfPace")
        assert "HC" not in name and "NLS" not in name and not wb.LEAK_RE.search(name)
    assert wb.assign_codes(subs) != wb.assign_codes(subs)                # OS randomness: not re-derivable
    with pytest.raises(ValueError):
        wb.blinded_name("HC01", "SelfPace")


def test_leak_check(tmp_path):
    clean = tmp_path / "a.csv"
    clean.write_text("Time,L1,L2\n0.00,1,2\n")
    assert not wb.leak_check(clean)
    for head in ("Subject: NLS12\nTime,L1\n", "Patient,HC 7\n", "group=Control\n", "Parkinson cohort\n"):
        bad = tmp_path / "b.csv"
        bad.write_text(head)
        assert wb.leak_check(bad)


def test_weargait_root_must_be_outside_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("WEARGAIT_ROOT", str(paths.REPO_ROOT / "weargait-pd"))
    with pytest.raises(RuntimeError, match="outside the repo"):
        paths.weargait_root()
    monkeypatch.setenv("WEARGAIT_ROOT", str(tmp_path / "wg"))
    assert paths.weargait_root(create=True) == (tmp_path / "wg").resolve()
