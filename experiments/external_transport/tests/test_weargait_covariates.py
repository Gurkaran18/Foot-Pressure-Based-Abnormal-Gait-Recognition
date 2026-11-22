"""B4 blinded covariates: ID normalisation, harmonised demographics, PKMAS speed, footprint speed, no group."""
import numpy as np
import pandas as pd
import pytest

from experiments.external_transport.etx import weargait_covariates as wc


def test_norm_id():
    assert wc.norm_id("NLS003") == "NLS03" and wc.norm_id("hc5") == "HC05" and wc.norm_id("NLS139") == "NLS139"
    assert wc.norm_id("NLS012_2") == "NLS12" and wc.norm_id("Subject ID") is None


def test_demographics_harmonised(tmp_path):
    a = tmp_path / "c.csv"
    a.write_text(",,,,\nSubject ID,Release,Age,Gender,Sex\nHC1,R1,70,Male,Male \nNLS145,R1,81,Female,Female\n")
    b = tmp_path / "p.csv"
    b.write_text("-,-,-,-,-,-\nSubject ID,Release,Age (years),Sex,DBS?,Years since PD diagnosis\nNLS7,R1,66.6,male,No,3\n")
    d = pd.concat([wc.read_demographics(a), wc.read_demographics(b)])
    assert list(d.columns) == ["original_subject", "age_y", "sex"]              # PD-only columns never kept
    assert d.sex.tolist() == ["M", "F", "M"] and d.original_subject.tolist() == ["HC01", "NLS145", "NLS07"]


def test_speed_and_build(tmp_path):
    pk = tmp_path / "pk.csv"
    pk.write_text("Task,Participant ID,PD vs Control,Velocity (cm./sec.)\n,,,\nSelfPace ,HC001,Control,100\n"
                  "SelfPace,NLS7,PD,80\nHurriedPace,NLS7,PD,120\n")
    demo = tmp_path / "d.csv"
    demo.write_text(",,\nSubject ID,Age,Sex\nHC1,70.4,Male\nNLS7,66.6,Female\n")
    key = pd.DataFrame({"code": ["WG002", "WG001", "WG003"], "original_subject": ["HC01", "NLS07", "NLS09"]})
    out = wc.build(key, [demo], pk)
    assert out.code.tolist() == ["WG001", "WG002", "WG003"] and "group" not in out.columns
    assert out.speed_selfpace_mps.tolist()[:2] == [0.8, 1.0] and np.isnan(out.speed_selfpace_mps.iloc[2])
    assert out.age_y.tolist()[:2] == [67, 70]
    out2 = wc.build(key, [demo], pk, {"WG001": 0.81234, "WG002": 0.99, "WG003": 1.1})
    assert out2.speed_selfpace_mps.tolist() == [0.812, 0.99, 1.1]               # footprint speed is the covariate
    with pytest.raises(ValueError):
        wc.build(key.assign(group=0), [demo], pk)


def test_walkway_speed_from_footprints(tmp_path):
    """Left foot lands every 1.0 s, 100 cells further on: 100 cells x 0.0127 m / 1.0 s = 1.27 m/s."""
    n, rows = 400, []
    for i in range(n):
        k, ph = divmod(i, 100)
        on = ph < 60
        x = "|".join(str(10 + 100 * k + dx) for dx in (0, 1, 2)) if on else ""
        rows.append({"L Foot Contact": int(on), "R Foot Contact": 0, "Walkway_X": x,
                     "Walkway_Y": "5|5|5" if on else "", "WalkwayPressureLevel": "3|3|3" if on else "",
                     "WalkwayFoot": "L|L|L" if on else ""})
    p = tmp_path / "WG001_SelfPace.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    s, k = wc.walkway_speed(p)
    assert k == 3 and abs(s - 1.27) < 1e-9
