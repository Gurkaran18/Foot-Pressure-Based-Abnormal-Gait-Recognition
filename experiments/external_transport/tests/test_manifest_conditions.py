"""T1 (usual-walking filter), T2 (no cued/dual-task in primary), checksum verification."""
import pytest

from experiments.external_transport.etx import conditions, manifest
from experiments.external_transport.tests.conftest import requires_gaitpdb


# ---- T1(a): synthetic filenames covering every pattern ------------------------------------
@pytest.mark.parametrize("name,cond,inc", [
    ("GaCo01_01.txt", "usual_primary", True),
    ("GaPt03_02.txt", "ga02_unverified", False),
    ("GaPt03_10.txt", "dual_task", False),
    ("JuCo01_01.txt", "usual_primary", True),
    *[(f"JuPt05_0{w}.txt", "ju_ras_or_carryover", False) for w in range(2, 8)],
    ("SiPt40_01.txt", "usual_primary", True),
])
def test_t1_filter_patterns(name, cond, inc):
    info = manifest.parse_name(name)
    assert info["condition"] == cond and info["include_primary"] is inc
    assert info["label"] == int("Pt" in name)


@pytest.mark.parametrize("name", ["demographics.txt", "SHA256SUMS.txt", "format.txt", "gaitpd.png",
                                  "GaCo01_01.csv", "XxCo01_01.txt", "GaCo01.txt", "gaco01_01.txt"])
def test_t1_non_trial_files_ignored(name):
    assert manifest.parse_name(name) is None


@pytest.mark.parametrize("study,walk", [("Si", "02"), ("Ga", "03"), ("Ju", "08"), ("Ju", "10")])
def test_t1_unknown_condition_raises(study, walk):
    with pytest.raises(ValueError):
        conditions.classify(study, walk)


# ---- T1(b) + T2: real gaitpdb manifest ------------------------------------------------------
@requires_gaitpdb
def test_t1_real_counts(real_manifest):
    rec, _ = real_manifest
    prim = rec[rec.include_primary]
    assert len(rec) == 306 and rec.subject_id.nunique() == 165
    assert len(prim) == 165 and prim.subject_id.nunique() == 165
    assert prim.groupby("study").size().to_dict() == {"Ga": 47, "Ju": 54, "Si": 64}
    assert prim.groupby("label").size().to_dict() == {0: 72, 1: 93}


@requires_gaitpdb
def test_t2_no_cued_or_dual_task_in_primary(real_manifest):
    rec, exc = real_manifest
    assert (rec.loc[rec.include_primary, "walk"] == "01").all()
    counts = exc.groupby("condition").size().to_dict()
    assert counts == {"ga02_unverified": 39, "dual_task": 27, "ju_ras_or_carryover": 75}
    assert len(exc) + rec.include_primary.sum() == 306


@requires_gaitpdb
def test_checksums_match_official(real_manifest):
    rec, _ = real_manifest
    assert rec.sha256_official_match.all()


@requires_gaitpdb
def test_real_sampling_is_uniform_100hz(real_manifest):
    rec, _ = real_manifest
    assert rec.fs_uniform.all()
    assert ((rec.fs_hz - 100.0).abs() < 1e-6).all()


def test_uniformity_check_rounding_vs_dropout(tmp_path):
    import numpy as np
    t = np.cumsum(np.full(3000, 0.01)) - 0.01                  # float-accumulated 100 Hz clock
    ok = tmp_path / "ok.txt"
    ok.write_text("\n".join(f"{x:.2f}\t1" for x in t))         # printed to 2 decimals
    assert manifest.time_info(ok)["fs_uniform"]
    gap = tmp_path / "gap.txt"
    gap.write_text("\n".join(f"{x:.2f}\t1" for x in np.delete(t, range(1000, 1005))))  # 5 dropped samples
    assert not manifest.time_info(gap)["fs_uniform"]
