"""T10: protected results/checkpoints/scripts are never modified; safe_write refuses them."""
import pytest

from experiments.external_transport.etx import guards, paths


def test_t10_safe_write_refuses_protected_paths(tmp_path):
    for target in [paths.REPO_ROOT / "results_new" / "x.csv", paths.REPO_ROOT / "main.py",
                   paths.REPO_ROOT / "checkpoints_pd" / "sub" / "m.pth"]:
        with pytest.raises(PermissionError):
            guards.safe_write(target, "should never be written")
    assert not (paths.REPO_ROOT / "results_new" / "x.csv").exists()


def test_t10_safe_write_refuses_readonly_data_root(tmp_path):
    with pytest.raises(PermissionError):
        guards.safe_write(tmp_path / "data" / "f.txt", "x", protected=[], readonly_roots=[tmp_path / "data"])


def test_t10_safe_write_allows_outputs(tmp_path):
    p = guards.safe_write(tmp_path / "out" / "ok.txt", "fine", protected=[paths.REPO_ROOT / "results_new"])
    assert p.read_text() == "fine"


def test_t10_verify_detects_change_add_remove(tmp_path):
    prot = tmp_path / "prot"
    prot.mkdir()
    (prot / "a.csv").write_text("1")
    (prot / "b.csv").write_text("2")
    before = guards.snapshot([prot], repo_root=tmp_path)
    guards.verify(before, [prot], repo_root=tmp_path)            # unchanged -> no error
    (prot / "a.csv").write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        guards.verify(before, [prot], repo_root=tmp_path)
    (prot / "a.csv").write_text("1")
    (prot / "b.csv").unlink()
    (prot / "c.csv").write_text("3")
    with pytest.raises(RuntimeError, match="removed"):
        guards.verify(before, [prot], repo_root=tmp_path)


def test_t10_real_protected_snapshot_nonempty():
    snap = guards.snapshot()
    assert any(k.startswith("results_new") for k in snap) and "main.py" in snap
