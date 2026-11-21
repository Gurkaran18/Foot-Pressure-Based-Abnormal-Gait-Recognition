"""Data/outputs roots (from env vars) and the protected-path list.

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PKG_ROOT = REPO_ROOT / "experiments" / "external_transport"
CONFIG_PATH = PKG_ROOT / "config" / "a2.yaml"
OUTPUTS_DIR = PKG_ROOT / "outputs"
LABELS_DIR = REPO_ROOT / "labels"          # git-ignored; WearGait-PD diagnosis key. Never read before `model-frozen`.

# Existing results and scripts that A2 must never modify (CLAUDE.md §4).
PROTECTED_PATHS = ["results_new", "results_pd", "checkpoints_pd", "main.py", "gait_common.py",
                   "gait_features.py", "cross_study.py", "explain.py", "leakage_demo.py", "app.py"]


def gaitpdb_root() -> Path:
    """Return $GAITPDB_ROOT (folder holding Ga*/Ju*/Si*.txt). Raise if unset/missing."""
    val = os.environ.get("GAITPDB_ROOT")
    if not val:
        raise RuntimeError("Set GAITPDB_ROOT to the gaitpdb folder (holds GaCo01_01.txt, SHA256SUMS.txt)")
    root = Path(val).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"GAITPDB_ROOT does not exist: {root}")
    return root


def weargait_root(create: bool = False) -> Path:
    """$WEARGAIT_ROOT (default ~/Downloads/weargait-pd). Must be outside the repo (licence: no re-upload)."""
    root = Path(os.environ.get("WEARGAIT_ROOT", "~/Downloads/weargait-pd")).expanduser().resolve()
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        raise RuntimeError(f"WEARGAIT_ROOT must be outside the repo: {root}")
    if create:
        root.mkdir(parents=True, exist_ok=True)
    elif not root.is_dir():
        raise RuntimeError(f"WEARGAIT_ROOT does not exist: {root} (run b_01_weargait_download.py first)")
    return root


def outputs_root(run_id: str) -> Path:
    """experiments/external_transport/outputs/<run_id>/ (git-ignored). Created if missing."""
    if not run_id or any(c in run_id for c in "/\\") or run_id in (".", ".."):
        raise ValueError(f"invalid run_id: {run_id!r}")
    out = OUTPUTS_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_config(path=CONFIG_PATH) -> dict:
    """Parse config/a2.yaml (the ${GAITPDB_ROOT} placeholder is left as text; use gaitpdb_root())."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def protected_abs_paths(repo_root: Path = REPO_ROOT, names=PROTECTED_PATHS):
    return [(Path(repo_root) / n).resolve() for n in names]
