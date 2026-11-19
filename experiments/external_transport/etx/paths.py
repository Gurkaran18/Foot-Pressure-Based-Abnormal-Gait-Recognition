"""Data/outputs roots (from env vars) and the protected-path list.

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PROTECTED_PATHS = ["results_new", "results_pd", "checkpoints_pd", "main.py", "gait_common.py",
                   "gait_features.py", "cross_study.py", "explain.py", "leakage_demo.py", "app.py"]

def gaitpdb_root() -> Path:
    """Return $GAITPDB_ROOT (folder holding Ga*/Ju*/Si*.txt). Raise if unset/missing."""
    raise NotImplementedError

def outputs_root(run_id: str) -> Path:
    """experiments/external_transport/outputs/<run_id>/ (git-ignored)."""
    raise NotImplementedError
