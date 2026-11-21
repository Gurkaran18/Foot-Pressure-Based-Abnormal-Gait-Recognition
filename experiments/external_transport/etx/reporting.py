"""Environment capture, run manifest, tables.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §5, E10
"""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .guards import safe_write


def _git_info(repo_root: Path) -> dict:
    """Return {commit, dirty} from git; both None if git unavailable."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"commit": commit, "dirty": bool(dirty_out)}
    except Exception:
        return {"commit": None, "dirty": None}


def _pip_freeze() -> list:
    """Return list of installed package strings."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return out.splitlines()
    except Exception:
        return []


def run_manifest(run_id: str, cfg: dict) -> dict:
    """Capture environment and produce the run manifest dict.

    Includes: run_id, start_time, python version, platform, packages, git commit,
    config hash, cfg snapshot.
    """
    import hashlib

    repo_root = Path(__file__).resolve().parents[3]
    git = _git_info(repo_root)

    # Stable config hash (sorted keys, no data paths)
    cfg_bytes = json.dumps(cfg, sort_keys=True, default=str).encode()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

    manifest = {
        "run_id": run_id,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "config_hash": cfg_hash,
        "config": cfg,
        "packages": _pip_freeze(),
    }
    return manifest


def save_manifest(manifest: dict, out_dir: Path) -> Path:
    """Write run_manifest.json to out_dir using safe_write. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.json"
    safe_write(path, json.dumps(manifest, indent=2, default=str))
    return path


def save_metrics_json(metrics: dict, out_dir: Path, name: str = "metrics.json") -> Path:
    """Write a metrics dict to out_dir/<name> using safe_write. Returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    safe_write(path, json.dumps(metrics, indent=2, default=str))
    return path


def finalize_manifest(manifest: dict, out_dir: Path, protected_before: dict, protected_after: dict) -> Path:
    """Add end_time and protected-path hash comparison; write updated manifest."""
    manifest["end_time_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["protected_paths_before"] = protected_before
    manifest["protected_paths_after"] = protected_after
    manifest["protected_paths_changed"] = {
        k: v for k, v in protected_after.items()
        if protected_before.get(k) != v
    }
    return save_manifest(manifest, out_dir)
