"""E0: environment lockfile + protected-path snapshot.

Owner: Lead (data/*). Run from repo root:
    python -m experiments.external_transport.scripts.a2_00_env_and_guard --run-id <id>
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime

from experiments.external_transport.etx import guards, paths


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=default_run_id())
    args = ap.parse_args(argv)
    out = paths.outputs_root(args.run_id)

    snap = guards.snapshot()
    guards.safe_write(out / "env" / "protected_snapshot.json", json.dumps(snap, indent=1))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    guards.safe_write(out / "env" / "pip_freeze.txt", freeze)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=paths.REPO_ROOT).stdout.strip()
    env = {"python": sys.version, "platform": platform.platform(), "git_commit": git("rev-parse", "HEAD"),
           "git_dirty": bool(git("status", "--porcelain", "--", "experiments")),
           "n_protected_files": len(snap)}
    guards.safe_write(out / "env" / "environment.json", json.dumps(env, indent=1))
    print(f"[E0] run_id={args.run_id}  protected files snapshotted: {len(snap)}  -> {out/'env'}")


if __name__ == "__main__":
    main()
