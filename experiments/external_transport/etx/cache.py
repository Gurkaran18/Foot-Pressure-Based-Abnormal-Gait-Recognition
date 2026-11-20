"""Content-addressed cache (spec §7). Never reads results_new/gait_features_subject.csv.

Owner: Lead (data/*). Test T3.

Key = SHA-256 of canonical JSON of:
  1. sorted (recording_id, file_sha256) actually used
  2. the `signal`, `conditions` and `features` config blocks
  3. SHA-256 of the source text of signal.py, events.py, features.py, conditions.py
  4. SCHEMA_VERSION
Any change to recordings, file contents, parameters or code gives a new key -> recomputation.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import paths

SCHEMA_VERSION = 1
MODULE_FILES = ["signal.py", "events.py", "features.py", "conditions.py"]


def _src_hash(fn) -> str:
    return hashlib.sha256(Path(fn).read_bytes()).hexdigest()


def cache_key(recordings, cfg: dict, module_files=None) -> str:
    """recordings: iterable of (recording_id, sha256). cfg: full config dict."""
    module_files = module_files or [Path(__file__).parent / m for m in MODULE_FILES]
    payload = {
        "recordings": sorted([list(map(str, r)) for r in recordings]),
        "config": {k: cfg.get(k) for k in ("signal", "conditions", "features")},
        "modules": {Path(m).name: _src_hash(m) for m in module_files},
        "schema": SCHEMA_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def cache_dir(key: str) -> Path:
    return paths.OUTPUTS_DIR / "cache" / key


def load(key: str, name: str):
    p = cache_dir(key) / name
    return pd.read_csv(p) if p.exists() else None


def store(key: str, name: str, df: pd.DataFrame) -> Path:
    from .guards import safe_write
    p = cache_dir(key) / name
    if p.exists():                                       # entries are immutable
        return p
    return safe_write(p, df.to_csv(index=False))
