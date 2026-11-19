"""Content-addressed cache (spec §7). Never reads results_new/gait_features_subject.csv.

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

SCHEMA_VERSION = 1

def cache_key(recordings, cfg: dict, module_files) -> str:
    """SHA-256 of canonical JSON of (recording ids+hashes, config blocks, module source hashes, schema)."""
    raise NotImplementedError
