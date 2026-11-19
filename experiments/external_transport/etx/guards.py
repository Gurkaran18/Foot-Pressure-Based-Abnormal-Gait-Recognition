"""Protected-path SHA-256 snapshot/verify and safe_write (raises on protected paths).

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def snapshot(paths=None) -> dict:
    """{relative_path: sha256} for every file under the protected paths."""
    raise NotImplementedError

def verify(before: dict) -> None:
    """Raise RuntimeError if any protected file changed, appeared or disappeared."""
    raise NotImplementedError

def safe_write(path, data, mode="w"):
    """Write file; raise PermissionError if path is inside a protected path."""
    raise NotImplementedError
