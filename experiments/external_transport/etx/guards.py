"""Protected-path SHA-256 snapshot/verify and safe_write (raises on protected paths).

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §4, test T10.
"""
import hashlib
import os
from pathlib import Path

from . import paths as _paths


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _iter_files(p: Path):
    if p.is_file():
        yield p
    elif p.is_dir():
        for dirpath, _, files in os.walk(p):
            for fn in files:
                yield Path(dirpath) / fn


def snapshot(paths=None, repo_root=None) -> dict:
    """{relative_path: sha256} for every file under the protected paths (missing paths are skipped)."""
    repo_root = Path(repo_root or _paths.REPO_ROOT).resolve()
    targets = paths if paths is not None else _paths.protected_abs_paths(repo_root)
    snap = {}
    for t in targets:
        for f in _iter_files(Path(t)):
            if "__pycache__" in f.parts:
                continue
            snap[str(f.resolve().relative_to(repo_root))] = sha256_file(f)
    return dict(sorted(snap.items()))


def verify(before: dict, paths=None, repo_root=None) -> None:
    """Raise RuntimeError if any protected file changed, appeared or disappeared."""
    after = snapshot(paths, repo_root)
    changed = sorted(k for k in before.keys() & after.keys() if before[k] != after[k])
    removed = sorted(before.keys() - after.keys())
    added = sorted(after.keys() - before.keys())
    if changed or removed or added:
        raise RuntimeError(f"Protected paths modified! changed={changed} removed={removed} added={added}")


def _is_protected(path: Path, protected) -> bool:
    path = Path(path).resolve()
    return any(path == p or p in path.parents for p in protected)


def safe_write(path, data, mode="w", protected=None, readonly_roots=None) -> Path:
    """Write file; raise PermissionError if path is inside a protected path or a read-only data root."""
    protected = [Path(p).resolve() for p in (protected or _paths.protected_abs_paths())]
    if readonly_roots is None:
        readonly_roots = []
        if os.environ.get("GAITPDB_ROOT"):
            readonly_roots.append(Path(os.environ["GAITPDB_ROOT"]).expanduser().resolve())
    path = Path(path)
    if _is_protected(path, protected) or _is_protected(path, readonly_roots):
        raise PermissionError(f"Refusing to write protected/read-only path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        f.write(data)
    return path
