"""Shared pytest fixtures. Tests needing real data are skipped if GAITPDB_ROOT is unset.

The session-wide guard (T10) snapshots every protected path before the test session and
verifies it afterwards, so no test can silently modify existing results or scripts.
"""
import os

import pytest

from experiments.external_transport.etx import guards

requires_gaitpdb = pytest.mark.skipif(not os.environ.get("GAITPDB_ROOT"),
                                      reason="GAITPDB_ROOT not set (real-data test)")


@pytest.fixture(scope="session", autouse=True)
def protected_paths_unchanged():
    before = guards.snapshot()
    yield
    guards.verify(before)


@pytest.fixture(scope="session")
def real_manifest():
    from experiments.external_transport.etx import manifest, paths
    return manifest.build_manifest(paths.gaitpdb_root())
