"""Shared pytest fixtures. Tests needing real data are skipped if GAITPDB_ROOT is unset."""
import os
import pytest

requires_gaitpdb = pytest.mark.skipif(not os.environ.get("GAITPDB_ROOT"),
                                      reason="GAITPDB_ROOT not set (real-data test)")
