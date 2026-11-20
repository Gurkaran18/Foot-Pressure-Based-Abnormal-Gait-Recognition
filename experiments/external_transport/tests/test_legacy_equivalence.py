"""T5 (second half): legacy equivalence on all 306 real files (spec §12.2).

The rewritten pipeline, run in legacy mode (debounce bug, ddof=0, all walks averaged), must reproduce
gait_common stride indices exactly and results_new/gait_features_subject.csv within 1e-9 (read-only).
"""
import pandas as pd
import pytest

from experiments.external_transport.etx import paths
from experiments.external_transport.tests.conftest import requires_gaitpdb


@requires_gaitpdb
def test_t5_legacy_equivalence_all_306_files(real_manifest):
    pytest.importorskip("gait_common")
    from experiments.external_transport.scripts import a2_04_legacy_equivalence as eq
    rec, _ = real_manifest
    v1, mismatches, n_strides = eq.run_equivalence(paths.gaitpdb_root(), rec, paths.load_config())
    assert len(rec) == 306 and n_strides > 50000
    assert mismatches == []
    res = eq.compare(v1, pd.read_csv(eq.LEGACY_CSV))
    assert res["n_subjects_matched"] == 165 and res["nan_pattern_mismatches"] == 0
    assert all(d <= eq.TOL for d in res["max_abs_diff"].values()), res["max_abs_diff"]
