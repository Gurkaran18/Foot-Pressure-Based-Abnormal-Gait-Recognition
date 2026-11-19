"""T1 features: v1 (legacy-equivalent, for equivalence only) and v2 (spec §6).

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def walk_features_v2(strides_L, strides_R, cfg) -> dict:
    """T1-core + double_support_v2_pct + n_valid_L/R (CV with ddof=1)."""
    raise NotImplementedError

def walk_features_v1(trial) -> dict:
    """Reproduce gait_common.gait_features exactly (legacy equivalence test)."""
    raise NotImplementedError
