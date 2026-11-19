"""HS/TO detection (logic identical to gait_common.detect_strides, in seconds). Output: events/strides.parquet.

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def detect_strides(force, fs, cfg) -> "pd.DataFrame":
    """Stride table: hs_s, to_s, next_hs_s, stride_s, stance_s, swing_s, valid, invalid_reason."""
    raise NotImplementedError
