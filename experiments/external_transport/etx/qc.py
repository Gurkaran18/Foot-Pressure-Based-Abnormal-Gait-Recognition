"""Per-walk QC flags, K-eligibility, subject aggregation -> features/subject_v2.csv (the hand-off file).

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def subject_table(walk_df, demographics_df) -> "pd.DataFrame":
    """One row per subject; columns per spec §5 features/subject_v2.csv."""
    raise NotImplementedError
