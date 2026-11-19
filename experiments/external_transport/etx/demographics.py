"""Robust demographics parser + HTML cross-check. NEVER fills values; UPDRS/H&Y description-only.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def load_demographics(root) -> "pd.DataFrame":
    """Columns per spec §5 demographics/clean.csv. Gender 1->M, 2->F; Ju height cm->m."""
    raise NotImplementedError
