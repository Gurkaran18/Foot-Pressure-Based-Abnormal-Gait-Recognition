"""Recording manifest (regex parse, SHA-256 vs SHA256SUMS.txt). Output: manifest/recordings.csv (spec §5).

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

TRIAL_RE = r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$"

def build_manifest(root) -> "pd.DataFrame":
    """One row per trial file; columns per spec §5 manifest/recordings.csv."""
    raise NotImplementedError
