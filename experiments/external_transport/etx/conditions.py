"""Walking-condition rules (version c1). The ONLY path to include_primary.

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

VERSION = "c1"

def classify(study: str, walk: str) -> tuple:
    """Return (condition, include_primary). usual_primary | ga02_unverified | dual_task | ju_ras_or_carryover."""
    raise NotImplementedError
