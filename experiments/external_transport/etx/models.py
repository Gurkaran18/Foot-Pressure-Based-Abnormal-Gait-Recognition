"""Pipeline factories M1, M1-fb, M0, M2, AS, X1, X13-ref (spec §8). All learned transforms inside the pipeline.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def make_pipeline(model_id: str, cfg, seed: int):
    """Return an unfitted sklearn Pipeline."""
    raise NotImplementedError
