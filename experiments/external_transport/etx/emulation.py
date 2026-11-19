"""Pseudo-SelfPace walks (4 segments x 6-7 consecutive valid strides).

Owner: Lead: walks / Teammate: evaluation. Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def emulate_walk(strides, cfg, rng):
    """Lead. Build one pseudo-SelfPace stride set (turn-adjacent strides dropped)."""
    raise NotImplementedError

def evaluate(feature_tables, cfg):
    """Teammate. Emulated internal AUC (train-full/test-K and train-K/test-K)."""
    raise NotImplementedError
