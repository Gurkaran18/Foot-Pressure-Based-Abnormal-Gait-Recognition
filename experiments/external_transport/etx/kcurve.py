"""K-curve: S1/S2 stride sampling (Lead) and evaluation + stability stats (Teammate).

Owner: Lead: samplers / Teammate: evaluation. Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def sample_strides(strides, K, scheme, rng):
    """Lead. S1_contiguous or S2_random subset of valid strides per foot."""
    raise NotImplementedError

def evaluate(feature_tables, cfg):
    """Teammate. CV AUC per (scheme, K, draw, regime); ICC(2,1), within-SD, bias, univariate AUC."""
    raise NotImplementedError
