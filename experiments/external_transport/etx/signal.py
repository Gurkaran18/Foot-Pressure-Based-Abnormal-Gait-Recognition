"""Trial loading, fs inference, smoothing/debounce in SECONDS, per-recording contact mask.

Owner: Lead (data/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def load_trial(path) -> dict:
    """{'t': seconds, 'fl': left total force, 'fr': right total force, 'fs': Hz, 'fs_uniform': bool}."""
    raise NotImplementedError

def samples(seconds: float, fs: float) -> int:
    """n = max(1, round(seconds * fs))."""
    raise NotImplementedError

def contact_mask(force, fs, cfg) -> "np.ndarray":
    """Boolean contact mask (smooth, p5/p95 relative threshold, debounce)."""
    raise NotImplementedError
