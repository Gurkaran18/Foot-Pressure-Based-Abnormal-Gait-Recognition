"""Trial loading, fs inference, smoothing/debounce in SECONDS, per-recording contact mask.

Owner: Lead (data/*). Spec §4; decision R3 (parameters in seconds, native fs, no resampling).

Logic is identical to gait_common.contact_mask, except that the smoothing window and the debounce
length are given in seconds and converted with the recording's own fs:
    n = max(1, round(seconds * fs))
At fs = 100 Hz this gives exactly the legacy values (5-sample smoothing, 10-sample debounce).
"""
from pathlib import Path

import numpy as np

COL_TIME, COL_TOTAL_L, COL_TOTAL_R = 0, 17, 18


def samples(seconds: float, fs: float) -> int:
    """n = max(1, round(seconds * fs))."""
    return max(1, int(round(seconds * fs)))


def infer_fs(t) -> float:
    """Sampling rate from the time column: 1 / median step (robust to 2-decimal printing drift)."""
    return 1.0 / float(np.median(np.diff(np.asarray(t, float))))


def load_trial(path) -> dict:
    """{'t': seconds, 'fl': left total force, 'fr': right total force, 'fs': Hz} for a gaitpdb file."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 19:
        raise ValueError(f"{path}: expected 19 columns, got shape {data.shape}")
    t = data[:, COL_TIME]
    return {"t": t, "fl": data[:, COL_TOTAL_L], "fr": data[:, COL_TOTAL_R], "fs": infer_fs(t),
            "recording_id": Path(path).stem}


def drop_short_runs(mask, min_len: int, legacy: bool = False):
    """Flip on/off runs shorter than min_len samples (removes short contacts, then short gaps).

    legacy=True reproduces gait_common._drop_short_runs exactly, INCLUDING its padding bug:
    it pads with `~target` (Python ints -2 / -1, not booleans), which adds spurious edges at the
    ends and mis-pairs runs, so only one of the two passes acts (which one depends on whether the
    recording starts in stance or swing). Used only for v1 legacy equivalence. v2 uses the fix.
    """
    mask = np.asarray(mask, bool).copy()
    for target in (True, False):
        pad = (~int(target)) if legacy else False          # ~int(True) == -2, as legacy ~True did
        edges = np.flatnonzero(np.diff(np.r_[pad, mask == target, pad].astype(int)))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start < min_len and start > 0 and stop < len(mask):
                mask[start:stop] = not target
    return mask


def odd_samples(seconds: float, fs: float) -> int:
    """Nearest-odd window length. An even moving-average window with mode='same' shifts the signal
    by half a sample (10 ms at 50 Hz), which would move every event at low fs. 100 Hz -> 5 (legacy)."""
    n = samples(seconds, fs)
    return n if n % 2 else n + 1


def smooth(force, fs: float, smooth_s: float):
    n = odd_samples(smooth_s, fs)
    return np.convolve(np.asarray(force, float), np.ones(n) / n, mode="same")


def contact_mask(force, fs: float, cfg: dict):
    """Boolean contact mask: smooth -> relative threshold (p5..p95 per recording) -> debounce.

    cfg = the `signal` block of config/a2.yaml (smooth_s, debounce_s, contact_fraction, percentiles).
    """
    f = smooth(force, fs, cfg["smooth_s"])
    lo, hi = np.percentile(f, cfg["percentiles"])
    loaded = f > lo + cfg["contact_fraction"] * (hi - lo)
    legacy = cfg.get("debounce_impl", "fixed") == "legacy"
    return drop_short_runs(loaded, samples(cfg["debounce_s"], fs), legacy=legacy)
