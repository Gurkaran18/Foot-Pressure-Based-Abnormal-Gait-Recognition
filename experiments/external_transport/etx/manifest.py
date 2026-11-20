"""Recording manifest (regex parse, SHA-256 vs SHA256SUMS.txt). Output: manifest/recordings.csv (spec §5).

Owner: Lead (data/*). Tests T1, T2 + checksum test.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import conditions
from .guards import sha256_file

TRIAL_RE = re.compile(r"^(Ga|Ju|Si)(Co|Pt)(\d+)_(\d+)\.txt$")
MANIFEST_COLUMNS = ["recording_id", "subject_id", "study", "group_code", "label", "walk", "condition",
                    "include_primary", "rel_path", "sha256", "sha256_official_match", "n_samples",
                    "duration_s", "fs_hz", "fs_uniform"]


def parse_name(filename: str):
    """Parse a trial filename -> dict, or None for non-trial files (demographics, SHA256SUMS, ...)."""
    m = TRIAL_RE.match(filename)
    if not m:
        return None
    study, group, num, walk = m.groups()
    cond, include = conditions.classify(study, walk)
    return {"recording_id": filename[:-4], "subject_id": f"{study}{group}{num}", "study": study,
            "group_code": group, "label": int(group == "Pt"), "walk": walk,
            "condition": cond, "include_primary": include}


def read_official_checksums(root: Path) -> dict:
    sums = {}
    for line in (Path(root) / "SHA256SUMS.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1].lstrip("*")] = parts[0].lower()
    return sums


def time_info(path: Path, tol: float = 0.01) -> dict:
    """n_samples, duration, fs inferred from the time column (col 0), and whether sampling is uniform.

    Uniformity = every timestamp lies within half a sample of the ideal grid t0 + i/fs. Comparing
    individual steps is too strict: gaitpdb times are printed with 2 decimals, so accumulated rounding
    shows some steps as 0.0099 s although the sampling is a steady 100 Hz. A dropped or duplicated
    sample still breaks the grid test. `tol` additionally bounds the median-vs-nominal step error.
    """
    t = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0]).iloc[:, 0].to_numpy(float)
    dt = np.diff(t)
    med = float(np.median(dt))
    fs_est = (t.size - 1) / (t[-1] - t[0])                 # overall rate over the whole trial
    grid = t[0] + np.arange(t.size) / fs_est
    uniform = bool(np.max(np.abs(t - grid)) <= 0.5 / fs_est and abs(1 / fs_est - med) <= tol * med)
    return {"n_samples": int(t.size), "duration_s": float(t[-1] - t[0]), "fs_hz": 1.0 / med,
            "fs_uniform": uniform}


def build_manifest(root, tol: float = 0.01, verify_checksums: bool = True):
    """Return (recordings_df, excluded_df). Raises on any checksum mismatch."""
    root = Path(root)
    official = read_official_checksums(root) if verify_checksums else {}
    rows = []
    for p in sorted(root.iterdir()):
        info = parse_name(p.name)
        if info is None:
            continue
        digest = sha256_file(p)
        info.update(rel_path=p.name, sha256=digest,
                    sha256_official_match=(official.get(p.name) == digest) if verify_checksums else None)
        info.update(time_info(p, tol))
        rows.append(info)
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if verify_checksums and not df["sha256_official_match"].all():
        bad = df.loc[~df["sha256_official_match"], "rel_path"].tolist()
        raise RuntimeError(f"Checksum mismatch vs SHA256SUMS.txt: {bad}")
    excluded = df.loc[~df["include_primary"], ["recording_id", "condition"]].copy()
    excluded["exclusion_reason"] = excluded["condition"].map({
        "ga02_unverified": "Ga _02: condition undocumented (D4)",
        "dual_task": "Ga _10: dual-task walking",
        "ju_ras_or_carryover": "Ju _02-_07: cued (RAS) or carryover walking"})
    return df, excluded.reset_index(drop=True)
