"""WearGait-PD download blinding: neutral subject codes, group-free file names, leak check.

Owner: Lead (data/*). Decision D10 (modified: no custodian; `labels/` unread before `model-frozen`).

Why: in WearGait-PD both the Synapse folder ("CONTROL"/"PD PARTICIPANTS") and the file prefix
(`HC##_` = control, `NLS##_` = PD) reveal the diagnosis. Every downloaded trial is therefore renamed
to `<code>_<task>.csv` with a random code `WG###`. The code -> original id -> group key is written
once to `labels/` and never read by any code before the freeze.

Codes come from the OS random source (not a fixed seed): the repo is public and the original
file list is public, so a seeded permutation could be re-derived by anyone.
"""
import re
import secrets

import numpy as np

# Synapse ids, WearGait-PD project syn52540892, "Version 1 (Cross-sectional)" = syn55052683.
SYN_VERSION1 = "syn55052683"
SYN_GROUP_FOLDERS = {"CONTROL PARTICIPANTS": "syn61370532", "PD PARTICIPANTS": "syn61370536"}
CSV_FOLDER_NAME = "CSV files"
# Group-split or group-revealing tables: stored under <root>/_restricted/ with original names,
# never inside blinded/. Turned into blinded tables only through an explicit column allow-list.
RESTRICTED_FILES = {
    "syn55105521": "CONTROLS - Demographic+Clinical - datasetV1.csv",
    "syn55105530": "PD - Demographic+Clinical - datasetV1.csv",
}
WALKWAY_METRICS_FOLDER = "syn64589874"          # 'PKMAS Walkway Gait Metrics - HP+SP.csv' (has subject ids)

DEFAULT_TASKS = ("SelfPace", "SelfPace_mat", "SelfPace_matTURN")
FILE_RE = re.compile(r"^(HC|NLS)(\d+)_([A-Za-z]+(?:_[A-Za-z]+)*)\.csv$")
GROUP_OF_PREFIX = {"HC": 0, "NLS": 1}           # 0 = control, 1 = PD (written to labels/ only)
CODE_RE = re.compile(r"^WG\d{3}$")
# Strings that would reveal group inside a blinded file (checked in the first lines only).
LEAK_RE = re.compile(r"\b(HC|NLS|WHC|WPD)\s?\d+|parkinson|control", re.IGNORECASE)


def parse_name(name: str):
    """'HC12_SelfPace.csv' -> {'subject': 'HC12', 'group': 0, 'task': 'SelfPace'}; None if not a trial."""
    m = FILE_RE.match(name)
    if not m:
        return None
    prefix, num, task = m.groups()
    return {"subject": f"{prefix}{int(num):02d}", "group": GROUP_OF_PREFIX[prefix], "task": task}


def select(names, tasks=DEFAULT_TASKS) -> list:
    """Trial files whose task is in `tasks`: [{'name', 'subject', 'group', 'task'}], sorted by name."""
    out = []
    for n in sorted(names):
        info = parse_name(n)
        if info and info["task"] in tasks:
            out.append({"name": n, **info})
    return out


def assign_codes(subjects, rng=None) -> dict:
    """{original_subject: 'WG###'}: a random one-to-one map onto WG001..WGnnn (n = #subjects).

    rng=None uses the OS random source (production). Tests pass a seeded np.random.Generator.
    """
    subs = sorted(set(subjects))
    if len(subs) > 999:
        raise ValueError("more than 999 subjects")
    if rng is None:
        rng = np.random.default_rng(secrets.randbits(128))
    perm = rng.permutation(len(subs)) + 1
    return {s: f"WG{int(k):03d}" for s, k in zip(subs, perm)}


def blinded_name(code: str, task: str) -> str:
    if not CODE_RE.match(code):
        raise ValueError(f"bad code {code!r}")
    return f"{code}_{task}.csv"


def leak_check(path, n_lines: int = 20) -> bool:
    """True if the first n_lines of a blinded file contain a group-revealing string."""
    with open(path, errors="replace") as f:
        head = "".join(line for _, line in zip(range(n_lines), f))
    return bool(LEAK_RE.search(head))
