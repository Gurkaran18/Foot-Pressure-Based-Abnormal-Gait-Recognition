"""Fake external feature table generator for test isolation (spec §5, §11).

Produces a DataFrame matching outputs/<run_id>/weargait/subject_external_v2.csv layout,
strictly label-free.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.external_transport.scripts import b_05_external_table as b05


def make_fake_external_table(n: int = 20, seed: int = 42, output_path=None) -> pd.DataFrame:
    """Generate fake WearGait-PD external subject table without labels."""
    rng = np.random.default_rng(seed)
    codes = [f"WG{i+1:03d}" for i in range(n)]

    sexes = rng.choice(["M", "F"], n)
    ages = rng.integers(50, 85, n).astype(float)
    speeds = np.round(rng.normal(1.0, 0.2, n), 3)

    # Realistic gait features
    stride_mean = np.round(rng.normal(1.10, 0.10, n), 3)
    stride_cv = np.round(np.clip(rng.normal(2.5, 0.8, n), 1.0, 8.0), 3)
    swing_cv = np.round(np.clip(rng.normal(4.8, 1.5, n), 1.5, 15.0), 3)
    stance_pct = np.round(np.clip(rng.normal(62.0, 2.0, n), 55.0, 72.0), 3)
    asym_swing = np.round(np.clip(rng.exponential(3.0, n), 0.1, 20.0), 3)
    double_sup = np.round(np.clip(rng.normal(23.0, 3.5, n), 15.0, 35.0), 3)

    n_valid_l = rng.integers(12, 25, n)
    n_valid_r = rng.integers(12, 25, n)

    has_insole = np.ones(n, dtype=bool)
    # A couple without insole / failing QC
    if n > 5:
        has_insole[-2:] = False

    qc_pass = has_insole.copy()
    qc_flags = ["no_insole" if not hi else "" for hi in has_insole]

    df = pd.DataFrame({
        "subject_id": codes,
        "study": "WG",
        "sex": sexes,
        "age_y": ages,
        "speed_01_mps": speeds,
        "speed_method_status": "walkway_footprint",
        "stride_time_mean": stride_mean,
        "stride_time_cv": stride_cv,
        "swing_time_cv": swing_cv,
        "stance_pct": stance_pct,
        "asym_swing": asym_swing,
        "double_support_v2_pct": double_sup,
        "n_valid_L": n_valid_l,
        "n_valid_R": n_valid_r,
        "qc_pass": qc_pass,
        "qc_flags": qc_flags,
        "has_insole": has_insole,
        "source_recordings": [f"{c}_SelfPace.csv" for c in codes],
    })

    # Ensure column order matches b05.COLUMNS
    df = df[b05.COLUMNS]

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)

    return df
