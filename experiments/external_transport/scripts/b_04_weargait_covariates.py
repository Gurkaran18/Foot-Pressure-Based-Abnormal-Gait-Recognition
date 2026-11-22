"""B4: blinded covariates (SelfPace walkway speed, age, sex) -> $WEARGAIT_ROOT/blinded/covariates.csv.

Owner: Lead (data/*). Blinding step: reads labels/weargait_key.csv columns `code`, `original_subject` only.
Speed is computed from walkway footprints for every trial (B-D9); PKMAS only for an agreement check.
Prints aggregate counts only. Run from repo root:
    python -m experiments.external_transport.scripts.b_04_weargait_covariates
"""
import glob
import os

import numpy as np
import pandas as pd

from experiments.external_transport.etx import guards, paths
from experiments.external_transport.etx import weargait_blind as wb
from experiments.external_transport.etx import weargait_covariates as wc


def main():
    root = paths.weargait_root()
    restricted, out = root / "_restricted", root / "blinded" / "covariates.csv"
    key = pd.read_csv(paths.LABELS_DIR / "weargait_key.csv", usecols=["code", "original_subject"])
    before = guards.snapshot()
    cell_m = paths.load_config()["weargait"]["walkway_cell_m"]
    ws = {}
    for f in sorted(glob.glob(str(root / "blinded" / "WG*_SelfPace.csv"))):
        ws[os.path.basename(f)[:5]] = wc.walkway_speed(f, cell_m)[0]
    cov = wc.build(key, [restricted / n for n in wb.RESTRICTED_FILES.values()],
                   restricted / "PKMAS Walkway Gait Metrics - HP+SP.csv", ws)
    chk = cov.dropna(subset=["_speed_pkmas_check", "speed_selfpace_mps"])
    diff = chk.speed_selfpace_mps - chk._speed_pkmas_check
    print(f"[B4] walkway speed vs PKMAS (n={len(chk)}): r={np.corrcoef(chk.speed_selfpace_mps, chk._speed_pkmas_check)[0, 1]:.4f} "
          f"bias={diff.mean() * 100:+.2f} cm/s  max|diff|={diff.abs().max() * 100:.2f} cm/s")
    cov = cov.drop(columns="_speed_pkmas_check")
    if out.exists():
        os.chmod(out, 0o644)
    guards.safe_write(out, cov.to_csv(index=False))
    os.chmod(out, 0o444)
    guards.verify(before)
    print(f"[B4] codes={len(cov)}  missing: {cov.isna().sum().drop('code').to_dict()}")
    print(f"     speed m/s: {cov.speed_selfpace_mps.describe()[['mean','min','max']].round(3).to_dict()}  "
          f"age: {cov.age_y.astype(float).describe()[['mean','min','max']].round(1).to_dict()}  "
          f"sex: {cov.sex.value_counts().to_dict()}")
    print(f"     written to {out}")


if __name__ == "__main__":
    main()
