"""E8: kcurve — Lead builds the feature tables; Teammate's kcurve.evaluate() scores them (gate).

Run from repo root (after a2_03 for the same run id):
    GAITPDB_ROOT=... python -m experiments.external_transport.scripts.a2_08_kcurve --run-id <id>
Reads  outputs/<id>/events/strides.csv and features/subject_v2.csv
Writes outputs/<id>/kcurve/feature_tables.csv, eligibility.csv, results.csv, stability.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import kcurve, guards, paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    run = paths.outputs_root(args.run_id)
    cfg = paths.load_config()

    before = guards.snapshot()
    strides = pd.read_csv(run / "events" / "strides.csv")
    subjects = pd.read_csv(run / "features" / "subject_v2.csv")
    feats, elig = kcurve.feature_tables(strides, subjects, cfg)
    guards.safe_write(run / "kcurve" / "feature_tables.csv", feats.to_csv(index=False))
    guards.safe_write(run / "kcurve" / "eligibility.csv", elig.to_csv(index=False))
    print(f"[E8] kcurve feature tables: {len(feats)} rows -> {run / 'kcurve'}")
    print(elig.groupby([c for c in ("scheme", "K") if c in elig.columns] or ["draw"])
              [["n_eligible_Co", "n_eligible_PD"]].min().to_string())

    res_df, stab_df = kcurve.evaluate(feats, cfg, full_subjects_df=subjects, out_dir=run / "kcurve")
    print(f"[E8] Evaluation complete: {len(res_df)} result rows, {len(stab_df)} stability rows.")

    guards.verify(before)


if __name__ == "__main__":
    main()
