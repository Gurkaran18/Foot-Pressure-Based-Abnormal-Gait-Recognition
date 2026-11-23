"""E9: emulation — Lead builds the feature tables; Teammate's emulation.evaluate() scores them (gate).

Run from repo root (after a2_03 for the same run id):
    GAITPDB_ROOT=... python -m experiments.external_transport.scripts.a2_09_emulation --run-id <id>
Reads  outputs/<id>/events/strides.csv and features/subject_v2.csv
Writes outputs/<id>/emulation/feature_tables.csv, eligibility.csv, results.csv, summary.json
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import emulation, guards, paths


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    run = paths.outputs_root(args.run_id)
    cfg = paths.load_config()

    before = guards.snapshot()
    strides = pd.read_csv(run / "events" / "strides.csv")
    subjects = pd.read_csv(run / "features" / "subject_v2.csv")
    feats, elig = emulation.feature_tables(strides, subjects, cfg)
    guards.safe_write(run / "emulation" / "feature_tables.csv", feats.to_csv(index=False))
    guards.safe_write(run / "emulation" / "eligibility.csv", elig.to_csv(index=False))
    print(f"[E9] emulation feature tables: {len(feats)} rows -> {run / 'emulation'}")
    print(elig.groupby([c for c in ("scheme", "K") if c in elig.columns] or ["draw"])
              [["n_eligible_Co", "n_eligible_PD"]].min().to_string())

    res_df, summary = emulation.evaluate(feats, cfg, full_subjects_df=subjects, out_dir=run / "emulation")
    print(f"[E9] Evaluation complete: {len(res_df)} result rows.")
    print(f"     D15 Decision (M1-fb): {summary['d15_decision_m1_fb']['selected_training_regime']} "
          f"(train-K median = {summary['d15_decision_m1_fb']['train_k_median_auc']:.4f}, "
          f"train-full median = {summary['d15_decision_m1_fb']['train_full_median_auc']:.4f}, "
          f"diff = {summary['d15_decision_m1_fb']['difference']:.4f})")
    print(f"     H3 Internal Reference (M1-fb emulated AUC): "
          f"{summary['h3_internal_reference_m1_fb']['auc_mean']:.4f} ± {summary['h3_internal_reference_m1_fb']['auc_std']:.4f}")

    guards.verify(before)


if __name__ == "__main__":
    main()
