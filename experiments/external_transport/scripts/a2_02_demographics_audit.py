"""E2: clean demographics + missingness audit (T8).

Owner: Teammate (model/*). Run from repo root:
    python -m experiments.external_transport.scripts.a2_02_demographics_audit --run-id <id>
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import demographics, guards, paths


def main(argv=None):
    parser = argparse.ArgumentParser(description="E2: Demographics audit and clean export")
    parser.add_argument("--run-id", default="day1")
    args = parser.parse_args(argv)

    root = paths.gaitpdb_root()
    out_dir = paths.outputs_root(args.run_id) / "demographics"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = guards.snapshot()

    clean_df = demographics.load_demographics(root)
    audit_df = demographics.build_audit(clean_df)

    clean_path = out_dir / "clean.csv"
    audit_path = out_dir / "audit.csv"

    guards.safe_write(clean_path, clean_df.to_csv(index=False))
    guards.safe_write(audit_path, audit_df.to_csv(index=False))

    guards.verify(before)

    print(f"[E2] Demographics loaded: {len(clean_df)} records (HTML matches: {clean_df['html_match'].sum()}/{len(clean_df)})")
    print(f"     Clean CSV written -> {clean_path}")
    print(f"     Audit CSV written -> {audit_path}")
    print("\nSummary missingness (Overall):")
    print(audit_df[audit_df["study"] == "All"][["variable", "n", "n_missing", "pct_missing"]].to_string(index=False))


if __name__ == "__main__":
    main()
