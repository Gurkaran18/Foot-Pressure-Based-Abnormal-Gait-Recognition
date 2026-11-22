"""B5: WearGait-PD subject table for frozen-model scoring (same columns as features/subject_v2.csv, no label).

Owner: Lead (data/*). Hand-off file to the Teammate's external scoring. Run from repo root after b_02 and b_04:
    python -m experiments.external_transport.scripts.b_05_external_table --run-id <b_02 run id>
Reads  outputs/<id>/weargait/walk_features.csv  and  $WEARGAIT_ROOT/blinded/covariates.csv
Writes outputs/<id>/weargait/subject_external_v2.csv (+ .sha256). One row per code (all 123; trials without
insoles kept with qc_pass = False so the counts stay visible). `label` is absent: it lives only in labels/.
Columns: subject_id (= WG code), study ("WG"), sex (M/F), age_y, speed_01_mps (SelfPace walkway footprint
speed, B-D9), speed_method_status, the v2 features, n_valid_L, n_valid_R, qc_pass, qc_flags, has_insole,
source_recordings.
"""
import argparse

import pandas as pd

from experiments.external_transport.etx import features, guards, paths

COLUMNS = (["subject_id", "study", "sex", "age_y", "speed_01_mps", "speed_method_status"]
           + features.WALK_V2_FEATURES
           + ["n_valid_L", "n_valid_R", "qc_pass", "qc_flags", "has_insole", "source_recordings"])


def build(walks: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    if "label" in walks.columns or "label" in cov.columns or "group" in cov.columns:
        raise ValueError("external table inputs must be label-free")
    w = walks.rename(columns={"code": "subject_id"})
    w["source_recordings"] = w["subject_id"] + "_" + w["task"] + ".csv"
    c = cov.rename(columns={"code": "subject_id", "speed_selfpace_mps": "speed_01_mps"})
    out = w.merge(c, on="subject_id", how="left", validate="one_to_one")
    out["study"] = "WG"
    out["speed_method_status"] = "walkway_footprint"
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["qc_pass"] = out["qc_pass"].fillna(False).astype(bool)
    return out[COLUMNS].sort_values("subject_id").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args(argv)
    out = paths.outputs_root(args.run_id) / "weargait"
    walks = pd.read_csv(out / "walk_features.csv")
    cov = pd.read_csv(paths.weargait_root() / "blinded" / "covariates.csv")
    before = guards.snapshot()
    tab = build(walks, cov)
    p = guards.safe_write(out / "subject_external_v2.csv", tab.to_csv(index=False))
    sha = guards.sha256_file(p)
    guards.safe_write(out / "subject_external_v2.csv.sha256", f"{sha}  subject_external_v2.csv\n")
    guards.verify(before)
    ok = tab[tab.qc_pass]
    print(f"[B5] rows={len(tab)} qc_pass={len(ok)} (no insole: {int((~tab.has_insole.astype(bool)).sum())})  "
          f"missing in qc_pass rows: {ok[COLUMNS[2:5] + features.T1_CORE].isna().sum().sum()}")
    print(f"     sha256={sha}")
    print(ok[["age_y", "speed_01_mps"] + features.T1_CORE].describe().loc[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
