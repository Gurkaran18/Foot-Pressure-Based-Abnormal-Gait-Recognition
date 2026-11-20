"""E4: v1 reproduces the legacy all-walk subject table (equivalence gate, spec §12.2).

Owner: Lead (data/*). Run from repo root:
    GAITPDB_ROOT=... python -m experiments.external_transport.scripts.a2_04_legacy_equivalence --run-id <id>

1. Stride events: new code (debounce_impl='legacy') vs gait_common.detect_strides on ALL 306 files,
   both feet -> identical sample indices and validity.
2. Features: v1 per file, averaged per subject over ALL recordings (legacy behaviour, issue I1) vs
   results_new/gait_features_subject.csv (read-only) -> all 13 features within 1e-9.
"""
import argparse
import json

import numpy as np
import pandas as pd

from experiments.external_transport.etx import events, features, guards, manifest, paths, signal
from experiments.external_transport.scripts.a2_00_env_and_guard import default_run_id

LEGACY_CSV = paths.REPO_ROOT / "results_new" / "gait_features_subject.csv"
TOL = 1e-9


def run_equivalence(root, rec: pd.DataFrame, cfg: dict, check_events_with_legacy=True):
    import gait_common as gc                                   # legacy reference (read-only use)
    sig = {**cfg["signal"], "debounce_impl": "legacy"}
    rows, event_mismatch, n_strides = [], [], 0
    for r in rec.itertuples(index=False):
        tr = signal.load_trial(root / r.rel_path)
        L, R = events.trial_strides(tr, sig)
        if check_events_with_legacy:
            for foot, new, force in (("L", L, tr["fl"]), ("R", R, tr["fr"])):
                old = gc.detect_strides(force, 100)
                n_strides += len(new)
                same = (new[["hs_idx", "to_idx", "next_hs_idx"]].to_numpy().tolist()
                        == old[["hs", "to", "next_hs"]].to_numpy().tolist()
                        and new["valid"].tolist() == old["valid"].tolist())
                if not same:
                    event_mismatch.append(f"{r.recording_id}:{foot}")
        rows.append({"subject_id": r.subject_id,
                     **features.walk_features_v1(L, R, tr["fl"], tr["fr"], tr["fs"], cfg["signal"])})
    v1 = pd.DataFrame(rows).groupby("subject_id", as_index=False).mean(numeric_only=True)
    return v1, event_mismatch, n_strides


def compare(v1: pd.DataFrame, legacy: pd.DataFrame) -> dict:
    m = legacy.merge(v1, on="subject_id", suffixes=("_old", "_new"), validate="one_to_one")
    per_feature = {}
    for f in features.V1_FEATURES + ["n_strides"]:
        d = np.abs(m[f + "_old"].to_numpy(float) - m[f + "_new"].to_numpy(float))
        per_feature[f] = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
    nan_mismatch = int(sum((m[f + "_old"].isna() != m[f + "_new"].isna()).sum() for f in features.V1_FEATURES))
    return {"n_subjects_legacy": len(legacy), "n_subjects_matched": len(m), "max_abs_diff": per_feature,
            "nan_pattern_mismatches": nan_mismatch,
            "pass": len(m) == len(legacy) and nan_mismatch == 0
                    and all(v <= TOL for v in per_feature.values())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=default_run_id())
    args = ap.parse_args(argv)
    out = paths.outputs_root(args.run_id) / "equivalence"
    cfg = paths.load_config()
    root = paths.gaitpdb_root()

    before = guards.snapshot()
    rec, _ = manifest.build_manifest(root)
    v1, mism, n = run_equivalence(root, rec, cfg)
    res = compare(v1, pd.read_csv(LEGACY_CSV))
    res.update({"files": len(rec), "feet_checked": 2 * len(rec), "strides_checked": n,
                "event_mismatches": mism, "events_pass": not mism})
    res["gate_pass"] = res["pass"] and res["events_pass"]
    guards.safe_write(out / "v1_subject_table.csv", v1.to_csv(index=False))
    guards.safe_write(out / "equivalence_report.json", json.dumps(res, indent=1))
    guards.verify(before)

    print(f"[E4] events: {res['files']} files, {n} strides, mismatches={len(mism)}")
    print(f"     features: {res['n_subjects_matched']}/{res['n_subjects_legacy']} subjects, "
          f"worst |diff| = {max(res['max_abs_diff'].values()):.2e}, NaN mismatches={res['nan_pattern_mismatches']}")
    print(f"     EQUIVALENCE GATE: {'PASS' if res['gate_pass'] else 'FAIL'}")
    if not res["gate_pass"]:
        print("     per-feature max diff:", {k: f"{v:.2e}" for k, v in res["max_abs_diff"].items()})


if __name__ == "__main__":
    main()
