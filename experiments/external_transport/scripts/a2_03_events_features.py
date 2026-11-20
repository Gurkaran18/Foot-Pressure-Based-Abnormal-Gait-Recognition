"""E3: strides, walk/subject features v2, QC -> features/subject_v2.csv (hand-off file) (T4, T5).

Owner: Lead (data/*). Run from repo root:
    GAITPDB_ROOT=... python -m experiments.external_transport.scripts.a2_03_events_features --run-id <id>
Optional: --demographics <path to teammate's demographics/clean.csv> to add sex/age/speed columns.
"""
import argparse

import pandas as pd

from experiments.external_transport.etx import cache, events, features, guards, manifest, paths, qc, signal
from experiments.external_transport.scripts.a2_00_env_and_guard import default_run_id


def compute(root, rec: pd.DataFrame, cfg: dict):
    stride_rows, walk_rows = [], []
    for r in rec.itertuples(index=False):
        trial = signal.load_trial(root / r.rel_path)
        L, R = events.trial_strides(trial, cfg["signal"])
        for foot, st in (("L", L), ("R", R)):
            stride_rows.append(st.assign(recording_id=r.recording_id, foot=foot,
                                         stride_idx=range(len(st))))
        w = features.walk_features_v2(L, R, cfg["features"])
        flags = qc.qc_flags(w, cfg["features"]["min_strides_per_foot"])
        walk_rows.append({"recording_id": r.recording_id, "subject_id": r.subject_id, "study": r.study,
                          "label": r.label, "walk": r.walk, "include_primary": r.include_primary,
                          "feature_version": "v2", **w, "qc_pass": not flags, "qc_flags": ";".join(flags),
                          "fs_hz": trial["fs"]})
    return pd.concat(stride_rows, ignore_index=True), pd.DataFrame(walk_rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=default_run_id())
    ap.add_argument("--demographics", default=None)
    args = ap.parse_args(argv)
    out = paths.outputs_root(args.run_id)
    cfg = paths.load_config()
    root = paths.gaitpdb_root()

    before = guards.snapshot()
    rec, _ = manifest.build_manifest(root)
    key = cache.cache_key(rec[["recording_id", "sha256"]].itertuples(index=False, name=None), cfg)
    strides, walks = cache.load(key, "strides.csv"), cache.load(key, "walk_v2.csv")
    hit = strides is not None and walks is not None
    if not hit:
        strides, walks = compute(root, rec, cfg)
        cache.store(key, "strides.csv", strides)
        cache.store(key, "walk_v2.csv", walks)
    demo = pd.read_csv(args.demographics) if args.demographics else None
    subj = qc.subject_table(walks, demo)

    guards.safe_write(out / "events" / "strides.csv", strides.to_csv(index=False))
    guards.safe_write(out / "features" / "walk_v2.csv", walks.to_csv(index=False))
    guards.safe_write(out / "features" / "subject_v2.csv", subj.to_csv(index=False))
    guards.safe_write(out / "features" / "cache_key.txt", key + "\n")
    guards.verify(before)

    print(f"[E3] cache {'hit' if hit else 'miss'} key={key[:12]}  walks={len(walks)}  "
          f"valid strides={int(strides.valid.sum())}")
    print(f"     subject_v2: {len(subj)} subjects, qc_pass={int(subj.qc_pass.sum())}, "
          f"by label {subj.groupby('label').size().to_dict()}")
    print(subj[features.WALK_V2_FEATURES].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
