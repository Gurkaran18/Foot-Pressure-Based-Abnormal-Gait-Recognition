"""B2: WearGait-PD SelfPace strides + v2 walk features + QC (label-free).

Owner: Lead (data/*). Run from repo root:
    python -m experiments.external_transport.scripts.b_02_weargait_features --run-id <id>

Outputs (outputs/<run_id>/weargait/, git-ignored; codes only, no labels):
    strides.csv       one row per detected stride (code, foot, stride_idx, pass_id, valid, invalid_reason, ...)
    walk_features.csv one row per code: has_insole, the v2 features, n_valid_L/R, qc_pass, qc_flags
"""
import argparse
import glob

import numpy as np
import pandas as pd

from experiments.external_transport.etx import features, guards, paths, qc, weargait
from experiments.external_transport.scripts.a2_00_env_and_guard import default_run_id


def compute(files, cfg):
    stride_rows, walk_rows = [], []
    task, min_k = cfg["weargait"]["task"], cfg["features"]["min_strides_per_foot"]
    for f in files:
        code = weargait.code_of(f)
        if not weargait.has_insole(f):
            walk_rows.append({"code": code, "task": task, "has_insole": False, "qc_pass": False,
                              "qc_flags": "no_insole"})
            continue
        trial = weargait.load_trial(f, cfg["weargait"]["never_read_columns"])
        L, R = weargait.trial_strides(trial, cfg)
        for foot, st in (("L", L), ("R", R)):
            stride_rows.append(st.assign(code=code, foot=foot, stride_idx=range(len(st))))
        w = features.walk_features_v2(L, R, cfg["features"])
        flags = qc.qc_flags(w, min_k)
        walk_rows.append({"code": code, "task": task, "has_insole": True, "feature_version": "v2", **w,
                          "qc_pass": not flags, "qc_flags": ";".join(flags), "fs_hz": trial["fs"],
                          "n_passes": int(trial["pass_id"].max() + 1),
                          "nan_frac_L": float(np.isnan(trial["fl"]).mean()),
                          "nan_frac_R": float(np.isnan(trial["fr"]).mean())})
    return pd.concat(stride_rows, ignore_index=True), pd.DataFrame(walk_rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=default_run_id())
    args = ap.parse_args(argv)
    cfg = paths.load_config()
    out = paths.outputs_root(args.run_id) / "weargait"
    files = sorted(glob.glob(str(paths.weargait_root() / "blinded" / f"WG*_{cfg['weargait']['task']}.csv")))

    before = guards.snapshot()
    strides, walks = compute(files, cfg)
    guards.safe_write(out / "strides.csv", strides.to_csv(index=False))
    guards.safe_write(out / "walk_features.csv", walks.to_csv(index=False))
    guards.verify(before)

    ins = walks[walks.has_insole]
    print(f"[B2] files={len(walks)} with insole={len(ins)} qc_pass={int(walks.qc_pass.sum())} "
          f"(min strides/foot = {cfg['features']['min_strides_per_foot']})")
    print("     invalid reasons:", strides.invalid_reason.replace("", "valid").value_counts().to_dict())
    print("     passes per trial:", ins.n_passes.value_counts().to_dict(), "| fs:", sorted(ins.fs_hz.unique()))
    m = ins[["n_valid_L", "n_valid_R"]].min(axis=1)
    print(f"     valid strides min(L,R): median {m.median():.0f}  p10 {m.quantile(.1):.0f}  "
          f"min {m.min()}  max {m.max()}")
    print(ins.loc[ins.qc_pass, features.WALK_V2_FEATURES].describe()
          .loc[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
