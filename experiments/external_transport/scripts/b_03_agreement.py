"""B3 / R6: insole-walkway agreement on blinded WearGait-PD SelfPace trials (label-free).

Owner: Lead (data/*). Run from repo root:
    python -m experiments.external_transport.scripts.b_03_agreement --run-id <id>
Outputs (outputs/<run_id>/agreement/): trials.csv (offsets, match rates, DS, drift), pairs.csv (per stride),
summary.json (bias / limits / pass-fail per D8 tolerance).
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

from experiments.external_transport.etx import agreement as ag
from experiments.external_transport.etx import guards, paths, weargait
from experiments.external_transport.scripts.a2_00_env_and_guard import default_run_id

WW_COLS = ["L Foot Contact", "R Foot Contact", "Walkway_X"]


def one_trial(path, cfg):
    a, sig, wg = cfg["agreement"], cfg["signal"], cfg["weargait"]
    tr = weargait.load_trial(path, wg["never_read_columns"])
    fs, n = tr["fs"], len(tr["t"])
    w = pd.read_csv(path, usecols=WW_COLS, low_memory=False)
    ww = {s: w[f"{s} Foot Contact"].fillna(0).to_numpy() == 1 for s in "LR"}
    onmat = w["Walkway_X"].notna().to_numpy()
    ins = {s: ag.insole_contact(tr["f" + s.lower()], fs, sig, wg) for s in "LR"}
    off = ag.estimate_offset(ins["L"], ins["R"], ww["L"], ww["R"], onmat, fs, a["max_lag_s"])
    strides = dict(zip("LR", weargait.trial_strides(tr, cfg)))

    pairs, row = [], {"code": tr["code"], **off, "duration_s": n / fs}
    lag = int(round(off["offset_s"] * fs))
    for s in "LR":
        wst, ic = ag.walkway_strides(ww[s], fs, a["walkway_max_stride_s"])
        p, cnt = ag.pair_strides(strides[s], wst, ic, off["offset_s"], a["pair_tol_s"])
        row.update({f"{k}_{s}": v for k, v in cnt.items()})
        if len(p):
            o = "R" if s == "L" else "L"
            hs_i = np.round(p["hs_s"].to_numpy() * fs).astype(int)
            ins_end = hs_i + np.round(p["ins_stride_s"].to_numpy() * fs).astype(int)
            ic_i = np.round(p["ic_s"].to_numpy() * fs).astype(int)
            ww_end = ic_i + np.round(p["ww_stride_s"].to_numpy() * fs).astype(int)
            row[f"ds_ins_{s}"] = ag.ds_pct(ins[s], ins[o], hs_i, ins_end)
            row[f"ds_ww_{s}"] = ag.ds_pct(ww[s], ww[o], ic_i, ww_end)
            pairs.append(p.assign(code=tr["code"], foot=s))
    pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
    if len(pairs):
        row["drift_ms_per_s"] = ag.drift_slope(pairs["t_s"], pairs["res_hs_ms"])
        sl = {s: ag.drift_slope(g["t_s"], g["res_hs_ms"]) for s, g in pairs.groupby("foot")}
        row["drift_LR_ms_per_s"] = sl.get("L", np.nan) - sl.get("R", np.nan)
    row["ds_ins"] = np.nanmean([row.get("ds_ins_L", np.nan), row.get("ds_ins_R", np.nan)])
    row["ds_ww"] = np.nanmean([row.get("ds_ww_L", np.nan), row.get("ds_ww_R", np.nan)])
    return row, pairs


def summarise(trials, pairs, a):
    out = {"n_trials": int(len(trials)), "n_paired_strides": int(len(pairs)),
           "ic_match_rate": float((trials.n_ww_ic_matched_L + trials.n_ww_ic_matched_R).sum()
                                  / (trials.n_ww_ic_L + trials.n_ww_ic_R).sum()),
           "offset_s": trials.offset_s.describe().round(3).to_dict(),
           "inter_insole_desync_s": (trials.offset_L_s - trials.offset_R_s).describe().round(3).to_dict(),
           "res_hs_ms": {"mean": float(pairs.res_hs_ms.mean()), "sd": float(pairs.res_hs_ms.std())},
           "res_to_ms": {"mean": float(pairs.res_to_ms.mean()), "sd": float(pairs.res_to_ms.std())}}
    for m in ("stride", "stance", "swing"):
        out[m] = {"stride_level": ag.rm_loa(pairs[f"d_{m}_ms"], pairs.code),
                  "participant_mean": ag.mean_loa(pairs[f"d_{m}_ms"], pairs.code)}
    d_ds = (trials.ds_ins - trials.ds_ww).dropna()
    out["ds_pp"] = {"mean": float(d_ds.mean()), "sd": float(d_ds.std()),
                    "lower": float(d_ds.mean() - 1.96 * d_ds.std()),
                    "upper": float(d_ds.mean() + 1.96 * d_ds.std()), "k": int(len(d_ds))}
    flag = (trials.drift_ms_per_s.abs() * trials.duration_s) > a["tol_drift_ms"]
    flag_lr = (trials.drift_LR_ms_per_s.abs() * trials.duration_s) > a["tol_drift_ms"]
    out["drift"] = {"n_flag": int(flag.sum()), "n_flag_inter_insole": int(flag_lr.sum()),
                    "slope_ms_per_s": trials.drift_ms_per_s.describe().round(3).to_dict()}
    out["checks"] = {
        "stride_bias": bool(abs(out["stride"]["stride_level"]["mean"]) <= a["tol_stride_bias_ms"]),
        "stance_bias": bool(abs(out["stance"]["stride_level"]["mean"]) <= a["tol_stance_bias_ms"]),
        "swing_sd": bool(out["swing"]["stride_level"]["sd_total"] <= a["tol_swing_sd_ms"]),
        "ds_bias": bool(abs(out["ds_pp"]["mean"]) <= a["tol_ds_bias_pp"]),
        "no_drift_flags": bool(flag.sum() == 0 and flag_lr.sum() == 0)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=default_run_id())
    args = ap.parse_args(argv)
    cfg = paths.load_config()
    out = paths.outputs_root(args.run_id) / "agreement"
    files = [f for f in sorted(glob.glob(str(paths.weargait_root() / "blinded" / "WG*_SelfPace.csv")))
             if weargait.has_insole(f)]
    before = guards.snapshot()
    rows, pairs = zip(*(one_trial(f, cfg) for f in files))
    trials, pairs = pd.DataFrame(rows), pd.concat(pairs, ignore_index=True)
    summ = summarise(trials, pairs, cfg["agreement"])
    guards.safe_write(out / "trials.csv", trials.to_csv(index=False))
    guards.safe_write(out / "pairs.csv", pairs.to_csv(index=False))
    guards.safe_write(out / "summary.json", json.dumps(summ, indent=2, default=float))
    guards.verify(before)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
