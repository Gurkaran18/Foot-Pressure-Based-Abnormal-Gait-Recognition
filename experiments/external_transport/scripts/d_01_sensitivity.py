"""D1: post-unblinding sensitivity analyses of the external result (EXPLORATORY, after `model-frozen`).

Owner: Lead. These never replace the frozen confirmatory result (FREEZE_RECORD.md); they test whether the
conclusion "the gaitpdb model does not transport" is robust, or an artefact of one analysis choice.

  S1  signal inside WearGait-PD: repeated 20 x 5-fold subject-level CV of the M1-fb / M1 procedures trained
      on WearGait-PD itself (inner C by log loss); univariate AUCs. Is there any learnable PD signal?
  S2  repeat trials: frozen models on the other two SelfPace trials (SelfPace_mat, SelfPace_matTURN) and on
      features pooled over all three trials (about 3x more strides per person).
  S3  device: stride features from the ProtoKinetics walkway contacts instead of the insoles, frozen M1-fb.
  S4  data choices: >= 15 valid strides per foot; clean trials only (4 passes, < 1 % missing samples).
  S5  case-mix: PD without DBS; PD ON / OFF medication; PD H&Y >= 2.5; no assistive device; age-adjusted
      AAUC (control score ~ age, Janes & Pepe).
Frozen models are refitted with the frozen code (c_01) on the frozen training table and checked against
frozen_manifest.json, so every score below comes from exactly the frozen model.

    python -m experiments.external_transport.scripts.d_01_sensitivity --run-id d_sensitivity
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from experiments.external_transport.etx import agreement, features, guards, metrics, models, paths, weargait
from experiments.external_transport.etx.weargait_covariates import norm_id
from experiments.external_transport.scripts import b_02_weargait_features as b02
from experiments.external_transport.scripts import c_01_frozen_scoring as c01

O = paths.OUTPUTS_DIR
EXT = O / "b02_20261122" / "weargait" / "subject_external_v2.csv"
TRAIN = O / "c_prefreeze" / "features" / "subject_v2.csv"
SCORES = O / "c_frozen" / "frozen_scores"
TASKS = ("SelfPace", "SelfPace_mat", "SelfPace_matTURN")
MODEL_FEATS = None  # filled from cfg


def auc_row(name, y, s, note=""):
    y, s = np.asarray(y, int), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(np.unique(y)) < 2:
        return {"analysis": name, "n": int(len(y)), "note": "one class only"}
    d = metrics.delong(y, s)
    return {"analysis": name, "n": int(len(y)), "n_pd": int(y.sum()), "n_co": int((y == 0).sum()),
            "auc": round(d["auc"], 3), "ci_lower": round(d["ci_lower"], 3), "ci_upper": round(d["ci_upper"], 3),
            "note": note}


def frozen_pipes(cfg):
    train = pd.read_csv(TRAIN)
    man = json.loads((SCORES / "frozen_manifest.json").read_text())
    assert c01.compute_sha256(TRAIN) == man["train_sha256"], "training table changed since the freeze"
    pipes = {}
    for mid, key, mk in (("M1-fb", "t1_fallback", "primary_model"), ("M1", "t1_core", "secondary_model")):
        feats = list(cfg["features"][key])
        pipe, meta = c01.fit_frozen_pipeline(mid, train, feats, cfg, cfg["cv"]["base_seed"])
        frozen = man[mk]["coefficients"]
        assert all(abs(meta["coefficients"][f] - frozen[f]) < 1e-9 for f in feats), f"{mid} differs from frozen"
        pipes[mid] = (pipe, feats)
    return pipes


def score(pipes, df):
    out = {}
    for mid, (pipe, feats) in pipes.items():
        s = np.full(len(df), np.nan)
        ok = df[feats].notna().all(axis=1).to_numpy()
        if ok.any():
            s[ok] = pipe.decision_function(df.loc[ok, feats].to_numpy())
        out[mid] = s
    return out


def within_cv(df, feats, cfg, repeats=20):
    grid = models._lr_C_grid(cfg)
    X, y = df[feats].to_numpy(), df["label"].to_numpy(int)
    aucs, coefs = [], []
    for r in range(repeats):
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + r).split(X, y):
            gs = GridSearchCV(models.make_pipeline("M1", cfg), {"clf__C": grid}, scoring="neg_log_loss",
                              cv=StratifiedKFold(5, shuffle=True, random_state=r)).fit(X[tr], y[tr])
            oof[te] = gs.predict_proba(X[te])[:, 1]
        aucs.append(metrics.auc(y, oof))
    gs = GridSearchCV(models.make_pipeline("M1", cfg), {"clf__C": grid}, scoring="neg_log_loss",
                      cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(X, y)
    coefs = dict(zip(feats, np.round(gs.best_estimator_.named_steps["clf"].coef_[0], 3)))
    return {"auc_mean": round(float(np.mean(aucs)), 3), "auc_sd": round(float(np.std(aucs, ddof=1)), 3),
            "auc_p2_5": round(float(np.percentile(aucs, 2.5)), 3),
            "auc_p97_5": round(float(np.percentile(aucs, 97.5)), 3), "coefficients_full_fit": coefs}


def walkway_tables(path, fs, cfg):
    """Stride tables (one per foot) from the walkway contact columns, same stride range rule."""
    d = pd.read_csv(path, usecols=lambda c: c in ("L Foot Contact", "R Foot Contact"), low_memory=False)
    if d.shape[1] < 2:
        return None
    lo, hi = cfg["signal"]["stride_range_s"]
    out = []
    for foot in ("L", "R"):
        st, _ = agreement.walkway_strides(d[f"{foot} Foot Contact"].fillna(0).to_numpy() > 0, fs,
                                          cfg["agreement"]["walkway_max_stride_s"])
        st = st.rename(columns={"ic_s": "hs_s", "fc_s": "to_s", "next_ic_s": "next_hs_s"})
        st["stride_s"] = st.next_hs_s - st.hs_s
        st["stance_s"] = st.to_s - st.hs_s
        st["swing_s"] = st.next_hs_s - st.to_s
        st["valid"] = st.stride_s.between(lo, hi)
        out.append(st)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="d_sensitivity")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args(argv)
    cfg = paths.load_config()
    root = paths.weargait_root() / "blinded"
    before = guards.snapshot()

    key = pd.read_csv(paths.LABELS_DIR / "weargait_key.csv", usecols=["code", "original_subject", "group", "task"])
    key = key[key.task == "SelfPace"].set_index("code")
    ext = pd.read_csv(EXT)
    ext["label"] = ext.subject_id.map(key.group)
    prim = ext[ext.qc_pass].reset_index(drop=True)
    pipes = frozen_pipes(cfg)
    rows, res = [], {}

    # reference: the frozen primary result, recomputed
    sc = score(pipes, prim)
    for m in pipes:
        rows.append(auc_row(f"REF frozen {m}, SelfPace, 113", prim.label, sc[m]))

    # S1 signal inside WearGait-PD
    res["S1_within_weargait_cv"] = {}
    for m, (_, feats) in pipes.items():
        res["S1_within_weargait_cv"][m] = within_cv(prim, feats, cfg, args.repeats)
    res["S1_univariate_auc_pd_high"] = {f: round(metrics.auc(prim.label, prim[f]), 3)
                                        for f in features.WALK_V2_FEATURES + ["speed_01_mps", "age_y"]}

    # S2 repeat trials + pooled; S3 walkway device
    task_walks, task_strides, ww_rows = {}, {}, []
    for t in TASKS:
        files = sorted(glob.glob(str(root / f"WG*_{t}.csv")))
        st, w = b02.compute(files, cfg)
        task_walks[t], task_strides[t] = w, st.assign(task=t)
        for f in files:
            tabs = walkway_tables(f, 100.0, cfg)
            if tabs is not None:
                ww_rows.append(pd.concat([tabs[0].assign(foot="L"), tabs[1].assign(foot="R")])
                               .assign(code=weargait.code_of(f), task=t))
    for t in TASKS[1:]:
        w = task_walks[t]
        w = w[w.qc_pass].copy()
        w["label"] = w.code.map(key.group)
        s = score(pipes, w)
        for m in pipes:
            rows.append(auc_row(f"S2 frozen {m}, {t}", w.label, s[m]))
    # pooled = mean of per-trial features over the QC-passing trials of each person (pooling raw strides
    # across trials would mix trial conditions into the CV features)
    allw = pd.concat([w[w.qc_pass].assign(task=t) for t, w in task_walks.items()], ignore_index=True)
    feats_all = features.WALK_V2_FEATURES
    pooled = allw.groupby("code")[feats_all].mean().reset_index()
    pooled["n_trials"] = pooled.code.map(allw.groupby("code").size())
    pooled["label"] = pooled.code.map(key.group)
    s = score(pipes, pooled)
    for m in pipes:
        rows.append(auc_row(f"S2 frozen {m}, mean of QC-pass trials", pooled.label, s[m],
                            f"trials/person: {pooled.n_trials.value_counts().sort_index().to_dict()}"))

    ww = pd.concat(ww_rows, ignore_index=True)
    per_trial = []
    for (code, t), g in ww.groupby(["code", "task"]):
        f = features.walk_features_v2(g[g.foot == "L"], g[g.foot == "R"], cfg["features"])
        per_trial.append({"code": code, "task": t, **f})
    pt = pd.DataFrame(per_trial)
    pt = pt[pt[["n_valid_L", "n_valid_R"]].min(axis=1) >= 4]                # walkway: ~2-3 strides per pass
    for label, fw in (*((t, pt[pt.task == t].copy()) for t in TASKS),
                      ("mean of 3 trials", pt.groupby("code")[["stride_time_mean", "stride_time_cv"]].mean()
                       .reset_index())):
        fw["label"] = fw.code.map(key.group)
        s = score({"M1-fb": pipes["M1-fb"]}, fw)
        rows.append(auc_row(f"S3 walkway strides, frozen M1-fb, {label}", fw.label, s["M1-fb"],
                            "all trials with walkway contacts incl. no-insole; >= 4 strides/foot per trial"))
        rows.append(auc_row(f"S3 walkway stride_time_cv univariate, {label}", fw.label, fw.stride_time_cv))

    # S4 data choices (primary SelfPace scores)
    wf = pd.read_csv(O / "b02_20261122" / "weargait" / "walk_features.csv").set_index("code")
    p = prim.copy()
    p["n_min"] = p[["n_valid_L", "n_valid_R"]].min(axis=1)
    nanf = p.subject_id.map(wf[["nan_frac_L", "nan_frac_R"]].max(axis=1))   # whole trial, incl. non-walk time
    p["clean"] = (p.subject_id.map(wf.n_passes) == 4) & (nanf <= nanf.median())
    for name, mask in (("S4 >= 15 valid strides/foot", p.n_min >= 15), ("S4 4 passes + missing data <= median", p.clean)):
        for m in pipes:
            rows.append(auc_row(f"{name}, frozen {m}", p.label[mask], sc[m][mask.to_numpy()]))

    # S5 case-mix
    pd_demo = pd.read_csv(paths.weargait_root() / "_restricted" / "PD - Demographic+Clinical - datasetV1.csv", header=1)
    co_demo = pd.read_csv(paths.weargait_root() / "_restricted" / "CONTROLS - Demographic+Clinical - datasetV1.csv",
                          header=1)
    pd_demo["sid"] = pd_demo["Subject ID"].map(norm_id)
    co_demo["sid"] = co_demo["Subject ID"].map(norm_id)
    clin = pd_demo.drop_duplicates("sid").set_index("sid")
    p["sid"] = p.subject_id.map(key.original_subject)
    st = lambda c: p.sid.map(clin[c].astype(str).str.strip().str.lower())        # noqa: E731
    dbs, med, hy, dev = st("DBS?"), st("3b"), pd.to_numeric(p.sid.map(clin["Modified Hoehn & Yahr Score"]),
                                                           errors="coerce"), st("Assistive Device used during testing?")
    co_dev = p.sid.map(co_demo.drop_duplicates("sid").set_index("sid")
                       .get("Assistive Device used during testing?", pd.Series(dtype=str)).astype(str).str.strip().str.lower())
    is_co = p.label == 0
    subsets = {"S5 PD without DBS": is_co | (dbs == "no"),
               "S5 PD medication ON": is_co | med.isin(["on"]),
               "S5 PD medication OFF": is_co | med.isin(["off", "med off/dbs on", "wearing-off"]),
               "S5 PD H&Y >= 2.5": is_co | (hy >= 2.5),
               "S5 no assistive device": (is_co & ~co_dev.fillna("no").str.startswith(("yes", "cane"))) |
                                         (~is_co & (dev == "no"))}
    for name, mask in subsets.items():
        for m in pipes:
            rows.append(auc_row(f"{name}, frozen {m}", p.label[mask], sc[m][mask.to_numpy()]))
    # 1:1 nearest age matching without replacement (caliper 3 years), PD -> controls
    rng = np.random.default_rng(cfg["cv"]["base_seed"])
    co_pool = p[is_co][["age_y"]].copy()
    keep = []
    for i in rng.permutation(p.index[~is_co]):
        if co_pool.empty:
            break
        dist = (co_pool.age_y - p.at[i, "age_y"]).abs()
        j = dist.idxmin()
        if dist[j] <= 3:
            keep += [i, j]
            co_pool = co_pool.drop(j)
    am = p.index.isin(keep)
    for m in pipes:
        rows.append(auc_row(f"S5 age-matched 1:1 (caliper 3 y), frozen {m}", p.label[am], sc[m][am],
                            f"median age PD {p.age_y[am & ~is_co].median():.0f} / Co {p.age_y[am & is_co].median():.0f}"))
    from scipy.stats import spearmanr
    res["S5_score_vs_age_in_controls_spearman"] = {
        m: round(float(spearmanr(p.age_y[is_co], sc[m][is_co.to_numpy()]).correlation), 3) for m in pipes}
    res["S5_age_by_group"] = {"weargait_median_co": float(p.age_y[is_co].median()),
                              "weargait_median_pd": float(p.age_y[~is_co].median())}
    res["S5_age_adjusted_aauc"] = {}
    for m in pipes:
        a = metrics.aauc(p.label.to_numpy(), sc[m], p.age_y.to_numpy(float), B=2000, seed=cfg["cv"]["base_seed"])
        res["S5_age_adjusted_aauc"][m] = {k: round(a[k], 3) for k in ("aauc", "ci_lower", "ci_upper")}
    res["S5_casemix_pd"] = {"n_pd": int((~is_co).sum()), "dbs_yes": int((dbs == "yes").sum()),
                            "med_on": int(med.isin(["on"]).sum()), "hy_median": float(hy.median()),
                            "hy_ge_2_5": int((hy >= 2.5).sum())}

    table = pd.DataFrame(rows)
    out = paths.outputs_root(args.run_id) / "sensitivity"
    guards.safe_write(out / "auc_table.csv", table.to_csv(index=False))
    guards.safe_write(out / "results.json", json.dumps(res, indent=2, default=float))
    guards.verify(before)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
