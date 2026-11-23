"""C2: Confirmatory hypothesis tests H1-H3 (Holm, M1-fb primary) + separate recalibration.

Post-freeze only. Run ONLY by the Lead after the `model-frozen` tag is pushed (CLAUDE.md §3):
    python -m experiments.external_transport.scripts.c_02_hypothesis_testing \
        --scores-csv outputs/<id>/frozen_scores/external_scores.csv \
        --key-csv labels/weargait_key.csv \
        --external-table outputs/<id>/weargait/subject_external_v2.csv \
        --emulation-summary outputs/<id>/emulation/summary.json \
        --run-id <id>

Methods (A1.1 R1, R2.2, R5, R10; decision register D6, D13, D14, B-D8):
  H1  frozen-model external AUC > 0.5: one-sided DeLong z-test; stratified bootstrap p as a check.
  H2  speed-adjusted AAUC (Janes & Pepe) of the frozen LINEAR PREDICTOR, adjusted for WearGait-PD walkway
      speed, > 0.5: one-sided stratified bootstrap p (B = 2,000). Precondition: the PD and control speed
      ranges overlap; otherwise H2 is "not testable" (kept in the Holm family with p = 1, conservative).
      An H2 rejection is interpreted only if H1 is also rejected.
  H3  gap = internal emulated AUC - external AUC > 0: one-sided bootstrap p, gap distribution formed by
      independently combining the internal bootstrap-of-CV draws (E9, h3_internal_bootstrap.csv) with the
      external stratified bootstrap draws.
  Holm-Bonferroni over H1-H3 for M1-fb (family alpha 0.05). M1 = same procedures, secondary, no Holm.
  Recalibration (intercept, slope, recalibrated Brier) is reported separately and never replaces the
  primary result. The B-D1 group split of the non-scored (no-insole / QC-fail) codes is reported.
Fails loudly on any missing input; there are no built-in fallback numbers.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import figures, guards, metrics, paths  # noqa: E402

FREEZE_TAG = "model-frozen"
MODELS = (("M1-fb", "m1_fb_score", "m1_fb_logit", "h3_internal_reference_m1_fb"),
          ("M1", "m1_score", "m1_logit", "h3_internal_reference_m1"))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def require_freeze_tag():
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "-q", "--verify", f"refs/tags/{FREEZE_TAG}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git tag '{FREEZE_TAG}' not found: C2 reads labels and may only run after the freeze.")


def load_labels(key_csv, task) -> pd.DataFrame:
    """One row per code from the blinding key: code -> label (group 0 = control, 1 = PD)."""
    key = pd.read_csv(key_csv, usecols=["code", "group", "task"])
    key = key[key["task"] == task]
    if key["code"].duplicated().any():
        raise ValueError(f"Key has more than one {task} row for some codes.")
    if not set(key["group"].unique()) <= {0, 1}:
        raise ValueError("Key group values must be 0/1.")
    return key.rename(columns={"code": "subject_id", "group": "label"})[["subject_id", "label"]]


def holm_bonferroni(p_dict, alpha=0.05):
    """Holm step-down over {name: p}. Returns per-hypothesis p, rank, threshold, rejected."""
    items = sorted(p_dict.items(), key=lambda x: x[1])
    m, out, stopped = len(items), {}, False
    for rank, (name, p) in enumerate(items):
        thr = alpha / (m - rank)
        rej = (not stopped) and p <= thr
        stopped = stopped or not rej
        out[name] = {"p_value": float(p), "rank": rank + 1, "threshold": float(thr), "rejected": bool(rej)}
    return out


def stratified_boot_auc(y, s, B, seed):
    """External bootstrap AUC draws, resampling within label (cases and controls kept)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = np.empty(B)
    for b in range(B):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        out[b] = metrics.auc(y[idx], s[idx])
    return out


def speed_overlap(y, speed):
    lo = max(speed[y == 1].min(), speed[y == 0].min())
    hi = min(speed[y == 1].max(), speed[y == 0].max())
    inside = (speed >= lo) & (speed <= hi)
    return {"pd_range": [float(speed[y == 1].min()), float(speed[y == 1].max())],
            "co_range": [float(speed[y == 0].min()), float(speed[y == 0].max())],
            "overlap": [float(lo), float(hi)], "testable": bool(lo < hi),
            "n_pd_in_overlap": int((inside & (y == 1)).sum()), "n_co_in_overlap": int((inside & (y == 0)).sum())}


def recalibration(y, prob, logit):
    cal = metrics.calibration(y, prob)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000).fit(logit.reshape(-1, 1), y)
    p_rec = lr.predict_proba(logit.reshape(-1, 1))[:, 1]
    return {"calibration_intercept": cal["intercept"], "calibration_slope": cal["slope"],
            "brier_frozen": metrics.brier(y, prob), "brier_recalibrated_in_sample": metrics.brier(y, p_rec),
            "note": "secondary; in-sample logistic recalibration; never replaces the primary result"}


def evaluate_model(df, prob_col, logit_col, ref, int_boot, B, seed):
    y = df["label"].to_numpy(int)
    lp = df[logit_col].to_numpy(float)
    prob = df[prob_col].to_numpy(float)

    # H1: one-sided DeLong z-test of AUC = 0.5 (+ stratified bootstrap check)
    dl = metrics.delong(y, lp)
    z = (dl["auc"] - 0.5) / dl["se"]
    ext_boot = stratified_boot_auc(y, lp, B, seed)
    h1 = {"auc": dl["auc"], "se_delong": dl["se"], "ci_lower": dl["ci_lower"], "ci_upper": dl["ci_upper"],
          "z": float(z), "p_value": float(stats.norm.sf(z)),
          "bootstrap_check_p": float(np.mean(ext_boot <= 0.5)),
          "bootstrap_ci": [float(np.percentile(ext_boot, 2.5)), float(np.percentile(ext_boot, 97.5))]}

    # H2: AAUC of the linear predictor adjusted for walkway speed
    speed = df["speed_01_mps"].to_numpy(float)
    if np.isnan(speed).any():
        raise ValueError("speed_01_mps missing for scored subjects.")
    ov = speed_overlap(y, speed)
    h2 = {"speed_overlap": ov}
    if ov["testable"]:
        a = metrics.aauc(y, lp, speed, strata=None, B=B, seed=seed, return_draws=True)
        h2.update({"aauc": a["aauc"], "ci_lower": a["ci_lower"], "ci_upper": a["ci_upper"],
                   "B_valid": int(len(a["draws"])), "p_value": float(np.mean(a["draws"] <= 0.5))})
    else:
        h2.update({"status": "not testable (speed ranges do not overlap)", "p_value": 1.0})

    # H3: gap = internal emulated AUC - external AUC, independent bootstrap combination
    rng = np.random.default_rng(seed + 1)
    ib = rng.choice(int_boot, size=len(ext_boot), replace=True)
    gap_b = ib - ext_boot
    gap = ref["auc_mean"] - dl["auc"]
    h3 = {"internal_emulated_auc": ref["auc_mean"], "internal_regime": ref["regime"], "external_auc": dl["auc"],
          "gap": float(gap), "ci_lower": float(np.percentile(gap_b, 2.5)),
          "ci_upper": float(np.percentile(gap_b, 97.5)), "p_value": float(np.mean(gap_b <= 0)),
          "B_internal": int(len(int_boot)), "B_external": int(len(ext_boot))}

    return {"n": int(len(y)), "n_pd": int(y.sum()), "n_co": int((y == 0).sum()),
            "H1": h1, "H2": h2, "H3": h3, "recalibration": recalibration(y, prob, lp)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="C2: post-freeze H1-H3 and recalibration")
    ap.add_argument("--scores-csv", required=True)
    ap.add_argument("--manifest", default=None, help="frozen_manifest.json (default: next to --scores-csv)")
    ap.add_argument("--key-csv", required=True, help="labels/weargait_key.csv")
    ap.add_argument("--external-table", required=True)
    ap.add_argument("--emulation-summary", required=True)
    ap.add_argument("--emulation-boot", default=None, help="h3_internal_bootstrap.csv (default: next to summary)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--skip-tag-check", action="store_true", help="tests only")
    args = ap.parse_args(argv)

    if not args.skip_tag_check:
        require_freeze_tag()
    cfg = paths.load_config()
    seed = cfg["cv"]["base_seed"]

    scores = pd.read_csv(args.scores_csv)
    man_p = Path(args.manifest) if args.manifest else Path(args.scores_csv).parent / "frozen_manifest.json"
    manifest = json.loads(man_p.read_text())
    if sha256(args.external_table) != manifest["external_sha256"]:
        raise RuntimeError("External table differs from the one scored at the freeze (SHA-256 mismatch).")
    if scores["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject_id in scores.")

    labels = load_labels(args.key_csv, cfg["weargait"]["task"])
    ext = pd.read_csv(args.external_table)
    df = scores.merge(labels, on="subject_id", how="left", validate="one_to_one")
    if df["label"].isna().any():
        raise ValueError(f"{int(df['label'].isna().sum())} scored subjects have no label in the key.")
    df = df.merge(ext[["subject_id", "speed_01_mps"]], on="subject_id", how="left", validate="one_to_one")

    not_scored = ext.loc[~ext["subject_id"].isin(scores["subject_id"]), ["subject_id", "qc_flags"]]
    not_scored = not_scored.merge(labels, on="subject_id", how="left")
    bd1 = {"n_not_scored": int(len(not_scored)), "n_pd": int((not_scored["label"] == 1).sum()),
           "n_co": int((not_scored["label"] == 0).sum()),
           "flags": not_scored["qc_flags"].fillna("").value_counts().to_dict()}

    em = json.loads(Path(args.emulation_summary).read_text())
    boot_p = Path(args.emulation_boot) if args.emulation_boot else \
        Path(args.emulation_summary).parent / "h3_internal_bootstrap.csv"
    boot = pd.read_csv(boot_p)

    results = {"freeze_tag": FREEZE_TAG, "external_sha256": manifest["external_sha256"],
               "train_sha256": manifest.get("train_sha256"), "B": args.B, "seed": seed,
               "bd1_not_scored_group_split": bd1}
    for model_id, prob_col, logit_col, ref_key in MODELS:
        if ref_key not in em:
            raise KeyError(f"{ref_key} missing from emulation summary (rerun a2_09).")
        ib = boot.loc[boot["model_id"] == model_id, "auc"].to_numpy(float)
        if len(ib) == 0:
            raise ValueError(f"No internal bootstrap draws for {model_id}.")
        results[model_id] = evaluate_model(df, prob_col, logit_col, em[ref_key], ib, args.B, seed)

    fb = results["M1-fb"]
    holm = holm_bonferroni({h: fb[h]["p_value"] for h in ("H1", "H2", "H3")})
    holm["H2"]["interpretable"] = bool(holm["H1"]["rejected"])      # D14 logical restriction
    fb["holm"] = holm
    results["M1"]["note"] = "secondary model (B-D8): same procedures, unadjusted p, outside the Holm family"

    out_dir = paths.outputs_root(args.run_id) / "hypothesis_testing"
    before = guards.snapshot()
    guards.safe_write(out_dir / "results.json", json.dumps(results, indent=2, default=float))
    y = df["label"].to_numpy(int)
    preds = {"M1-fb (primary)": (y, df["m1_fb_score"].to_numpy()), "M1 (secondary)": (y, df["m1_score"].to_numpy())}
    figures.plot_roc_curves(preds, out_dir=out_dir)
    figures.plot_calibration(preds, out_dir=out_dir)
    guards.verify(before)

    print(f"[C2] n = {fb['n']} (PD {fb['n_pd']}, Co {fb['n_co']}); not scored: {bd1}")
    for h in ("H1", "H2", "H3"):
        r = holm[h]
        print(f"  M1-fb {h}: p = {r['p_value']:.4f}  Holm threshold {r['threshold']:.4f}  "
              f"{'REJECT H0' if r['rejected'] else 'do not reject'}")
    print(f"  results -> {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
