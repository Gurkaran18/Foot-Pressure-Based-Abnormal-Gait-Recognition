"""E5: unweighted vs balanced LR — R8 rule (spec §9 step 5, R8).

Owner: Teammate (model/*).
Run from repo root:  python -m experiments.external_transport.scripts.a2_05_weighting_comparison --run-id <id>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import paths, splits, models, metrics, uncertainty
from experiments.external_transport.etx.guards import snapshot, verify, safe_write


def run_repeated_cv_weighting(subjects_df, weighting: str, cfg: dict, feature_cols: list):
    """Run repeated CV with inner C selection for a given LR weighting ('none' or 'balanced')."""
    cv_cfg = cfg["cv"]
    seed_base = cv_cfg["base_seed"]
    repeats = cv_cfg["repeats"]
    k_outer = cv_cfg["outer_folds"]
    k_inner = cv_cfg["inner_folds"]
    C_grid = models._lr_C_grid(cfg)

    cfg_w = {**cfg, "models": {**cfg["models"], "lr_weighting": weighting}}

    preds_all = []
    for repeat, fold, train_ids, test_ids in splits.outer_folds(
        subjects_df,
        repeats=repeats,
        k=k_outer,
        base_seed=seed_base,
        strata=cv_cfg.get("strata"),
    ):
        train_df = subjects_df[subjects_df["subject_id"].isin(train_ids)].reset_index(drop=True)
        test_df = subjects_df[subjects_df["subject_id"].isin(test_ids)].reset_index(drop=True)

        inner_seed = splits._seed_for_inner(seed_base, repeat, fold)
        inner_gen = list(
            splits.inner_folds(train_df, k=k_inner, seed=inner_seed, strata=cv_cfg.get("strata"))
        )

        best_C, best_ll = None, float("inf")
        for C in C_grid:
            lls = []
            for it_ids, iv_ids in inner_gen:
                it_df = train_df[train_df["subject_id"].isin(it_ids)]
                iv_df = train_df[train_df["subject_id"].isin(iv_ids)]
                p_tmp = models.make_pipeline("M1", cfg_w, seed=inner_seed)
                models.set_C(p_tmp, C)
                p_tmp.fit(it_df[feature_cols].values, it_df["label"].values)
                prob_iv = np.clip(p_tmp.predict_proba(iv_df[feature_cols].values)[:, 1], 1e-9, 1.0 - 1e-9)
                ll = -np.mean(
                    iv_df["label"].values * np.log(prob_iv)
                    + (1.0 - iv_df["label"].values) * np.log(1.0 - prob_iv)
                )
                lls.append(ll)
            mean_ll = float(np.mean(lls))
            if mean_ll < best_ll:
                best_ll, best_C = mean_ll, C

        pipe = models.make_pipeline("M1", cfg_w, seed=inner_seed)
        models.set_C(pipe, best_C)
        pipe.fit(train_df[feature_cols].values, train_df["label"].values)
        probs = pipe.predict_proba(test_df[feature_cols].values)[:, 1]

        for sid, y_i, p_i in zip(test_df["subject_id"], test_df["label"], probs):
            preds_all.append({"repeat": repeat, "fold": fold, "subject_id": sid, "y": int(y_i), "p": float(p_i)})

    preds_df = pd.DataFrame(preds_all)

    # Compute metrics per repeat
    repeat_aucs, repeat_briers, repeat_slopes, repeat_intercepts = [], [], [], []
    for r in range(repeats):
        r_df = preds_df[preds_df["repeat"] == r]
        y_r = r_df["y"].values
        p_r = r_df["p"].values
        repeat_aucs.append(metrics.auc(y_r, p_r))
        repeat_briers.append(metrics.brier(y_r, p_r))
        cal = metrics.calibration(y_r, p_r)
        repeat_slopes.append(cal["slope"])
        repeat_intercepts.append(cal["intercept"])

    return {
        "auc": uncertainty.summarize_repeats(repeat_aucs),
        "brier": uncertainty.summarize_repeats(repeat_briers),
        "calibration_slope": uncertainty.summarize_repeats(repeat_slopes),
        "calibration_intercept": uncertainty.summarize_repeats(repeat_intercepts),
    }


def main():
    parser = argparse.ArgumentParser(description="E5: LR weighting comparison (R8)")
    parser.add_argument("--run-id", default="day1")
    parser.add_argument("--subject-csv", default=None,
                        help="Path to subject_v2.csv")
    args = parser.parse_args()

    cfg = paths.load_config()
    out_dir = paths.outputs_root(args.run_id) / "weighting"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = snapshot()

    # Discover subject CSV from run-id first
    if args.subject_csv:
        csv_path = Path(args.subject_csv)
    else:
        run_csv = paths.outputs_root(args.run_id) / "features" / "subject_v2.csv"
        if run_csv.exists():
            csv_path = run_csv
        else:
            candidates = sorted(paths.OUTPUTS_DIR.glob("*/features/subject_v2.csv"))
            if not candidates:
                print("ERROR: subject_v2.csv not found. Run a2_03 first.", file=sys.stderr)
                sys.exit(1)
            csv_path = candidates[-1]

    print(f"Using subject CSV: {csv_path}")
    subjects_df = pd.read_csv(csv_path)
    if "qc_pass" in subjects_df.columns:
        subjects_df = subjects_df[subjects_df["qc_pass"]].reset_index(drop=True)

    feature_cols = cfg["features"]["t1_core"]

    results = {}
    print("\nRunning repeated CV with inner C selection for unweighted ('none') and 'balanced'...")
    for weighting in ["none", "balanced"]:
        print(f"  Evaluating weighting = {weighting} (R={cfg['cv']['repeats']} repeats)...")
        w_res = run_repeated_cv_weighting(subjects_df, weighting, cfg, feature_cols)
        results[weighting] = w_res
        print(f"    AUC:   {w_res['auc']['mean']:.4f} ± {w_res['auc']['std']:.4f}")
        print(f"    Brier: {w_res['brier']['mean']:.4f} ± {w_res['brier']['std']:.4f}")
        print(f"    Slope: {w_res['calibration_slope']['mean']:.4f} ± {w_res['calibration_slope']['std']:.4f}")

    # R8 rule:
    # Choose 'none' unless 'balanced' has LOWER Brier AND calibration slope at least as close to 1
    brier_none = results["none"]["brier"]["mean"]
    brier_bal = results["balanced"]["brier"]["mean"]
    slope_none = results["none"]["calibration_slope"]["mean"]
    slope_bal = results["balanced"]["calibration_slope"]["mean"]

    lower_brier = brier_bal < (brier_none - 1e-6)
    better_slope = abs(slope_bal - 1.0) <= (abs(slope_none - 1.0) + 1e-6)

    if lower_brier and better_slope:
        chosen = "balanced"
        threshold = 0.5
        reason = "balanced has lower Brier score AND slope at least as close to 1"
    else:
        chosen = "none"
        threshold = 0.564  # training prevalence 93/165
        reasons = []
        if not lower_brier:
            reasons.append(f"balanced Brier ({brier_bal:.4f}) >= none Brier ({brier_none:.4f})")
        if not better_slope:
            reasons.append(f"balanced slope dev ({abs(slope_bal - 1.0):.4f}) > none slope dev ({abs(slope_none - 1.0):.4f})")
        reason = "unweighted default chosen: " + "; ".join(reasons)

    print(f"\nR8 Rule Decision: {chosen} (threshold: {threshold})")
    print(f"Reason: {reason}")

    results["decision"] = {
        "chosen_weighting": chosen,
        "chosen_threshold": threshold,
        "reason": reason,
        "r8_rule": "choose 'none' unless 'balanced' has LOWER Brier AND calibration slope at least as close to 1",
    }

    out_path = out_dir / "weighting_results.json"
    safe_write(out_path, json.dumps(results, indent=2, default=str))
    print(f"Results written to {out_path}")

    after = snapshot()
    verify(before)
    print("E5 complete. Protected-path hashes unchanged ✓")


if __name__ == "__main__":
    main()
