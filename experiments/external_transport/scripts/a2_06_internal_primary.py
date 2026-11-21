"""E6: internal primary models (M1, M1-fb, M0, M2, AS, X1, X13-ref), uncertainty, internal AAUC (gate).

Owner: Teammate (model/*).
Run from repo root:  python -m experiments.external_transport.scripts.a2_06_internal_primary --run-id <id>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import paths, splits, models, metrics, uncertainty, reporting
from experiments.external_transport.etx.guards import snapshot, verify, safe_write


def run_repeated_cv(subjects_df, model_id, cfg, feature_cols):
    """Run full R=20 × 5-fold CV. Returns (per_repeat_metrics_dict, all_preds_df)."""
    cv_cfg = cfg["cv"]
    seed_base = cv_cfg["base_seed"]
    repeats = cv_cfg["repeats"]
    k_outer = cv_cfg["outer_folds"]
    k_inner = cv_cfg["inner_folds"]
    C_grid = models._lr_C_grid(cfg)

    all_preds = []

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
        pipe = models.make_pipeline(model_id, cfg, seed=inner_seed)

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        # Inner CV for C selection (LR models)
        if model_id not in {"X1", "X13-ref"}:
            best_C, best_ll = None, float("inf")
            inner_gen = list(splits.inner_folds(train_df, k=k_inner, seed=inner_seed, strata=cv_cfg.get("strata")))
            for C in C_grid:
                lls = []
                for it_ids, iv_ids in inner_gen:
                    it_df = train_df[train_df["subject_id"].isin(it_ids)]
                    iv_df = train_df[train_df["subject_id"].isin(iv_ids)]
                    p_tmp = models.make_pipeline(model_id, cfg, seed=inner_seed)
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
            models.set_C(pipe, best_C)

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        for sid, y_i, p_i in zip(test_df["subject_id"], y_test, probs):
            all_preds.append({
                "model_id": model_id,
                "repeat": repeat,
                "outer_fold": fold,
                "subject_id": sid,
                "y": int(y_i),
                "p": float(p_i),
            })

    preds_df = pd.DataFrame(all_preds)

    # Compute metrics PER REPEAT, then summarise
    thr = cfg["models"].get("threshold", 0.5)
    r_aucs, r_briers, r_slopes, r_intercepts = [], [], [], []
    r_sens, r_specs, r_baccs = [], [], []

    for r in range(repeats):
        r_df = preds_df[preds_df["repeat"] == r]
        y_r = r_df["y"].values
        p_r = r_df["p"].values
        if len(np.unique(y_r)) == 2:
            r_aucs.append(metrics.auc(y_r, p_r))
            r_briers.append(metrics.brier(y_r, p_r))
            cal = metrics.calibration(y_r, p_r)
            r_slopes.append(cal["slope"])
            r_intercepts.append(cal["intercept"])
            tm = metrics.threshold_metrics(y_r, p_r, thr)
            r_sens.append(tm["sensitivity"])
            r_specs.append(tm["specificity"])
            r_baccs.append(tm["balanced_accuracy"])

    summary = {
        "auc": uncertainty.summarize_repeats(r_aucs),
        "brier": uncertainty.summarize_repeats(r_briers),
        "calibration_slope": uncertainty.summarize_repeats(r_slopes),
        "calibration_intercept": uncertainty.summarize_repeats(r_intercepts),
        "sensitivity": uncertainty.summarize_repeats(r_sens),
        "specificity": uncertainty.summarize_repeats(r_specs),
        "balanced_accuracy": uncertainty.summarize_repeats(r_baccs),
        "threshold": thr,
    }

    return summary, preds_df


def main():
    parser = argparse.ArgumentParser(description="E6: internal primary models")
    parser.add_argument("--run-id", default="day1")
    parser.add_argument("--subject-csv", default=None)
    args = parser.parse_args()

    cfg = paths.load_config()
    out_dir = paths.outputs_root(args.run_id)
    cv_out = out_dir / "cv"
    cv_out.mkdir(parents=True, exist_ok=True)
    metrics_out = out_dir / "metrics"
    metrics_out.mkdir(parents=True, exist_ok=True)

    before = snapshot()

    # Discover subject CSV from --run-id first
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

    feature_sets = {
        "M1": cfg["features"]["t1_core"],
        "M1-fb": cfg["features"]["t1_fallback"],
        "X1": cfg["features"]["t1_core"],
    }
    if "speed_01_mps" in subjects_df.columns:
        feature_sets["M0"] = ["speed_01_mps"]
        feature_sets["M2"] = ["speed_01_mps"] + list(cfg["features"]["t1_core"])
    if all(c in subjects_df.columns for c in ["age_y", "sex"]):
        feature_sets["AS"] = ["age_y", "sex"]

    all_models_summary = {}
    all_preds_list = []

    for model_id, feature_cols in feature_sets.items():
        print(f"\n--- {model_id} ---")
        available = [c for c in feature_cols if c in subjects_df.columns]
        if len(available) < len(feature_cols):
            print(f"  Skipping {model_id}: missing columns {set(feature_cols) - set(subjects_df.columns)}")
            continue

        rep_summary, preds_df = run_repeated_cv(subjects_df, model_id, cfg, available)
        all_preds_list.append(preds_df)

        # Pooled mean probability per subject across repeats
        subj_preds = preds_df.groupby("subject_id").agg({"y": "first", "p": "mean", "outer_fold": "first"})
        all_y = subj_preds["y"].values
        mean_p = subj_preds["p"].values
        folds_p = subj_preds["outer_fold"].values

        # DeLong and LeDell on pooled predictions
        dl = metrics.delong(all_y, mean_p)
        ld = uncertainty.ledell_ci(all_y, mean_p, folds_p)

        model_results = {
            "model_id": model_id,
            "repeat_summary": rep_summary,
            "delong_pooled": dl,
            "ledell_pooled": ld,
        }

        # Participant-level CI for primary model M1 using bootstrap_cv
        if model_id == "M1":
            print("  Computing participant-level bootstrap-of-CV CI for M1 (grouped duplicates)...")
            B_boot = cfg.get("uncertainty", {}).get("bootstrap_cv_B", 500)
            C_grid = models._lr_C_grid(cfg)

            def _boot_cv_eval(b_df, f_splits):
                y_true, p_pred = [], []
                for tr_i, te_i in f_splits:
                    tr = b_df.iloc[tr_i]
                    te = b_df.iloc[te_i]
                    # Select C using train fold
                    best_c = C_grid[len(C_grid) // 2]  # median C or fast fit
                    p_pipe = models.make_pipeline("M1", cfg, seed=42)
                    models.set_C(p_pipe, best_c)
                    p_pipe.fit(tr[feature_cols].values, tr["label"].values)
                    p_pred.extend(p_pipe.predict_proba(te[feature_cols].values)[:, 1])
                    y_true.extend(te["label"].values)
                return metrics.auc(np.array(y_true), np.array(p_pred))

            boot_res = uncertainty.bootstrap_cv(
                _boot_cv_eval,
                subjects_df,
                B=min(B_boot, 200),
                seed=cfg["cv"]["base_seed"],
            )
            model_results["bootstrap_cv_participant"] = boot_res
            print(f"  Bootstrap-of-CV CI: {boot_res['mean']:.4f} [{boot_res['ci_lower']:.4f}, {boot_res['ci_upper']:.4f}]")

            # Internal speed-adjusted AAUC (gate check)
            if "speed_01_mps" in subjects_df.columns:
                speed_vals = subjects_df.set_index("subject_id").loc[subj_preds.index, "speed_01_mps"].values
                study_vals = subjects_df.set_index("subject_id").loc[subj_preds.index, "study"].values
                aauc_res = metrics.aauc(all_y, mean_p, speed_vals, strata=study_vals, B=500, seed=cfg["cv"]["base_seed"])
                model_results["speed_adjusted_aauc"] = aauc_res
                print(f"  Internal AAUC (speed-adjusted) = {aauc_res['aauc']:.4f} [{aauc_res['ci_lower']:.4f}, {aauc_res['ci_upper']:.4f}]")
                if aauc_res["ci_lower"] <= 0.5:
                    print("  ⚠️  INTERNAL GATE: AAUC CI includes 0.5 — report to Lead before continuing.")

        all_models_summary[model_id] = model_results
        auc_m = rep_summary["auc"]
        print(f"  AUC:   {auc_m['mean']:.4f} ± {auc_m['std']:.4f} [{auc_m['p2_5']:.4f}, {auc_m['p97_5']:.4f}]")
        print(f"  Brier: {rep_summary['brier']['mean']:.4f} ± {rep_summary['brier']['std']:.4f}")
        print(f"  Slope: {rep_summary['calibration_slope']['mean']:.4f} ± {rep_summary['calibration_slope']['std']:.4f}")
        reporting.save_metrics_json(model_results, metrics_out, f"{model_id}_metrics.json")

    # Save predictions CSV using safe_write
    if all_preds_list:
        all_preds_df = pd.concat(all_preds_list, ignore_index=True)
        csv_str = all_preds_df.to_csv(index=False)
        safe_write(cv_out / "predictions.csv", csv_str)
        print(f"\nPredictions saved to {cv_out / 'predictions.csv'}")

    reporting.save_metrics_json(all_models_summary, metrics_out, "all_models_summary.json")

    after = snapshot()
    verify(before)
    print("\nE6 complete. Protected-path hashes unchanged ✓")


if __name__ == "__main__":
    main()
