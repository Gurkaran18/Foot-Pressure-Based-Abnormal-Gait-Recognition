"""E7: Ga+Ju sensitivity analysis and leave-one-study-out (spec §9 step 9).

Owner: Teammate (model/*).
Run from repo root:  python -m experiments.external_transport.scripts.a2_07_sensitivity_loso --run-id <id>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import paths, splits, models, metrics, reporting
from experiments.external_transport.etx.guards import snapshot, verify


def loso_run(subjects_df, model_id, feature_cols, cfg):
    """Leave-one-study-out: train on 2 studies, test on the third. No standardisation."""
    studies = subjects_df["study"].unique()
    results = {}
    seed = cfg["cv"]["base_seed"]

    for held_out in studies:
        train_df = subjects_df[subjects_df["study"] != held_out].reset_index(drop=True)
        test_df = subjects_df[subjects_df["study"] == held_out].reset_index(drop=True)

        pipe = models.make_pipeline(model_id, cfg, seed=seed)

        # Inner CV for C selection (LR models)
        if model_id not in {"X1", "X13-ref"}:
            best_C, best_ll = None, float("inf")
            C_grid = np.logspace(-3, 2, 11)
            inner_gen = list(splits.inner_folds(train_df, k=cfg["cv"]["inner_folds"],
                                                seed=seed, strata=cfg["cv"].get("strata")))
            for C in C_grid:
                lls = []
                for it_ids, iv_ids in inner_gen:
                    it_df = train_df[train_df["subject_id"].isin(it_ids)]
                    iv_df = train_df[train_df["subject_id"].isin(iv_ids)]
                    p_tmp = models.make_pipeline(model_id, cfg, seed=seed)
                    models.set_C(p_tmp, C)
                    p_tmp.fit(it_df[feature_cols].values, it_df["label"].values)
                    prob_iv = np.clip(
                        p_tmp.predict_proba(iv_df[feature_cols].values)[:, 1], 1e-9, 1 - 1e-9
                    )
                    ll = -np.mean(
                        iv_df["label"].values * np.log(prob_iv)
                        + (1 - iv_df["label"].values) * np.log(1 - prob_iv)
                    )
                    lls.append(ll)
                if np.mean(lls) < best_ll:
                    best_ll, best_C = np.mean(lls), C
            models.set_C(pipe, best_C)

        pipe.fit(train_df[feature_cols].values, train_df["label"].values)
        probs = pipe.predict_proba(test_df[feature_cols].values)[:, 1]
        y_test = test_df["label"].values
        thr = cfg["models"].get("threshold", 0.5)

        auc_val = metrics.auc(y_test, probs)
        thr_m = metrics.threshold_metrics(y_test, probs, thr)
        results[held_out] = {
            "held_out_study": held_out,
            "train_studies": [s for s in studies if s != held_out],
            "n_test": int(len(y_test)),
            "n_train": int(len(train_df)),
            "auc": auc_val,
            **thr_m,
        }
        print(f"  LOSO held={held_out}: AUC={auc_val:.4f} "
              f"sens={thr_m['sensitivity']:.3f} spec={thr_m['specificity']:.3f}")

    # Mean across held-out studies
    auc_vals = [v["auc"] for v in results.values()]
    results["mean"] = {
        "auc": float(np.mean(auc_vals)),
        "auc_std": float(np.std(auc_vals, ddof=1)),
        "mean_sensitivity": float(np.mean([v["sensitivity"] for v in results.values() if "sensitivity" in v])),
        "mean_specificity": float(np.mean([v["specificity"] for v in results.values() if "specificity" in v])),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="E7: Sensitivity analyses")
    parser.add_argument("--run-id", default="e7_sensitivity")
    parser.add_argument("--subject-csv", default=None)
    args = parser.parse_args()

    cfg = paths.load_config()
    out_dir = paths.outputs_root(args.run_id) / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = snapshot()

    if args.subject_csv:
        csv_path = Path(args.subject_csv)
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

    feature_cols = [c for c in cfg["features"]["t1_core"] if c in subjects_df.columns]
    model_id = "M1"

    # --- Sensitivity 1: Ga + Ju only (no Si) ---
    ga_ju = subjects_df[subjects_df["study"].isin(["Ga", "Ju"])].reset_index(drop=True)
    print(f"\n--- Ga+Ju only ({len(ga_ju)} subjects) ---")
    auc_gaju = []
    for _, fold, train_ids, test_ids in splits.outer_folds(
        ga_ju, repeats=5, k=cfg["cv"]["outer_folds"],
        base_seed=cfg["cv"]["base_seed"], strata=cfg["cv"].get("strata"),
    ):
        train_df = ga_ju[ga_ju["subject_id"].isin(train_ids)]
        test_df = ga_ju[ga_ju["subject_id"].isin(test_ids)]
        pipe = models.make_pipeline(model_id, cfg, seed=cfg["cv"]["base_seed"])
        pipe.fit(train_df[feature_cols].values, train_df["label"].values)
        probs = pipe.predict_proba(test_df[feature_cols].values)[:, 1]
        if len(np.unique(test_df["label"].values)) == 2:
            auc_gaju.append(metrics.auc(test_df["label"].values, probs))
    gaju_result = {
        "subset": "Ga+Ju",
        "n_subjects": int(len(ga_ju)),
        "auc_mean": float(np.mean(auc_gaju)),
        "auc_std": float(np.std(auc_gaju, ddof=1)),
    }
    print(f"  Ga+Ju AUC = {gaju_result['auc_mean']:.4f} ± {gaju_result['auc_std']:.4f}")

    # --- Sensitivity 2: Leave-one-study-out ---
    print(f"\n--- Leave-one-study-out (M1, {len(subjects_df)} subjects) ---")
    loso_result = loso_run(subjects_df, model_id, feature_cols, cfg)

    result = {
        "ga_ju_sensitivity": gaju_result,
        "loso": loso_result,
    }
    out_path = out_dir / "sensitivity_results.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nResults written to {out_path}")

    after = snapshot()
    verify(before)
    print("E7 complete. Protected-path hashes unchanged ✓")


if __name__ == "__main__":
    main()
