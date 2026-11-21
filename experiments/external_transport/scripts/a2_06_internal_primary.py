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
from experiments.external_transport.etx.guards import snapshot, verify


def run_repeated_cv(subjects_df, model_id, cfg, feature_cols):
    """Run the full R=20 × 5-fold CV. Returns list of per-repeat pooled-OOF AUC values and predictions."""
    cv_cfg = cfg["cv"]
    seed_base = cv_cfg["base_seed"]
    n = len(subjects_df)
    all_repeat_aucs = []
    all_preds = []  # list of (y, p, repeat) rows

    for repeat, fold, train_ids, test_ids in splits.outer_folds(
        subjects_df,
        repeats=cv_cfg["repeats"],
        k=cv_cfg["outer_folds"],
        base_seed=seed_base,
        strata=cv_cfg.get("strata"),
    ):
        train_df = subjects_df[subjects_df["subject_id"].isin(train_ids)].reset_index(drop=True)
        test_df = subjects_df[subjects_df["subject_id"].isin(test_ids)].reset_index(drop=True)

        # Inner CV to select C (LR models only)
        inner_seed = int(hash(f"{seed_base}_{repeat}_{fold}")) % (2 ** 31)
        pipe = models.make_pipeline(model_id, cfg, seed=inner_seed)

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["label"].values

        # C selection for LR via inner CV
        if model_id not in {"X1", "X13-ref"}:
            best_C, best_ll = None, float("inf")
            C_grid = np.logspace(-3, 2, 11)
            if "logspace" in str(cfg["models"].get("lr_C_grid", "")):
                pass  # already default
            inner_gen = list(splits.inner_folds(train_df, k=cv_cfg["inner_folds"],
                                                seed=inner_seed,
                                                strata=cv_cfg.get("strata")))
            for C in C_grid:
                lls = []
                for it_ids, iv_ids in inner_gen:
                    it_df = train_df[train_df["subject_id"].isin(it_ids)]
                    iv_df = train_df[train_df["subject_id"].isin(iv_ids)]
                    p_tmp = models.make_pipeline(model_id, cfg, seed=inner_seed)
                    models.set_C(p_tmp, C)
                    p_tmp.fit(it_df[feature_cols].values, it_df["label"].values)
                    prob_iv = p_tmp.predict_proba(iv_df[feature_cols].values)[:, 1]
                    prob_iv = np.clip(prob_iv, 1e-9, 1 - 1e-9)
                    ll = -np.mean(
                        iv_df["label"].values * np.log(prob_iv)
                        + (1 - iv_df["label"].values) * np.log(1 - prob_iv)
                    )
                    lls.append(ll)
                mean_ll = float(np.mean(lls))
                if mean_ll < best_ll:
                    best_ll, best_C = mean_ll, C
            models.set_C(pipe, best_C)

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        for sid, y_i, p_i in zip(test_df["subject_id"], y_test, probs):
            all_preds.append({"model_id": model_id, "repeat": repeat, "outer_fold": fold,
                              "subject_id": sid, "y": int(y_i), "p": float(p_i)})

    preds_df = pd.DataFrame(all_preds)

    # Per-repeat pooled AUC
    for r in range(cv_cfg["repeats"]):
        r_df = preds_df[preds_df["repeat"] == r]
        if len(r_df["y"].unique()) == 2:
            all_repeat_aucs.append(metrics.auc(r_df["y"].values, r_df["p"].values))

    return all_repeat_aucs, preds_df


def main():
    parser = argparse.ArgumentParser(description="E6: internal primary models")
    parser.add_argument("--run-id", default="e6_internal")
    parser.add_argument("--subject-csv", default=None)
    args = parser.parse_args()

    cfg = paths.load_config()
    out_dir = paths.outputs_root(args.run_id)
    cv_out = out_dir / "cv"
    cv_out.mkdir(parents=True, exist_ok=True)
    metrics_out = out_dir / "metrics"
    metrics_out.mkdir(parents=True, exist_ok=True)

    before = snapshot()

    # Discover subject CSV
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

    feature_sets = {
        "M1": cfg["features"]["t1_core"],
        "M1-fb": cfg["features"]["t1_fallback"],
        "X1": cfg["features"]["t1_core"],
    }
    # Add optional models if features available
    if "speed_01_mps" in subjects_df.columns:
        feature_sets["M0"] = ["speed_01_mps"]
        feature_sets["M2"] = ["speed_01_mps"] + list(cfg["features"]["t1_core"])
    if all(c in subjects_df.columns for c in ["age_y", "sex"]):
        feature_sets["AS"] = ["age_y", "sex"]

    summary = {}
    all_preds_list = []

    for model_id, feature_cols in feature_sets.items():
        print(f"\n--- {model_id} ---")
        # Only run if all features present
        available = [c for c in feature_cols if c in subjects_df.columns]
        if len(available) < len(feature_cols):
            print(f"  Skipping {model_id}: missing columns {set(feature_cols) - set(subjects_df.columns)}")
            continue

        aucs, preds_df = run_repeated_cv(subjects_df, model_id, cfg, available)
        all_preds_list.append(preds_df)
        summ = uncertainty.summarize_repeats(aucs)
        thr = cfg["models"].get("threshold", 0.5)

        # Pooled OOF predictions (last repeat × all folds)
        last_rep = preds_df[preds_df["repeat"] == preds_df["repeat"].max()]
        thr_mets = metrics.threshold_metrics(last_rep["y"].values, last_rep["p"].values, thr)

        # DeLong CI on pooled (across all repeats)
        all_y = preds_df.groupby("subject_id")["y"].first().values
        # Use mean p over repeats for each subject
        mean_p = preds_df.groupby("subject_id")["p"].mean().values
        dl = metrics.delong(all_y, mean_p)
        cal = metrics.calibration(all_y, mean_p)
        bs = metrics.brier(all_y, mean_p)

        # Internal AAUC (speed-adjusted) if speed column present
        aauc_result = None
        if "speed_01_mps" in subjects_df.columns and model_id == "M1":
            speed = subjects_df.set_index("subject_id")["speed_01_mps"]
            speed_aligned = subjects_df.set_index("subject_id").loc[
                preds_df.groupby("subject_id")["p"].mean().index, "speed_01_mps"
            ].values
            study_aligned = subjects_df.set_index("subject_id").loc[
                preds_df.groupby("subject_id")["p"].mean().index, "study"
            ].values
            aauc_result = metrics.aauc(all_y, mean_p, speed_aligned, strata=study_aligned)
            print(f"  Internal AAUC (speed-adjusted) = {aauc_result['aauc']:.4f} "
                  f"[{aauc_result['ci_lower']:.4f}, {aauc_result['ci_upper']:.4f}]")
            if aauc_result["ci_lower"] <= 0.5:
                print("  ⚠️  INTERNAL GATE: AAUC CI includes 0.5 — report to Lead before continuing.")

        result = {
            "model_id": model_id,
            "auc_summary": summ,
            "delong": dl,
            "brier": bs,
            "calibration": cal,
            "threshold_metrics": thr_mets,
            "aauc": aauc_result,
        }
        summary[model_id] = result
        print(f"  AUC mean={summ['mean']:.4f} ± {summ['std']:.4f} "
              f"[{summ['p2_5']:.4f}, {summ['p97_5']:.4f}]")
        reporting.save_metrics_json(result, metrics_out, f"{model_id}_metrics.json")

    # Save all predictions
    if all_preds_list:
        all_preds_df = pd.concat(all_preds_list, ignore_index=True)
        all_preds_df.to_csv(cv_out / "predictions.csv", index=False)
        print(f"\nPredictions saved to {cv_out / 'predictions.csv'}")

    reporting.save_metrics_json(summary, metrics_out, "all_models_summary.json")

    after = snapshot()
    verify(before)   # protected paths unchanged
    print("\nE6 complete. Protected-path hashes unchanged ✓")


if __name__ == "__main__":
    main()
