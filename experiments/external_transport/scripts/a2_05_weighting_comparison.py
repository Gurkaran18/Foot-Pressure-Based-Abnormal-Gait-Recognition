"""E5: unweighted vs balanced LR — R8 rule (spec §9 step 5).

Owner: Teammate (model/*).
Run from repo root:  python -m experiments.external_transport.scripts.a2_05_weighting_comparison --run-id <id>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import paths, splits, models, metrics, reporting
from experiments.external_transport.etx.guards import snapshot


def run_cv_auc(subjects_df, model_id, cfg, seed, feature_cols):
    """Run 5-fold CV once and return pooled out-of-fold AUC."""
    from sklearn.pipeline import Pipeline

    cv_cfg = cfg["cv"]
    all_probs = np.full(len(subjects_df), np.nan)
    y_all = subjects_df["label"].values

    for _, fold, train_ids, test_ids in splits.outer_folds(
        subjects_df, repeats=1, k=cv_cfg["outer_folds"], base_seed=seed,
        strata=cv_cfg.get("strata"),
    ):
        train_df = subjects_df[subjects_df["subject_id"].isin(train_ids)]
        test_df = subjects_df[subjects_df["subject_id"].isin(test_ids)]
        test_idx = subjects_df.index[subjects_df["subject_id"].isin(test_ids)]

        pipe = models.make_pipeline(model_id, cfg, seed=seed)
        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test = test_df[feature_cols].values

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        all_probs[test_idx] = probs

    mask = ~np.isnan(all_probs)
    return metrics.auc(y_all[mask], all_probs[mask])


def main():
    parser = argparse.ArgumentParser(description="E5: LR weighting comparison")
    parser.add_argument("--run-id", default="e5_weighting")
    parser.add_argument("--subject-csv", default=None,
                        help="Path to subject_v2.csv (auto-discovered if omitted)")
    args = parser.parse_args()

    cfg = paths.load_config()
    out_dir = paths.outputs_root(args.run_id) / "weighting"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover subject CSV
    if args.subject_csv:
        csv_path = Path(args.subject_csv)
    else:
        # Look for the most recent outputs/*/features/subject_v2.csv
        candidates = sorted(paths.OUTPUTS_DIR.glob("*/features/subject_v2.csv"))
        if not candidates:
            print("ERROR: subject_v2.csv not found. Run a2_03 first.", file=sys.stderr)
            sys.exit(1)
        csv_path = candidates[-1]
    print(f"Using subject CSV: {csv_path}")

    subjects_df = pd.read_csv(csv_path)
    # Filter to QC-passing subjects
    if "qc_pass" in subjects_df.columns:
        subjects_df = subjects_df[subjects_df["qc_pass"]].reset_index(drop=True)

    feature_cols = cfg["features"]["t1_core"]

    results = {}
    for weighting in ["none", "balanced"]:
        cfg_w = {**cfg, "models": {**cfg["models"], "lr_weighting": weighting}}
        auc_val = run_cv_auc(subjects_df, "M1", cfg_w, cfg["cv"]["base_seed"], feature_cols)
        results[weighting] = auc_val
        print(f"  LR weighting={weighting:10s}  AUC={auc_val:.4f}")

    # R8 rule: use whichever gives higher AUC; break ties with 'none'
    if results.get("balanced", 0) > results.get("none", 0) + 1e-6:
        chosen = "balanced"
    else:
        chosen = "none"
    print(f"\nR8 rule → chosen weighting: {chosen}")
    results["chosen"] = chosen

    out_path = out_dir / "weighting_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
