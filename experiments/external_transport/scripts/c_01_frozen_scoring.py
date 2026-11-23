"""C1: Frozen-model fitting and external scoring (pre-freeze critical deliverable).

Fits final frozen pipelines (M1-fb primary, M1 secondary) on all 165 gaitpdb subjects.
Scores blinded external table (subject_external_v2.csv).
Strict rules:
  1. Never reads labels/ or accepts label inputs.
  2. Verifies SHA-256 fingerprint against recorded freeze value.
  3. Uses only qc_pass == True rows.
  4. Writes one score per subject_id for each model.

Owner: Teammate (model/*). Run from repo root:
    python -m experiments.external_transport.scripts.c_01_frozen_scoring --external-csv <path> --run-id <id>
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import guards, models, paths, splits

# Pre-declared frozen fingerprint from Lead B5 step
EXPECTED_EXTERNAL_SHA256 = "ac1fe306e2079c1531d80ae64d7784684506bf3b26f0cc002b52bf2ce4ea6a49"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def select_best_C(model_id, X, y, study, C_grid, cfg, seed):
    """Inner CV (log loss) to select C on the training set; folds stratified label x study as in a2_06."""
    best_c, best_ll = C_grid[len(C_grid) // 2], float("inf")
    inner_splits = list(splits.inner_folds(
        pd.DataFrame({"subject_id": range(len(y)), "label": y, "study": study}),
        k=cfg["cv"].get("inner_folds", 5),
        seed=seed,
        strata=cfg["cv"].get("strata"),
    ))
    for C in C_grid:
        lls = []
        for it_ids, iv_ids in inner_splits:
            X_tr, y_tr = X[it_ids], y[it_ids]
            X_val, y_val = X[iv_ids], y[iv_ids]
            p_tmp = models.make_pipeline(model_id, cfg, seed=seed)
            models.set_C(p_tmp, C)
            p_tmp.fit(X_tr, y_tr)
            prob_iv = np.clip(p_tmp.predict_proba(X_val)[:, 1], 1e-9, 1.0 - 1e-9)
            ll = -np.mean(y_val * np.log(prob_iv) + (1.0 - y_val) * np.log(1.0 - prob_iv))
            lls.append(ll)
        mean_ll = np.mean(lls)
        if mean_ll < best_ll:
            best_ll, best_c = mean_ll, C
    return best_c


def fit_frozen_pipeline(model_id, train_df, feature_cols, cfg, seed):
    """Fit final frozen pipeline on all training subjects with tuned C."""
    X = train_df[feature_cols].values
    y = train_df["label"].values
    C_grid = models._lr_C_grid(cfg)

    pipe = models.make_pipeline(model_id, cfg, seed=seed)
    best_c = select_best_C(model_id, X, y, train_df["study"].values, C_grid, cfg, seed)
    models.set_C(pipe, best_c)
    pipe.fit(X, y)

    # Extract fitted parameters
    lr_step = pipe.named_steps.get("clf")
    coefs = lr_step.coef_[0].tolist() if hasattr(lr_step, "coef_") else []
    intercept = float(lr_step.intercept_[0]) if hasattr(lr_step, "intercept_") else 0.0

    meta = {
        "model_id": model_id,
        "features": feature_cols,
        "best_C": float(best_c),
        "intercept": intercept,
        "coefficients": dict(zip(feature_cols, coefs)),
        "n_train_subjects": int(len(train_df)),
        "n_train_pd": int((y == 1).sum()),
        "n_train_co": int((y == 0).sum()),
    }
    return pipe, meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="C1: Frozen Model Fitting and External Scoring")
    parser.add_argument("--external-csv", required=True, help="Path to subject_external_v2.csv")
    parser.add_argument("--run-id", default="day2")
    parser.add_argument("--train-csv", required=True,
                        help="Path to the gaitpdb subject_v2.csv the frozen model is trained on (recorded by SHA-256)")
    parser.add_argument("--expected-train-sha256", default=None,
                        help="If given (as at the freeze), stop unless the training table matches")
    parser.add_argument("--expected-n-train", type=int, default=165, help="Primary cohort size (CLAUDE.md §3)")
    parser.add_argument("--expected-sha256", default=EXPECTED_EXTERNAL_SHA256)
    parser.add_argument("--allow-sha-mismatch", action="store_true", help="Allow test run on fake table")
    args = parser.parse_args(argv)

    ext_path = Path(args.external_csv)
    # Integrity check 1: never read from labels/
    for arg_str in (str(ext_path), sys.argv):
        if "labels" in str(arg_str).lower():
            raise RuntimeError("CRITICAL INTEGRITY VIOLATION: labels/ access forbidden in frozen scoring!")

    if not ext_path.exists():
        raise FileNotFoundError(f"External table not found at: {ext_path}")

    # Integrity check 2: SHA-256 fingerprint check
    actual_sha = compute_sha256(ext_path)
    if actual_sha != args.expected_sha256:
        if not args.allow_sha_mismatch:
            raise RuntimeError(
                f"SHA-256 verification FAILED!\n"
                f"  Actual:   {actual_sha}\n"
                f"  Expected: {args.expected_sha256}\n"
                f"Stopping before scoring to protect study integrity."
            )
        else:
            print(f"⚠️  WARNING: Running with SHA-256 mismatch allowed (testing mode). SHA: {actual_sha}")
    else:
        print(f"✓ SHA-256 fingerprint verified: {actual_sha}")

    cfg = paths.load_config()
    out_dir = paths.outputs_root(args.run_id) / "frozen_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    before = guards.snapshot()

    # Load external table
    ext_df = pd.read_csv(ext_path)
    if "label" in ext_df.columns:
        raise RuntimeError("CRITICAL INTEGRITY VIOLATION: External table contains 'label' column!")

    # Filter to qc_pass == True only
    if "qc_pass" not in ext_df.columns:
        raise ValueError("External table missing required 'qc_pass' column.")

    scored_cohort = ext_df[ext_df["qc_pass"]].copy().reset_index(drop=True)
    print(f"Loaded external table: {len(ext_df)} rows, {len(scored_cohort)} passing QC for scoring.")

    # Load training table (gaitpdb subject_v2.csv) -- explicit path only (pre-freeze fix 5)
    train_path = Path(args.train_csv)
    train_sha = compute_sha256(train_path)
    if args.expected_train_sha256 and train_sha != args.expected_train_sha256:
        raise RuntimeError(f"Training table SHA-256 mismatch: {train_sha} != {args.expected_train_sha256}")
    print(f"Training on gaitpdb cohort: {train_path} (sha256 {train_sha})")
    train_df = pd.read_csv(train_path)
    if "qc_pass" in train_df.columns:
        train_df = train_df[train_df["qc_pass"]].reset_index(drop=True)
    if train_df["subject_id"].duplicated().any():
        raise ValueError("Training table has duplicate subject_id rows (one row per subject expected).")
    if len(train_df) != args.expected_n_train:
        raise ValueError(f"Training cohort has {len(train_df)} subjects, expected {args.expected_n_train}.")
    if not set(train_df["label"].unique()) <= {0, 1}:
        raise ValueError("Training labels must be 0/1.")

    seed = cfg["cv"]["base_seed"]

    # 1. Primary External Model: M1-fb (B-D8)
    feat_m1_fb = list(cfg["features"]["t1_fallback"])
    pipe_m1_fb, meta_m1_fb = fit_frozen_pipeline("M1-fb", train_df, feat_m1_fb, cfg, seed)
    scores_fb = pipe_m1_fb.predict_proba(scored_cohort[feat_m1_fb].values)[:, 1]
    logits_fb = pipe_m1_fb.decision_function(scored_cohort[feat_m1_fb].values)

    # 2. Secondary External Model: M1 (T1-core)
    feat_m1 = list(cfg["features"]["t1_core"])
    pipe_m1, meta_m1 = fit_frozen_pipeline("M1", train_df, feat_m1, cfg, seed)
    scores_m1 = pipe_m1.predict_proba(scored_cohort[feat_m1].values)[:, 1]
    logits_m1 = pipe_m1.decision_function(scored_cohort[feat_m1].values)

    # Write output scores
    out_scores = pd.DataFrame({
        "subject_id": scored_cohort["subject_id"],
        "study": scored_cohort.get("study", "WG"),
        "m1_fb_score": np.round(scores_fb, 6),
        "m1_fb_logit": np.round(logits_fb, 6),
        "m1_score": np.round(scores_m1, 6),
        "m1_logit": np.round(logits_m1, 6),
    })

    # Also melted tall format per spec
    tall_rows = []
    for s_id, s_fb, l_fb, s_m1, l_m1 in zip(
        scored_cohort["subject_id"], scores_fb, logits_fb, scores_m1, logits_m1
    ):
        tall_rows.append({"subject_id": s_id, "model_id": "M1-fb", "score": round(s_fb, 6), "linear_predictor": round(l_fb, 6)})
        tall_rows.append({"subject_id": s_id, "model_id": "M1", "score": round(s_m1, 6), "linear_predictor": round(l_m1, 6)})
    tall_scores = pd.DataFrame(tall_rows)

    scores_path = out_dir / "external_scores.csv"
    tall_path = out_dir / "external_scores_tall.csv"
    manifest_path = out_dir / "frozen_manifest.json"

    manifest = {
        "external_table_path": str(ext_path),
        "external_sha256": actual_sha,
        "train_table_path": str(train_path),
        "train_sha256": train_sha,
        "n_external_total": int(len(ext_df)),
        "n_external_scored": int(len(scored_cohort)),
        "primary_model": meta_m1_fb,
        "secondary_model": meta_m1,
    }

    guards.safe_write(scores_path, out_scores.to_csv(index=False))
    guards.safe_write(tall_path, tall_scores.to_csv(index=False))
    guards.safe_write(manifest_path, json.dumps(manifest, indent=2))

    guards.verify(before)

    print(f"\n✓ Frozen scoring complete!")
    print(f"  Scores written -> {scores_path}")
    print(f"  Manifest written -> {manifest_path}")
    print("\nSummary of Scored Probabilities (M1-fb primary):")
    print(out_scores[["m1_fb_score", "m1_score"]].describe().loc[["mean", "std", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
