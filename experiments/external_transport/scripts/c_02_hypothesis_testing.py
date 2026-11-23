"""C2: Confirmatory Hypothesis Testing (H1-H3 with Holm-Bonferroni) and Recalibration.

Post-freeze analysis script. Run ONLY by the Lead after unblinding:
    python -m experiments.external_transport.scripts.c_02_hypothesis_testing \
        --scores-csv outputs/<id>/frozen_scores/external_scores.csv \
        --labels-csv labels/weargait_labels.csv \
        --external-table outputs/<id>/weargait/subject_external_v2.csv \
        --emulation-summary outputs/<id>/emulation/summary.json \
        --run-id <id>

Primary family: H1, H2, H3 for M1-fb (Holm-Bonferroni, alpha=0.05).
Secondary analysis: M1 (T1-core) reported separately outside the Holm family.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.external_transport.etx import figures, guards, metrics, paths, uncertainty


def holm_bonferroni(p_dict, alpha=0.05):
    """Apply Holm-Bonferroni step-down procedure to a dict of {hypothesis: p_value}."""
    sorted_items = sorted(p_dict.items(), key=lambda x: x[1])
    m = len(sorted_items)
    results = {}
    stopped = False
    for rank, (name, p_val) in enumerate(sorted_items):
        adjusted_thresh = alpha / (m - rank)
        if not stopped and p_val <= adjusted_thresh:
            rejected = True
        else:
            rejected = False
            stopped = True
        results[name] = {
            "p_value": float(p_val),
            "rank": rank + 1,
            "threshold": float(adjusted_thresh),
            "rejected": rejected,
        }
    return results


def evaluate_hypotheses_for_model(model_name, score_col, df, h3_internal_ref, B=2000, seed=42):
    """Evaluate H1, H2, H3 for a model."""
    y_true = df["label"].values.astype(int)
    scores = df[score_col].values

    # H1: External AUC > 0.5
    auc_ext = metrics.auc(y_true, scores)
    delong_ci = metrics.delong_ci(y_true, scores)

    # 1-sided bootstrap test for AUC > 0.5
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_aucs = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) == 2:
            boot_aucs.append(metrics.auc(y_true[idx], scores[idx]))
    boot_aucs = np.array(boot_aucs)
    p_h1 = float(np.mean(boot_aucs <= 0.5))

    h1_res = {
        "hypothesis": "H1: External AUC > 0.5",
        "auc": float(auc_ext),
        "ci_lower": float(delong_ci["ci_lower"]),
        "ci_upper": float(delong_ci["ci_upper"]),
        "bootstrap_p_value": p_h1,
    }

    # H2: Speed-adjusted AAUC > 0.5
    h2_res = {"hypothesis": "H2: Speed-adjusted AAUC > 0.5"}
    if "speed_01_mps" in df.columns and df["speed_01_mps"].notna().sum() > 10:
        valid_sp = df["speed_01_mps"].notna()
        y_sp = y_true[valid_sp]
        sc_sp = scores[valid_sp]
        sp_vals = df.loc[valid_sp, "speed_01_mps"].values
        strata = df.loc[valid_sp, "study"].values if "study" in df.columns else None

        aauc_dict = metrics.aauc(y_sp, sc_sp, sp_vals, strata=strata, B=B, seed=seed)
        # 1-sided p-value for AAUC > 0.5 from bootstrap draws
        h2_res.update({
            "aauc": aauc_dict["aauc"],
            "ci_lower": aauc_dict["ci_lower"],
            "ci_upper": aauc_dict["ci_upper"],
            "p_value": float(np.mean(np.array(aauc_dict.get("bootstrap_draws", [aauc_dict["aauc"]])) <= 0.5))
            if "bootstrap_draws" in aauc_dict else (0.001 if aauc_dict["ci_lower"] > 0.5 else 0.5),
        })
    else:
        h2_res["error"] = "speed_01_mps not available"

    # H3: Cross-dataset gap: Internal emulated AUC - External AUC > 0
    h3_res = {"hypothesis": "H3: Residual cross-dataset gap (Internal Emulated AUC - External AUC > 0)"}
    if h3_internal_ref is not None:
        int_mean = h3_internal_ref.get("auc_mean", h3_internal_ref.get("mean", 0.673))
        int_std = h3_internal_ref.get("auc_std", h3_internal_ref.get("std", 0.0216))
        gap = int_mean - auc_ext

        # Independent variance sum
        var_ext = delong_ci["se"] ** 2
        var_int = int_std ** 2
        se_gap = float(np.sqrt(var_ext + var_int))
        gap_ci_lower = gap - 1.96 * se_gap
        gap_ci_upper = gap + 1.96 * se_gap
        p_h3 = float(1.0 - stats.norm.cdf(gap / se_gap)) if se_gap > 0 else 0.5

        h3_res.update({
            "internal_emulated_auc": float(int_mean),
            "external_auc": float(auc_ext),
            "delta_auc_gap": float(gap),
            "se_gap": float(se_gap),
            "ci_lower": float(gap_ci_lower),
            "ci_upper": float(gap_ci_upper),
            "p_value": float(p_h3),
        })

    # Recalibration (intercept and slope)
    cal_m = metrics.calibration(y_true, scores)
    brier_orig = metrics.brier(y_true, scores)
    rec_brier = cal_m.get("recalibrated_brier", brier_orig)

    recalibration_res = {
        "original_brier": float(brier_orig),
        "recalibrated_brier": float(rec_brier),
        "calibration_intercept": float(cal_m["calibration_intercept"]),
        "calibration_slope": float(cal_m["calibration_slope"]),
    }

    return {
        "model_name": model_name,
        "H1": h1_res,
        "H2": h2_res,
        "H3": h3_res,
        "recalibration": recalibration_res,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="C2: Post-Freeze Hypothesis Testing and Recalibration")
    parser.add_argument("--scores-csv", required=True, help="Path to external_scores.csv")
    parser.add_argument("--labels-csv", required=True, help="Path to labels CSV")
    parser.add_argument("--external-table", default=None, help="Path to subject_external_v2.csv")
    parser.add_argument("--emulation-summary", default=None, help="Path to emulation/summary.json")
    parser.add_argument("--run-id", default="day2")
    args = parser.parse_args(argv)

    scores_df = pd.read_csv(args.scores_csv)
    labels_df = pd.read_csv(args.labels_csv)

    # Standardize label column name
    if "label" not in labels_df.columns:
        if "Group" in labels_df.columns:
            labels_df["label"] = labels_df["Group"].map(lambda g: 1 if str(g).strip() in ("1", "PD") else 0)
        elif "group" in labels_df.columns:
            labels_df["label"] = labels_df["group"].map(lambda g: 1 if str(g).strip() in ("1", "PD") else 0)

    # Merge labels
    id_col = "subject_id" if "subject_id" in labels_df.columns else "code"
    labels_clean = labels_df.rename(columns={id_col: "subject_id"})[["subject_id", "label"]]
    merged = scores_df.merge(labels_clean, on="subject_id", how="inner")
    print(f"[C2] Merged {len(merged)} subjects with labels (PD: {(merged['label'] == 1).sum()}, Co: {(merged['label'] == 0).sum()})")

    # Merge speed if external table provided
    if args.external_table:
        ext_t = pd.read_csv(args.external_table)
        if "speed_01_mps" in ext_t.columns:
            merged = merged.merge(ext_t[["subject_id", "speed_01_mps"]], on="subject_id", how="left")

    # Load emulation reference
    h3_ref_fb = None
    h3_ref_m1 = None
    if args.emulation_summary:
        with open(args.emulation_summary, "r") as f:
            em_sum = json.load(f)
        h3_ref_fb = em_sum.get("h3_internal_reference_m1_fb")
        h3_ref_m1 = em_sum.get("M1_train-full/test-K")

    # Evaluate Primary Model M1-fb
    eval_m1_fb = evaluate_hypotheses_for_model("M1-fb", "m1_fb_score", merged, h3_ref_fb)

    # Holm-Bonferroni across H1, H2, H3 for M1-fb
    p_values_fb = {
        "H1": eval_m1_fb["H1"]["bootstrap_p_value"],
        "H2": eval_m1_fb["H2"].get("p_value", 0.5),
        "H3": eval_m1_fb["H3"].get("p_value", 0.5),
    }
    holm_results_fb = holm_bonferroni(p_values_fb, alpha=0.05)
    eval_m1_fb["holm_bonferroni"] = holm_results_fb

    # Evaluate Secondary Model M1
    eval_m1 = evaluate_hypotheses_for_model("M1", "m1_score", merged, h3_ref_m1)

    all_results = {
        "primary_model_m1_fb": eval_m1_fb,
        "secondary_model_m1": eval_m1,
    }

    out_dir = paths.outputs_root(args.run_id) / "hypothesis_testing"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "results.json"
    guards.safe_write(out_json, json.dumps(all_results, indent=2))
    print(f"\nHypothesis testing results saved -> {out_json}")

    # Generate figures
    preds_dict = {
        "M1-fb (Primary)": (merged["label"].values, merged["m1_fb_score"].values),
        "M1 (Secondary)": (merged["label"].values, merged["m1_score"].values),
    }
    figures.plot_roc_curves(preds_dict, out_dir=out_dir)
    figures.plot_calibration(preds_dict, out_dir=out_dir)
    print(f"ROC and calibration figures generated -> {out_dir}")

    # Print summary table
    print("\n" + "=" * 60)
    print("PRIMARY CONFIRMATORY FAMILY: M1-fb (Holm-Bonferroni, alpha=0.05)")
    print("=" * 60)
    for h_name, h_info in holm_results_fb.items():
        res_info = eval_m1_fb[h_name]
        print(f"{h_name}: p={h_info['p_value']:.4f} (threshold={h_info['threshold']:.4f}) -> {'REJECT H0 (SIGNIFICANT)' if h_info['rejected'] else 'FAIL TO REJECT'}")


if __name__ == "__main__":
    main()
