"""Figures module for Phase B audit, QC, agreement, and post-freeze performance.

Owner: Teammate (model/*).
Functions accept file paths or DataFrames to allow running on remote or local data.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def plot_stride_qc(strides_input, out_dir=None, sample_n: int = 20):
    """Blinded stride-detection QC plots on trials."""
    if isinstance(strides_input, (str, Path)):
        df = pd.read_csv(strides_input)
    else:
        df = strides_input.copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Stride duration distributions
    valid_strides = df[df["valid"]] if "valid" in df.columns else df
    durations = valid_strides["stride_s"] if "stride_s" in valid_strides.columns else valid_strides.get("stride_time_s", pd.Series())
    if len(durations) > 0:
        axes[0].hist(durations, bins=40, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].set_title(f"Valid Stride Durations (N={len(durations)})")
        axes[0].set_xlabel("Stride Duration (s)")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Foot split comparison (L vs R)
    if "foot" in valid_strides.columns and len(durations) > 0:
        l_dur = valid_strides[valid_strides["foot"] == "L"]["stride_s"]
        r_dur = valid_strides[valid_strides["foot"] == "R"]["stride_s"]
        axes[1].boxplot([l_dur.dropna(), r_dur.dropna()], tick_labels=["Left Foot", "Right Foot"])
        axes[1].set_title("Stride Duration by Foot")
        axes[1].set_ylabel("Duration (s)")
        axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Sample trial stride counts
    rec_col = "recording_id" if "recording_id" in df.columns else "trial_id"
    if rec_col in df.columns:
        counts = valid_strides.groupby(rec_col).size()
        sample_counts = counts.head(sample_n)
        axes[2].bar(range(len(sample_counts)), sample_counts.values, color="teal")
        axes[2].axhline(y=10, color="red", linestyle="--", label="Min Stride QC (10)")
        axes[2].set_title(f"Valid Strides per Trial (Sample of {len(sample_counts)})")
        axes[2].set_xlabel("Sample Trial Index")
        axes[2].set_ylabel("Valid Strides")
        axes[2].legend()
        axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = out_p / "stride_detection_qc.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return fig


def plot_bland_altman_agreement(pairs_input, out_dir=None):
    """Bland-Altman plots for Stride, Stance, and Swing differences (insole minus walkway)."""
    if isinstance(pairs_input, (str, Path)):
        df = pd.read_csv(pairs_input)
    else:
        df = pairs_input.copy()

    # Identify metric columns
    metrics = [
        ("stride", "diff_stride_ms" if "diff_stride_ms" in df.columns else "diff_stride_s", 10.0, "Stride Duration"),
        ("stance", "diff_stance_ms" if "diff_stance_ms" in df.columns else "diff_stance_s", 20.0, "Stance Duration"),
        ("swing", "diff_swing_ms" if "diff_swing_ms" in df.columns else "diff_swing_s", 15.0, "Swing Duration"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, col, tol, title) in zip(axes, metrics):
        if col in df.columns:
            diffs = df[col].dropna().values
            # Convert to ms if in seconds
            if "diff" in col and "_s" in col:
                diffs = diffs * 1000.0

            # Mean/reference value
            mean_col = f"mean_{name}_ms" if f"mean_{name}_ms" in df.columns else f"{name}_mean_s"
            if mean_col in df.columns:
                means = df[mean_col].dropna().values
                if "_s" in mean_col:
                    means = means * 1000.0
            else:
                means = np.arange(len(diffs))

            bias = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1))
            loa_upper = bias + 1.96 * sd
            loa_lower = bias - 1.96 * sd

            ax.scatter(means, diffs, alpha=0.4, color="navy", s=15)
            ax.axhline(bias, color="red", linestyle="-", label=f"Bias: {bias:+.1f} ms")
            ax.axhline(loa_upper, color="gray", linestyle="--", label=f"+1.96 SD: {loa_upper:+.1f} ms")
            ax.axhline(loa_lower, color="gray", linestyle="--", label=f"-1.96 SD: {loa_lower:+.1f} ms")

            # Tolerance band
            ax.axhspan(-tol, tol, color="green", alpha=0.15, label=f"Tolerance: ±{tol:.0f} ms")

            ax.set_title(f"{title} (N={len(diffs)})")
            ax.set_xlabel("Mean (ms)" if mean_col in df.columns else "Sample Index")
            ax.set_ylabel("Insole − Walkway Difference (ms)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.5)
        else:
            ax.text(0.5, 0.5, f"Column '{col}' not found", ha="center", va="center")
            ax.set_title(title)

    plt.tight_layout()
    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = out_p / "bland_altman_agreement.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return fig


def plot_drift(pairs_input, out_dir=None):
    """Drift plot of timing differences across trial duration."""
    if isinstance(pairs_input, (str, Path)):
        df = pd.read_csv(pairs_input)
    else:
        df = pairs_input.copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    time_col = "trial_time_s" if "trial_time_s" in df.columns else "time_s"
    diff_col = "diff_stride_ms" if "diff_stride_ms" in df.columns else "diff_hs_ms"

    if time_col in df.columns and diff_col in df.columns:
        sub = df[[time_col, diff_col]].dropna()
        x = sub[time_col].values
        y = sub[diff_col].values

        ax.scatter(x, y, alpha=0.3, color="darkslateblue", s=15, label="Paired Strides")
        # Linear drift slope
        poly = np.polyfit(x, y, 1)
        x_grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_grid, np.polyval(poly, x_grid), color="crimson", linewidth=2,
                label=f"Drift Slope: {poly[0]:+.3f} ms/s")

        ax.set_title("Insole-Walkway Timing Difference vs. Trial Time")
        ax.set_xlabel("Trial Time (s)")
        ax.set_ylabel("Timing Difference (ms)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "Time or difference columns not found for drift plot", ha="center", va="center")

    plt.tight_layout()
    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = out_p / "timing_drift.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return fig


def plot_roc_curves(predictions_dict: dict, out_dir=None):
    """Plot ROC curves for models with AUC in legend."""
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, (m_id, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{m_id} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = out_p / "roc_curves.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return fig


def plot_calibration(predictions_dict: dict, out_dir=None, n_bins: int = 10):
    """Plot calibration curve comparing predicted vs observed event frequencies."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for m_id, (y_true, y_pred) in predictions_dict.items():
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_pred, bins) - 1
        bin_pred, bin_obs = [], []
        for b in range(n_bins):
            mask = bin_ids == b
            if mask.sum() > 0:
                bin_pred.append(np.mean(y_pred[mask]))
                bin_obs.append(np.mean(y_true[mask]))

        ax.plot(bin_pred, bin_obs, marker="o", linewidth=2, label=f"{m_id}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_path = out_p / "calibration_curves.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
    plt.close(fig)
    return fig
