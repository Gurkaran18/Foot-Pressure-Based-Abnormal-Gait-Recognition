"""
evaluate.py — Standalone evaluation script for MSCAF-Gait
==========================================================

Usage:
    python evaluate.py --checkpoint ./checkpoints_pd/best_model_pd.pth \\
                       --data_root  /path/to/gait-in-parkinsons-disease-1.0.0 \\
                       --task       pd_detection

If --data_root is omitted, the DATA_ROOT environment variable is used.
If that is also unset, defaults to './gait-in-parkinsons-disease-1.0.0'.

This script scores the SINGLE-SPLIT checkpoint, i.e. the model main.py trains
when CONFIG['use_cv'] is False. When main.py is run with cross-validation (the
default), the headline numbers are the per-fold and pooled out-of-fold results
in results_pd/cv_results.json, not this script's output.

Evaluation runs on the HELD-OUT TEST SUBJECTS ONLY. The subject-level split is
rebuilt with main.make_subject_splits() using the same CONFIG and seed that
training used, so the train and validation subjects the model has already seen
are excluded. Reporting on the full dataset leaks training data into the metrics
and inflates them.
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from torch.utils.data import DataLoader

# Re-use the model and dataset classes from main.py
from main import (
    MSCAFGaitPD,
    PhysioNetGaitDataset,
    load_subject_info,
    make_subject_splits,
    aggregate_by_subject,
    classification_metrics,
    _json_safe,
    CONFIG,
    set_seed,
)


# ────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MSCAF-Gait checkpoint."
    )
    parser.add_argument(
        '--checkpoint', type=str,
        default='./checkpoints_pd/best_model_pd.pth',
        help='Path to the saved checkpoint (.pth file)',
    )
    parser.add_argument(
        '--data_root', type=str,
        default=os.environ.get('DATA_ROOT', CONFIG['data_root']),
        help='Root directory of the PhysioNet gaitpdb dataset',
    )
    parser.add_argument(
        '--task', type=str,
        default=CONFIG['task'],
        choices=['pd_detection', 'severity', 'hy_stage'],
        help='Task the model was trained on',
    )
    parser.add_argument(
        '--output_dir', type=str,
        default='./eval_results',
        help='Directory to save evaluation outputs',
    )
    parser.add_argument(
        '--batch_size', type=int, default=CONFIG['batch_size'],
    )
    return parser.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curve(y_true, y_scores, save_path,
                   title='ROC Curve — PD vs. Healthy'):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return auc


# ────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(CONFIG['seed'])
    device = CONFIG['device']

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MSCAF-Gait — Evaluation")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data root  : {args.data_root}")
    print(f"Task       : {args.task}")
    print(f"Device     : {device}")
    print(f"Fusion     : {CONFIG['fusion']} | use_lstm: {CONFIG['use_lstm']}")
    print("=" * 60)

    if CONFIG['use_cv']:
        print("\n  Note: CONFIG['use_cv'] is True, so main.py's headline results come")
        print("        from cross-validation (results_pd/cv_results.json). This script")
        print("        scores the single-split checkpoint on that split's test subjects.")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"\nCheckpoint trained to epoch {ckpt.get('epoch', '?')}")

    # ── Build model ───────────────────────────────────────────────────────────
    num_classes = 5 if args.task == 'hy_stage' else CONFIG['num_classes']
    model = MSCAFGaitPD(
        num_classes=num_classes,
        in_channels=CONFIG['in_channels'],
        task=args.task,
        fusion=CONFIG['fusion'],
        use_lstm=CONFIG['use_lstm'],
        attn_dim=CONFIG['attn_dim'],
        spatial_kernel=CONFIG['spatial_kernel'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ── Load subjects → HELD-OUT TEST SUBJECTS ONLY ──────────────────────────
    print("\nLoading dataset...")
    subjects_df = load_subject_info(args.data_root)
    if len(subjects_df) == 0:
        print("❌ No data found. Check --data_root path.")
        return

    # Rebuild the exact split training used — same CONFIG, same seed — and keep
    # only the test subjects. Scoring the full dataset would include the train
    # and validation subjects the model was fitted on and inflate every metric.
    train_ids, val_ids, test_ids = make_subject_splits(subjects_df, CONFIG)
    test_df = subjects_df[subjects_df['subject_id'].isin(test_ids)]

    assert not (set(test_ids) & set(train_ids)), "train/test subject overlap"
    assert not (set(test_ids) & set(val_ids)),   "val/test subject overlap"

    dataset = PhysioNetGaitDataset(
        args.data_root, test_df,
        CONFIG['sequence_length'], CONFIG['stride'], args.task,
    )
    if len(dataset) == 0:
        print("❌ The test split produced zero sequences. Check --data_root.")
        return

    # shuffle=False so predictions line up with dataset.sequences for
    # aggregate_by_subject()
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=CONFIG['num_workers'],
    )
    print(f"Held-out test subjects : {len(test_ids)}  ({len(test_df)} files)")
    print(f"Test windows           : {len(dataset)}")
    print("  (cross-check against main.py's 'Test: N subjects' and "
          "'Test sequences' lines)")

    # ── Run inference ─────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    all_probs  = []   # for ROC-AUC (classification only)

    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            outputs, _ = model(data)

            if args.task == 'severity':
                all_preds.extend(outputs.cpu().numpy())
            else:
                probs  = torch.softmax(outputs, dim=1)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            all_labels.extend(labels.numpy())

    # ── Compute & print metrics (held-out test subjects only) ────────────────
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    results = {
        'n_test_subjects': int(len(test_ids)),
        'n_test_windows':  int(len(dataset)),
    }

    if args.task == 'severity':
        mae  = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2   = r2_score(all_labels, all_preds)
        results.update({'MAE': mae, 'RMSE': rmse, 'R2': r2})

        print("\n── Regression Metrics — Held-out test subjects ──")
        print(f"  Window-level MAE  : {mae:.4f}")
        print(f"  Window-level RMSE : {rmse:.4f}")
        print(f"  Window-level R²   : {r2:.4f}")

        # Subject level: mean of that subject's window predictions
        yt, yp = aggregate_by_subject(dataset, all_preds, all_labels, args.task)
        subj_mae = mean_absolute_error(yt, yp)
        print(f"  Subject-level MAE : {subj_mae:.4f}   ({len(yt)} subjects)")
        results['subject_MAE'] = float(subj_mae)

    else:
        num_cls = num_classes
        report  = classification_report(all_labels, all_preds, digits=4)

        # Shared metric helper, so these numbers are computed exactly the way
        # main.py computes them for the single-split and cross-validation paths.
        win = classification_metrics(all_labels, all_preds, all_probs, num_cls)

        # Subject level: one majority-vote prediction per person, which is the
        # clinically meaningful unit — a clinician diagnoses a patient, not a
        # 10-second window.
        yt, yp = aggregate_by_subject(dataset, all_preds, all_labels, args.task)
        subj = classification_metrics(yt, yp, None, num_cls)

        class_names = (['Healthy', 'PD'] if args.task == 'pd_detection'
                       else [f'H&Y {i}' for i in range(num_cls)])
        binary = (num_cls == 2)

        results.update({'window': win, 'subject': subj})

        print("\n── Window-level Metrics — Held-out test subjects ─")
        print(f"  Windows           : {win['n']}")
        print(f"  Accuracy          : {win['accuracy']:.2f}%")
        print(f"  Balanced accuracy : {win['balanced_accuracy']:.2f}%")
        if binary:
            print(f"  Sensitivity       : {win['sensitivity']:.2f}%")
            print(f"  Specificity       : {win['specificity']:.2f}%")
            print(f"  ROC-AUC           : {win['roc_auc']:.4f}")
        print(f"  Majority baseline : {win['majority_baseline']:.2f}%")
        print("  Confusion matrix (windows):")
        print(np.array(win['confusion_matrix']))

        print("\n── Subject-level Metrics (majority vote) ────────")
        print(f"  Subjects          : {subj['n']}")
        print(f"  Accuracy          : {subj['accuracy']:.2f}%")
        print(f"  Balanced accuracy : {subj['balanced_accuracy']:.2f}%")
        if binary:
            print(f"  Sensitivity       : {subj['sensitivity']:.2f}%")
            print(f"  Specificity       : {subj['specificity']:.2f}%")
        print(f"  Majority baseline : {subj['majority_baseline']:.2f}%")
        print("  Confusion matrix (subjects):")
        print(np.array(subj['confusion_matrix']))

        print("\n── Classification Report (windows) ──────────────")
        print(report)

        if binary:
            plot_roc_curve(
                all_labels, np.asarray(all_probs)[:, 1],
                Path(args.output_dir) / 'roc_curve.png',
                title='ROC Curve — PD vs. Healthy (Held-out test subjects)',
            )

        plot_confusion_matrix(
            np.array(win['confusion_matrix']), class_names,
            Path(args.output_dir) / 'confusion_matrix.png',
            title='Confusion Matrix — Held-out test subjects (windows)',
        )
        plot_confusion_matrix(
            np.array(subj['confusion_matrix']), class_names,
            Path(args.output_dir) / 'confusion_matrix_subjects.png',
            title='Confusion Matrix — Held-out test subjects (majority vote)',
        )

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = Path(args.output_dir) / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(_json_safe(results), f, indent=2)
    print(f"\n  Saved metrics: {results_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
