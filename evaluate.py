"""
evaluate.py — Standalone evaluation script for MSCAF-Gait
==========================================================

Usage:
    python evaluate.py --checkpoint ./checkpoints_pd/best_model_pd.pth \\
                       --data_root  /path/to/gait-in-parkinsons-disease-1.0.0 \\
                       --task       pd_detection

If --data_root is omitted, the DATA_ROOT environment variable is used.
If that is also unset, defaults to './gait-in-parkinsons-disease-1.0.0'.
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

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_roc_curve(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {auc:.3f}', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — PD vs. Healthy')
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
    print("=" * 60)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"\nCheckpoint trained to epoch {ckpt.get('epoch', '?')}")

    # ── Build model ───────────────────────────────────────────────────────────
    num_classes = 5 if args.task == 'hy_stage' else CONFIG['num_classes']
    model = MSCAFGaitPD(
        num_classes=num_classes,
        in_channels=CONFIG['in_channels'],
        task=args.task,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ── Load all subjects → use the full dataset as test set ─────────────────
    print("\nLoading dataset...")
    subjects_df = load_subject_info(args.data_root)
    if len(subjects_df) == 0:
        print("❌ No data found. Check --data_root path.")
        return

    dataset = PhysioNetGaitDataset(
        args.data_root, subjects_df,
        CONFIG['sequence_length'], CONFIG['stride'], args.task,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=CONFIG['num_workers'],
    )
    print(f"Total sequences: {len(dataset)}")

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

    # ── Compute & print metrics ───────────────────────────────────────────────
    results = {}

    if args.task == 'severity':
        mae  = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2   = r2_score(all_labels, all_preds)
        results = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

        print("\n── Regression Metrics ───────────────────────────")
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R²   : {r2:.4f}")

    else:
        acc    = accuracy_score(all_labels, all_preds) * 100
        report = classification_report(all_labels, all_preds, digits=4)
        cm     = confusion_matrix(all_labels, all_preds)

        if args.task == 'pd_detection':
            class_names = ['Healthy', 'PD']
            # Sensitivity / Specificity from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) * 100
            specificity = tn / (tn + fp) * 100
            results.update({'Sensitivity': sensitivity, 'Specificity': specificity})

            # ROC-AUC (binary)
            pd_probs = np.array(all_probs)[:, 1]
            auc = plot_roc_curve(
                all_labels, pd_probs,
                Path(args.output_dir) / 'roc_curve.png'
            )
            results['ROC_AUC'] = auc
        else:
            class_names = [f'H&Y {i}' for i in range(num_classes)]

        results['Accuracy'] = acc

        print("\n── Classification Metrics ───────────────────────")
        print(f"  Accuracy    : {acc:.2f}%")
        if args.task == 'pd_detection':
            print(f"  Sensitivity : {sensitivity:.2f}%")
            print(f"  Specificity : {specificity:.2f}%")
            print(f"  ROC-AUC     : {auc:.4f}")
        print("\n── Classification Report ────────────────────────")
        print(report)

        plot_confusion_matrix(
            cm, class_names,
            Path(args.output_dir) / 'confusion_matrix.png'
        )

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = Path(args.output_dir) / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved metrics: {results_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
