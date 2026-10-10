"""
leakage_demo.py — how much a window-level split inflates accuracy
=================================================================

The same model (gait_common.PRIMARY, on the same gait features computed per
10-second window with main.py's windowing) is evaluated twice:

  A. RANDOM WINDOW-LEVEL split — StratifiedKFold over windows. Windows of
     one person land in both train and test, so the model can recognise the
     PERSON (body weight, walking rhythm) instead of the disease. Leaky.
  B. SUBJECT-LEVEL split — main.py's folds: every window of a test subject
     is unseen during training. Honest.

Both are scored at WINDOW level, so the only thing that differs is the split.

Run (Kaggle):   !python leakage_demo.py
Outputs      :  /kaggle/working/results_new/
    leakage_demo.json   metrics for both splits
    leakage_demo.png    bar chart of the inflation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import gait_common as gc


def main():
    gc.apply_style()
    print("=" * 60)
    print("LEAKAGE DEMO — window-level vs subject-level split")
    print("=" * 60)

    subjects_df = gc.load_subjects()
    windows = gc.window_feature_table(subjects_df)
    print(f"Windows with ≥{gc.MIN_STRIDES} clean strides per foot: {len(windows)} "
          f"from {windows['subject_id'].nunique()} subjects")

    X = windows[gc.FEATURE_NAMES].to_numpy(float)
    y = windows['label'].to_numpy(int)
    sids = windows['subject_id'].to_numpy()
    name = gc.PRIMARY
    k = gc.CONFIG['cv_folds']

    # ── A. random window-level split (leaky) ─────────────────────────────
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=gc.SEED)
    p_win = np.full(len(y), np.nan)
    shared = []
    for tr, te in skf.split(X, y):
        model = gc.make_model(name, y[tr]).fit(X[tr], y[tr])
        p_win[te] = model.predict_proba(X[te])[:, 1]
        shared.append(np.isin(sids[te], sids[tr]).mean() * 100)
    leaky = gc.binary_metrics(y, p_win)

    # ── B. subject-level split (main.py folds) ───────────────────────────
    p_subj, _ = gc.cv_oof_probs(X, y, sids, gc.subject_folds(subjects_df), name)
    honest = gc.binary_metrics(y, p_subj)

    print(f"\n  Model: {gc.PRETTY[name]}, window-level metrics, {k}-fold")
    for label, m in (('Random window split', leaky), ('Subject-level split', honest)):
        print(f"  {label:<20} acc {m['accuracy']:6.2f}% | bal.acc "
              f"{m['balanced_accuracy']:6.2f}% | AUC {m['roc_auc']:.4f}")
    gap = leaky['balanced_accuracy'] - honest['balanced_accuracy']
    print(f"  Inflation from leakage: +{gap:.2f} balanced-accuracy points")
    print(f"  Test windows whose subject was also in training (window split): "
          f"{np.mean(shared):.1f}%")

    gc.save_json({
        'model': name, 'level': 'window', 'n_windows': int(len(y)),
        'n_subjects': int(windows['subject_id'].nunique()),
        'random_window_split': leaky, 'subject_level_split': honest,
        'inflation_balanced_accuracy_points': float(gap),
        'pct_test_windows_with_subject_in_train': float(np.mean(shared)),
    }, 'leakage_demo.json')

    # ── Plot ─────────────────────────────────────────────────────────────
    keys = [('accuracy', 'Accuracy'), ('balanced_accuracy', 'Balanced accuracy'),
            ('roc_auc', 'ROC-AUC ×100')]
    vals = [[leaky[k_] * (100 if k_ == 'roc_auc' else 1),
             honest[k_] * (100 if k_ == 'roc_auc' else 1)] for k_, _ in keys]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    gc.grouped_bars(ax, [lab for _, lab in keys],
                    ['Random WINDOW split (leaky)', 'SUBJECT-level split (honest)'],
                    vals, ylim=(0, 108), colors=[gc.SERIES[1], gc.SERIES[0]])
    ax.axhline(50, ls=':', color=gc.MUTED, lw=1.2, label='Chance (balanced acc. / AUC)')
    ax.set_ylabel('%')
    ax.set_title(f'Data leakage: +{gap:.1f} pts balanced accuracy from a window split')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    gc.save_fig(fig, 'leakage_demo.png')


if __name__ == '__main__':
    main()
