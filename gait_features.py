"""
gait_features.py — interpretable gait features + late fusion with MSCAF
=======================================================================

1. From the two total-force columns (L/R), detect heel strikes and toe-offs
   and compute per-subject stride time mean / CV, swing %, stance %, double
   support %, cadence, L/R asymmetry and peak forces (gait_common.py).
2. Train RandomForest and XGBoost on those features under the SAME 5-fold
   stratified subject-level CV as main.py (main.make_cv_folds, same seed).
3. Late-fuse the feature model's out-of-fold P(PD) with MSCAF's out-of-fold
   P(PD):  p_fused = (p_features + p_mscaf) / 2,  threshold 0.5.
   Fixed equal weights, so the fusion has no parameter fitted on test folds.

All metrics are pooled out-of-fold at SUBJECT level (one prediction per
subject), the same headline protocol as main.py's cv_results.json.

Run (Kaggle):   !python gait_features.py      (after main.py has run)
Outputs      :  /kaggle/working/results_new/
    gait_features_subject.csv   feature table, one row per subject
    feature_oof_probs.csv       out-of-fold P(PD) per subject, per model
    feature_model_<name>.joblib model refit on ALL subjects (for app.py)
    gait_features_metrics.json  all metrics, incl. fusion
    gait_features_roc.png       ROC curves
    gait_features_metrics.png   metric bars per model
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import gait_common as gc


def report(name, m):
    print(f"  {name:<28} bal.acc {m['balanced_accuracy']:6.2f}% | "
          f"sens {m['sensitivity']:6.2f}% | spec {m['specificity']:6.2f}% | "
          f"AUC {m['roc_auc']:.4f}")


def main():
    gc.apply_style()
    print("=" * 60)
    print("GAIT FEATURES + LATE FUSION  (5-fold subject-level CV, main.py folds)")
    print("=" * 60)

    subjects_df = gc.load_subjects()
    print(f"Data root: {gc.CONFIG['data_root']}")
    print(f"Subjects : {subjects_df['subject_id'].nunique()}  "
          f"({len(subjects_df)} trial files)")

    table = gc.subject_feature_table(subjects_df)
    folds = gc.subject_folds(subjects_df)

    X = table[gc.FEATURE_NAMES].to_numpy(float)
    y = table['label'].to_numpy(int)
    sids = table['subject_id'].to_numpy()

    results = {'protocol': 'pooled out-of-fold, subject level, main.py folds '
                           f"(StratifiedKFold k={gc.CONFIG['cv_folds']}, seed={gc.SEED})",
               'n_subjects': int(len(y)), 'features': gc.FEATURE_NAMES,
               'primary_model': gc.PRIMARY, 'models': {}}
    oof = pd.DataFrame({'subject_id': sids, 'true': y})

    print("\n── Feature models ───────────────────────────────")
    for name in gc.model_names():
        p, per_fold = gc.cv_oof_probs(X, y, sids, folds, name)
        oof[f'prob_{name}'] = p
        m = gc.binary_metrics(y, p)
        results['models'][name] = {'pooled_subject': m, 'per_fold': gc.fold_summary(per_fold)}
        report(name, m)

        # Refit on every subject for the demo app (not used for any metric)
        final = gc.make_model(name, y).fit(X, y)
        joblib.dump({'model': final, 'features': gc.FEATURE_NAMES},
                    gc.results_dir() / f'feature_model_{name}.joblib')

    # ── Late fusion with MSCAF ───────────────────────────────────────────
    print("\n── MSCAF out-of-fold probabilities ──────────────")
    mscaf = gc.load_mscaf_oof(subjects_df)
    if mscaf is not None:
        oof = oof.merge(mscaf[['subject_id', 'prob_pd', 'pred']].rename(
            columns={'prob_pd': 'prob_mscaf', 'pred': 'pred_mscaf_vote'}),
            on='subject_id', how='left')
        missing = oof['prob_mscaf'].isna().sum()
        if missing:
            print(f"  ⚠️  {missing} subjects have no MSCAF prediction; fusion uses the rest")
        ok = oof['prob_mscaf'].notna().to_numpy()
        yo = y[ok]

        from main import classification_metrics
        vote = classification_metrics(yo, oof.loc[ok, 'pred_mscaf_vote'].to_numpy(int), None, 2)
        results['mscaf'] = {
            'majority_vote': vote,                       # = main.py's headline
            'mean_probability': gc.binary_metrics(yo, oof.loc[ok, 'prob_mscaf']),
        }
        print("\n── Results (pooled out-of-fold, subject level) ──")
        print(f"  {'MSCAF (majority vote)':<28} bal.acc {vote['balanced_accuracy']:6.2f}% | "
              f"sens {vote['sensitivity']:6.2f}% | spec {vote['specificity']:6.2f}%")
        report('MSCAF (mean probability)', results['mscaf']['mean_probability'])

        results['fusion'] = {}
        for name in gc.model_names():
            report(f'{name} features', results['models'][name]['pooled_subject'])
            pf = 0.5 * oof.loc[ok, f'prob_{name}'] + 0.5 * oof.loc[ok, 'prob_mscaf']
            oof.loc[ok, f'prob_fused_{name}'] = pf
            m = gc.binary_metrics(yo, pf)
            results['fusion'][name] = {'weights': {'features': 0.5, 'mscaf': 0.5},
                                       'pooled_subject': m}
            report(f'MSCAF + {name} (fused)', m)
        results['n_subjects_fused'] = int(ok.sum())
    else:
        results['mscaf'] = None
        results['fusion'] = None
        print("  Fusion skipped — feature-model results only.")

    oof.to_csv(gc.results_dir() / 'feature_oof_probs.csv', index=False)
    gc.save_json(results, 'gait_features_metrics.json')
    plot_roc(oof, y, results)
    plot_metric_bars(results)


def _curves(oof, results):
    """(label, probability column) for every model that has OOF probabilities."""
    out = [(f"{gc.PRETTY[n]} (features)", f'prob_{n}') for n in gc.model_names()]
    if results.get('fusion'):
        out.insert(0, ('MSCAF-Gait', 'prob_mscaf'))
        out += [(f"MSCAF + {gc.PRETTY[n]}", f'prob_fused_{n}')
                for n in gc.model_names()]
    return [(lab, col) for lab, col in out if col in oof.columns]


def plot_roc(oof, y, results):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for (label, col), color in zip(_curves(oof, results), gc.SERIES):
        ok = oof[col].notna().to_numpy()
        fpr, tpr, _ = roc_curve(y[ok], oof.loc[ok, col])
        auc = gc.binary_metrics(y[ok], oof.loc[ok, col])['roc_auc']
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{label}  AUC {auc:.3f}')
    ax.plot([0, 1], [0, 1], ls='--', lw=1, color=gc.MUTED)
    ax.set_xlabel('False positive rate (1 − specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_title('ROC — pooled out-of-fold, subject level')
    ax.grid(axis='both')
    ax.legend(loc='lower right')
    gc.save_fig(fig, 'gait_features_roc.png')


def plot_metric_bars(results):
    rows = []
    if results.get('mscaf'):
        rows.append(('MSCAF-Gait', results['mscaf']['majority_vote']))
    for n in gc.model_names():
        rows.append((f"{gc.PRETTY[n]}\n(features)",
                     results['models'][n]['pooled_subject']))
    for n in gc.model_names() if results.get('fusion') else []:
        rows.append((f"MSCAF +\n{gc.PRETTY[n]}",
                     results['fusion'][n]['pooled_subject']))

    keys = ['balanced_accuracy', 'sensitivity', 'specificity']
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(rows)), 4.4))
    gc.grouped_bars(ax, [r[0] for r in rows],
                    ['Balanced accuracy', 'Sensitivity', 'Specificity'],
                    [[r[1][k] for k in keys] for r in rows], ylim=(0, 105))
    base = rows[0][1]['majority_baseline']
    ax.axhline(base, ls=':', color=gc.MUTED, lw=1.2, label=f'Majority baseline ({base:.1f}%)')
    ax.set_ylabel('%')
    ax.set_title('PD detection — pooled out-of-fold, subject level')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=4)
    gc.save_fig(fig, 'gait_features_metrics.png')


if __name__ == '__main__':
    main()
