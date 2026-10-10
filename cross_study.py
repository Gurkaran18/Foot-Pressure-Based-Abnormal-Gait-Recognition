"""
cross_study.py — leave-one-study-out generalisation
===================================================

The cohort pools three studies (filename prefix):
    Ga — Yogev et al. 2005   (overground walking, incl. dual-task trials)
    Ju — Hausdorff et al. 2007 (overground, rhythmic auditory stimulation)
    Si — Frenkel-Toledo et al. 2005 (treadmill walking)

Train the gait-feature model on two studies, test on the third. This asks
whether the model learned PD gait or each lab's protocol.

Two variants:
  raw         — features as computed
  per-study z — each study's features standardised with that study's own
                mean/SD (unsupervised; uses no labels from the test study),
                which removes protocol-level offsets such as treadmill vs
                overground cadence

Run (Kaggle):   !python cross_study.py
Outputs      :  /kaggle/working/results_new/
    cross_study.json, cross_study.png
"""

import numpy as np
import matplotlib.pyplot as plt

import gait_common as gc

STUDIES = ['Ga', 'Ju', 'Si']


def per_study_z(table):
    t = table.copy()
    for st in STUDIES:
        m = t['study'] == st
        mu = t.loc[m, gc.FEATURE_NAMES].mean()
        sd = t.loc[m, gc.FEATURE_NAMES].std().replace(0, 1)
        t.loc[m, gc.FEATURE_NAMES] = (t.loc[m, gc.FEATURE_NAMES] - mu) / sd
    return t


def main():
    gc.apply_style()
    print("=" * 60)
    print("CROSS-STUDY — leave-one-study-out (Ga / Ju / Si)")
    print("=" * 60)

    subjects_df = gc.load_subjects()
    table = gc.subject_feature_table(subjects_df)
    name = gc.PRIMARY
    print(f"Model: {gc.PRETTY[name]}")
    for st in STUDIES:
        t = table[table['study'] == st]
        print(f"  {st}: {len(t)} subjects ({int(t['label'].sum())} PD / "
              f"{int((t['label'] == 0).sum())} control)")

    results = {'model': name, 'variants': {}}
    for variant, tab in (('raw', table), ('per_study_z', per_study_z(table))):
        X = tab[gc.FEATURE_NAMES].to_numpy(float)
        y = tab['label'].to_numpy(int)
        study = tab['study'].to_numpy()
        res = {}
        print(f"\n── {variant} ─────────────────────────────────────")
        for held in STUDIES:
            tr, te = study != held, study == held
            model = gc.make_model(name, y[tr]).fit(X[tr], y[tr])
            m = gc.binary_metrics(y[te], model.predict_proba(X[te])[:, 1])
            res[held] = m
            print(f"  train {'+'.join(s for s in STUDIES if s != held)} → test {held}: "
                  f"bal.acc {m['balanced_accuracy']:6.2f}% | sens {m['sensitivity']:6.2f}% | "
                  f"spec {m['specificity']:6.2f}% | AUC {m['roc_auc']:.4f}")
        res['mean'] = {k: float(np.mean([res[s][k] for s in STUDIES]))
                       for k in ('balanced_accuracy', 'sensitivity', 'specificity', 'roc_auc')}
        print(f"  mean over held-out studies: bal.acc {res['mean']['balanced_accuracy']:.2f}% | "
              f"AUC {res['mean']['roc_auc']:.4f}")
        results['variants'][variant] = res

    # Reference: the within-cohort subject-level CV of the same model
    ref = None
    path = gc.results_dir() / 'gait_features_metrics.json'
    if path.exists():
        import json
        with open(path) as f:
            ref = json.load(f)['models'][name]['pooled_subject']['balanced_accuracy']
        results['reference_subject_cv_balanced_accuracy'] = ref
    gc.save_json(results, 'cross_study.json')

    # ── Plot: balanced accuracy per held-out study ───────────────────────
    groups = [f'Test on {s}' for s in STUDIES] + ['Mean']
    vals = [[results['variants'][v][s]['balanced_accuracy'] for v in ('raw', 'per_study_z')]
            for s in STUDIES + ['mean']]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    gc.grouped_bars(ax, groups, ['Raw features', 'Per-study standardised'], vals, ylim=(0, 105))
    ax.axhline(50, ls=':', color=gc.MUTED, lw=1.2, label='Chance (50%)')
    if ref is not None:
        ax.axhline(ref, ls='--', color=gc.INK_2, lw=1.2,
                   label=f'Pooled 5-fold subject CV ({ref:.1f}%)')
    ax.set_ylabel('Balanced accuracy (%)')
    ax.set_title(f'Leave-one-study-out — {gc.PRETTY[name]} on gait features')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    gc.save_fig(fig, 'cross_study.png')


if __name__ == '__main__':
    main()
