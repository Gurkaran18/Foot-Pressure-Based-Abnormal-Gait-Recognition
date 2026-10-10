"""
results_summary.py — one slide-ready table from every results JSON
==================================================================

Reads (whatever exists):
    results_pd/cv_results.json            main.py, MSCAF-Gait 5-fold CV
    results_new/gait_features_metrics.json
    results_new/leakage_demo.json
    results_new/cross_study.json
    results_new/shap_importance.json
    results_new/occlusion_importance.json

Writes to /kaggle/working/results_new/:
    results_summary.md    Markdown tables (paste into slides / README)
    results_summary.csv   main table as CSV
    results_summary.png   main table as an image
    results_summary.json

Run last:   !python results_summary.py
"""

import json

import pandas as pd
import matplotlib.pyplot as plt

import gait_common as gc


def _load(name):
    path = gc.results_dir() / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def to_markdown(df):
    """Markdown table without the optional `tabulate` dependency."""
    def cell(v, col):
        if isinstance(v, float):
            return '—' if pd.isna(v) else (f'{v:.3f}' if 'AUC' in col else f'{v:.2f}')
        return str(v)
    lines = ['| ' + ' | '.join(df.columns) + ' |',
             '|' + '|'.join('---' for _ in df.columns) + '|']
    lines += ['| ' + ' | '.join(cell(v, c) for v, c in zip(row, df.columns)) + ' |'
              for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def _row(model, m, note=''):
    auc = m.get('roc_auc')
    return {'Model': model,
            'Balanced acc. (%)': m['balanced_accuracy'],
            'Sensitivity (%)': m['sensitivity'],
            'Specificity (%)': m['specificity'],
            'ROC-AUC': auc if auc is not None else float('nan'),
            'Note': note}


def main():
    gc.apply_style()
    feats = _load('gait_features_metrics.json')
    leak = _load('leakage_demo.json')
    cross = _load('cross_study.json')
    shap_res = _load('shap_importance.json')
    occl = _load('occlusion_importance.json')

    cv_path = gc.find_cv_results()
    cv = json.load(open(cv_path)) if cv_path else None

    # ── Main table: pooled out-of-fold, subject level ────────────────────
    rows = []
    mscaf = (cv or {}).get('pooled_out_of_fold_subject')
    if mscaf is None and feats and feats.get('mscaf'):
        mscaf = feats['mscaf']['majority_vote']
    n = (mscaf or {}).get('n') or (feats or {}).get('n_subjects')
    base = (mscaf or (feats['models'][gc.PRIMARY]['pooled_subject'] if feats else {})).get(
        'majority_baseline')
    if base is not None:
        rows.append({'Model': 'Majority-class baseline', 'Balanced acc. (%)': 50.0,
                     'Sensitivity (%)': float('nan'), 'Specificity (%)': float('nan'),
                     'ROC-AUC': 0.5, 'Note': f'accuracy {base:.2f}%'})
    if mscaf:
        rows.append(_row('MSCAF-Gait (deep, raw 16 sensors)', mscaf, 'majority vote'))
    if feats:
        for name, res in feats['models'].items():
            rows.append(_row(f'{gc.PRETTY[name]} (13 gait features)', res['pooled_subject'],
                             'primary' if name == feats.get('primary_model') else ''))
        for name, res in (feats.get('fusion') or {}).items():
            rows.append(_row(f'MSCAF + {gc.PRETTY[name]} (late fusion)', res['pooled_subject'],
                             'mean of probabilities'))
    main_tab = pd.DataFrame(rows)

    md = [f"## PD detection — 5-fold stratified subject-level CV (pooled out-of-fold, "
          f"{n} subjects)\n"]
    if not main_tab.empty:
        md.append(to_markdown(main_tab))
        main_tab.to_csv(gc.results_dir() / 'results_summary.csv', index=False)
    if feats and not feats.get('fusion'):
        md.append("\n_Fusion rows missing: run main.py (or attach its checkpoints) and "
                  "re-run gait_features.py._")

    # ── Robustness ───────────────────────────────────────────────────────
    if leak:
        a, b = leak['random_window_split'], leak['subject_level_split']
        md.append(f"\n## Data leakage — {gc.PRETTY[leak['model']]}, window level "
                  f"({leak['n_windows']} windows)\n")
        md.append(to_markdown(pd.DataFrame([
            {'Split': 'Random window-level (leaky)', 'Accuracy (%)': a['accuracy'],
             'Balanced acc. (%)': a['balanced_accuracy'], 'ROC-AUC': a['roc_auc']},
            {'Split': 'Subject-level (honest)', 'Accuracy (%)': b['accuracy'],
             'Balanced acc. (%)': b['balanced_accuracy'], 'ROC-AUC': b['roc_auc']},
        ])))
        md.append(f"\nInflation from leakage: **+{leak['inflation_balanced_accuracy_points']:.1f} "
                  f"balanced-accuracy points**.")

    if cross:
        md.append(f"\n## Cross-study generalisation — leave-one-study-out, "
                  f"{gc.PRETTY[cross['model']]}\n")
        crows = []
        for st in ('Ga', 'Ju', 'Si', 'mean'):
            r = cross['variants']['raw'][st]
            crows.append({'Held-out study': st if st != 'mean' else 'Mean',
                          'Balanced acc. (%)': r['balanced_accuracy'],
                          'Sensitivity (%)': r['sensitivity'],
                          'Specificity (%)': r['specificity'], 'ROC-AUC': r['roc_auc']})
        md.append(to_markdown(pd.DataFrame(crows)))

    # ── Explainability ───────────────────────────────────────────────────
    if shap_res or occl:
        md.append("\n## What the models look at\n")
    if shap_res:
        top = list(shap_res['mean_abs_shap'])[:3]
        md.append("- Top gait features (mean |SHAP|): " +
                  ", ".join(gc.FEATURE_LABELS[f] for f in top))
    if occl:
        md.append("- Most important MSCAF sensors (occlusion): " + ", ".join(occl['ranking'][:4]))

    text = "\n".join(md) + "\n"
    (gc.results_dir() / 'results_summary.md').write_text(text)
    print(text)
    print(f"  Saved: {gc.results_dir() / 'results_summary.md'}")
    gc.save_json({'main_table': rows, 'leakage': leak, 'cross_study': cross},
                 'results_summary.json')

    if not main_tab.empty:
        table_png(main_tab, n)


def table_png(tab, n):
    show = tab.drop(columns=['Note']).copy()
    for c in show.columns[1:]:
        fmt = '{:.3f}' if c == 'ROC-AUC' else '{:.1f}'
        show[c] = [('—' if pd.isna(v) else fmt.format(v)) for v in show[c]]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(show) + 1.2))
    ax.axis('off')
    t = ax.table(cellText=show.values, colLabels=show.columns, loc='center',
                 cellLoc='center', colLoc='center')
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.6)
    best = tab['Balanced acc. (%)'].iloc[1:].idxmax() if len(tab) > 1 else None
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(gc.GRID)
        if r == 0:
            cell.set_text_props(weight='bold', color=gc.INK)
            cell.set_facecolor('#f0efec')
        else:
            cell.set_facecolor(gc.SURFACE)
            cell.set_text_props(color=gc.INK_2 if r - 1 == 0 else gc.INK)
            if best is not None and r - 1 == best:
                cell.set_text_props(weight='bold', color=gc.INK)
                cell.set_facecolor('#cde2fb')
        if c == 0:
            cell.set_text_props(ha='left')
            cell._loc = 'left'
    t.auto_set_column_width(list(range(len(show.columns))))
    ax.set_title(f'PD detection — 5-fold subject-level CV, pooled out-of-fold ({n} subjects)',
                 fontweight='bold', pad=4)
    gc.save_fig(fig, 'results_summary.png')


if __name__ == '__main__':
    main()
