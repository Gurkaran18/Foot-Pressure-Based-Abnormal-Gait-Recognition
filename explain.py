"""
explain.py — what the two models look at
========================================

1. SHAP for the gait-feature model (gait_common.PRIMARY). Out-of-fold: each
   fold's model, trained on that fold's training subjects, explains only that
   fold's test subjects, using main.py's folds. Beeswarm + mean |SHAP| bar.
2. Sensor-occlusion importance for MSCAF-Gait. For each fold's best checkpoint
   (from main.py's CV run) and that fold's held-out test windows, one sensor
   channel at a time is replaced by 0 — its per-recording mean after main.py's
   z-scoring, i.e. "this sensor carries no signal" — and the drop in the
   model's probability for the TRUE class is measured. Drawn as a heatmap on
   the 16-sensor insole layout (8 per foot).

Run (Kaggle):   !python explain.py      (after main.py and gait_features.py)
Outputs      :  /kaggle/working/results_new/
    shap_summary.png, shap_bar.png, shap_importance.json
    occlusion_foot_heatmap.png, occlusion_importance.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gait_common as gc


# ============================================================================
# 1. SHAP — feature model
# ============================================================================

def shap_explain(subjects_df):
    try:
        import shap
    except ImportError:
        print("  ⚠️  shap not installed (pip install shap) — skipping SHAP.")
        return None

    table = gc.subject_feature_table(subjects_df)
    X = table[gc.FEATURE_NAMES].to_numpy(float)
    y = table['label'].to_numpy(int)
    sids = table['subject_id'].to_numpy()
    name = gc.PRIMARY

    sv_all = np.zeros_like(X)
    X_shown = np.zeros_like(X)
    for fit_ids, test_ids in gc.subject_folds(subjects_df):
        tr, te = np.isin(sids, fit_ids), np.isin(sids, test_ids)
        model = gc.make_model(name, y[tr]).fit(X[tr], y[tr])
        Xt = model.named_steps['impute'].transform(X[te])
        sv = shap.TreeExplainer(model.named_steps['clf']).shap_values(Xt)
        if isinstance(sv, list):          # older shap, RF: one array per class
            sv = sv[1]
        elif np.ndim(sv) == 3:            # newer shap, RF: (n, features, classes)
            sv = sv[..., 1]
        sv_all[te], X_shown[te] = sv, Xt

    labels = [gc.FEATURE_LABELS[f] for f in gc.FEATURE_NAMES]
    X_df = pd.DataFrame(X_shown, columns=labels)
    unit = 'log-odds of PD' if name == 'xgboost' else 'P(PD)'

    shap.summary_plot(sv_all, X_df, show=False, plot_size=(8, 5.5))
    fig = plt.gcf()
    fig.axes[0].set_title(f'SHAP — {gc.PRETTY[name]} on gait features (out-of-fold)')
    fig.axes[0].set_xlabel(f'SHAP value (impact on {unit}; > 0 pushes towards PD)')
    gc.save_fig(fig, 'shap_summary.png')

    mean_abs = np.abs(sv_all).mean(axis=0)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(labels)[order], mean_abs[order], color=gc.SERIES[0], height=0.6)
    for i, v in enumerate(mean_abs[order]):
        ax.text(v, i, f' {v:.3f}', va='center', fontsize=8, color=gc.INK_2)
    ax.grid(axis='x')
    ax.grid(axis='y', visible=False)
    ax.set_xlabel(f'mean |SHAP| ({unit})')
    ax.set_title('Which gait features drive the PD prediction')
    gc.save_fig(fig, 'shap_bar.png')

    ranking = sorted(zip(gc.FEATURE_NAMES, mean_abs.tolist()), key=lambda t: -t[1])
    print("  Top features by mean |SHAP|:")
    for f, v in ranking[:5]:
        print(f"    {gc.FEATURE_LABELS[f]:<26} {v:.4f}")
    return {'model': name, 'unit': unit, 'out_of_fold': True,
            'mean_abs_shap': dict(ranking)}


# ============================================================================
# 2. Sensor occlusion — MSCAF
# ============================================================================

def _probs(model, X, device, batch=128):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            logits, _ = model(X[i:i + batch].to(device))
            out.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(out).numpy()


def occlusion(subjects_df):
    import torch
    from sklearn.metrics import roc_auc_score

    ckpt_dir = gc.find_checkpoint_dir()
    if ckpt_dir is None:
        print("  ⚠️  No MSCAF fold checkpoints (best_model_pd_fold*.pth) found in\n"
              "      checkpoints_pd/ or /kaggle/input — skipping occlusion. Run main.py.")
        return None

    device = gc.CONFIG['device']
    n_sensors = gc.CONFIG['in_channels']
    drop_sum = np.zeros(n_sensors)
    all_y, all_p0, all_ps = [], [], [[] for _ in range(n_sensors)]

    for k, model, test_ds in gc.mscaf_fold_models(subjects_df, ckpt_dir):
        X = torch.from_numpy(np.stack([s['data'] for s in test_ds.sequences])).float()
        y = np.array([s['label'] for s in test_ds.sequences])
        p0 = _probs(model, X, device)
        t0 = np.where(y == 1, p0, 1 - p0)            # probability of the true class
        for s in range(n_sensors):
            Xo = X.clone()
            Xo[:, s, :] = 0.0
            ps = _probs(model, Xo, device)
            drop_sum[s] += (t0 - np.where(y == 1, ps, 1 - ps)).sum()
            all_ps[s].append(ps)
        all_y.append(y)
        all_p0.append(p0)
        print(f"  fold {k}: {len(y)} test windows occluded × {n_sensors} sensors")

    y = np.concatenate(all_y)
    p0 = np.concatenate(all_p0)
    base_auc = roc_auc_score(y, p0)
    drop = drop_sum / len(y) * 100                   # percentage points
    auc_drop = np.array([base_auc - roc_auc_score(y, np.concatenate(ps)) for ps in all_ps])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax, vals, title, fmt in (
            (axes[0], drop, 'Drop in true-class probability (pp)', '{:.1f}'),
            (axes[1], auc_drop, 'Drop in window ROC-AUC', '{:.3f}')):
        sm = gc.draw_foot_heatmap(ax, vals, title=title, fmt=fmt, vmin=min(0, vals.min()))
        fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    fig.suptitle('MSCAF-Gait sensor-occlusion importance (out-of-fold, all 5 folds)',
                 fontweight='bold', y=1.0)
    fig.subplots_adjust(top=0.9, wspace=0.05)
    gc.save_fig(fig, 'occlusion_foot_heatmap.png')

    order = np.argsort(-drop)
    print(f"  Baseline window AUC {base_auc:.4f}. Most important sensors:")
    for i in order[:5]:
        print(f"    {gc.SENSOR_NAMES[i]}  −{drop[i]:.2f} pp true-class prob, "
              f"−{auc_drop[i]:.4f} AUC")
    return {
        'method': 'zero one z-scored sensor channel on held-out test windows',
        'n_windows': int(len(y)), 'baseline_window_auc': float(base_auc),
        'sensors': {n: {'xy': gc.SENSOR_XY[n],
                        'true_class_prob_drop_pp': float(drop[i]),
                        'auc_drop': float(auc_drop[i])}
                    for i, n in enumerate(gc.SENSOR_NAMES)},
        'ranking': [gc.SENSOR_NAMES[i] for i in order],
    }


def main():
    gc.apply_style()
    print("=" * 60)
    print("EXPLAINABILITY — SHAP (features) + sensor occlusion (MSCAF)")
    print("=" * 60)
    subjects_df = gc.load_subjects()

    print("\n── SHAP ─────────────────────────────────────────")
    res = shap_explain(subjects_df)
    if res is not None:
        gc.save_json(res, 'shap_importance.json')

    gc.apply_style()          # shap.summary_plot changes rcParams
    print("\n── Sensor occlusion ─────────────────────────────")
    res = occlusion(subjects_df)
    if res is not None:
        gc.save_json(res, 'occlusion_importance.json')


if __name__ == '__main__':
    main()
