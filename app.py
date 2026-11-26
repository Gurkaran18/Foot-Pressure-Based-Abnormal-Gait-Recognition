"""
app.py — Streamlit demo: gait file in, PD probability out
=========================================================

Upload a PhysioNet gaitpdb trial (.txt, 19 columns), or pick one from the
dataset, and see:
  * P(PD) from the gait-feature model (and MSCAF-Gait + fusion if the fold
    checkpoints from main.py are available)
  * the extracted gait features next to the cohort's control / PD medians
  * mean plantar force on the 16-sensor foot layout
  * the total-force traces with the detected heel strikes and toe-offs

Run:
    streamlit run app.py
  Needs results_new/ from gait_features.py (the app retrains the feature model
  in a few seconds if the saved one is missing or will not load). On Kaggle,
  expose the port with a tunnel, e.g.
      !streamlit run app.py --server.port 8501 &>/kaggle/working/app.log &
      !npx localtunnel --port 8501
  For a live presentation it is more reliable to download results_new/ and
  run it locally with DATA_ROOT pointing at the dataset.
"""

import io
import json
import os
from pathlib import Path

# On macOS, scikit-learn/XGBoost and PyTorch each bring an OpenMP runtime;
# with several threads, loading the MSCAF checkpoints then segfaults (exit 139)
# and the app hangs at start-up. One OpenMP thread avoids it, and is plenty
# for scoring one recording. Must be set before torch / sklearn are imported.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import torch  # noqa: F401  (import before gait_common, see above)
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st

import gait_common as gc


st.set_page_config(page_title='MSCAF-Gait demo', layout='wide')
gc.apply_style()


# ============================================================================
# CACHED RESOURCES
# ============================================================================

@st.cache_resource
def subjects():
    try:
        return gc.load_subjects()
    except Exception:
        return None


@st.cache_resource
def feature_model():
    """The model gait_features.py refit on all subjects, or a fresh refit."""
    for name in [gc.PRIMARY] + gc.model_names():
        path = gc.results_dir() / f'feature_model_{name}.joblib'
        if path.exists():
            try:
                return joblib.load(path)['model'], name, 'loaded'
            except Exception:
                pass
    df = subjects()
    if df is None:
        return None, None, 'no model and no dataset to train one'
    table = gc.subject_feature_table(df)
    y = table['label'].to_numpy(int)
    model = gc.make_model(gc.PRIMARY, y).fit(table[gc.FEATURE_NAMES].to_numpy(float), y)
    return model, gc.PRIMARY, 'trained now on all subjects'


@st.cache_resource
def mscaf_models():
    """The K fold checkpoints from main.py, used as an ensemble. None if absent."""
    ckpt_dir = gc.find_checkpoint_dir()
    if ckpt_dir is None:
        return None
    from main import build_model, restore_best
    models = []
    for k in range(1, gc.CONFIG['cv_folds'] + 1):
        m = build_model(gc.CONFIG, gc.CONFIG['num_classes'])
        restore_best(m, ckpt_dir / f'best_model_pd_fold{k}.pth', gc.CONFIG)
        models.append(m.eval())
    return models


@st.cache_data
def oof_table():
    """Out-of-fold probabilities, for honest scores on files from the dataset."""
    path = gc.results_dir() / 'feature_oof_probs.csv'
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def cohort_medians():
    path = gc.results_dir() / 'gait_features_subject.csv'
    if not path.exists():
        return None
    t = pd.read_csv(path)
    return t.groupby('label')[gc.FEATURE_NAMES].median()


# ============================================================================
# INFERENCE
# ============================================================================

def mscaf_prob(data, models):
    """main.py preprocessing (per-channel z-score, 10 s windows, 50% overlap)."""
    import torch
    sensors = data[:, 1:17]
    sensors = (sensors - sensors.mean(0)) / (sensors.std(0) + 1e-8)
    L, hop = gc.CONFIG['sequence_length'], gc.CONFIG['stride']
    wins = [sensors[s:s + L].T for s in range(0, len(sensors) - L, hop)]
    if not wins:
        return None
    X = torch.from_numpy(np.stack(wins)).float().to(gc.CONFIG['device'])
    with torch.no_grad():
        probs = [torch.softmax(m(X)[0], 1)[:, 1].mean().item() for m in models]
    return float(np.mean(probs))


def foot_plot(data):
    fig, ax = plt.subplots(figsize=(5, 5.6))
    sm = gc.draw_foot_heatmap(ax, data[:, 1:17].mean(0), title='Mean plantar force (N)',
                              fmt='{:.0f}', vmin=0)
    fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    return fig


def trace_plot(data, L, R, seconds=10):
    n = min(len(data), seconds * gc.FS)
    t = np.arange(n) / gc.FS
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
    for ax, col, strides, side, color in (
            (axes[0], gc.COL_TOTAL_L, L, 'Left', gc.SERIES[0]),
            (axes[1], gc.COL_TOTAL_R, R, 'Right', gc.SERIES[1])):
        f = data[:n, col]
        ax.plot(t, f, color=color, lw=1.6)
        hs = strides['hs'][strides['hs'] < n].to_numpy()
        to = strides['to'][strides['to'] < n].to_numpy()
        ax.scatter(hs / gc.FS, f[hs], marker='v', s=40, color=gc.INK, zorder=3,
                   label='heel strike')
        ax.scatter(to / gc.FS, f[to], marker='^', s=40, color=gc.MUTED, zorder=3,
                   label='toe-off')
        ax.set_ylabel(f'{side} (N)')
    axes[0].legend(loc='lower right', bbox_to_anchor=(1, 1.0), ncol=2)
    axes[1].set_xlabel('time (s)')
    axes[0].set_title(f'Total vertical force, first {seconds} s, with detected gait events',
                      loc='left')
    fig.tight_layout()
    return fig


# ============================================================================
# PHASE 2 — FROZEN EXTERNAL TEST ON WEARGAIT-PD (aggregate results only)
# ============================================================================

ETX_OUT = Path(__file__).resolve().parent / 'experiments' / 'external_transport' / 'outputs'
M1FB = ['stride_time_mean', 'stride_time_cv']


@st.cache_data
def phase2_results():
    """results.json + frozen manifest from the post-freeze run, or None if absent."""
    res = ETX_OUT / 'c_frozen' / 'hypothesis_testing' / 'results.json'
    man = ETX_OUT / 'c_frozen' / 'frozen_scores' / 'frozen_manifest.json'
    if not (res.exists() and man.exists()):
        return None, None
    return json.loads(res.read_text()), json.loads(man.read_text())


@st.cache_resource
def frozen_m1fb(manifest):
    """Refit M1-fb exactly as c_01 did (median impute, z-score, L2 LR, tuned C)
    on the 165-subject training table, and check it reproduces the frozen
    coefficients before using it."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    root = Path(__file__).resolve().parent
    train = root / manifest['train_table_path']
    if not train.exists():
        return None
    d = pd.read_csv(train)
    pm = manifest['primary_model']
    pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()),
                     ('clf', LogisticRegression(C=pm['best_C'], max_iter=5000))])
    pipe.fit(d[M1FB].to_numpy(float), d['label'].to_numpy(int))
    got = np.r_[pipe['clf'].intercept_, pipe['clf'].coef_[0]]
    want = np.r_[pm['intercept'], [pm['coefficients'][f] for f in M1FB]]
    if not np.allclose(got, want, atol=1e-6):
        return None
    return pipe, d


def render_phase2():
    res, man = phase2_results()
    if res is None:
        st.info('Phase 2 outputs not found (experiments/external_transport/outputs/c_frozen). '
                'They are local-only and never committed; run the c_01/c_02 scripts first.')
        return
    r = res['M1-fb']
    pm = man['primary_model']
    st.markdown(f"**Frozen model M1-fb** (logistic regression on stride-time mean and "
                f"stride-time variability), trained on all {pm['n_train_subjects']} gaitpdb people, "
                f"locked with git tag `{res['freeze_tag']}` before any WearGait-PD diagnosis was read, "
                f"then scored once on **{r['n']} WearGait-PD people** ({r['n_pd']} PD, {r['n_co']} controls).")
    h1, h2, h3 = r['H1'], r['H2'], r['H3']
    # Holm on (H1, H3); H2 is interpreted only if H1 passes (protocol).
    h1_ok, h3_ok = h1['p_value'] <= 0.025, h3['p_value'] <= 0.0167
    tab = pd.DataFrame({
        'Hypothesis': ['H1: external AUC better than chance?',
                       'H2: better than chance at equal walking speed?',
                       'H3: real gap between internal and external AUC?'],
        'Result': [f"AUC {h1['auc']:.3f} [{h1['ci_lower']:.3f}, {h1['ci_upper']:.3f}], p = {h1['p_value']:.2f}",
                   f"AAUC {h2['aauc']:.3f} [{h2['ci_lower']:.3f}, {h2['ci_upper']:.3f}], p = {h2['p_value']:.2f}",
                   f"gap {h3['gap']:.3f} [{h3['ci_lower']:.3f}, {h3['ci_upper']:.3f}], p = {h3['p_value']:.4f}"],
        'Supported?': ['Yes' if h1_ok else 'No',
                       'Yes' if (h1_ok and h2['p_value'] <= 0.05) else 'No',
                       'Yes' if h3_ok else 'No'],
    })
    st.dataframe(tab, hide_index=True, width='stretch')
    st.caption(f"Internal AUC on gaitpdb (short walks): {h3['internal_emulated_auc']:.3f}. "
               f"Holm-corrected; H2 is only interpreted if H1 passes. "
               f"Reading: the model works inside gaitpdb but does not transfer to WearGait-PD.")

    c1, c2 = st.columns([1, 1])
    with c1:
        roc = ETX_OUT / 'c_frozen' / 'hypothesis_testing' / 'roc_curves.png'
        if roc.exists():
            st.image(str(roc), caption='External ROC curves (frozen models)')
    with c2:
        st.markdown('**Try the frozen model**')
        fm = frozen_m1fb(man)
        if fm is None:
            st.warning('Could not rebuild the frozen model from the training table.')
        else:
            pipe, d = fm
            med = d.groupby('label')[M1FB].median()
            mean_s = st.number_input('Stride time mean (s)', 0.5, 3.0,
                                     float(round(d['stride_time_mean'].median(), 2)), 0.01)
            cv = st.number_input('Stride time variability, CV (%)', 0.0, 20.0,
                                 float(round(d['stride_time_cv'].median(), 2)), 0.1)
            p = float(pipe.predict_proba([[mean_s, cv]])[0, 1])
            st.metric('Frozen M1-fb', f'{100 * p:.0f}% PD')
            st.caption(f"gaitpdb medians: controls {med.loc[0, 'stride_time_mean']:.2f} s / "
                       f"{med.loc[0, 'stride_time_cv']:.2f}%, PD {med.loc[1, 'stride_time_mean']:.2f} s / "
                       f"{med.loc[1, 'stride_time_cv']:.2f}%. Coefficients verified against the "
                       f"frozen manifest.")


# ============================================================================
# PAGE
# ============================================================================

st.title('Parkinson\'s detection from foot pressure')
st.caption('Research prototype. Not a diagnostic device.')

tab1, tab2 = st.tabs(['Phase 1: score a gaitpdb recording',
                      'Phase 2: external test (WearGait-PD)'])
with tab2:
    render_phase2()

# Phase 1 page (st.stop() inside only ends this tab; Phase 2 is already drawn)
with tab1:

    df = subjects()
    left, right = st.columns([2, 1])
    with left:
        upload = st.file_uploader('Upload a gaitpdb trial (.txt, 19 columns)', type=['txt'])
    with right:
        example = None
        if df is not None:
            example = st.selectbox('…or pick a file from the dataset',
                                   [''] + sorted(df['filename'].tolist()))

    data, fname = None, None
    try:
        if upload is not None:
            data, fname = np.loadtxt(io.BytesIO(upload.getvalue())), upload.name.removesuffix('.txt')
        elif example:
            data, fname = gc.load_trial(gc.resolve_data_root() / f'{example}.txt'), example
    except Exception as e:
        st.error(f'Could not read the file: {e}')

    if data is None:
        st.info('Upload or pick a recording to start.')
        st.stop()
    if data.ndim != 2 or data.shape[1] < 19:
        st.error(f'Expected 19 columns (time, 16 sensors, L/R totals); got shape {data.shape}.')
        st.stop()

    L, R, fl, fr = gc.trial_strides(data)
    feats = gc.gait_features(L, R, fl, fr)
    x = np.array([[feats[f] for f in gc.FEATURE_NAMES]], dtype=float)

    model, model_name, how = feature_model()
    p_feat = float(model.predict_proba(x)[0, 1]) if model is not None else None
    mscaf = mscaf_models()
    p_mscaf = mscaf_prob(data, mscaf) if mscaf else None

    # Files from the cohort were in the refit model's training data, so show the
    # out-of-fold (held-out) probability for that subject instead.
    sid = fname.split('_')[0]
    oof = oof_table()
    held_out = None
    if oof is not None and sid in set(oof['subject_id']):
        held_out = oof.set_index('subject_id').loc[sid]
        if f'prob_{model_name}' in held_out:
            p_feat = float(held_out[f'prob_{model_name}'])
        if 'prob_mscaf' in held_out and pd.notna(held_out['prob_mscaf']):
            p_mscaf = float(held_out['prob_mscaf'])

    # ── Scores ───────────────────────────────────────────────────────────────
    st.subheader(f'Recording {fname}  ·  {len(data) / gc.FS:.0f} s  ·  '
                 f'{feats["n_strides"]} clean strides')
    if held_out is not None:
        truth = 'PD' if held_out['true'] == 1 else 'Control'
        st.caption(f'Subject {sid} is in the cohort (ground truth: **{truth}**). Scores shown are '
                   f'its out-of-fold predictions, from models that never saw this subject.')

    cols = st.columns(3)
    cols[0].metric(f'Gait features ({gc.PRETTY.get(model_name, "n/a")})',
                   '—' if p_feat is None else f'{100 * p_feat:.0f}% PD')
    cols[1].metric('MSCAF-Gait', '—' if p_mscaf is None else f'{100 * p_mscaf:.0f}% PD',
                   help=None if p_mscaf is not None else 'fold checkpoints not found')
    if p_feat is not None and p_mscaf is not None:
        p_fused = 0.5 * (p_feat + p_mscaf)
        cols[2].metric('Fused (mean)', f'{100 * p_fused:.0f}% PD')
        final = p_fused
    else:
        final = p_feat if p_feat is not None else p_mscaf
    if final is not None:
        verdict = 'PD-like gait' if final >= 0.5 else 'Control-like gait'
        st.progress(min(max(final, 0.0), 1.0), text=f'{verdict}  (threshold 50%)')

    # ── Features, foot map, traces ───────────────────────────────────────────
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown('**Gait features**')
        tab = pd.DataFrame({'Feature': [gc.FEATURE_LABELS[f] for f in gc.FEATURE_NAMES],
                            'This recording': [feats[f] for f in gc.FEATURE_NAMES]})
        med = cohort_medians()
        if med is not None:
            tab['Control median'] = med.loc[0].to_numpy()
            tab['PD median'] = med.loc[1].to_numpy()
        st.dataframe(tab.style.format(precision=2), hide_index=True)
    with c2:
        st.pyplot(foot_plot(data))

    st.pyplot(trace_plot(data, L, R))
    st.caption(f'Feature model: {how}. Event detection: contact = total force above '
               f'{int(gc.CONTACT_FRACTION * 100)}% of the swing-to-peak range; strides outside '
               f'{gc.STRIDE_RANGE_S[0]}–{gc.STRIDE_RANGE_S[1]} s or > {gc.OUTLIER_MADS:.0f} robust SD '
               f'from the median (turns) are excluded.')
