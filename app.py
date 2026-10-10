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
# PAGE
# ============================================================================

st.title('MSCAF-Gait — Parkinson\'s detection from foot pressure')
st.caption('Research prototype. Not a diagnostic device.')

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
