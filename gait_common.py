"""
gait_common.py — shared helpers for the post-hoc analysis scripts
==================================================================

Used by gait_features.py, leakage_demo.py, explain.py, cross_study.py and
app.py. Everything that has to stay identical across those scripts lives here:

  * paths      — the dataset root main.py uses, and the Kaggle output folder
  * folds      — the SAME 5-fold stratified subject-level folds as main.py
                 (main.make_cv_folds, same CONFIG, same seed)
  * gait cycle — heel-strike / toe-off detection from the L/R total-force
                 columns, and the stride-based gait features
  * models     — the RandomForest / XGBoost feature classifiers
  * MSCAF      — out-of-fold per-subject probabilities from main.py's run
  * plotting   — one shared style and the 16-sensor foot layout

Paths
-----
Data   : CONFIG['data_root'] from main.py (set with the DATA_ROOT env var). If
         that folder holds no trial files, the first folder under /kaggle/input
         that contains GaCo01_01.txt is used instead.
Outputs: /kaggle/working/results_new/ on Kaggle (./results_new/ elsewhere).
"""

import os
import json
from pathlib import Path

# xgboost must be imported BEFORE torch (pulled in by main.py): on macOS the
# two ship different OpenMP runtimes and fitting segfaults the other way round.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:          # Kaggle ships xgboost; RandomForest still runs without it
    HAS_XGB = False

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless — safe on Kaggle and in Streamlit
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from main import (
    CONFIG,
    load_subject_info,
    make_cv_folds,
    classification_metrics,
    _json_safe,
)


# ============================================================================
# PATHS
# ============================================================================

KAGGLE_INPUT   = Path('/kaggle/input')
KAGGLE_WORKING = Path('/kaggle/working')
ON_KAGGLE      = KAGGLE_WORKING.exists()

RESULTS_DIR = Path(os.environ.get(
    'RESULTS_NEW_DIR',
    (KAGGLE_WORKING if ON_KAGGLE else Path('.')) / 'results_new'))

SEED = CONFIG['seed']
FS   = CONFIG['sampling_rate']        # 100 Hz


def _has_trials(path):
    return Path(path).is_dir() and any(Path(path).glob('GaCo01_*.txt'))


def resolve_data_root():
    """
    The dataset root main.py uses (CONFIG['data_root'], i.e. DATA_ROOT).
    Falls back to searching /kaggle/input when that path holds no trial files,
    and writes the resolved path back into CONFIG so main.py's own dataset
    builders read from the same place.
    """
    root = Path(CONFIG['data_root'])
    if not _has_trials(root):
        hits = sorted(KAGGLE_INPUT.glob('**/GaCo01_01.txt')) if KAGGLE_INPUT.exists() else []
        if not hits:
            raise FileNotFoundError(
                f"No gaitpdb trial files under CONFIG['data_root'] = {root} "
                f"or /kaggle/input. Set DATA_ROOT to the dataset folder.")
        root = hits[0].parent
    CONFIG['data_root'] = str(root)
    return root


def results_dir():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_json(obj, name):
    path = results_dir() / name
    with open(path, 'w') as f:
        json.dump(_json_safe(obj), f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def load_subjects():
    """main.load_subject_info() on the resolved data root, plus the study prefix."""
    root = resolve_data_root()
    df = load_subject_info(root)
    if df.empty:
        raise RuntimeError(f"load_subject_info() found no subjects under {root}")
    df['study'] = df['subject_id'].str[:2]          # 'Ga' | 'Ju' | 'Si'
    return df


# ============================================================================
# FOLDS — identical to main.py
# ============================================================================

def subject_folds(subjects_df):
    """
    main.py's folds, exactly: main.make_cv_folds() with the same CONFIG (seed,
    cv_folds, val_size). Returns a list of (fit_ids, test_ids).

    main.py splits each fold's non-test subjects into train + val because it
    needs a validation set for early stopping. The feature models here have no
    early stopping, so they fit on train + val together. The TEST fold — the
    subjects that are scored — is identical to main.py's, which is what makes
    the late fusion and the side-by-side comparison valid.
    """
    folds = make_cv_folds(subjects_df, CONFIG)
    return [(np.concatenate([tr, va]), te) for tr, va, te in folds]


# ============================================================================
# GAIT-CYCLE DETECTION FROM THE TOTAL-FORCE COLUMNS
# ============================================================================
#
# gaitpdb trial files have 19 columns (see format.txt in the dataset):
#   0 time | 1-8 left sensors L1..L8 | 9-16 right sensors R1..R8
#   17 total force left | 18 total force right      (Newton, 100 Hz)

COL_TOTAL_L, COL_TOTAL_R = 17, 18

SMOOTH_SAMPLES   = 5       # 50 ms moving average before thresholding
CONTACT_FRACTION = 0.10    # contact = force above 10% of the swing-to-peak range
MIN_PHASE_S      = 0.10    # drop contacts / gaps shorter than 100 ms (noise)
STRIDE_RANGE_S   = (0.5, 3.0)   # physiologically plausible stride times
OUTLIER_MADS     = 3.0     # drop strides > 3 robust SDs from the median (turns)


def load_trial(path):
    """A gaitpdb trial file as an (N, 19) float array."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 19:
        raise ValueError(f"{path}: expected 19 columns, got shape {data.shape}")
    return data


def _drop_short_runs(mask, min_len):
    """Flip on/off runs shorter than min_len samples (debounces threshold chatter)."""
    mask = mask.copy()
    for target in (True, False):        # remove short contacts, then short gaps
        edges = np.flatnonzero(np.diff(np.r_[~target, mask == target, ~target].astype(int)))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start < min_len and start > 0 and stop < len(mask):
                mask[start:stop] = not target
    return mask


def contact_mask(force, fs=FS):
    """Boolean stance mask for one foot's total vertical force."""
    f = np.convolve(force, np.ones(SMOOTH_SAMPLES) / SMOOTH_SAMPLES, mode='same')
    lo, hi = np.percentile(f, [5, 95])          # swing baseline, typical peak
    loaded = f > lo + CONTACT_FRACTION * (hi - lo)
    return _drop_short_runs(loaded, int(MIN_PHASE_S * fs))


def detect_strides(force, fs=FS):
    """
    Heel strikes (contact onsets) and toe-offs (contact ends) for one foot, and
    one row per stride:  heel strike -> toe-off -> next heel strike.

    Columns: hs, to, next_hs (sample indices), stride, stance, swing (seconds),
    peak (N, max force during that stance) and valid (plausible duration and
    not an outlier, e.g. a turn at the end of the walkway).
    """
    mask = contact_mask(force, fs)
    d = np.diff(mask.astype(int))
    hs = np.flatnonzero(d == 1) + 1
    to = np.flatnonzero(d == -1) + 1

    rows = []
    for a, b in zip(hs[:-1], hs[1:]):
        t = to[(to > a) & (to < b)]
        if len(t) != 1:              # zero or several toe-offs: not a clean cycle
            continue
        t = t[0]
        rows.append((a, t, b, (b - a) / fs, (t - a) / fs, (b - t) / fs,
                     float(force[a:t].max())))
    st = pd.DataFrame(rows, columns=['hs', 'to', 'next_hs', 'stride',
                                     'stance', 'swing', 'peak'])
    if st.empty:
        st['valid'] = pd.Series(dtype=bool)
        return st

    ok = st['stride'].between(*STRIDE_RANGE_S)
    if ok.sum() >= 3:
        med = st.loc[ok, 'stride'].median()
        mad = 1.4826 * np.median(np.abs(st.loc[ok, 'stride'] - med)) + 1e-6
        ok &= (st['stride'] - med).abs() <= OUTLIER_MADS * mad
    st['valid'] = ok
    return st


def trial_strides(data, fs=FS):
    """(left strides, right strides, left force, right force) for one trial."""
    fl, fr = data[:, COL_TOTAL_L], data[:, COL_TOTAL_R]
    return detect_strides(fl, fs), detect_strides(fr, fs), fl, fr


# ============================================================================
# GAIT FEATURES
# ============================================================================

FEATURE_NAMES = [
    'stride_time_mean',     # s
    'stride_time_cv',       # %   — stride-to-stride variability, a classic PD marker
    'swing_time_cv',        # %
    'stance_pct',           # % of the gait cycle on the ground
    'swing_pct',            # % of the gait cycle in the air
    'double_support_pct',   # % of the window with both feet loaded
    'cadence',              # steps / min
    'asym_stride',          # %   |L - R| / mean(L, R)
    'asym_swing',           # %   gait asymmetry, 100 * |ln(swing_R / swing_L)|
    'asym_peak',            # %
    'peak_force_L',         # N
    'peak_force_R',         # N
    'peak_force_norm',      # peak / mean total force  (≈ body-weight units)
]

# Slide-friendly names for plots
FEATURE_LABELS = {
    'stride_time_mean':   'Stride time (s)',
    'stride_time_cv':     'Stride time CV (%)',
    'swing_time_cv':      'Swing time CV (%)',
    'stance_pct':         'Stance (% cycle)',
    'swing_pct':          'Swing (% cycle)',
    'double_support_pct': 'Double support (%)',
    'cadence':            'Cadence (steps/min)',
    'asym_stride':        'Stride asymmetry (%)',
    'asym_swing':         'Swing asymmetry (%)',
    'asym_peak':          'Peak-force asymmetry (%)',
    'peak_force_L':       'Peak force L (N)',
    'peak_force_R':       'Peak force R (N)',
    'peak_force_norm':    'Peak force (× body wt)',
}

MIN_STRIDES = 3            # per foot; fewer than this and the window is NaN


def _cv(x):
    return float(100 * np.std(x) / np.mean(x)) if len(x) >= 2 and np.mean(x) > 0 else np.nan


def _asym(a, b):
    return float(100 * abs(a - b) / (0.5 * (a + b))) if (a + b) > 0 else np.nan


def gait_features(left, right, fl, fr, start=0, end=None, fs=FS):
    """
    Gait features from the strides whose whole cycle lies in [start, end).

    `left` / `right` are detect_strides() tables computed once over the whole
    trial, so a 10-second window sees exactly the same events the full
    recording does — the window boundary never creates a fake heel strike.
    Returns a dict keyed by FEATURE_NAMES (NaN when too few strides).
    """
    end = len(fl) if end is None else end
    sel = [s[(s['valid']) & (s['hs'] >= start) & (s['next_hs'] <= end)] for s in (left, right)]
    feats = dict.fromkeys(FEATURE_NAMES, np.nan)
    feats['n_strides'] = int(sum(len(s) for s in sel))
    if min(len(s) for s in sel) < MIN_STRIDES:
        return feats
    L, R = sel

    stride = np.r_[L['stride'], R['stride']]
    stance_pct = 100 * np.r_[L['stance'] / L['stride'], R['stance'] / R['stride']]
    swing_L, swing_R = L['swing'].mean(), R['swing'].mean()
    peak_L, peak_R = L['peak'].mean(), R['peak'].mean()

    # Both feet loaded at once, over the stretch actually covered by strides
    a = int(min(L['hs'].min(), R['hs'].min()))
    b = int(max(L['next_hs'].max(), R['next_hs'].max()))
    both = contact_mask(fl[a:b], fs) & contact_mask(fr[a:b], fs)

    # Over whole gait cycles the vertical force averages to body weight, so
    # mean(L + R) is a body-weight proxy that makes peak force scale-free.
    body = float(np.mean(fl[a:b] + fr[a:b]))

    feats.update({
        'stride_time_mean':   float(stride.mean()),
        'stride_time_cv':     float(np.mean([_cv(L['stride']), _cv(R['stride'])])),
        'swing_time_cv':      float(np.mean([_cv(L['swing']), _cv(R['swing'])])),
        'stance_pct':         float(stance_pct.mean()),
        'swing_pct':          float(100 - stance_pct.mean()),
        'double_support_pct': float(100 * both.mean()),
        'cadence':            float(120.0 / stride.mean()),      # 2 steps per stride
        'asym_stride':        _asym(L['stride'].mean(), R['stride'].mean()),
        'asym_swing':         float(100 * abs(np.log(swing_R / swing_L))) if swing_L > 0 and swing_R > 0 else np.nan,
        'asym_peak':          _asym(peak_L, peak_R),
        'peak_force_L':       float(peak_L),
        'peak_force_R':       float(peak_R),
        'peak_force_norm':    float(0.5 * (peak_L + peak_R) / body) if body > 0 else np.nan,
    })
    return feats


def subject_feature_table(subjects_df, cache=True):
    """
    One row of gait features per SUBJECT: features are computed per trial file
    over the whole recording, then averaged over that subject's trials.
    Cached as results_new/gait_features_subject.csv.
    """
    path = results_dir() / 'gait_features_subject.csv'
    if cache and path.exists():
        cached = pd.read_csv(path)
        if set(cached['subject_id']) == set(subjects_df['subject_id']):
            return cached

    root = resolve_data_root()
    per_file = []
    for _, row in subjects_df.iterrows():
        try:
            L, R, fl, fr = trial_strides(load_trial(root / f"{row['filename']}.txt"))
        except Exception as e:
            print(f"  Warning: skipping {row['filename']}: {e}")
            continue
        per_file.append({'subject_id': row['subject_id'], **gait_features(L, R, fl, fr)})

    pf = pd.DataFrame(per_file)
    meta = subjects_df.drop_duplicates('subject_id')[['subject_id', 'label', 'study']]
    table = meta.merge(pf.groupby('subject_id', as_index=False).mean(numeric_only=True),
                       on='subject_id', how='left')
    table.to_csv(path, index=False)
    print(f"  Saved: {path}  ({len(table)} subjects, {len(pf)} trial files)")
    return table


def window_feature_table(subjects_df, seq_len=None, stride=None):
    """
    Gait features per 10-second WINDOW, using main.py's windowing exactly
    (CONFIG['sequence_length'] samples, CONFIG['stride'] hop, same range()).
    Windows with fewer than MIN_STRIDES clean strides per foot are dropped.
    """
    seq_len = seq_len or CONFIG['sequence_length']
    stride = stride or CONFIG['stride']
    root = resolve_data_root()
    rows = []
    for _, row in subjects_df.iterrows():
        try:
            L, R, fl, fr = trial_strides(load_trial(root / f"{row['filename']}.txt"))
        except Exception as e:
            print(f"  Warning: skipping {row['filename']}: {e}")
            continue
        for start in range(0, len(fl) - seq_len, stride):
            f = gait_features(L, R, fl, fr, start, start + seq_len)
            if np.isfinite(f['stride_time_mean']):
                rows.append({'subject_id': row['subject_id'], 'label': row['label'],
                             'study': row['study'], 'filename': row['filename'],
                             'start': start, **f})
    return pd.DataFrame(rows)


# ============================================================================
# MODELS
# ============================================================================

# Pre-declared before looking at any test fold: XGBoost when it is installed,
# RandomForest otherwise. Both are always reported; this one drives explain.py,
# leakage_demo.py, cross_study.py and the demo app.
PRIMARY = 'xgboost' if HAS_XGB else 'random_forest'


def make_model(name, y_train):
    """
    A fresh, untuned classifier. Fixed hyper-parameters on purpose: with ~165
    subjects there is no data to spare for tuning without nested CV.
    Class imbalance is handled the same way as main.py (inverse frequency),
    from the TRAINING labels only.
    """
    y_train = np.asarray(y_train)
    if name == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced', random_state=SEED, n_jobs=-1)
    elif name == 'xgboost':
        if not HAS_XGB:
            raise ImportError("xgboost is not installed (pip install xgboost)")
        pos = max(int((y_train == 1).sum()), 1)
        neg = int((y_train == 0).sum())
        clf = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            scale_pos_weight=neg / pos, eval_metric='logloss',
            random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(name)
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('clf', clf)])


def model_names():
    return ['random_forest', 'xgboost'] if HAS_XGB else ['random_forest']


PRETTY = {'random_forest': 'Random Forest', 'xgboost': 'XGBoost'}


def binary_metrics(y_true, p_pd, threshold=0.5):
    """main.classification_metrics() from a P(PD) vector (adds ROC-AUC)."""
    y_true = np.asarray(y_true, dtype=int)
    p_pd = np.asarray(p_pd, dtype=float)
    y_pred = (p_pd >= threshold).astype(int)
    return classification_metrics(y_true, y_pred, np.c_[1 - p_pd, p_pd], 2)


def cv_oof_probs(X, y, groups_ids, folds, name):
    """
    Out-of-fold P(PD) for each row of X, fitting on each fold's fit_ids and
    predicting its test_ids. `groups_ids` gives the subject of each row, so this
    works for subject-level and window-level tables alike.
    Returns (oof_probs, per_fold_metrics).
    """
    oof = np.full(len(X), np.nan)
    per_fold = []
    for fit_ids, test_ids in folds:
        tr = np.isin(groups_ids, fit_ids)
        te = np.isin(groups_ids, test_ids)
        model = make_model(name, y[tr]).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
        per_fold.append(binary_metrics(y[te], oof[te]))
    return oof, per_fold


def fold_summary(per_fold, keys=('accuracy', 'balanced_accuracy', 'sensitivity',
                                 'specificity', 'roc_auc')):
    return {k: {'mean': float(np.nanmean([m[k] for m in per_fold])),
                'std':  float(np.nanstd([m[k] for m in per_fold])),
                'per_fold': [m[k] for m in per_fold]} for k in keys}


# ============================================================================
# MSCAF OUT-OF-FOLD PROBABILITIES
# ============================================================================

def _first_existing(candidates):
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _search_input(pattern):
    if not KAGGLE_INPUT.exists():
        return None
    hits = sorted(KAGGLE_INPUT.glob(f'**/{pattern}'))
    return hits[0] if hits else None


def find_cv_results():
    """main.py's cv_results.json: its results_dir, /kaggle/working, or an attached input."""
    return _first_existing([
        Path(CONFIG['results_dir']) / 'cv_results.json',
        KAGGLE_WORKING / 'results_pd' / 'cv_results.json',
    ]) or _search_input('cv_results.json')


def find_checkpoint_dir():
    """Folder holding best_model_pd_fold{1..K}.pth from main.py's CV run."""
    name = 'best_model_pd_fold1.pth'
    hit = _first_existing([
        Path(CONFIG['save_dir']) / name,
        KAGGLE_WORKING / 'checkpoints_pd' / name,
    ]) or _search_input(name)
    if hit is None:
        return None
    k = CONFIG['cv_folds']
    if all((hit.parent / f'best_model_pd_fold{i}.pth').exists() for i in range(1, k + 1)):
        return hit.parent
    return None


def mscaf_fold_models(subjects_df, ckpt_dir):
    """
    Yield (fold, model, test_dataset) for every fold of main.py's CV run: the
    fold's best-validation checkpoint and that fold's held-out test subjects.
    CONFIG (fusion, use_lstm, ...) must match the run that wrote the checkpoints.
    """
    from main import build_model, restore_best, PhysioNetGaitDataset
    folds = make_cv_folds(subjects_df, CONFIG)
    for k, (_, _, test_ids) in enumerate(folds, start=1):
        model = build_model(CONFIG, CONFIG['num_classes'])
        restore_best(model, ckpt_dir / f'best_model_pd_fold{k}.pth', CONFIG)
        model.eval()
        test_ds = PhysioNetGaitDataset(
            CONFIG['data_root'], subjects_df[subjects_df['subject_id'].isin(test_ids)],
            CONFIG['sequence_length'], CONFIG['stride'], CONFIG['task'])
        yield k, model, test_ds


def window_probs(model, dataset, batch_size=None):
    """P(PD) for every window of `dataset`, in dataset.sequences order."""
    import torch
    from torch.utils.data import DataLoader
    from main import collect_predictions
    loader = DataLoader(dataset, batch_size=batch_size or CONFIG['batch_size'],
                        shuffle=False, num_workers=0)
    _, labels, probs = collect_predictions(
        model, loader, CONFIG['device'], CONFIG['task'], return_probs=True)
    return probs[:, 1], labels


def load_mscaf_oof(subjects_df):
    """
    One out-of-fold MSCAF P(PD) per subject (mean of that subject's window
    probabilities), as a DataFrame [subject_id, true, pred, prob_pd].

    Looked up in this order:
      1. results_new/mscaf_oof_probs.csv (cache from an earlier call)
      2. cv_results.json from main.py, if it already stores 'prob_pd'
      3. recomputed from main.py's fold checkpoints on each fold's test
         subjects — no retraining, same folds, so still out-of-fold
    Returns None if none of these exist (run main.py first).
    """
    cache = results_dir() / 'mscaf_oof_probs.csv'
    if cache.exists():
        print(f"  MSCAF out-of-fold probabilities: {cache}")
        return pd.read_csv(cache)

    cv_path = find_cv_results()
    cv_preds = None
    if cv_path is not None:
        with open(cv_path) as f:
            cv_preds = pd.DataFrame(json.load(f)['pooled_out_of_fold_predictions'])
        if 'prob_pd' in cv_preds.columns:
            print(f"  MSCAF out-of-fold probabilities: {cv_path}")
            cv_preds.to_csv(cache, index=False)
            return cv_preds

    ckpt_dir = find_checkpoint_dir()
    if ckpt_dir is None:
        print("  ⚠️  No MSCAF out-of-fold probabilities found: cv_results.json has no\n"
              "      'prob_pd' and no fold checkpoints were found. Run main.py (it now\n"
              "      saves prob_pd) or attach checkpoints_pd/ as a Kaggle input.")
        return None

    print(f"  Recomputing MSCAF out-of-fold probabilities from {ckpt_dir} ...")
    rows = []
    for k, model, test_ds in mscaf_fold_models(subjects_df, ckpt_dir):
        p, y = window_probs(model, test_ds)
        sids = np.array([s['subject_id'] for s in test_ds.sequences])
        for sid in pd.unique(sids):
            m = sids == sid
            vote = int(np.bincount((p[m] >= 0.5).astype(int), minlength=2).argmax())
            rows.append({'subject_id': sid, 'true': int(y[m][0]), 'pred': vote,
                         'prob_pd': float(p[m].mean()), 'fold': k})
    out = pd.DataFrame(rows)

    # Sanity check against the run that produced the headline numbers
    if cv_preds is not None:
        merged = out.merge(cv_preds[['subject_id', 'pred']], on='subject_id',
                           suffixes=('', '_cv'))
        agree = (merged['pred'] == merged['pred_cv']).mean() * 100
        print(f"  Recomputed majority votes match cv_results.json for "
              f"{agree:.1f}% of {len(merged)} subjects")
        if agree < 100:
            print("  ⚠️  Not a perfect match — check that CONFIG matches the training run.")

    out.to_csv(cache, index=False)
    print(f"  Saved: {cache}")
    return out


# ============================================================================
# PLOTTING — one style for every PNG
# ============================================================================
#
# Light chart surface, recessive grid, categorical hues in fixed order, and a
# one-hue blue ramp for magnitudes (heatmaps).

SURFACE   = '#fcfcfb'
INK       = '#0b0b0b'
INK_2     = '#52514e'
MUTED     = '#898781'
GRID      = '#e1e0d9'
AXIS      = '#c3c2b7'
SERIES    = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
BLUE_RAMP = LinearSegmentedColormap.from_list(
    'blue_seq', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'axes.edgecolor': AXIS,
        'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
        'xtick.color': MUTED, 'ytick.color': MUTED,
        'xtick.labelcolor': INK_2, 'ytick.labelcolor': INK_2,
        'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'legend.fontsize': 9,
        'font.family': 'sans-serif', 'font.size': 10, 'text.color': INK,
    })


def save_fig(fig, name, dpi=200):
    path = results_dir() / name
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def grouped_bars(ax, groups, series, values, ylim=(0, 100), fmt='{:.1f}', colors=None):
    """
    Grouped bars with a value label on each bar.
    values[i][j] = value of series j in group i.
    """
    colors = colors or SERIES
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for j, name in enumerate(series):
        vals = [values[i][j] for i in range(len(groups))]
        xs = x - 0.4 + width * (j + 0.5)
        bars = ax.bar(xs, vals, width * 0.92, color=colors[j], label=name,
                      edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            if v is not None and np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.01,
                        fmt.format(v), ha='center', va='bottom', fontsize=8, color=INK_2,
                        zorder=5, bbox=dict(facecolor=SURFACE, edgecolor='none', pad=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylim(*ylim)


# The 16-sensor layout, from the dataset's format.txt (Infotronic insole).
# X < 0 is the left foot, Y runs heel (-800) -> toe (+800).
SENSOR_XY = {
    'L1': (-500, -800), 'L2': (-700, -400), 'L3': (-300, -400), 'L4': (-700, 0),
    'L5': (-300, 0),    'L6': (-700, 400),  'L7': (-300, 400),  'L8': (-500, 800),
    'R1': (500, -800),  'R2': (700, -400),  'R3': (300, -400),  'R4': (700, 0),
    'R5': (300, 0),     'R6': (700, 400),   'R7': (300, 400),   'R8': (500, 800),
}
SENSOR_NAMES = list(SENSOR_XY)          # column order of main.py's 16 channels


def draw_foot_heatmap(ax, values, title='', label='', fmt='{:.2f}', vmin=None, vmax=None,
                      cmap=BLUE_RAMP):
    """
    One value per sensor (16, in L1..L8, R1..R8 order) drawn as coloured
    circles on two insole outlines. Returns the ScalarMappable for a colorbar.
    """
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values) if vmin is None else vmin
    vmax = np.nanmax(values) if vmax is None else vmax
    norm = plt.Normalize(vmin, vmax)

    for cx in (-500, 500):
        ax.add_patch(Ellipse((cx, 0), 760, 2150, facecolor='#f0efec',
                             edgecolor=AXIS, linewidth=1.5, zorder=0))
    for (name, (x, y)), v in zip(SENSOR_XY.items(), values):
        rgba = cmap(norm(v))
        ax.scatter(x, y, s=1100, color=rgba, edgecolor=SURFACE, linewidth=2, zorder=2)
        # dark ink on light cells, white on dark cells
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ink = INK if lum > 0.5 else 'white'
        ax.text(x, y + 35, name, ha='center', va='bottom', fontsize=7, color=ink, zorder=3)
        ax.text(x, y - 35, fmt.format(v), ha='center', va='top', fontsize=7,
                color=ink, fontweight='bold', zorder=3)
    ax.text(-500, -1180, 'Left foot', ha='center', color=INK_2, fontsize=9)
    ax.text(500, -1180, 'Right foot', ha='center', color=INK_2, fontsize=9)
    ax.text(0, 1000, 'toe ↑', ha='center', color=MUTED, fontsize=8)
    ax.text(0, -1000, 'heel ↓', ha='center', color=MUTED, fontsize=8)
    ax.set_xlim(-1050, 1050)
    ax.set_ylim(-1260, 1180)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm
