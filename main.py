"""
PART 2: PARKINSON'S DISEASE DETECTION
Foot Pressure-Based Gait Recognition with MSCAF-Gait
======================================================

Dataset: PhysioNet - Gait in Parkinson's Disease
- 16 Ground Reaction Force (GRF) sensors (8 per foot)
- 100 Hz sampling rate
- ~2 minutes of walking per subject

NOTE: Severity regression (UPDRS) and H&Y staging tasks use synthetic labels
      when the real demographics file (demographics.xls) is not found in the
      dataset root. Binary PD classification always uses real ground-truth labels
      derived from the PhysioNet file-naming convention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import random
import re
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, roc_auc_score,
)

# ============================================================================
# CONFIGURATION
# DATA PATH: set the DATA_ROOT environment variable, or edit 'data_root' below.
#   e.g.  export DATA_ROOT=/path/to/gait-in-parkinsons-disease-1.0.0
# ============================================================================

CONFIG = {
    # ==================== PATHS ====================
    'data_root': os.environ.get(
        'DATA_ROOT',
        './gait-in-parkinsons-disease-1.0.0'   # relative fallback
    ),
    'save_dir': './checkpoints_pd',
    'results_dir': './results_pd',

    # ==================== TASKS ====================
    'task': 'pd_detection',  # 'pd_detection' | 'severity' | 'hy_stage'

    # ==================== MODEL ====================
    'num_classes': 2,
    'in_channels': 16,
    'reduction_ratio': 16,

    # ---- fusion / architecture switches ----
    'fusion': 'cross',       # 'cross' = paper's cross-attention fusion (Exp VII)
                             # 'add'   = h + s baseline, for the Table VI ablation
    'use_lstm': False,       # paper topology (no LSTM); set True for the LSTM variant
    'attn_dim': 64,          # d_k for the cross-attention q/k projections
    'spatial_kernel': 1,     # 1 = paper's Conv(1x1); 7 = CBAM-style temporal context

    # ==================== TRAINING ====================
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,

    # ==================== DATA ====================
    'sequence_length': 1000,  # 10 seconds at 100 Hz
    'stride': 500,            # 50% overlap for data augmentation

    # ---- three-way SUBJECT-LEVEL split (used when use_cv is False) ----
    'test_size': 0.2,            # fraction of SUBJECTS held out for the final test
    'val_size': 0.2,             # fraction of SUBJECTS used for model selection
    'early_stop_patience': 15,   # stop after this many epochs with no val gain
    'min_epochs': 20,            # never stop before this many epochs

    # ---- SUBJECT-LEVEL cross-validation ----
    'use_cv': True,              # True = 5-fold CV; False = the old single split
    'cv_folds': 5,               # number of stratified subject-level folds

    # ---- class imbalance / model selection ----
    'class_weights': 'inverse_frequency',  # per-fold weights from TRAIN windows only
    'selection_metric': 'balanced_accuracy',  # val metric for early stop + checkpoint
    'sampling_rate': 100,     # Hz

    # ==================== SYSTEM ====================
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,
    'seed': 42,
}

# ============================================================================
# SET RANDOM SEEDS
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG['seed'])

# ============================================================================
# MODEL ARCHITECTURE (1D VERSION FOR TIME SERIES)
# ============================================================================

class ShallowFeatureExtraction1D(nn.Module):
    """Shallow Feature Extraction for 1D time series (GRF sensors)"""
    def __init__(self, in_channels=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.2)
        self.pool4 = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
        return x


class MultiScaleConvolution1D(nn.Module):
    """Multi-Scale Convolution for 1D time series"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different temporal scales for gait analysis
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        return torch.cat([f1, f3, f5, f7], dim=1)


class ChannelAttention1D(nn.Module):
    """Channel Attention for 1D time series"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * self.sigmoid(y).expand_as(x)


class SpatialAttention1D(nn.Module):
    """
    Temporal ("spatial") attention -- Eqs. (6)-(8) of MSCAF-Gait.

        D_avg = (1/C) * sum_i M_i                      -> (B, 1, W)
        D_max = max_i M_i                              -> (B, 1, W)
        D_o   = ReLU[ BN( Conv(Concat(D_avg, D_max)) ) ]  2 -> C channels
        S     = sigmoid(D_o) (*) M

    The paper expands 2 -> C, so every channel receives its OWN positional
    gate (Fig. 5). Standard CBAM uses 2 -> 1 and broadcasts a single shared
    gate across all channels -- strictly less expressive.

    kernel_size=1 reproduces the paper's Conv(1x1), which gives each
    timestep's gate no temporal context. kernel_size=7 is the CBAM-style
    variant that lets the gate see a +/-3 neighbourhood.
    """
    def __init__(self, channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)          # (B, 1, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # (B, 1, W)
        d = torch.cat([avg_out, max_out], dim=1)              # (B, 2, W)
        d = self.relu(self.bn(self.conv(d)))                  # (B, C, W)
        return x * self.sigmoid(d)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion -- Algorithm 1 and Eq. (9) of MSCAF-Gait.

    Takes the three views M, H, S (each (B, C, W)), permutes them into
    (B, W, C) token sequences, projects each into queries / keys / values,
    and computes all SIX ordered cross-domain pairs. No self-attention
    terms: the information the paper is after lives in the relationships
    BETWEEN domains.

        O_c1 = Att(q1, k2, v2)    O_c2 = Att(q1, k3, v3)
        O_c3 = Att(q2, k1, v1)    O_c4 = Att(q2, k3, v3)
        O_c5 = Att(q3, k1, v1)    O_c6 = Att(q3, k2, v2)

        Att(Q, K, V) = Softmax( Q K^T / sqrt(d_k) ) V

    The six outputs are concatenated along the TEMPORAL axis and permuted
    back, giving O_f of shape (B, C, 6W).
    """
    def __init__(self, channels, attn_dim=64, dropout=0.1):
        super().__init__()
        self.scale = attn_dim ** -0.5
        # One (W_Q, W_K, W_V) triplet per view. Values project back to C so
        # the concatenated output keeps the channel width the paper specifies.
        self.q_proj = nn.ModuleList(
            [nn.Linear(channels, attn_dim, bias=False) for _ in range(3)])
        self.k_proj = nn.ModuleList(
            [nn.Linear(channels, attn_dim, bias=False) for _ in range(3)])
        self.v_proj = nn.ModuleList(
            [nn.Linear(channels, channels, bias=False) for _ in range(3)])
        self.attn_drop = nn.Dropout(dropout)

    def _attend(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, W, W)
        weights = self.attn_drop(torch.softmax(scores, dim=-1))
        return torch.matmul(weights, v)                              # (B, W, C)

    def forward(self, m, h, s):
        views = [t.permute(0, 2, 1) for t in (m, h, s)]              # (B, W, C)
        q = [p(v) for p, v in zip(self.q_proj, views)]
        k = [p(v) for p, v in zip(self.k_proj, views)]
        val = [p(v) for p, v in zip(self.v_proj, views)]

        pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        outs = [self._attend(q[i], k[j], val[j]) for i, j in pairs]

        o_c = torch.cat(outs, dim=1)          # (B, 6W, C)
        return o_c.permute(0, 2, 1)           # (B, C, 6W)


class MSCAFGaitPD(nn.Module):
    """
    MSCAF-Gait for Parkinson's Disease detection.

    fusion='cross' -> the paper's cross-attention fusion (Exp VII)
    fusion='add'   -> h + s additive baseline, for the Table VI ablation
    use_lstm=False -> the paper's own topology (no recurrent layer)
    """
    def __init__(self, num_classes=2, in_channels=16, task='pd_detection',
                 fusion='cross', use_lstm=True, attn_dim=64, spatial_kernel=1):
        super().__init__()
        self.task = task
        self.fusion = fusion
        self.use_lstm = use_lstm

        self.sfe = ShallowFeatureExtraction1D(in_channels)
        self.multi_scale = MultiScaleConvolution1D(128, 64)

        # pixel-level enhancement -> M
        self.pixel_conv = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.channel_attention = ChannelAttention1D(128)
        self.spatial_attention = SpatialAttention1D(128, kernel_size=spatial_kernel)

        if fusion == 'cross':
            self.cross_attention = CrossAttentionFusion(128, attn_dim=attn_dim)
        elif fusion != 'add':
            raise ValueError("fusion must be 'cross' or 'add'")

        # Post-fusion refinement: Conv(1x3) + BN + ReLU + MaxPool  (Fig. 2)
        self.post_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        if use_lstm:
            self.lstm = nn.LSTM(128, 128, num_layers=2,
                                batch_first=True, bidirectional=True)
            feat_dim = 256
        else:
            feat_dim = 128
        self.feat_dim = feat_dim

        # Task-specific heads
        if task in ('pd_detection', 'hy_stage'):
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        elif task == 'severity':
            self.regressor = nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, x):
        # x: (B, C_in, T) -- batch, 16 sensors, timesteps
        f = self.sfe(x)                       # (B, 128, T/8)
        fm = self.multi_scale(f)              # (B, 256, T/16)
        m = self.pixel_conv(fm)               # (B, 128, W)   <- M
        h = self.channel_attention(m)         # (B, 128, W)   <- H
        s = self.spatial_attention(m)         # (B, 128, W)   <- S

        if self.fusion == 'cross':
            fused = self.cross_attention(m, h, s)   # (B, 128, 6W)  <- O_f
        else:
            fused = h + s                            # (B, 128, W)

        fused = self.post_conv(fused)                # (B, 128, L)

        seq = fused.permute(0, 2, 1)                 # (B, L, 128)
        if self.use_lstm:
            seq, _ = self.lstm(seq)                  # (B, L, 256)

        pooled = torch.mean(seq, dim=1)              # (B, feat_dim)

        if self.task in ('pd_detection', 'hy_stage'):
            output = self.classifier(pooled)
        else:
            output = self.regressor(pooled).squeeze()

        return output, pooled


# ============================================================================
# DATASET LOADER
# ============================================================================

class PhysioNetGaitDataset(Dataset):
    """PhysioNet Gait in Parkinson's Disease Dataset"""
    def __init__(self, data_root, subjects_df, sequence_length=1000, stride=500, task='pd_detection'):
        self.data_root = Path(data_root)
        self.subjects_df = subjects_df
        self.sequence_length = sequence_length
        self.stride = stride
        self.task = task
        self.sequences = self._build_sequences()

    def _load_grf_data(self, filename):
        """Load GRF sensor data using the exact filename"""
        path = self.data_root / f"{filename}.txt"

        if path.exists():
            try:
                # Load the data (18 columns: time + 16 sensors + total)
                data = np.loadtxt(path)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                return data
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")

        return None

    def _build_sequences(self):
        """Build sequences with sliding window"""
        sequences = []

        for idx, row in self.subjects_df.iterrows():
            subject_id = row['subject_id']
            label = row['label']

            # Load GRF data
            data = self._load_grf_data(row['filename'])

            if data is None:
                continue

            if data.shape[1] >= 16:
                sensors = data[:, 1:17]  # Columns 1-16 are the sensors
            else:
                continue

            # Z-score normalization per sensor channel
            sensors = (sensors - sensors.mean(axis=0)) / (sensors.std(axis=0) + 1e-8)

            # Create sliding windows
            for start_idx in range(0, len(sensors) - self.sequence_length, self.stride):
                end_idx = start_idx + self.sequence_length
                window = sensors[start_idx:end_idx]

                sequences.append({
                    'data': window.T,  # (16, seq_len)
                    'label': label,
                    'subject_id': subject_id,
                    'updrs': row.get('updrs', 0),
                    'hy_stage': row.get('hy_stage', 0)
                })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        data = torch.from_numpy(seq_info['data']).float()

        if self.task == 'pd_detection':
            label = seq_info['label']
        elif self.task == 'severity':
            label = seq_info['updrs']
        elif self.task == 'hy_stage':
            label = seq_info['hy_stage']

        return data, label


def _try_load_demographics(data_path):
    """
    Try to load real UPDRS / H&Y scores from the PhysioNet demographics file.

    The PhysioNet gaitpdb dataset ships with a demographics spreadsheet
    (demographics.xls or demographics.xlsx) that contains per-subject
    UPDRS scores and H&Y staging.

    Returns a dict: { subject_id -> {'updrs': float, 'hy': int} }
    or an empty dict if the file cannot be found / parsed.
    """
    demo_map = {}
    for name in ('demographics.xls', 'demographics.xlsx'):
        demo_file = data_path / name
        if demo_file.exists():
            try:
                df = pd.read_excel(demo_file)
                # Column names vary slightly between dataset versions; try common ones
                id_col    = next((c for c in df.columns if 'id'    in c.lower()), None)
                updrs_col = next((c for c in df.columns if 'updrs' in c.lower()), None)
                hy_col    = next((c for c in df.columns if 'hoehn' in c.lower()
                                                        or 'yahr'  in c.lower()
                                                        or 'h&y'   in c.lower()
                                                        or 'stage' in c.lower()), None)

                if id_col:
                    for _, row in df.iterrows():
                        sid = str(row[id_col]).strip()
                        demo_map[sid] = {
                            'updrs': float(row[updrs_col]) if updrs_col and pd.notna(row[updrs_col]) else 0.0,
                            'hy':    int(row[hy_col])      if hy_col    and pd.notna(row[hy_col])    else 0,
                        }
                    print(f"  ✅ Loaded real demographics from {demo_file.name} ({len(demo_map)} subjects)")
                    return demo_map
            except Exception as e:
                print(f"  Warning: could not parse {demo_file.name}: {e}")

    print("  ⚠️  demographics file not found — UPDRS and H&Y labels will be SYNTHETIC.")
    print("      Severity regression and H&Y staging results are NOT clinically valid.")
    return demo_map


# Only files named like a real recording trial are subjects. The dataset root
# also ships demographics.txt, SHA256SUMS.txt and format.txt, which are not
# recordings and were previously picked up and mislabelled as PD patients.
_SUBJECT_FILE_RE = re.compile(r'^(Ga|Ju|Si)(Co|Pt)\d+_\d+\.txt$')


def load_subject_info(data_root):
    """
    Load subject information from the PhysioNet gaitpdb dataset.

    Binary PD/Healthy labels are derived from the file-naming convention
    (e.g. GaCo01 → Healthy Control, GaPt01 → PD Patient).

    UPDRS and H&Y labels are read from demographics.xls when available;
    otherwise synthetic values are assigned and a warning is printed.
    """
    subjects = []
    data_path = Path(data_root)

    # Try to load real demographics first
    demo_map = _try_load_demographics(data_path)

    if data_path.exists():
        for subject_file in sorted(data_path.glob("*.txt")):
            # Skip non-recording text files (demographics.txt, SHA256SUMS.txt,
            # format.txt, ...) silently — only trial files are subjects.
            if not _SUBJECT_FILE_RE.match(subject_file.name):
                continue

            filename  = subject_file.stem          # e.g. 'GaCo01_01'
            subject_id = filename.split('_')[0] if '_' in filename else filename  # e.g. 'GaCo01'
            group      = subject_id[:2]            # 'Ga', 'Ju', or 'Si'

            # PD / Healthy label from filename convention
            if 'Co' in subject_id or 'co' in subject_id:
                label = 0  # Healthy control
            else:
                label = 1  # PD patient

            # UPDRS & H&Y — real values where available, synthetic fallback
            real = demo_map.get(subject_id, {})
            if real:
                updrs    = real['updrs']
                hy_stage = real['hy']
            else:
                if label == 1:
                    # Synthetic fallback — clearly marked in console output above
                    updrs    = random.randint(10, 50)
                    hy_stage = random.randint(1, 4)
                else:
                    updrs    = 0
                    hy_stage = 0

            subjects.append({
                'subject_id': subject_id,
                'filename':   filename,
                'label':      label,
                'updrs':      updrs,
                'hy_stage':   hy_stage,
                'group':      group,
            })

    return pd.DataFrame(subjects)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, task='pd_detection'):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for data, labels in tqdm(train_loader, desc="Training"):
        data   = data.to(device)
        labels = labels.to(device)

        if task == 'severity':
            labels = labels.float()
        else:
            labels = labels.long()

        optimizer.zero_grad()
        outputs, _ = model(data)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if task != 'severity':
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)

    if task == 'severity':
        return avg_loss, 0.0
    else:
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc * 100


def evaluate(model, test_loader, device, task='pd_detection'):
    model.eval()
    all_preds    = []
    all_labels   = []
    all_features = []

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc="Evaluating"):
            data   = data.to(device)
            labels = labels.to(device)

            outputs, features = model(data)

            if task == 'severity':
                all_preds.extend(outputs.cpu().numpy())
            else:
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())

            all_labels.extend(labels.cpu().numpy())
            all_features.append(features.cpu())

    if task == 'severity':
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        r2   = r2_score(all_labels, all_preds)
        return mae, rmse, r2, torch.cat(all_features)
    else:
        acc    = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, output_dict=True)
        cm     = confusion_matrix(all_labels, all_preds)
        return acc * 100, report, cm, torch.cat(all_features)


def collect_predictions(model, loader, device, task='pd_detection', return_probs=False):
    """
    Return (preds, labels) as numpy arrays, in dataset order.
    With return_probs=True, also return the class probabilities (classification
    only) so ROC-AUC can be computed without a second pass.

    Requires the loader to have been built with shuffle=False.
    """
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for data, y in loader:
            out, _ = model(data.to(device))
            if task == 'severity':
                preds.extend(np.atleast_1d(out.cpu().numpy()).tolist())
            else:
                preds.extend(out.argmax(1).cpu().numpy().tolist())
                if return_probs:
                    probs.extend(torch.softmax(out, dim=1).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())

    if return_probs:
        return np.array(preds), np.array(labels), np.array(probs)
    return np.array(preds), np.array(labels)


def aggregate_by_subject(dataset, preds, labels, task='pd_detection', return_ids=False):
    """
    Collapse window-level predictions into ONE prediction per subject.

    A clinician diagnoses a person, not a 10-second window, so this is the
    clinically meaningful metric. Classification uses a majority vote across
    that subject's windows; regression uses the mean.

    With return_ids=True the subject IDs are returned alongside, which is what
    lets cross-validation pool one out-of-fold prediction per subject.

    Relies on `preds` lining up with dataset.sequences, i.e. the loader must
    have shuffle=False.
    """
    from collections import defaultdict
    bucket, truth = defaultdict(list), {}
    for i, seq in enumerate(dataset.sequences):
        sid = seq['subject_id']
        bucket[sid].append(preds[i])
        truth[sid] = labels[i]

    y_true, y_pred, sids = [], [], []
    for sid, plist in bucket.items():
        sids.append(sid)
        y_true.append(truth[sid])
        if task == 'severity':
            y_pred.append(float(np.mean(plist)))
        else:
            y_pred.append(int(np.bincount(np.array(plist, dtype=int)).argmax()))

    if return_ids:
        return np.array(y_true), np.array(y_pred), np.array(sids)
    return np.array(y_true), np.array(y_pred)


# ============================================================================
# CLASS IMBALANCE & METRICS
# ============================================================================

def _window_label(seq, task):
    """The label a single window contributes, for the given task."""
    return int(seq['hy_stage']) if task == 'hy_stage' else int(seq['label'])


def compute_class_weights(dataset, num_classes, task, device):
    """
    Inverse-frequency CrossEntropy weights from the TRAINING windows only.

    Computed per fold, from that fold's training windows and nothing else:
    deriving weights from the val or test windows would leak information about
    the held-out subjects into the loss.

    weight_c = N / (num_classes * count_c), so the weights average to 1 and an
    under-represented class is up-weighted in proportion to how rare it is.

    Returns (weight_tensor, per_class_window_counts).
    """
    labels = np.array([_window_label(s, task) for s in dataset.sequences], dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # A class with no training windows in this fold would divide by zero; clamp
    # it so the run continues (its weight is then meaningless but unused).
    safe = np.maximum(counts, 1.0)
    weights = safe.sum() / (num_classes * safe)
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


def classification_metrics(y_true, y_pred, probs=None, num_classes=2):
    """
    Accuracy, balanced accuracy, sensitivity, specificity, ROC-AUC (binary),
    and the confusion matrix — all as plain floats/lists, ready for JSON.

    Balanced accuracy is the mean of the per-class recalls, i.e. the mean of
    sensitivity and specificity in the binary case. Unlike plain accuracy it
    cannot be gamed by always predicting the majority class.
    """
    labels_arg = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_arg)

    m = {
        'accuracy': float(accuracy_score(y_true, y_pred) * 100),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred) * 100),
        'confusion_matrix': cm.tolist(),
        'n': int(len(y_true)),
    }

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        m['sensitivity'] = float(tp / (tp + fn) * 100) if (tp + fn) else float('nan')
        m['specificity'] = float(tn / (tn + fp) * 100) if (tn + fp) else float('nan')
        if probs is not None and len(np.unique(y_true)) == 2:
            m['roc_auc'] = float(roc_auc_score(y_true, np.asarray(probs)[:, 1]))
        else:
            m['roc_auc'] = float('nan')

    counts = np.bincount(np.asarray(y_true, dtype=int), minlength=num_classes)
    m['majority_baseline'] = float(100.0 * counts.max() / counts.sum())
    return m


def _json_safe(obj):
    """
    Replace non-finite floats with None so the output is strict-JSON valid.

    Metrics like ROC-AUC are NaN when a split contains a single class, and
    json.dump would happily write a bare NaN, which most JSON parsers reject.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _fmt(m, keys):
    """One-line metric summary for console output."""
    out = []
    for k in keys:
        v = m.get(k)
        if v is None:
            continue
        out.append(f"{k.replace('_', ' ')} {v:.2f}" if k != 'roc_auc' else f"{k} {v:.4f}")
    return " | ".join(out)


# ============================================================================
# SUBJECT-LEVEL SPLITS
# ============================================================================

def make_subject_splits(subjects_df, config):
    """
    Three-way SUBJECT-LEVEL train / val / test split (the use_cv=False path).

    Split on unique subject IDs, never on files or windows: different trials
    of the same person must not straddle a split, or the model can recognise
    the individual instead of the disease.

    Three-way, so that model selection -- early stopping and checkpointing --
    happens on VALIDATION data. The test set is evaluated exactly once, after
    training, which is what makes the reported number an unbiased estimate of
    generalisation rather than a maximum over epochs.

    Deterministic given the same subjects_df and config, so that evaluate.py
    can reconstruct exactly the same held-out test subjects that training used.

    Returns (train_ids, val_ids, test_ids) as numpy arrays of subject IDs.
    """
    unique_subjects = subjects_df[['subject_id', 'label']].drop_duplicates('subject_id')
    ids   = unique_subjects['subject_id'].values
    strat = unique_subjects['label'].values

    trainval_ids, test_ids, trainval_strat, _ = train_test_split(
        ids, strat,
        test_size=config['test_size'],
        random_state=config['seed'],
        stratify=strat
    )
    val_fraction = config['val_size'] / (1.0 - config['test_size'])
    train_ids, val_ids = train_test_split(
        trainval_ids,
        test_size=val_fraction,
        random_state=config['seed'],
        stratify=trainval_strat
    )

    assert not (set(train_ids) & set(val_ids)),  "train/val subject overlap"
    assert not (set(train_ids) & set(test_ids)), "train/test subject overlap"
    assert not (set(val_ids)   & set(test_ids)), "val/test subject overlap"

    return train_ids, val_ids, test_ids


def make_cv_folds(subjects_df, config):
    """
    K-fold stratified SUBJECT-LEVEL cross-validation splits.

    StratifiedKFold over unique subject IDs (stratified on the PD/healthy
    label), so every subject is held out exactly once across the K folds and
    each fold keeps the class balance of the cohort. Within a fold, a
    stratified validation set is carved out of the remaining subjects — early
    stopping never sees the test fold.

    The validation fraction is expressed relative to the whole cohort, matching
    make_subject_splits(): config['val_size'] of all subjects.

    Returns a list of K (train_ids, val_ids, test_ids) tuples.
    """
    unique_subjects = subjects_df[['subject_id', 'label']].drop_duplicates('subject_id')
    ids   = unique_subjects['subject_id'].values
    strat = unique_subjects['label'].values

    k = config['cv_folds']
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=config['seed'])

    # val_size is a fraction of ALL subjects, but it is carved out of the
    # (1 - 1/k) that are not in the test fold, so rescale accordingly.
    val_fraction = config['val_size'] / (1.0 - 1.0 / k)

    folds = []
    for rest_idx, test_idx in skf.split(ids, strat):
        test_ids   = ids[test_idx]
        rest_ids   = ids[rest_idx]
        rest_strat = strat[rest_idx]

        train_ids, val_ids = train_test_split(
            rest_ids,
            test_size=val_fraction,
            random_state=config['seed'],
            stratify=rest_strat
        )

        assert not (set(train_ids) & set(val_ids)),  "train/val subject overlap"
        assert not (set(train_ids) & set(test_ids)), "train/test subject overlap"
        assert not (set(val_ids)   & set(test_ids)), "val/test subject overlap"

        folds.append((train_ids, val_ids, test_ids))

    # Every subject is held out exactly once, and no more than once
    all_test = np.concatenate([f[2] for f in folds])
    assert len(all_test) == len(set(all_test)) == len(ids), \
        "CV test folds must partition the subjects exactly once"

    return folds


# ============================================================================
# BUILD / TRAIN / EVALUATE ONE MODEL
# ============================================================================

def build_datasets(subjects_df, ids_tuple, config):
    """Build the (train, val, test) datasets for one split of subject IDs."""
    def build(ids):
        df = subjects_df[subjects_df['subject_id'].isin(ids)]
        return PhysioNetGaitDataset(
            config['data_root'], df,
            config['sequence_length'], config['stride'], config['task']
        )
    return tuple(build(ids) for ids in ids_tuple)


def build_loaders(train_dataset, val_dataset, test_dataset, config):
    """shuffle=False on val/test so predictions line up with dataset.sequences."""
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=config['num_workers'])
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])
    test_loader  = DataLoader(test_dataset,  batch_size=config['batch_size'],
                              shuffle=False, num_workers=config['num_workers'])
    return train_loader, val_loader, test_loader


def build_model(config, num_classes):
    """A fresh, untrained model. Architecture is entirely config-driven."""
    return MSCAFGaitPD(
        num_classes,
        config['in_channels'],
        config['task'],
        fusion=config['fusion'],
        use_lstm=config['use_lstm'],
        attn_dim=config['attn_dim'],
        spatial_kernel=config['spatial_kernel']
    ).to(config['device'])


def make_criterion(train_dataset, num_classes, config, verbose=True):
    """
    Loss for this fold. Classification uses inverse-frequency class weights
    computed from this fold's TRAINING windows only.
    """
    if config['task'] == 'severity':
        return nn.MSELoss(), None

    weights, counts = compute_class_weights(
        train_dataset, num_classes, config['task'], config['device'])
    if verbose:
        print(f"  Train window counts per class: {counts.astype(int).tolist()}")
        print(f"  Inverse-frequency loss weights: "
              f"{[round(w, 3) for w in weights.cpu().numpy().tolist()]}")
    return nn.CrossEntropyLoss(weight=weights), weights.cpu().numpy().tolist()


def train_model(model, train_loader, val_loader, criterion, config, ckpt_path):
    """
    Train with early stopping and checkpointing on VALIDATION data.

    For classification the selection metric is BALANCED accuracy (the mean of
    sensitivity and specificity). Plain accuracy would let a model that always
    predicts the majority class look good on an imbalanced validation set and
    get checkpointed for it.

    Returns (history, best_epoch, best_val).
    """
    is_reg = config['task'] == 'severity'
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'],
                           weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['num_epochs'], eta_min=1e-6)

    best_val = float('inf') if is_reg else -1.0
    best_epoch, no_improve = 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_metric': []}

    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            config['device'], config['task'])

        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        if is_reg:
            mae, rmse, r2, _ = evaluate(model, val_loader, config['device'], config['task'])
            val_metric, improved = mae, mae < best_val
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
            payload = {'mae': mae, 'rmse': rmse, 'r2': r2}
        else:
            preds, labels = collect_predictions(
                model, val_loader, config['device'], config['task'])
            val_acc = accuracy_score(labels, preds) * 100
            val_bal = balanced_accuracy_score(labels, preds) * 100
            val_metric, improved = val_bal, val_bal > best_val
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Balanced Acc: {val_bal:.2f}%  (plain acc {val_acc:.2f}%)")
            if config['num_classes'] == 2 and config['task'] == 'pd_detection':
                cm = confusion_matrix(labels, preds, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) * 100 if (tp + fn) else float('nan')
                spec = tn / (tn + fp) * 100 if (tn + fp) else float('nan')
                print(f"  Val Sensitivity: {sens:.2f}% | Val Specificity: {spec:.2f}%")
            payload = {'val_balanced_accuracy': val_bal, 'val_accuracy': val_acc}

        if improved:
            best_val, best_epoch, no_improve = val_metric, epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), **payload},
                       ckpt_path)
            print(f"  ✓ new best on VALIDATION ({best_val:.2f}) — checkpoint saved")
        else:
            no_improve += 1
            print(f"  no val improvement for {no_improve} epoch(s) "
                  f"(best {best_val:.2f} @ epoch {best_epoch})")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metric'].append(val_metric)
        scheduler.step()

        if epoch >= config['min_epochs'] and no_improve >= config['early_stop_patience']:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best validation result was at epoch {best_epoch}.")
            break

    return history, best_epoch, best_val


def restore_best(model, ckpt_path, config):
    """Load the best-validation checkpoint back into the model."""
    # weights_only=False: our checkpoints also carry metric payloads, which the
    # PyTorch >=2.6 safe loader rejects.
    ckpt = torch.load(ckpt_path, map_location=config['device'], weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return ckpt


def evaluate_test_split(model, test_dataset, test_loader, config, num_classes):
    """
    Score one held-out test split, at window level and subject level.

    Returns a JSON-ready dict plus the per-subject out-of-fold predictions, so
    the caller can pool them across folds. The subject prediction is still the
    majority vote; the subject probability (mean of the window class
    probabilities) is returned alongside for ROC-AUC and late fusion.
    """
    preds, labels, probs = collect_predictions(
        model, test_loader, config['device'], config['task'], return_probs=True)

    window = classification_metrics(labels, preds, probs, num_classes)

    yt, yp, sids = aggregate_by_subject(
        test_dataset, preds, labels, config['task'], return_ids=True)

    # aggregate_by_subject() visits subjects in dataset order, so the means
    # below line up with sids
    win_sids = np.array([s['subject_id'] for s in test_dataset.sequences])
    subj_probs = np.array([probs[win_sids == sid].mean(axis=0) for sid in sids])
    subject = classification_metrics(yt, yp, subj_probs, num_classes)

    return {'window': window, 'subject': subject}, (sids, yt, yp, subj_probs)


def print_split_metrics(res, title, binary=True):
    w, s = res['window'], res['subject']
    print(f"\n── {title} ─────────────────────────")
    print(f"  Window level  ({w['n']} windows)")
    print(f"    Accuracy          : {w['accuracy']:.2f}%")
    print(f"    Balanced accuracy : {w['balanced_accuracy']:.2f}%")
    if binary:
        print(f"    Sensitivity       : {w['sensitivity']:.2f}%")
        print(f"    Specificity       : {w['specificity']:.2f}%")
        print(f"    ROC-AUC           : {w['roc_auc']:.4f}")
    print(f"    Majority baseline : {w['majority_baseline']:.2f}%")
    print(f"    Confusion matrix  : {w['confusion_matrix']}")
    print(f"  Subject level ({s['n']} subjects, majority vote)")
    print(f"    Accuracy          : {s['accuracy']:.2f}%")
    print(f"    Balanced accuracy : {s['balanced_accuracy']:.2f}%")
    if binary:
        print(f"    Sensitivity       : {s['sensitivity']:.2f}%")
        print(f"    Specificity       : {s['specificity']:.2f}%")
    print(f"    Majority baseline : {s['majority_baseline']:.2f}%")
    print(f"    Confusion matrix  : {s['confusion_matrix']}")


# ============================================================================
# RUN CONFIGURATION LOGGING
# ============================================================================

def log_run_config(config):
    """Print exactly how this run is set up, so a logged run is reproducible."""
    print("\n── Run configuration ────────────────────────────")
    print(f"  Task              : {config['task']}")
    print(f"  Fusion            : {config['fusion']}")
    print(f"  use_lstm          : {config['use_lstm']}"
          f"{'  (paper topology)' if not config['use_lstm'] else '  (LSTM variant)'}")
    print(f"  attn_dim          : {config['attn_dim']}")
    print(f"  spatial_kernel    : {config['spatial_kernel']}")
    if config['task'] == 'severity':
        print("  Loss              : MSELoss (regression)")
        print("  Selection metric  : validation MAE (lower is better)")
    else:
        print(f"  Loss              : CrossEntropyLoss, class weights = "
              f"{config['class_weights']} (from TRAIN windows of each fold only)")
        print(f"  Selection metric  : validation {config['selection_metric']}")
    if config['use_cv']:
        eval_desc = f"{config['cv_folds']}-fold stratified subject-level CV"
    else:
        eval_desc = "single subject-level train/val/test split"
    print(f"  Evaluation        : {eval_desc}")
    print(f"  Early stopping    : patience {config['early_stop_patience']}, "
          f"min {config['min_epochs']} epochs")
    print(f"  Epochs (max)      : {config['num_epochs']}")
    print(f"  Batch size        : {config['batch_size']}")
    print(f"  Learning rate     : {config['learning_rate']}")
    print(f"  Seed              : {config['seed']}")
    print(f"  Device            : {config['device']}")
    print("─" * 49)


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

# The metrics summarised as mean ± std across folds
_CV_WINDOW_KEYS  = ['accuracy', 'balanced_accuracy', 'sensitivity',
                    'specificity', 'roc_auc', 'majority_baseline']
_CV_SUBJECT_KEYS = ['accuracy', 'balanced_accuracy', 'sensitivity',
                    'specificity', 'roc_auc', 'majority_baseline']


def summarise_folds(fold_results):
    """mean ± std across folds for every reported metric."""
    summary = {}
    for level, keys in (('window', _CV_WINDOW_KEYS), ('subject', _CV_SUBJECT_KEYS)):
        summary[level] = {}
        for key in keys:
            vals = np.array([f[level][key] for f in fold_results], dtype=float)
            summary[level][key] = {
                'mean':      float(np.nanmean(vals)),
                'std':       float(np.nanstd(vals)),
                'per_fold':  [float(v) for v in vals],
            }
    return summary


def plot_cv_summary(fold_results, summary, save_path, binary=True):
    """Per-fold bars with the across-fold mean drawn on top."""
    keys = (['accuracy', 'balanced_accuracy', 'sensitivity', 'specificity']
            if binary else ['accuracy', 'balanced_accuracy'])
    n_folds = len(fold_results)
    x = np.arange(n_folds)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, level in zip(axes, ('window', 'subject')):
        width = 0.8 / len(keys)
        for i, key in enumerate(keys):
            vals = [f[level][key] for f in fold_results]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=key.replace('_', ' '))
        bal = summary[level]['balanced_accuracy']
        ax.axhline(bal['mean'], ls='--', c='grey', lw=1.2,
                   label=f"mean bal. acc {bal['mean']:.1f}%")
        ax.axhline(np.mean([f[level]['majority_baseline'] for f in fold_results]),
                   ls=':', c='red', lw=1.2, label='majority baseline')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
        ax.set_title(f'{level.capitalize()}-level metrics per fold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('%')

    # One shared legend below the axes, so it never sits on top of the bars
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle('Stratified subject-level cross-validation')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_cross_validation(subjects_df, config, num_classes):
    """
    K-fold stratified subject-level cross-validation.

    A fresh model is trained per fold; each fold's test subjects are scored
    exactly once, after restoring that fold's best-validation checkpoint.
    """
    binary = (num_classes == 2)
    folds = make_cv_folds(subjects_df, config)

    fold_results, fold_meta = [], []
    oof_sids, oof_true, oof_pred, oof_prob = [], [], [], []

    for k, (train_ids, val_ids, test_ids) in enumerate(folds, start=1):
        print("\n" + "=" * 60)
        print(f"FOLD {k}/{len(folds)}")
        print("=" * 60)

        train_dataset, val_dataset, test_dataset = build_datasets(
            subjects_df, (train_ids, val_ids, test_ids), config)

        print(f"  Subjects  train/val/test: "
              f"{len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
        print(f"  Windows   train/val/test: "
              f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}")
        print("  verified: no subject appears in more than one split")

        if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
            print(f"❌ Fold {k} produced an empty split. Check data_root / cv_folds.")
            return None

        set_seed(config['seed'] + k)          # fresh but reproducible init per fold
        model = build_model(config, num_classes)
        criterion, weights = make_criterion(train_dataset, num_classes, config)

        train_loader, val_loader, test_loader = build_loaders(
            train_dataset, val_dataset, test_dataset, config)

        ckpt_path = os.path.join(config['save_dir'], f'best_model_pd_fold{k}.pth')
        history, best_epoch, best_val = train_model(
            model, train_loader, val_loader, criterion, config, ckpt_path)

        # The test fold is touched exactly once, here, with the best-val weights
        restore_best(model, ckpt_path, config)
        res, (sids, yt, yp, sp) = evaluate_test_split(
            model, test_dataset, test_loader, config, num_classes)

        print_split_metrics(res, f"Fold {k} — held-out test subjects", binary)

        fold_results.append(res)
        fold_meta.append({
            'fold': k,
            'n_train_subjects': int(len(train_ids)),
            'n_val_subjects':   int(len(val_ids)),
            'n_test_subjects':  int(len(test_ids)),
            'n_train_windows':  int(len(train_dataset)),
            'n_val_windows':    int(len(val_dataset)),
            'n_test_windows':   int(len(test_dataset)),
            'test_subject_ids': sorted(test_ids.tolist()),
            'class_weights':    weights,
            'best_epoch':       int(best_epoch),
            'best_val_metric':  float(best_val),
            'epochs_run':       len(history['train_loss']),
            'history':          history,
            'window':           res['window'],
            'subject':          res['subject'],
        })

        oof_sids.extend(sids.tolist())
        oof_true.extend(yt.tolist())
        oof_pred.extend(yp.tolist())
        oof_prob.extend(sp.tolist())

    # ── Across-fold summary ──────────────────────────────────────────────
    summary = summarise_folds(fold_results)

    # ── Pooled out-of-fold subject-level metrics ─────────────────────────
    # Every subject was held out exactly once, so pooling gives one prediction
    # per subject over the whole cohort — a single number on all subjects,
    # rather than an average of five fold-sized estimates.
    assert len(oof_sids) == len(set(oof_sids)), "a subject was scored in two folds"
    pooled = classification_metrics(
        np.array(oof_true), np.array(oof_pred), np.array(oof_prob), num_classes)

    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION SUMMARY  ({len(folds)} folds)")
    print("=" * 60)
    for level, keys in (('window', _CV_WINDOW_KEYS), ('subject', _CV_SUBJECT_KEYS)):
        print(f"\n  {level.capitalize()}-level, mean ± std across folds:")
        for key in keys:
            if not binary and key in ('sensitivity', 'specificity', 'roc_auc'):
                continue
            st = summary[level][key]
            unit = '' if key == 'roc_auc' else '%'
            fmt = '.4f' if key == 'roc_auc' else '.2f'
            print(f"    {key.replace('_', ' '):<18}: "
                  f"{st['mean']:{fmt}}{unit} ± {st['std']:{fmt}}{unit}")

    print(f"\n  Pooled out-of-fold, subject level ({pooled['n']} subjects, "
          f"one prediction each):")
    print(f"    accuracy          : {pooled['accuracy']:.2f}%")
    print(f"    balanced accuracy : {pooled['balanced_accuracy']:.2f}%")
    if binary:
        print(f"    sensitivity       : {pooled['sensitivity']:.2f}%")
        print(f"    specificity       : {pooled['specificity']:.2f}%")
        print(f"    ROC-AUC           : {pooled['roc_auc']:.4f}  (mean window probability)")
    print(f"    majority baseline : {pooled['majority_baseline']:.2f}%")
    print(f"    confusion matrix  : {pooled['confusion_matrix']}")

    # ── Save ─────────────────────────────────────────────────────────────
    results = {
        'config': {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                   for k, v in config.items()},
        'n_folds': len(folds),
        'n_subjects': int(subjects_df['subject_id'].nunique()),
        'folds': fold_meta,
        'summary': summary,
        'pooled_out_of_fold_subject': pooled,
        'pooled_out_of_fold_predictions': [
            # prob_pd: mean window P(PD), used by gait_features.py for fusion
            {'subject_id': s, 'true': int(t), 'pred': int(p),
             'prob_pd': float(pr[1]) if num_classes == 2 else None}
            for s, t, p, pr in zip(oof_sids, oof_true, oof_pred, oof_prob)
        ],
    }
    out_path = os.path.join(config['results_dir'], 'cv_results.json')
    with open(out_path, 'w') as f:
        json.dump(_json_safe(results), f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")

    plot_cv_summary(fold_results, summary,
                    os.path.join(config['results_dir'], 'cv_summary.png'), binary)

    return results


# ============================================================================
# SINGLE SPLIT (use_cv = False)
# ============================================================================

def run_single_split(subjects_df, config, num_classes):
    """The original single subject-level train/val/test split path."""
    binary = (num_classes == 2)
    is_reg = config['task'] == 'severity'

    train_ids, val_ids, test_ids = make_subject_splits(subjects_df, config)
    train_dataset, val_dataset, test_dataset = build_datasets(
        subjects_df, (train_ids, val_ids, test_ids), config)

    print(f"\nTrain: {len(train_ids):3d} subjects  ({len(train_dataset)} windows)")
    print(f"Val:   {len(val_ids):3d} subjects  ({len(val_dataset)} windows)")
    print(f"Test:  {len(test_ids):3d} subjects  ({len(test_dataset)} windows)")
    print("  verified: no subject appears in more than one split")

    if min(len(train_dataset), len(val_dataset), len(test_dataset)) == 0:
        print("❌ A split produced zero sequences. Check data_root / split sizes.")
        return None

    print("\nBuilding model...")
    model = build_model(config, num_classes)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    criterion, weights = make_criterion(train_dataset, num_classes, config)

    train_loader, val_loader, test_loader = build_loaders(
        train_dataset, val_dataset, test_dataset, config)

    print("\nStarting training...\n")
    ckpt_path = os.path.join(config['save_dir'], 'best_model_pd.pth')
    history, best_epoch, best_val = train_model(
        model, train_loader, val_loader, criterion, config, ckpt_path)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON THE HELD-OUT TEST SET")
    print("The test set has not influenced training or model selection.")
    print("=" * 60)
    restore_best(model, ckpt_path, config)

    results = {'best_epoch': best_epoch, 'best_val_metric': best_val,
               'class_weights': weights,
               'n_train_subjects': int(len(train_ids)),
               'n_val_subjects': int(len(val_ids)),
               'n_test_subjects': int(len(test_ids))}

    if is_reg:
        mae, rmse, r2, _ = evaluate(model, test_loader, config['device'], config['task'])
        print(f"\nWindow-level  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
        preds, labels = collect_predictions(model, test_loader, config['device'], config['task'])
        yt, yp = aggregate_by_subject(test_dataset, preds, labels, config['task'])
        from sklearn.metrics import mean_absolute_error
        subj_mae = mean_absolute_error(yt, yp)
        print(f"Subject-level MAE: {subj_mae:.2f}   ({len(yt)} subjects)")
        results.update({'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2,
                        'test_subject_mae': float(subj_mae)})
    else:
        res, _ = evaluate_test_split(
            model, test_dataset, test_loader, config, num_classes)
        print_split_metrics(res, "Held-out test subjects", binary)
        results.update({'test_window': res['window'], 'test_subject': res['subject']})

    with open(os.path.join(config['results_dir'], 'history_pd.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(config['results_dir'], 'final_results_pd.json'), 'w') as f:
        json.dump(_json_safe(results), f, indent=2, default=str)

    # ── Plot ─────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    if not is_reg:
        plt.plot(history['train_acc'], label='Train accuracy')
    plt.plot(history['val_metric'], label='Validation')
    if best_epoch:
        plt.axvline(best_epoch - 1, ls='--', c='grey', lw=1, label=f'best (epoch {best_epoch})')
    plt.title('Validation Performance'); plt.xlabel('Epoch')
    plt.ylabel('MAE' if is_reg else 'Balanced accuracy (%)'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config['results_dir'], 'training_curves_pd.png'), dpi=150)
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("PART 2: PARKINSON'S DISEASE DETECTION")
    print("=" * 60)
    print(f"Device:  {CONFIG['device']}")
    print(f"Dataset: PhysioNet Gait in PD")
    print(f"Task:    {CONFIG['task']}")
    print("=" * 60)

    os.makedirs(CONFIG['save_dir'],   exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    if CONFIG['task'] == 'hy_stage':
        CONFIG['num_classes'] = 5
    num_classes = CONFIG['num_classes']

    log_run_config(CONFIG)

    # ── Load subject information ──────────────────────────────────────────────
    print("\nLoading dataset...")
    subjects_df = load_subject_info(CONFIG['data_root'])

    if len(subjects_df) == 0:
        print("❌ No data found! Please check data_root path.")
        print(f"   Current path: {CONFIG['data_root']}")
        print("   Set the DATA_ROOT environment variable, e.g.:")
        print("     export DATA_ROOT=/path/to/gait-in-parkinsons-disease-1.0.0")
        return

    print(f"Found {len(subjects_df)} files across {subjects_df['subject_id'].nunique()} subjects")
    print(f"  PD patients:      {subjects_df[subjects_df['label']==1]['subject_id'].nunique()} subjects")
    print(f"  Healthy controls: {subjects_df[subjects_df['label']==0]['subject_id'].nunique()} subjects")

    if CONFIG['use_cv'] and CONFIG['task'] != 'severity':
        run_cross_validation(subjects_df, CONFIG, num_classes)
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION COMPLETE")
        print(f"Checkpoints: {CONFIG['save_dir']}")
        print(f"Results:     {CONFIG['results_dir']}")
        print("=" * 60)
        return

    if CONFIG['use_cv']:
        print("\n  Note: cross-validation is implemented for the classification "
              "tasks only; falling back to the single split for 'severity'.")

    run_single_split(subjects_df, CONFIG, num_classes)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Checkpoints: {CONFIG['save_dir']}")
    print(f"Results:     {CONFIG['results_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
