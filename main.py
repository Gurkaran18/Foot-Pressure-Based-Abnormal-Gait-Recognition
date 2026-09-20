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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

    # ---- three-way SUBJECT-LEVEL split ----
    'test_size': 0.2,            # fraction of SUBJECTS held out for the final test
    'val_size': 0.2,             # fraction of SUBJECTS used for model selection
    'early_stop_patience': 15,   # stop after this many epochs with no val gain
    'min_epochs': 20,            # never stop before this many epochs
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



def collect_predictions(model, loader, device, task='pd_detection'):
    """
    Return (preds, labels) as numpy arrays, in dataset order.
    Requires the loader to have been built with shuffle=False.
    """
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for data, y in loader:
            out, _ = model(data.to(device))
            if task == 'severity':
                preds.extend(np.atleast_1d(out.cpu().numpy()).tolist())
            else:
                preds.extend(out.argmax(1).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
    return np.array(preds), np.array(labels)


def aggregate_by_subject(dataset, preds, labels, task='pd_detection'):
    """
    Collapse window-level predictions into ONE prediction per subject.

    A clinician diagnoses a person, not a 10-second window, so this is the
    clinically meaningful metric. Classification uses a majority vote across
    that subject's windows; regression uses the mean.

    Relies on `preds` lining up with dataset.sequences, i.e. the loader must
    have shuffle=False.
    """
    from collections import defaultdict
    bucket, truth = defaultdict(list), {}
    for i, seq in enumerate(dataset.sequences):
        sid = seq['subject_id']
        bucket[sid].append(preds[i])
        truth[sid] = labels[i]

    y_true, y_pred = [], []
    for sid, plist in bucket.items():
        y_true.append(truth[sid])
        if task == 'severity':
            y_pred.append(float(np.mean(plist)))
        else:
            y_pred.append(int(np.bincount(np.array(plist, dtype=int)).argmax()))
    return np.array(y_true), np.array(y_pred)


# ============================================================================
# SUBJECT-LEVEL SPLIT
# ============================================================================

def make_subject_splits(subjects_df, config):
    """
    Three-way SUBJECT-LEVEL train / val / test split.

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

    # ── Subject-level train / val / test split ───────────────────────
    train_ids, val_ids, test_ids = make_subject_splits(subjects_df, CONFIG)

    train_df = subjects_df[subjects_df['subject_id'].isin(train_ids)]
    val_df   = subjects_df[subjects_df['subject_id'].isin(val_ids)]
    test_df  = subjects_df[subjects_df['subject_id'].isin(test_ids)]

    print(f"\nTrain: {len(train_ids):3d} subjects  ({len(train_df)} files)")
    print(f"Val:   {len(val_ids):3d} subjects  ({len(val_df)} files)")
    print(f"Test:  {len(test_ids):3d} subjects  ({len(test_df)} files)")
    print("  verified: no subject appears in more than one split")

    # ── Datasets ──────────────────────────────────────────────
    def build(df):
        return PhysioNetGaitDataset(
            CONFIG['data_root'], df,
            CONFIG['sequence_length'], CONFIG['stride'], CONFIG['task']
        )

    train_dataset, val_dataset, test_dataset = build(train_df), build(val_df), build(test_df)

    print(f"\nTrain sequences: {len(train_dataset)}")
    print(f"Val   sequences: {len(val_dataset)}")
    print(f"Test  sequences: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("❌ A split produced zero sequences. Check data_root / split sizes.")
        return

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True,  num_workers=CONFIG['num_workers'])
    # shuffle=False on val/test so predictions line up with dataset.sequences
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=CONFIG['num_workers'])
    test_loader  = DataLoader(test_dataset,  batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=CONFIG['num_workers'])

    # ── Build model ─────────────────────────────────────────
    print("\nBuilding model...")
    if CONFIG['task'] == 'hy_stage':
        CONFIG['num_classes'] = 5

    model = MSCAFGaitPD(
        CONFIG['num_classes'],
        CONFIG['in_channels'],
        CONFIG['task'],
        fusion=CONFIG['fusion'],
        use_lstm=CONFIG['use_lstm'],
        attn_dim=CONFIG['attn_dim'],
        spatial_kernel=CONFIG['spatial_kernel']
    ).to(CONFIG['device'])

    print(f"Fusion: {CONFIG['fusion']} | LSTM: {CONFIG['use_lstm']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss() if CONFIG['task'] == 'severity' else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=CONFIG['learning_rate'],
                           weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)

    # ── Training loop ── model selection on VALIDATION only ──────────────
    print("\nStarting training...\n")
    is_reg = CONFIG['task'] == 'severity'
    best_val = float('inf') if is_reg else 0.0
    best_epoch, no_improve = 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_metric': []}
    ckpt_path = os.path.join(CONFIG['save_dir'], 'best_model_pd.pth')

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            CONFIG['device'], CONFIG['task'])

        if is_reg:
            mae, rmse, r2, _ = evaluate(model, val_loader, CONFIG['device'], CONFIG['task'])
            val_metric, improved = mae, mae < best_val
            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
            payload = {'mae': mae, 'rmse': rmse, 'r2': r2}
        else:
            val_acc, report, cm, _ = evaluate(model, val_loader, CONFIG['device'], CONFIG['task'])
            val_metric, improved = val_acc, val_acc > best_val
            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Acc: {val_acc:.2f}%")
            if CONFIG['task'] == 'pd_detection':
                print(f"  Val Sensitivity: {report['1']['recall']*100:.2f}% | "
                      f"Val Specificity: {report['0']['recall']*100:.2f}%")
            payload = {'val_accuracy': val_acc, 'report': report, 'confusion_matrix': cm}

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

        if epoch >= CONFIG['min_epochs'] and no_improve >= CONFIG['early_stop_patience']:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best validation result was at epoch {best_epoch}.")
            break

    # ── FINAL TEST ── the test set is used exactly ONCE, here ────────────
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON THE HELD-OUT TEST SET")
    print(f"Restoring the checkpoint from epoch {best_epoch}, selected on validation.")
    print("The test set has not influenced training or model selection.")
    print("=" * 60)

    # weights_only=False: our own checkpoint also stores the confusion matrix
    # and classification report, which the PyTorch >=2.6 safe loader rejects.
    ckpt = torch.load(ckpt_path, map_location=CONFIG['device'], weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    results = {'best_epoch': best_epoch, 'best_val_metric': best_val,
               'n_train_subjects': int(len(train_ids)),
               'n_val_subjects': int(len(val_ids)),
               'n_test_subjects': int(len(test_ids))}

    if is_reg:
        mae, rmse, r2, _ = evaluate(model, test_loader, CONFIG['device'], CONFIG['task'])
        print(f"\nWindow-level  MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f}")
        preds, labels = collect_predictions(model, test_loader, CONFIG['device'], CONFIG['task'])
        yt, yp = aggregate_by_subject(test_dataset, preds, labels, CONFIG['task'])
        from sklearn.metrics import mean_absolute_error
        subj_mae = mean_absolute_error(yt, yp)
        print(f"Subject-level MAE: {subj_mae:.2f}   ({len(yt)} subjects)")
        results.update({'test_mae': mae, 'test_rmse': rmse, 'test_r2': r2,
                        'test_subject_mae': float(subj_mae)})
    else:
        test_acc, report, cm, _ = evaluate(model, test_loader, CONFIG['device'], CONFIG['task'])
        print(f"\nWindow-level accuracy: {test_acc:.2f}%   ({len(test_dataset)} windows)")
        if CONFIG['task'] == 'pd_detection':
            print(f"  Sensitivity (PD recall)      : {report['1']['recall']*100:.2f}%")
            print(f"  Specificity (healthy recall) : {report['0']['recall']*100:.2f}%")
        print("  Confusion matrix (windows):")
        print(cm)

        preds, labels = collect_predictions(model, test_loader, CONFIG['device'], CONFIG['task'])
        yt, yp = aggregate_by_subject(test_dataset, preds, labels, CONFIG['task'])
        subj_acc = accuracy_score(yt, yp) * 100
        subj_cm = confusion_matrix(yt, yp)
        print(f"\nSubject-level accuracy: {subj_acc:.2f}%   "
              f"({len(yt)} subjects, majority vote over their windows)")
        print("  Confusion matrix (subjects):")
        print(subj_cm)

        # context: what a constant majority-class predictor would score
        counts = np.bincount(labels.astype(int))
        baseline = 100.0 * counts.max() / counts.sum()
        print(f"\n  For context, always predicting the majority class "
              f"would score {baseline:.2f}% at window level.")

        results.update({
            'test_window_accuracy': test_acc,
            'test_subject_accuracy': float(subj_acc),
            'majority_baseline': float(baseline),
            'test_report': report,
            'test_confusion_matrix': cm.tolist(),
            'test_subject_confusion_matrix': subj_cm.tolist(),
        })

    # ── Save history + results ──────────────────────────────────
    with open(os.path.join(CONFIG['results_dir'], 'history_pd.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(CONFIG['results_dir'], 'final_results_pd.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # ── Plot ─────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    if not is_reg:
        plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_metric'], label='Validation')
    if best_epoch:
        plt.axvline(best_epoch - 1, ls='--', c='grey', lw=1, label=f'best (epoch {best_epoch})')
    plt.title('Validation Performance'); plt.xlabel('Epoch')
    plt.ylabel('MAE' if is_reg else 'Accuracy (%)'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_dir'], 'training_curves_pd.png'), dpi=150)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best validation {'MAE' if is_reg else 'accuracy'}: {best_val:.2f} (epoch {best_epoch})")
    print(f"Checkpoints: {CONFIG['save_dir']}")
    print(f"Results:     {CONFIG['results_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
