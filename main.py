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

    # ==================== TRAINING ====================
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,

    # ==================== DATA ====================
    'sequence_length': 1000,  # 10 seconds at 100 Hz
    'stride': 500,            # 50% overlap for data augmentation
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
    """Temporal Attention for 1D time series"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)


class MSCAFGaitPD(nn.Module):
    """MSCAF-Gait Model for Parkinson's Disease Detection"""
    def __init__(self, num_classes=2, in_channels=16, task='pd_detection'):
        super().__init__()
        self.task = task

        self.sfe = ShallowFeatureExtraction1D(in_channels)
        self.multi_scale = MultiScaleConvolution1D(128, 64)

        self.pixel_conv = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.channel_attention = ChannelAttention1D(128)
        self.spatial_attention = SpatialAttention1D()

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=True)

        # Task-specific heads
        if task == 'pd_detection':
            # Binary classification
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
        elif task == 'severity':
            # Regression for UPDRS score
            self.regressor = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        elif task == 'hy_stage':
            # Multi-class classification for H&Y stages
            self.classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        # x: (B, C, T) - Batch, Channels (16 sensors), Time
        f = self.sfe(x)
        fm = self.multi_scale(f)
        m = self.pixel_conv(fm)
        h = self.channel_attention(m)
        s = self.spatial_attention(m)

        # Combine features
        combined = h + s

        # LSTM for temporal modeling
        combined = combined.permute(0, 2, 1)  # (B, T, C)
        lstm_out, _ = self.lstm(combined)     # (B, T, 256)

        # Global average pooling
        pooled = torch.mean(lstm_out, dim=1)  # (B, 256)

        # Task-specific output
        if self.task == 'pd_detection' or self.task == 'hy_stage':
            output = self.classifier(pooled)
        elif self.task == 'severity':
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
        for subject_file in data_path.glob("*.txt"):
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

    # ── Subject-level train / test split ─────────────────────────────────────
    # IMPORTANT: split on unique subject IDs, not on individual files.
    # This prevents data leakage where different walking trials of the same
    # person appear in both train and test sets.
    unique_subjects = subjects_df[['subject_id', 'label']].drop_duplicates('subject_id')

    train_ids, test_ids = train_test_split(
        unique_subjects['subject_id'].values,
        test_size=0.3,
        random_state=CONFIG['seed'],
        stratify=unique_subjects['label'].values   # preserve PD / Healthy ratio
    )

    train_df = subjects_df[subjects_df['subject_id'].isin(train_ids)]
    test_df  = subjects_df[subjects_df['subject_id'].isin(test_ids)]

    print(f"\nTrain: {train_df['subject_id'].nunique()} subjects  ({len(train_df)} files)")
    print(f"Test:  {test_df['subject_id'].nunique()} subjects  ({len(test_df)} files)")

    # ── Create datasets ───────────────────────────────────────────────────────
    train_dataset = PhysioNetGaitDataset(
        CONFIG['data_root'], train_df,
        CONFIG['sequence_length'], CONFIG['stride'], CONFIG['task']
    )
    test_dataset = PhysioNetGaitDataset(
        CONFIG['data_root'], test_df,
        CONFIG['sequence_length'], CONFIG['stride'], CONFIG['task']
    )

    print(f"\nTrain sequences: {len(train_dataset)}")
    print(f"Test  sequences: {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("❌ No training sequences generated!")
        return

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=CONFIG['num_workers']
    )

    # ── Build model ───────────────────────────────────────────────────────────
    print("\nBuilding model...")
    if CONFIG['task'] == 'hy_stage':
        CONFIG['num_classes'] = 5  # H&Y stages 0-4

    model = MSCAFGaitPD(
        CONFIG['num_classes'],
        CONFIG['in_channels'],
        CONFIG['task']
    ).to(CONFIG['device'])

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.MSELoss() if CONFIG['task'] == 'severity' else nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # CosineAnnealingLR: smoothly decays the learning rate to near-zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("\nStarting training...\n")
    best_metric = 0 if CONFIG['task'] != 'severity' else float('inf')
    history = {'train_loss': [], 'train_acc': [], 'test_metric': []}

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            CONFIG['device'], CONFIG['task']
        )

        if CONFIG['task'] == 'severity':
            mae, rmse, r2, features = evaluate(
                model, test_loader, CONFIG['device'], CONFIG['task']
            )
            test_metric = mae

            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.3f}")

            if mae < best_metric:
                best_metric = mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'mae': mae, 'rmse': rmse, 'r2': r2
                }, os.path.join(CONFIG['save_dir'], 'best_model_pd.pth'))
                print(f"  ✓ New best MAE: {best_metric:.2f}")
        else:
            test_acc, report, cm, features = evaluate(
                model, test_loader, CONFIG['device'], CONFIG['task']
            )
            test_metric = test_acc

            print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Acc: {test_acc:.2f}%")
            if CONFIG['task'] == 'pd_detection':
                print(f"  Sensitivity: {report['1']['recall']*100:.2f}%")
                print(f"  Specificity: {report['0']['recall']*100:.2f}%")

            if test_acc > best_metric:
                best_metric = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'accuracy': test_acc,
                    'report': report,
                    'confusion_matrix': cm
                }, os.path.join(CONFIG['save_dir'], 'best_model_pd.pth'))
                print(f"  ✓ New best: {best_metric:.2f}%")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_metric'].append(test_metric)

        scheduler.step()

    # ── Save history ──────────────────────────────────────────────────────────
    with open(os.path.join(CONFIG['results_dir'], 'history_pd.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ── Plot results ──────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['test_metric'])
    plt.title('Test Performance')
    plt.xlabel('Epoch')
    plt.ylabel('MAE' if CONFIG['task'] == 'severity' else 'Accuracy (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['results_dir'], 'training_curves_pd.png'))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    if CONFIG['task'] == 'severity':
        print(f"Best MAE: {best_metric:.2f}")
    else:
        print(f"Best Accuracy: {best_metric:.2f}%")
    print(f"Checkpoints: {CONFIG['save_dir']}")
    print(f"Results:     {CONFIG['results_dir']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
