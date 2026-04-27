# 🧠 Parkinson's Disease Detection via Gait Analysis (MSCAF-Gait)

> **Deep learning model for non-invasive Parkinson's Disease detection using multi-sensor foot pressure signals**

---

## 📌 Overview

This project implements **MSCAF-Gait** — a Multi-Scale Convolutional Attention Fusion architecture for detecting and staging Parkinson's Disease (PD) using Ground Reaction Force (GRF) signals from foot pressure sensors.

Traditional PD diagnosis relies on subjective clinical assessments. This model offers an **objective, data-driven alternative** by analyzing gait dynamics — a clinically validated PD biomarker — using deep learning.

The system simultaneously handles:
| Task | Output |
|---|---|
| Binary Classification | PD vs. Healthy Control |
| Severity Regression | UPDRS Score Prediction |
| Stage Classification | Hoehn & Yahr (H&Y) Stage |

---

## 🧩 Model Architecture

```
Raw GRF Signal (16 channels × 1000 timesteps)
        │
        ▼
┌─────────────────────────┐
│  Shallow Feature         │  → 1D CNN for low-level temporal feature extraction
│  Extraction (1D CNN)     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Multi-Scale             │  → Parallel convolutions: kernels [1, 3, 5, 7]
│  Convolution Block       │    Captures short and long-range gait patterns
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Channel Attention       │  → Recalibrates feature maps channel-wise (SE-style)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Spatial (Temporal)      │  → Focuses on discriminative time-steps in the gait cycle
│  Attention               │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Bi-directional LSTM     │  → Models forward and backward temporal dependencies
└──────────┬──────────────┘
           │
      ┌────┴─────┐
      ▼          ▼
 Classification  Regression
 Head (PD/H&Y)   Head (UPDRS)
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [PhysioNet – Gait in Parkinson's Disease](https://physionet.org/content/gaitpdb/1.0.0/) |
| **Sensors** | 16 GRF sensors (8 per foot) |
| **Sampling Rate** | 100 Hz |
| **Recording Duration** | ~2 minutes per subject (normal walking) |
| **Subjects** | 93 PD patients + 73 Healthy Controls |

**Preprocessing Pipeline:**
- Sliding window segmentation: `window = 1000 samples (10s)`, `stride = 500`
- Z-score normalization per sensor channel
- Data augmentation via overlapping windows

---

## ⚙️ Configuration

```python
CONFIG = {
    'batch_size'      : 32,
    'epochs'          : 100,
    'learning_rate'   : 1e-4,
    'sequence_length' : 1000,   # 10 seconds at 100 Hz
    'stride'          : 500,    # 50% overlap
    'num_sensors'     : 16,
    'optimizer'       : 'Adam',
    'scheduler'       : 'CosineAnnealingLR',
}
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/parkinsons-detection.git
cd parkinsons-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
Download from [PhysioNet](https://physionet.org/content/gaitpdb/1.0.0/) and update the path:
```python
# In config.py or main.py
CONFIG['data_root'] = '/path/to/gait-in-parkinsons-disease'
```

### 4. Train the Model
```bash
python main.py
```

### 5. Evaluate
```bash
python evaluate.py --checkpoint ./checkpoints_pd/best_model.pth
```

---

## 📁 Project Structure

```
parkinsons-detection/
│
├── main.py                  # Entry point — training & evaluation
├── config.py                # Hyperparameters and path configuration
├── evaluate.py              # Standalone evaluation script
│
├── models/
│   ├── mscaf_gait.py        # Full MSCAF-Gait architecture
│   ├── attention.py         # Channel & Spatial Attention modules
│   └── multi_scale_conv.py  # Multi-scale convolution block
│
├── dataset/
│   ├── loader.py            # PhysioNet GRF data loader
│   └── preprocessing.py     # Normalization + sliding window
│
├── checkpoints_pd/          # Saved model weights (auto-created)
├── results_pd/              # Training curves, confusion matrices (auto-created)
│
├── requirements.txt
└── README.md
```

---

## 📈 Evaluation Metrics

**Classification (PD vs. Healthy / H&Y Staging)**
- Accuracy, Precision, Recall, F1-Score
- Sensitivity & Specificity
- Confusion Matrix (plotted + saved)
- ROC-AUC Curve

**Regression (UPDRS Score)**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 💾 Outputs

All outputs are auto-generated during training:

```
checkpoints_pd/
└── best_model.pth          # Best validation checkpoint

results_pd/
├── training_curves.png     # Loss & accuracy over epochs
├── confusion_matrix.png    # Classification results
└── metrics_summary.json    # Final scores (JSON)
```

---

## 🔬 Key Contributions

- **Multi-scale temporal modeling** — captures gait features at multiple timescales simultaneously
- **Dual attention mechanism** — channel attention for sensor importance, spatial attention for critical gait phases
- **Multi-task learning** — jointly optimizes classification and regression for richer representations
- **Clinical alignment** — outputs map directly to standard PD assessment tools (UPDRS, H&Y)
- **Sliding window augmentation** — effective data expansion for limited clinical datasets

---

## 🧪 Future Work

- [ ] Transformer-based temporal modeling (ViT / Temporal Transformer)
- [ ] Real-time gait inference pipeline
- [ ] Wearable/IoT deployment (edge inference)
- [ ] Generalization across multiple gait datasets
- [ ] Explainability module (GradCAM for temporal attention visualization)

---

## 🤝 Contributing

Contributions and collaborations are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is developed for **academic and research purposes**. Dataset usage is subject to [PhysioNet's data use agreement](https://physionet.org/content/gaitpdb/1.0.0/).

---

## 👨‍💻 Author

**Gurkarnbir Singh**  
B.Tech Computer Science & Engineering  
📧 [your-email@example.com] | 🔗 [LinkedIn] | 💻 [GitHub]

---

## ⭐ Acknowledgements

- [PhysioNet](https://physionet.org/) — for the publicly available gait dataset
- [PyTorch](https://pytorch.org/) — deep learning framework
- Original MSCAF paper authors — for the architectural inspiration

---

*If you find this project useful, please consider giving it a ⭐ on GitHub!*
