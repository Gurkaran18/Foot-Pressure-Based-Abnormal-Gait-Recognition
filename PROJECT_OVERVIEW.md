# Detecting Parkinson's Disease from Foot Pressure — Project Overview

**A comparative study of deep learning and clinical gait features for classifying Parkinsonian gait from plantar pressure signals**

| | |
|---|---|
| **Institution** | Dr. B.R. Ambedkar National Institute of Technology, Jalandhar — Department of CSE |
| **Team** | Gurkaranbir Singh (23103059), Janak Varshney (23103071) |
| **Guide** | Dr. Kunwar Pal |
| **Base paper** | M. Yuan et al., "Foot Pressure-Based Abnormal Gait Recognition With Multi-Scale Cross-Attention Fusion," *IEEE TNSRE*, vol. 33, pp. 3146–3159, 2025 |
| **Dataset** | PhysioNet *Gait in Parkinson's Disease* (gaitpdb 1.0.0) |

---

## 1. Summary

This project builds and tests a system that decides whether a person has Parkinson's disease (PD) from about two minutes of walking in pressure-sensing insoles. There are two models:

1. **MSCAF-Gait**, a re-implementation of the 2025 deep model in the base paper. It learns directly from raw 10-second windows of the 16 insole sensors.
2. **A clinical-feature model.** It detects every step, computes 13 standard gait measures and classifies them with XGBoost or Random Forest.

Both models are judged only on **people they have never seen**, using 5-fold cross-validation at the person level.

**Main results**

| Finding | Number |
|---|---|
| XGBoost on 13 gait features (best model) | **78.4%** balanced accuracy, AUC 0.86 |
| MSCAF-Gait on raw sensors | **65.3%** balanced accuracy, AUC 0.73 |
| Accuracy added by splitting windows randomly instead of by person (leakage) | **+19.2 points** (87.0% vs 67.8%) |
| Leave-one-study-out (train on two studies, test on the third) | **75.2%** mean balanced accuracy |
| Most important gait markers (SHAP) | swing asymmetry, swing-time variability, double support |

**What we learned:** how you test a model matters more than which model you pick. Splitting by window instead of by person adds about 19 points of fake accuracy, which explains much of the 95–99% reported in the literature. On a dataset of 165 people, 13 interpretable features beat a deep network by about 13 points.

---

## 2. Motivation and problem statement

- **Diagnosis is subjective.** PD is diagnosed by a neurologist watching the patient and scoring rating scales (UPDRS, Hoehn & Yahr). Specialists are scarce outside large hospitals.
- **Gait is an objective signal.** PD changes walking in measurable ways: shorter strides, longer double support, irregular step timing and left/right asymmetry. A cheap insole can record these anywhere, without cameras.
- **Published accuracies look too good.** Papers on this dataset report 91–99.6% accuracy. The base paper itself falls from **99.61%** with a random split to **82.31%** when tested leave-one-subject-out (its Table V).

**Problem statement:** given a plantar pressure recording from 16 insole sensors, classify the person as Parkinson's or healthy, with accuracy that holds for people the model has never seen.

**Objectives**

| # | Objective | Status |
|---|---|---|
| O1 | Re-implement MSCAF-Gait (multi-scale convolution, channel and spatial attention, cross-attention fusion) | Done — [main.py](main.py) |
| O2 | Evaluate at the person level with stratified 5-fold CV, no person in both train and test | Done |
| O3 | Build a clinical-feature model: step detection, 13 gait measures, XGBoost / Random Forest | Done — [gait_features.py](gait_features.py) |
| O4 | Measure leakage (random vs person-level split) and cross-study generalisation | Done — [leakage_demo.py](leakage_demo.py), [cross_study.py](cross_study.py) |
| O5 | Explain predictions: SHAP for features, sensor occlusion for the deep model | Done — [explain.py](explain.py) |
| O6 | Working demo app | Done — [app.py](app.py) (Streamlit) |

---

## 3. Dataset

**PhysioNet gaitpdb 1.0.0**, recorded at Tel Aviv Sourasky Medical Center.

| Property | Value |
|---|---|
| People | **165** with recordings: 93 PD, 72 healthy (the demographics sheet lists 166; one person has no recording) |
| Recordings | **306** walks |
| Sensors | 16 vertical-force sensors, 8 under each foot (L1–L8, R1–R8), in newtons |
| Sampling | 100 Hz, about 2 minutes per walk |
| File format | 19 columns: time, 8 left sensors, 8 right sensors, left total force, right total force |
| Extra data | Demographics: age, height, weight, Hoehn & Yahr stage, UPDRS, walking speed |

**Three studies, one gait lab**

| Code | Study | PD / healthy | Walks |
|---|---|---|---|
| Ga | Dual tasking (Yogev et al., 2005) | 29 / 18 | `_01`, `_02` normal; `_10` dual task |
| Ju | Rhythmic auditory cues (Hausdorff et al., 2007) | 29 / 25 | `_01` normal; `_02`–`_07` cued conditions |
| Si | Treadmill walking (Frenkel-Toledo et al., 2005) | 35 / 29 | `_01` |

**File names:** `GaPt03_10` means study **Ga**, **Pt** = patient (**Co** = healthy control), subject **03**, walk **10**. Only files matching `(Ga|Ju|Si)(Co|Pt)NN_NN.txt` are loaded. Earlier versions wrongly loaded `demographics.txt`, `SHA256SUMS.txt` and `format.txt` as subjects.

**Sensor layout** (from `format.txt`; +Y points toward the toes): L1/R1 at the heel (Y = −800), L8/R8 at the toes (Y = +800). Odd-numbered sensors L3, L5, L7 (and R3, R5, R7) lie on the inner side of the foot.

---

## 4. The base model: MSCAF-Gait

**Idea:** build three views of the gait signal and let them query each other with cross-attention instead of simply adding them. The views are:
- **M**, multi-scale convolution features
- **H**, channel attention ("which features matter")
- **S**, spatial attention ("which moments matter")

### 4.1 What the paper specifies

| Component | Paper definition |
|---|---|
| Multi-scale convolution | Four parallel branches with kernels 1×1, 1×3, 1×5, 1×7, each Conv → BN → ReLU → MaxPool (Eq. 1). Outputs are concatenated (Eq. 2), then a pixel-level Conv(1×1) + BN + ReLU gives **M**. |
| Channel attention | Global average pool over time → 1×1 conv (reduce) → ReLU → 1×1 conv (restore) → sigmoid → reweight M, giving **H** (Eqs. 3–5). |
| Spatial attention | Mean and max over channels → concatenate → Conv(1×1) from 2 to C channels → BN → ReLU → sigmoid → reweight M, giving **S** (Eqs. 6–8). |
| Cross-attention fusion | Q/K/V projections from M, H and S. All **6 ordered pairs** (M→H, M→S, H→M, H→S, S→M, S→H), no self-attention, each computing softmax(QKᵀ/√d_k)·V. The six outputs are concatenated in time, giving C × 6W (Algorithm 1). |
| Head | Conv(1×3) + BN + ReLU + MaxPool → fully connected layer → softmax |
| Paper setup on GaitinPD | 1,600 ms windows, 50% overlap, Min-Max normalisation, random 6:2:2 split, Adam, up to 100 epochs, Optuna tuning (lr 1.9×10⁻⁴, batch 128) |
| Paper results | 99.61% (PD vs healthy, random split), 98.88% (H&Y staging), **82.31% leave-one-subject-out**, 0.08 M parameters on GaitinPD |

### 4.2 Our implementation ([main.py](main.py))

| Choice | Our value | Note |
|---|---|---|
| Input | 16 sensors × 1,000 samples (10 s), stride 500 (50% overlap) | The paper uses 1.6 s windows |
| Normalisation | z-score per sensor, **per recording** | The paper uses Min-Max |
| Shallow CNN before multi-scale | 4 × [Conv1d + BN + ReLU + Dropout 0.2], 3 × MaxPool; channels 32/64/128/128, kernels 5/3/3/3; 1,000 → 125 time steps | **Not in MSCAF-Gait.** This is the SFE block from Chen et al. (IEEE TNNLS 2025, Table I), added by us |
| Multi-scale | 4 branches × 64 channels, each with MaxPool → 256 channels, 1×1 conv to 128 = M; W = 62 | One time step = 80 ms after the shallow CNN, so the kernels span 80–560 ms |
| Channel attention | Reduction r = 16 (128 → 8 → 128) | r is our choice |
| Spatial attention | Conv 2 → 128, kernel 1 (as in the paper) | `spatial_kernel = 7` gives a CBAM-style variant |
| Cross-attention | d_k = 64, attention dropout 0.1 | d_k is our choice |
| Head | Conv(k=3) + BN + ReLU + MaxPool → mean-pool over time → Linear 128 → 128 → 2 | Optional BiLSTM (`use_lstm`, off by default) |
| Size | **415,786 parameters (≈ 0.42 M)** | The paper's model is about 0.08 M; the difference is mainly the shallow CNN |
| Ablation flag | `fusion = 'add'` gives the H + S additive baseline | |

**Training (per fold):** Adam (lr 1×10⁻⁴, weight decay 1×10⁻⁴), cosine annealing, batch 32, up to 100 epochs. Early stopping has patience 15 and a minimum of 20 epochs. Cross-entropy is class-weighted with inverse-frequency weights computed from that fold's training windows only. The checkpoint is chosen on **validation balanced accuracy**. Seed 42.

---

## 5. The clinical-feature model ([gait_common.py](gait_common.py), [gait_features.py](gait_features.py))

### 5.1 Step detection
Applied to the left and right total-force columns:
1. Smooth with a 50 ms moving average.
2. **Contact** = force above 10% of the swing-to-peak range. Force rising above it is a **heel strike**; falling below it is a **toe-off**.
3. Drop contacts or gaps shorter than 100 ms (sensor chatter).
4. Keep strides lasting 0.5–3.0 s, and drop strides more than 3 robust SDs (MAD) from the median. This removes turns at the end of the walkway.

A recording gives a median of about 190 clean strides counting both feet, or about 95 per foot.

### 5.2 The 13 gait features
Computed per walk, then averaged over each person's walks:

| Group | Features |
|---|---|
| Timing | stride time (mean), cadence, stance %, swing %, double support % |
| Variability | stride-time CV, swing-time CV |
| Asymmetry and force | L/R stride asymmetry, swing asymmetry, peak-force asymmetry; peak force L, peak force R; peak force ÷ body weight |

### 5.3 Classifiers
Fixed hyper-parameters, with no tuning on the test folds. Missing values are filled with the median.
- **Random Forest:** 500 trees, min_samples_leaf 2, max_features √.
- **XGBoost:** 300 trees, depth 3, learning rate 0.05, subsample 0.8, colsample 0.8, min_child_weight 2.
- **Late fusion:** p = (p_features + p_MSCAF) / 2, threshold 0.5. The weights are fixed, so nothing is fitted on test data.

---

## 6. Evaluation design

- **5-fold stratified cross-validation at the person level.** Folds are built over the 165 subject IDs with `StratifiedKFold` (seed 42), so each fold keeps the PD/healthy ratio. Every person is tested exactly once, by a model that never saw them. Each fold has 99 training, 33 validation and 33 test people, and the code asserts that no person appears in two splits.
- **The test fold is scored once**, after restoring the best checkpoint by validation score.
- **Subject-level prediction:** MSCAF scores each 10 s window, and each person gets a majority vote over their windows. The ROC curve uses the mean window probability.
- **Pooled out-of-fold metrics:** one prediction per person across all 165, rather than an average of five small folds.
- **Metrics:** balanced accuracy (mean of sensitivity and specificity, the headline metric because classes are imbalanced), sensitivity, specificity, ROC-AUC and confusion matrix. The majority-class baseline is 50% balanced accuracy, or 56.4% plain accuracy.
- **All post-hoc scripts use the same folds** (`make_cv_folds`) as MSCAF, so every comparison is like for like.

**Integrity fixes made during the project.** These are documented because earlier numbers were inflated:
1. The split was by recording file rather than by person, so the same person appeared in train and test. **Fixed.**
2. The model was selected on the test set, and `evaluate.py` scored training subjects. **Fixed.**
3. Plain accuracy was used for model selection, which rewards collapsing to the majority class. **Switched to balanced accuracy.**
4. Non-subject `.txt` files were loaded as PD subjects. **Fixed.**

---

## 7. Results

All results are pooled out-of-fold, at subject level: 165 people (93 PD, 72 healthy).

### 7.1 Every model under the same protocol

| Model | Balanced acc. | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Majority-class baseline | 50.0 | — | — | 0.50 |
| MSCAF-Gait (raw 16 sensors) | 65.3 | 66.7 | 63.9 | 0.73 |
| Random Forest (13 features) | 75.2 | 79.6 | 70.8 | 0.84 |
| **XGBoost (13 features)** | **78.4** | **81.7** | **75.0** | **0.86** |
| MSCAF + Random Forest (fusion) | 71.1 | 74.2 | 68.1 | 0.81 |
| MSCAF + XGBoost (fusion) | 74.8 | 77.4 | 72.2 | 0.83 |

- **XGBoost:** catches 76 of 93 patients and clears 54 of 72 healthy people. Confusion matrix [[54, 18], [17, 76]].
- **MSCAF-Gait:** catches 62 of 93 and clears 46 of 72. Confusion matrix [[46, 26], [31, 62]].
- **Fold stability:** XGBoost 78.4 ± 2.6, MSCAF 65.5 ± 8.0. MSCAF folds range from 55.1% to 74.4%, so a single train/test split could mislead in either direction.
- **Fusion did not help.** Averaging with the weaker MSCAF probability pulled XGBoost down. This is an honest negative result.

### 7.2 Leakage experiment ([leakage_demo.py](leakage_demo.py))
Same XGBoost, same 6,185 ten-second windows with features computed per window, scored at window level:

| Split | Accuracy | Balanced acc. | AUC |
|---|---|---|---|
| Random window split (leaky) | 86.7 | **87.0** | 0.945 |
| Person-level split (honest) | 68.7 | **67.8** | 0.733 |

In the leaky split, 100% of test windows belong to a person who also appears in training. The extra **19.2 points** come from recognising people (body weight, rhythm), not disease.

### 7.3 Cross-study generalisation ([cross_study.py](cross_study.py))
XGBoost trained on two studies and tested on the third:

| Tested on | Trained on | Balanced acc. | Sensitivity | Specificity | AUC |
|---|---|---|---|---|---|
| Ga (47) | Ju + Si | 77.1 | 93.1 | 61.1 | 0.82 |
| Ju (54) | Ga + Si | 73.7 | 79.3 | 68.0 | 0.83 |
| Si (64) | Ga + Ju | 74.8 | 60.0 | 89.7 | 0.83 |
| **Mean** | | **75.2** | 77.5 | 72.9 | 0.83 |

This is close to the 78.4% from cross-validation, so the model learns PD gait rather than one protocol. The decision threshold does shift between studies: Ga leans toward sensitivity, Si toward specificity. A variant that standardises features within each study scored lower, at 70.8% mean.

### 7.4 Explainability ([explain.py](explain.py))
Both analyses are out-of-fold: each person is explained by a model that never saw them.

**SHAP, XGBoost (mean |SHAP|, log-odds of PD):**
swing asymmetry 1.08 › swing-time CV 1.04 › double support 0.65 › peak-force asymmetry 0.30 › stride asymmetry 0.29 › peak force R 0.27 › stride time 0.27 › stride-time CV 0.21 › … › cadence 0.07 › swing % 0.06.
The top three are established clinical markers of Parkinsonian gait.

**Sensor occlusion, MSCAF-Gait:** each z-scored sensor is zeroed on held-out windows (6,229 windows; baseline window AUC 0.711).
- By **drop in true-class probability**, the most important sensors are **L3** (5.7 pp), **R5** (4.1), **L8** (3.7) and **L5** (2.7). These are inner rear-foot and midfoot sensors.
- By **drop in window AUC**, the order differs: **L1** (heel, 0.099), L3 (0.071), L8 (0.064), L6 (0.060).

### 7.5 Why the deep model lags
1. A 10 s window holds only about 8 strides per foot, too few to measure the stride-to-stride variability that is the strongest PD signal.
2. 165 people is little data for 0.42 M parameters.
3. Per-recording z-scoring removes absolute force and body-weight information.
4. Outputs are over-confident (many window probabilities above 0.999), so AUC 0.73 is the fairer view.
5. Compared with the paper's 82.31% leave-one-subject-out: our architecture differs (added shallow CNN, 10 s vs 1.6 s windows, z-score vs Min-Max), we report balanced accuracy over 5 folds rather than plain accuracy over leave-one-subject-out, and we select checkpoints on a validation set.

---

## 8. Literature context

Reported accuracy on gaitpdb, as compiled in Table III of the base paper:

| Work | Method | PD vs healthy | H&Y staging |
|---|---|---|---|
| Daliri, 2013 | SVM, chi-square kernel | 91.20 | — |
| Açıcı et al., 2017 | Random Forest | 98.04 | — |
| Veeraragavan et al., 2020 | ANN | 97.40 | 87.10 |
| Xia et al., 2020 | CNN-LSTM with attention | 99.07 | 98.03 |
| Liu et al., 2021 | Dual-branch CNN-BiLSTM | 99.22 | — |
| Nguyen et al., 2022 | Transformers | 95.2 | — |
| Naimi et al., 2023 | Hybrid ConvNet-Transformer | 97.00 | 87.00 |
| **Yuan et al., 2025** | **MSCAF-Gait (base paper)** | **99.61** | **98.88** |

**Gap:** these figures come from splits that can put one person in both train and test. The base paper's own leave-one-subject-out result is 82.31%. There is little person-level or cross-study testing in this literature, and little explanation of what models rely on. The survey by Sepas-Moghaddam & Etemad (IEEE TPAMI 2023) draws the same distinction between subject-dependent and subject-independent test protocols.

---

## 9. Demo app ([app.py](app.py))

```bash
streamlit run app.py
```
1. Upload a gaitpdb walk (.txt, 19 columns).
2. Steps are detected and the 13 gait features computed.
3. The app shows P(PD) from XGBoost, plus MSCAF-Gait and fusion when the fold checkpoint is available.
4. It shows a foot heatmap of mean plantar force and each feature next to the cohort's healthy and PD medians.

For files from the dataset, the app shows the **out-of-fold** prediction, so it never scores a person the model trained on.

---

## 10. Repository and how to run

```
main.py            MSCAF-Gait model, data loader, training, 5-fold CV
evaluate.py        Score a single-split checkpoint (use_cv = False path)
gait_common.py     Shared paths, folds, step detection, features, models
gait_features.py   RF / XGBoost on gait features + late fusion
leakage_demo.py    Window-level vs subject-level split
explain.py         SHAP + sensor occlusion
cross_study.py     Leave-one-study-out
results_summary.py Slide-ready summary table
app.py             Streamlit demo
results_pd/        cv_results.json, cv_summary.png   (MSCAF CV output)
results_new/       metrics JSON/CSV and figures from the post-hoc scripts
checkpoints_pd/    best_model_pd_fold{1..5}.pth
```

```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/gait-in-parkinsons-disease-1.0.0
python main.py             # 5-fold CV for MSCAF-Gait (GPU recommended)
python gait_features.py    # feature models + fusion
python leakage_demo.py
python explain.py
python cross_study.py
python results_summary.py
streamlit run app.py
```

**Dependencies:** PyTorch ≥ 2.0, NumPy, pandas, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, openpyxl, joblib, Streamlit. The final runs were done on Kaggle with a CUDA GPU.

---

## 11. Conclusions

**Delivered:** a full MSCAF-Gait implementation, leak-free 5-fold person-level evaluation, a 78.4% clinical-feature model, leakage and cross-study experiments, out-of-fold explainability, and a working demo.

**Learned:**
- Evaluation protocol matters more than architecture: leakage alone adds about 19 points.
- On small clinical datasets, interpretable gait features beat a deep model and generalise across studies.
- None of these results are clinically usable yet. They are an honest research baseline.

**Limitations:**
- One dataset, from one gait lab, with 165 people.
- MSCAF-Gait has high variance from fold to fold.
- The H&Y staging and UPDRS tasks exist in code (`CONFIG['task']`) but were not part of the final evaluation.

**Future work:**
- Longer windows, and gait features fed into MSCAF.
- Self-supervised pre-training.
- Severity (H&Y) staging.
- Calibrated probabilities.
- New cohorts and real wearable deployment.

---

## 12. References

1. M. Yuan et al., "Foot pressure-based abnormal gait recognition with multi-scale cross-attention fusion," *IEEE Trans. Neural Syst. Rehabil. Eng.*, vol. 33, pp. 3146–3159, 2025.
2. A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation*, vol. 101, no. 23, 2000. Dataset: Gait in Parkinson's Disease (gaitpdb 1.0.0).
3. G. Yogev et al., "Dual tasking, gait rhythmicity, and Parkinson's disease," *Eur. J. Neurosci.*, vol. 22, no. 5, 2005.
4. J. M. Hausdorff et al., "Rhythmic auditory stimulation modulates gait variability in Parkinson's disease," *Eur. J. Neurosci.*, vol. 26, no. 8, 2007.
5. S. Frenkel-Toledo et al., "Treadmill walking as an external pacemaker to improve gait rhythm and stability in Parkinson's disease," *Mov. Disord.*, vol. 20, no. 9, 2005.
6. C. G. Goetz et al., "Movement Disorder Society task force report on the Hoehn and Yahr staging scale," *Mov. Disord.*, vol. 19, no. 9, 2004.
7. X. Liu et al., "A dual-branch model for diagnosis of Parkinson's disease based on the independent and joint features of the left and right gait," *Appl. Intell.*, vol. 51, 2021.
8. Y. Xia et al., "A dual-modal attention-enhanced deep learning network for quantification of Parkinson's disease characteristics," *IEEE TNSRE*, vol. 28, no. 1, 2020.
9. A. Sepas-Moghaddam and A. Etemad, "Deep gait recognition: A survey," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 1, pp. 264–284, 2023.
10. Y. Chen, Y. Zhao and X. Li, "Adaptive gait feature learning using mixed gait sequence," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 36, no. 1, pp. 1545–1554, 2025. *(Source of the shallow feature-extraction block used before the multi-scale convolution.)*
11. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. ACM SIGKDD*, 2016.
12. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. NeurIPS*, 2017.
