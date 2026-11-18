# CASS-Net Feasibility and Novelty Review

**Date:** 18 November 2026 · **Scope:** a go/no-go review before any implementation · **Companion to:** [RESEARCH_EXTENSION_PLAN.md](RESEARCH_EXTENSION_PLAN.md)

**How claims were checked**
- **Primary sources.**
  - The WearGait-PD paper, full text via Europe PMC.
  - The Synapse REST API.
  - Crossref abstracts for the MDPI papers.
  - Full texts of SENN and Yu et al. via Europe PMC.
- **Our own measurements.**
  - A signal audit of the downloaded gaitpdb files using our own stride detector (`gait_common.trial_strides`).
  - A CPU benchmark of a concrete CASS-Net.
- **Tags.** [M] = measured by us in this review; [F] = full text read; [A] = abstract only; [K] = standard reference cited from knowledge.

---

## Verdict (summary)

| Decision | Verdict |
|---|---|
| CASS-Net **as specified**, as the paper's **primary contribution** | **NO-GO** |
| A **reduced** stride-set model (no auxiliary loss, cross-fitted clinical offset), as a **secondary arm behind three gates** | **CONDITIONAL GO** |
| Cross-device external validation on WearGait-PD (data verified: open, CC BY 4.0) | **GO**, as the new primary contribution, with a redesigned protocol (§8) |
| P0, the subject-level re-audit of MSCAF-Gait | **GO**, unchanged |

**Why not CASS-Net as the primary contribution**
1. **Most of its architecture already exists.**
   - **PAS-MIL** (Farfoura et al., *Applied Sciences*, Aug 2026) built the same core idea on the same dataset with the same protocol family: session-level attention-MIL over gait segments, plus predefined physiological descriptors, plus static-feature experts and distillation from a tree model.
   - The pooling operator is standard attentive statistics pooling.
   - The gated residual is a standard offset/ReZero design.
2. **The closest prior attempt did worse than simple features.** PAS-MIL reached a pooled AUC of **0.771**, below our 13-feature XGBoost (0.86) and the COP-feature models (0.89–0.91).
3. **Walking speed alone gets AUC 0.779 on gaitpdb [M].** This is a number from a stopwatch. Any architecture paper that does not beat "speed + clinical features" has little to say.
4. **The external test set mostly offers short, interrupted walks.**
   - On WearGait-PD the straight-walking tasks are 4 passes over a 16-ft walkway, with turns. So the "whole-walk, many-strides" premise of CASS-Net largely fails there.
   - The dataset's FreeWalk task is longer, but each site used a different route with stairs and chairs.
5. **The design has internal flaws** that would bias its key hypothesis tests (§7: flaws 1–3).

**What is genuinely open, and does not need CASS-Net:** a *documented* cross-device, cross-site external validation of plantar-pressure PD classifiers. It should:
- compare representations tiered by how device-invariant they are
- match the number of strides to what the target protocol provides
- control for speed and age

---

## 1. WearGait-PD — verified facts

| Item | Verified finding | Source |
|---|---|---|
| Existence and official links | Synapse project **syn52540892**: "Wearables for gait in Parkinson's Disease and age-matched controls (WearGait-PD)". Created 2023-09-22; modified 2026-06-12 | Synapse API [F] |
| Peer-reviewed publication | Anderson AJ, Eguren D, Gonzalez MA, …, Butala A, Kontson K. *Scientific Data* **13**:440 (2026). [doi:10.1038/s41597-026-06806-2](https://doi.org/10.1038/s41597-026-06806-2). PMID 41680227, **PMC13009270**. Preprint: medRxiv [10.1101/2024.09.11.24313476](https://doi.org/10.1101/2024.09.11.24313476) | Europe PMC [F] |
| Regulatory record | FDA Catalog of Regulatory Science Tools, **RST24NO01.1** (cross-sectional) and **RST26IP02.01** (longitudinal) | Synapse wiki [F]; FDA page [F] |
| Licence | "licensed under the Creative Commons Attribution 4.0 International (**CC BY 4.0**) license" | Paper, Usage Notes [F] |
| Access | Register on Synapse (name + valid e-mail), accept the Synapse Terms of Use, and take the "Synapse Pledge". The API lists **no extra access requirements** on the project or the Version 1 folder. I did not check access requirements on individual files | Paper [F]; Synapse API `accessRequirement` returned 0 [F] |
| Documentation | Wiki pages: data collection plan, data description, data access, publications. Supplementary Table S5 describes the variables and coordinate systems. MATLAB quality-control code is in the `Code` folder | Wiki + paper [F] |
| Structure | *Version 1 (Cross-sectional)*: `CONTROL PARTICIPANTS/`, `PD PARTICIPANTS/`, demographic + clinical CSVs, `Real-world context tasks/`, `Walkway-derived metrics/`. *Version 2 (Longitudinal)*: 47 PD participants re-assessed | Synapse API [F] |
| Participants: PD | **100** (35 F / 65 M). Age **67 ± 8**. Modified H&Y **2.15 ± 0.61**. MDS-UPDRS III 6–59, mean 25 ± 10 | Paper [F] |
| Participants: controls | **85** (48 F / 37 M). Age **74 ± 9**, so controls are older than the PD group | Paper [F] |
| Subsets other papers could analyse | 181 (100 PD / 81 controls) in an IMU study. 133 (65 / 68) for self-paced walkway metrics. 87 PD with both of two tasks | Herath 2026 [A]; Sifat 2610.16220 [F, repo]; Yu 2026 [F] |
| Insoles | **Moticon OpenGo**: "sixteen distributed pressure sensors" per insole ("absolute pressure"), plus an on-board 3-axis accelerometer and gyroscope. App version 03.11.00 | Paper [F] |
| Insole sampling rate | **Not stated in the main text** (IMUs: 100 Hz). OpenGo supports 10–100 Hz. The survey lists 100 Hz, synchronised. **Confirm in Table S5 or the files** | Paper [F]; Moticon [A]; survey [S] |
| Other sensors | 13 Xsens MTw Awinda IMUs; ProtoKinetics Zeno walkway (16 ft × 2 ft); GoPro video with frame-by-frame annotation | Paper [F] |
| Walking tasks | SelfPace (SP) and HurriedPace (HP): 4 passes each, starting 5 ft before the mat and ending 5 ft after it, turning off the mat. SP_mat and HP_mat: 4 passes on the mat only, stopping at the ends. SP_matTURN; TandemGait; TUG (3 trials); Balance (6 × 10 s); **FreeWalk**: hallways, stairs if available, sit/stand, on a site-specific route | Paper [F] |
| Labels | MDS-UPDRS; modified H&Y; medication type and dose; **time since last dose** (the protocol did not control ON/OFF state); DBS; height, weight, sex; freezing-of-gait annotations (145 events in 12 PD participants) | Paper [F] |
| Sites | JHOC (ID prefix NLS), Bayview (HC), VA-Seattle (WPD/WHC) | Paper [F] |
| Data quality | 1,568 trials. **83% complete**. 259 files had significant loss (dropout ≥ 7 frames or a lost sensor). Insole desynchronisation was corrected by shifting and trimming | Paper [F] |

**Consequences for our design** (interpretation)
- **Walk length.**
  - *Estimate from the protocol geometry:* each SP pass covers about 7.9 m, roughly 6–7 strides per foot including starting and stopping. So 4 passes give about 24–28 strides per foot, perhaps 16–20 of them steady-state.
  - gaitpdb usual walks give 74–193 strides per subject [M].
  - *Verify after download.*
- **Age confound runs in the opposite direction.** WearGait-PD controls are about 7 years older than its PD group. In gaitpdb the medians are 64 vs 68 [M]. Older controls walk slower, so speed-based markers may lose discrimination for reasons unrelated to the device.
- **Site confound.** Only VA-Seattle recruited both groups. Site alone predicts diagnosis with AUC 0.955 in walkway data (Sifat et al., arXiv 2610.16220).
- **Medication state is uncontrolled.** Analyse by time since last dose.
- **Blinding is hard.** Group is visible in the folder names (`PD PARTICIPANTS`, `CONTROL PARTICIPANTS`) and in the VA-Seattle ID prefixes (WPD/WHC). A label-blind harmonisation step needs one teammate to make a recoded, label-stripped copy.

---

## 2. Prior work on stride or sequence modelling combined with clinical features

| Work | Instance unit | Clinical / handcrafted input | Aggregation | Protocol | Result |
|---|---|---|---|---|---|
| **PAS-MIL** — Farfoura, Alkhatib, Elkhodr, El Didi, Al-Sabbagh, *Applied Sciences* 16(17):8354, Aug 2026 [A] | "bag of temporal windows" per session | 8 predefined bilateral descriptors + 8 learned latent dimensions; "static-feature expert" pathway | Attention pooling; concept-guided and prototype pathways; distillation from inner out-of-fold ExtraTrees probabilities | gaitpdb, 5-fold person-grouped CV, 306 sessions / 165 participants | Pooled session AUC **0.771**, mean fold AUC 0.826 ± 0.041; per-study AUC 0.74–0.79; ECE **0.291** after temperature scaling |
| **SENN** — Farfoura, Alkhatib, Connie, *Sensors* 26(9):2671, Apr 2026 [F] | 128-sample windows | 16-concept bottleneck (concepts linked to midfoot loading, step variability, cadence asymmetry) | Mean of calibrated window probabilities | "306 subjects", 70/15/15 split | Subject-level AUC 0.916. **But** 306 is the number of *recordings* in gaitpdb; there are 165 people. A "subject" split over 306 units can put the same person in train and test (Ga and Ju subjects have 2–7 recordings each). Probably inflated |
| Xia et al. 2020 (*IEEE TNSRE*); Liu et al. 2021 | Gait cycles | None | Per-cycle classification | Random splits | 99%+, per the base paper, which notes both "depend on extracting gait cycles" [B] |
| Choi et al. 2026 (*Bioengineering*) [P] | 1-s windows, tokens of per-sensor mean and SD | Summary statistics per window | TCN / BiGRU / Transformer | 5 random subject splits | AUC 0.93–0.94 |
| Herath et al. 2026 (*Bioengineering*, WearGait-PD IMUs) [A] | Task-level tokens from a frozen ST-GCN | — | 3-layer Transformer | Nested, subject-disjoint | BAcc 0.834. "the aggregator contributes gains inseparable from noise" |
| Sifat et al., arXiv 2609.22956 [P] | Stance-indexed AP-COP descriptors (handcrafted) | Yes (all handcrafted) | Median across stances | 20×5 nested participant CV | AUC 0.894–0.908 |

**Conclusion.**
- **Stride segmentation as a model input** on gaitpdb: *not new* (Xia 2020; Liu 2021).
- **Session-level attention-MIL fused with physiological descriptors** on gaitpdb with person-grouped CV: *already done* (PAS-MIL), with weaker results than features alone.
- **Walk-level attention over detected strides with explicit variability and asymmetry statistics:** *not found* in the plantar-pressure PD literature. However, the operator is attentive statistics pooling, i.e. attention-weighted mean + SD (Okabe et al., Interspeech 2018; x-vector statistics pooling, Snyder et al., ICASSP 2018 [K]). So the novelty is application-level only.

---

## 3. Gated residual fusion and device-agnostic representations

**Gated residual fusion — not novel as a method.**
- It is the classic "offset" model: boosting from a baseline logit, or stacking.
- Related architectures: Wide & Deep (Cheng et al. 2016), ReZero-style zero-initialised gates (Bachlechner et al. 2021) and gated multimodal units (Arevalo et al. 2017) [K].
- Within gaitpdb, PAS-MIL already anchors its deep model to a tree model's out-of-fold probabilities (confidence strata and distillation targets).
- What would be new is the *use*: a formal test of incremental value over a strong clinical + COP + speed baseline. That is an evaluation contribution, not an architectural one.

**Device-agnostic plantar-pressure representations — open, but only partly.**
- No published harmonisation method between insole systems was found.
- Yu et al. 2026 (*Diagnostics*, [F]) report "external validation" on gaitpdb (84.3% accuracy, PD-only severity). However:
  - their model takes **66 channels** (30 IMU, 32 Moticon pressure, 4 COP), while gaitpdb provides 16 force channels and no IMU
  - the paper does not describe how this mismatch was resolved
  - the severity-mapping text contradicts itself

  The transfer is therefore undocumented and not reproducible from the paper.
- A course project (UCalgary BMEN 600, GitHub, unpublished) plans a gaitpdb → WearGait-PD test using timing features only.

**Conclusion.** A documented, reproducible cross-device transfer with representations tiered by how invariant they are is still open. This is the most defensible novelty available to us, and it does not depend on CASS-Net.

---

## 4. CASS-Net vs the closest architectures

| | MSCAF-Gait (base) | PAS-MIL | SENN | CASS-Net (proposed) |
|---|---|---|---|---|
| Instance | 1.6-s window | Window, in a session bag | 1.28-s window | Detected stride (101-point resampled) + duration features |
| Clinical knowledge | None | 8 descriptors + static-feature expert + tree distillation | Concept bottleneck | Clinical + COP logistic anchor |
| Aggregation | None (window classifier) | Attention-MIL | Mean probability | Attentive statistics pooling + L/R asymmetry |
| Fusion | — | Multi-pathway experts | — | Zero-initialised gated residual on the anchor logit |
| Subject-level evaluation | LOSO only as a secondary result | Yes | Unit likely mislabelled | Yes (nested), plus leave-one-study-out |
| External / cross-device | No | No | No | Planned (WearGait-PD) |
| Best reported subject-level AUC on gaitpdb | 82.31% accuracy (LOSO) | 0.771 pooled | 0.916 (doubtful) | — |

**Genuinely novel, in descending order of strength:**
1. A cross-device, cross-site external test with tiered invariance, strides matched to the target protocol, and age/speed control. *(Independent of CASS-Net.)*
2. A test of incremental value over **walking speed** + clinical/COP features. *(Independent of CASS-Net.)*
3. A curve of AUC against the number of strides (K-curve), measured honestly. This tells clinicians how much walking is needed. *(Any model can produce it; features are enough.)*
4. Stride tokens with attentive statistics pooling for plantar pressure. *(Application-level novelty only, and only if it beats the controls in §7.)*

**Not novel:** session MIL, physiological descriptors combined with learned features, attention pooling, statistics pooling, gated or residual fusion, temperature scaling, and stride segmentation on gaitpdb.

---

## 5. Does gaitpdb contain enough information for stride-level modelling? [M]

Audit of all **306 recordings from 165 subjects**, using our own stride detector:

| Check | Result | Meaning |
|---|---|---|
| Sampling | 100.0 Hz in every file | — |
| Bandwidth of total force | 99% of power below **4.6 Hz** (median; maximum 9.3). 99.9% below **11.2 Hz** (maximum 26). Power above 15 Hz: **0.04%** of the total | 100 Hz (Nyquist 50 Hz) is far more than needed. No aliasing concern |
| Total-force columns | Exactly equal to the sum of the 8 sensors (maximum difference 0.000 N) | The "total" channel carries no independent information. Canonical channels can be derived without loss |
| Signal hygiene | 0 negative samples. 0 constant channels. Median swing-phase baseline = 0 (maximum 2.6% of peak) | Clean zero baseline. No offset correction needed |
| Samples per stride | Median minimum 101, median maximum 114 (extremes 91–226) | 101 points is about the native resolution; a slight downsample for slow strides |
| Loss from 101-point time normalisation | Force shape: **0.39%** relative RMS (p90 0.60%, maximum 1.1%). AP-COP: **0.16%** (p90 2.0%). At 51 points: 1.1% | Effectively lossless *for shape*. **Duration is discarded**, so it must be kept as separate features (τ) |
| Valid strides per foot, usual walking | Ga 98 [86–110], Si 98 [88–109], Ju 74 [42–101] (median [p10–p90]) | Enough for variability statistics |
| Usual-walking strides per subject | Ga median 193 (two walks), Ju 74 (minimum 31), Si 98. Only **12 of 165** subjects have < 64; 1 has < 32 | — |
| Continuity | 4–6 uninterrupted runs per walk; the **longest run is only 21–28 strides** (turns and quality-control drops) | The order of strides is broken at turns. This favours **order-free (set) pooling** over sequence Transformers. Long-range fractal measures (DFA) are not reliable within runs of about 20 strides |
| Spatial resolution | 8 sensors per foot at **5 front-to-back levels** (Y = −800, −400, 0, 400, 800) and 2 side-to-side columns | AP-COP is a coarse load-centre proxy. Side-to-side COP is very coarse |
| Demographics | Weight missing for 1–2%. Speed_01 missing for 1%. PD H&Y only **2.0 / 2.5 / 3.0** (55 / 28 / 10) | Body-weight normalisation is feasible. **Severity range too narrow for staging** |
| Speed confound | **Speed_01 alone: AUC 0.779** (Ga 0.792, Ju 0.836, Si 0.721; n = 165) | Must be a mandatory baseline |

**Answer.** Yes. The recordings hold enough information for stride-level modelling with negligible signal loss, provided stride durations are kept as explicit features. Three things are lost or limited, and none is fixable by architecture:
- stride order across turns
- fine spatial COP detail, because there are only 8 sensors per foot
- clinical severity range

---

## 6. Compute — is about 15k parameters realistic? [M]

A concrete specification was benchmarked on an Apple M2 CPU with 8 threads:
- encoder: 3 convolutions (4→16→32→32, kernel 5)
- duration MLP
- gated attentive statistics pooling
- MLP head, auxiliary head and logistic anchor

| Quantity | Measured |
|---|---|
| Parameters | **16,815**: encoder 9,888; pooling 1,073; head 5,185; auxiliary 644; anchor 24 |
| Encoder cost | about 0.3 M multiply-accumulates per stride |
| One training epoch (about 170 walks, batch 16, both feet), all 96 strides per foot | **9.5 s** |
| Same, with training-time subsampling to 32 strides per foot | **3.2 s** |
| Inference per walk | 41 ms (K = 96); 16 ms (K = 32) |
| Encoding one new stride (streaming use) | **0.3 ms**. This answers the base paper's real-time objection |
| Full repeated nested search (5×5 outer × 12 configurations × 3–5 seeds × 80 epochs) | **190–316 CPU-hours** at K = 96; **64–106 CPU-hours** at K = 32 |

**Answer.**
- The parameter count is realistic: 16.8k measured against the roughly 15k estimate.
- **The plan's compute claim was wrong.** It said a full nested CV on CPU "should take minutes to a few hours".
- A single configuration trains in minutes on a CPU. The full repeated nested search needs one of the following:
  - a Kaggle GPU (not measured here; expect a large speed-up for this convolution-dominated workload)
  - training-time stride subsampling
  - a pre-declared grid of 4 configurations or fewer, with 3 seeds and 1 repeat for selection

---

## 7. Methodological flaws in the proposed design

| # | Severity | Flaw | Fix |
|---|---|---|---|
| 1 | **Critical** | **Stacking leakage in the residual.** The clinical anchor β is fitted on the training fold *in-sample*. The deep residual is then trained on the same subjects against overfitted anchor logits, so it learns to correct in-sample errors that will not occur on test subjects | Use **cross-fitted offsets**: anchor logits for training subjects come from inner out-of-fold models. Refit the anchor on the full training fold only for test-time prediction |
| 2 | **Critical** | **The auxiliary loss works against the residual hypothesis.** Forcing z to predict the clinical features x makes the deep branch *redundant with the anchor*. That biases H2 (incremental value) towards a null result | Drop the auxiliary loss from the residual model. Keep probing as a separate, post-hoc analysis, or keep the auxiliary loss only in the standalone variant |
| 3 | **Critical** | **The temporal-context test is confounded.** "10-s windows vs stride tokens" changes several things at once: segment unit, alignment, time normalisation, duration features, channels and pooling. A win cannot be attributed to temporal context | Use a factorial design with a **window-MIL control**: pool all windows of a walk with the *same* attentive statistics pooling. Add stride sets with and without τ. Only stride-set > window-MIL shows that stride alignment matters |
| 4 | **Critical** | **Mismatch with the target protocol.** CASS-Net assumes long continuous walks. WearGait-PD's straight-walking tasks give an estimated 16–28 interrupted strides per foot. Variability statistics from about 20 non-steady strides are noisy and biased | Make the **K-curve the gate**. Train with stride subsampling and evaluate on gaitpdb at the K that WearGait-PD actually provides, *before* the external test. Exclude the first and last stride of each pass |
| 5 | **Major** | **Speed confound not in the design.** Speed alone gives AUC 0.779 [M] | Add speed-only and clinical + speed baselines. The primary incremental-value test should be over **clinical + COP + speed**. WearGait-PD has walkway-derived speed, so this can be repeated externally |
| 6 | **Major** | **Overstated device-agnostic claim.** Force amplitude is not comparable across devices: OpenGo measures pressure at 16 discrete sensors with partial area coverage, and gaitpdb sums 8 force sensors. The COP references also differ (sensor-coordinate span vs % of insole length) | Define **invariance tiers**: (T1) timing only; (T2) amplitude-normalised shape (force divided by the walk's mean total force) and rank-normalised COP; (T3) raw sensors (gaitpdb only). Report transfer separately for each tier. Any label-free per-dataset standardisation is transductive and must be reported separately |
| 7 | **Major** | **Age and site confounding in the external set.** WearGait-PD controls are older (74 vs 67), and site predicts group | Primary external analysis: report pooled results, **within VA-Seattle**, and **age-matched** (1:1 or by inverse-probability weighting) |
| 8 | **Major** | **Underpowered primary comparison.** With n = 165 the AUC 95% CI half-width is about ±0.05. A ΔAUC of 0.02–0.03 cannot be detected | Pre-specify log-loss and the likelihood-ratio test as primary, with an **equivalence margin** (e.g. ΔAUC within ±0.03). Accept and report an inconclusive result |
| 9 | **Major** | **Tiny inner validation sets** (about 20–26 subjects) drive early stopping and gate selection, so selection is noisy | Use a fixed number of epochs chosen in the inner CV and average over seeds, or select on inner-CV log-loss rather than one split |
| 10 | Major | **Blinding is not actually possible as planned.** WearGait-PD folder names and ID prefixes reveal group | One teammate creates a recoded, label-stripped copy for harmonisation work. Labels are released only after the analysis is frozen |
| 11 | Minor | **Identity probe needs two walks per subject.** Only 39/165 subjects have two usual walks (all in Ga) [M] | Restrict the probe to them, or use split halves of a walk |
| 12 | Minor | **Unequal stride counts by study** (Ga about 193, Ju about 74, Si about 98) could act as a study proxy, and variance estimates are noisier when K is small | Use count-invariant pooling, cap K in training, and stratify folds by label × study |
| 13 | Minor | **Severity tasks.** gaitpdb PD H&Y is limited to 2–3, and WearGait-PD uses *modified* H&Y with MDS-UPDRS | No staging claims. At most, an exploratory within-PD correlation |
| 14 | Minor | **Ablation creep** (12 ablations) | Pre-declare 1 primary and ≤ 3 secondary comparisons. All other ablations are descriptive |

---

## 8. A more scientifically sound design

The evidence points to a ceiling of about AUC 0.9 for PD-vs-control on gaitpdb with *any* method that is honestly evaluated:

| Model | AUC |
|---|---|
| Speed alone | 0.78 |
| Our features | 0.86 |
| COP features (Sifat et al.) | 0.89–0.91 |
| PAS-MIL | 0.77 |

The open scientific question is therefore **what transfers to a new device, site and protocol, and how much walking is needed**, not which architecture squeezes out another 0.01 AUC internally.

**Proposed primary study**

*Working title:* "Cross-device external validation of plantar-pressure Parkinson's markers: what transfers, how many strides are needed, and what remains beyond walking speed"

| Element | Design |
|---|---|
| Representations | Three invariance tiers (flaw 6): **T1** timing and variability (stride, swing, stance, double support; CV; asymmetry). **T2** T1 + amplitude-normalised loading shape + rank-normalised AP-COP descriptors. **T3** raw 16-sensor input, gaitpdb only (internal reference) |
| Models | Penalised logistic regression and gradient boosting for each tier. **One** deep comparator: a stride-set encoder with attentive statistics pooling on T2 channels. MSCAF (ours + faithful configuration) as the windowed reference (P0) |
| Baselines | Speed only; age + sex; speed + T1 |
| Internal evaluation | gaitpdb: repeated nested subject-level CV, label × study stratified; leave-one-study-out; K-curve at K = 8, 16, 24, 32, 64, all |
| External evaluation (pre-registered, frozen) | gaitpdb → WearGait-PD on SP (+ SP_mat as a sensitivity analysis), K-matched. Reported pooled, **within VA-Seattle** and **age-matched**. Calibration before and after recalibration on k = 5, 10, 20 target subjects. Reverse direction as a secondary analysis |
| Primary hypotheses | (H1) T1/T2 features keep AUC ≥ 0.75 externally. (H2) Their incremental value over speed alone is greater than 0, internally and externally. (H3) The stride-set deep model is *equivalent within ±0.03 AUC* to T2 features — or it is not, and either answer is publishable |
| Why this is stronger | Every element tests something not yet published. It is robust to a null deep-learning result. Compute is modest (features plus one small deep model). It directly answers the base paper's stated future work (domain adaptation, sensor placement) |

---

## 9. GO / NO-GO with gates

| Item | Decision | Condition |
|---|---|---|
| P0 (MSCAF subject-level re-audit) | **GO** | — |
| Feature tiers T1/T2, speed baselines, K-curve on gaitpdb | **GO** | — |
| WearGait-PD access and label-blind harmonisation | **GO** | Register now; one teammate makes the recoded copy |
| Stride-set deep comparator (CASS-Net *reduced*: no auxiliary loss, set pooling, T2 channels) | **CONDITIONAL GO** | **Gate A:** beats window-MIL with the same pooling (ΔAUC CI excludes 0, or Δlog-loss < 0). **Gate B:** is within 0.03 AUC of the best T2 feature model on gaitpdb. **Gate C:** at the K available in WearGait-PD, its AUC does not drop more than the features' AUC |
| Gated residual (clinical anchor + deep residual) | **CONDITIONAL GO, only after Gates A–C** | Cross-fitted offsets (flaw 1); primary test of incremental value over clinical + COP + **speed**; equivalence margin pre-declared |
| CASS-Net as specified (auxiliary loss + in-sample anchor + window comparison only) as the primary contribution | **NO-GO** | — |

**If the gates fail,** the paper is still complete: an external, cross-device, confound-aware validation of interpretable plantar-pressure markers, with a well-controlled negative deep-learning result. Given the evidence, that is the more likely outcome.

---

## Sources

**WearGait-PD**
- Paper (CC BY 4.0, open access via PMC13009270): https://doi.org/10.1038/s41597-026-06806-2
- Synapse project: https://www.synapse.org/Synapse:syn52540892
- FDA catalogue: https://www.fda.gov/medical-devices/regulatory-science-tools-catalog/weargait-pd-longitudinal-multi-session-wearables-dataset-gait-parkinsons-disease

**Closest prior work**
- PAS-MIL (Farfoura et al. 2026): https://doi.org/10.3390/app16178354
- SENN (Farfoura et al. 2026): https://doi.org/10.3390/s26092671 (PMC13165671)
- Yu et al. 2026: https://doi.org/10.3390/diagnostics16152386 (PMC13464320)
- Jin 2026: https://doi.org/10.3390/app16136296. *Abstract: AUC 0.992 uses UPDRS and H&Y as inputs; the gait-only hold-out AUC is 0.844; leave-one-study-out AUC ranges 0.500–1.000*
- Herath et al. 2026: https://doi.org/10.3390/bioengineering13080860
- Ma & Huo 2025, plantar-pressure feature fusion: https://doi.org/10.3390/technologies13110522
- Sifat et al.: https://arxiv.org/abs/2609.22956 and https://arxiv.org/abs/2610.16220

**Data and measurements**
- gaitpdb: https://physionet.org/content/gaitpdb/1.0.0/
- Audit and benchmark scripts (scratch copies; ask to move them into the repo for reproducibility):
  - `/private/tmp/claude-501/-Users-gurkaranbirsingh-Foot-Pressure-Based-Abnormal-Gait-Recognition/47f658ac-dfc1-4f02-9399-2f555d3d9cb0/scratchpad/scripts/gaitpdb_signal_audit.py`
  - `/private/tmp/claude-501/-Users-gurkaranbirsingh-Foot-Pressure-Based-Abnormal-Gait-Recognition/47f658ac-dfc1-4f02-9399-2f555d3d9cb0/scratchpad/scripts/cassnet_bench.py`

**Standard references [K]:** Okabe et al. 2018 (attentive statistics pooling); Snyder et al. 2018 (x-vectors); Ilse et al. 2018 (attention MIL); Cheng et al. 2016 (Wide & Deep); Bachlechner et al. 2021 (ReZero); Arevalo et al. 2017 (gated multimodal units).
