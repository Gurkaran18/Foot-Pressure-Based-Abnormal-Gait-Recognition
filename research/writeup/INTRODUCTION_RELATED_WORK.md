# Introduction and related work (draft, 2026-11-23)

Companion to `RESULTS.md`. Every reference listed below was checked against the publication or its abstract.
Preprints are marked as such. Numbers quoted from other papers are as reported by their authors.

---

## 1. Introduction

### 1.1 Why gait, and why foot pressure

Parkinson's disease (PD) changes how people walk. Steps become shorter and slower, and the timing of
successive strides becomes less regular. Stride-to-stride variability is one of the most studied gait
markers of PD [8]. Walking can be measured objectively and cheaply. Pressure-sensing insoles record the force
under each foot at every instant, so stride, stance and swing times can be computed in daily life without a
laboratory. This has made plantar-pressure gait data an attractive target for machine-learning (ML) tools that
might support PD screening or monitoring.

### 1.2 The problem: impressive numbers that may not travel

Most published work uses one public dataset, the PhysioNet *Gait in Parkinson's Disease* database (gaitpdb)
[5]. A 2026 systematic review found that 82 of 102 plantar-pressure PD studies used gaitpdb. Only 6% tested a
model on a dataset other than the one it was trained on [20]. Reported accuracies are often 97–99%. Two
evaluation problems explain much of this:

1. **Leakage across the train/test split.** Many studies cut each walk into short windows and split the
   windows at random. The same person then appears in both training and test data. Because individual gait
   patterns are highly distinctive [9], a model can recognise the *person* rather than the disease. Our base
   paper illustrates the size of the effect: 99.61% accuracy with random window splits, against 82.31%
   with leave-one-subject-out validation on the same data [23]. Subject-level splitting is the minimum
   standard for clinical ML [16], and leakage is a recognised cause of irreproducible ML results [10].
2. **Validation inside a single dataset.** Even a correct participant-level cross-validation estimates
   performance only for new people *from the same source*: the same device, protocol, sites and recruitment.
   Clinical prediction research treats **external validation** on independently collected data as the real
   test of a model [3, 4]. ML models are known to fail under *dataset shift*, when the population or
   measurement process changes [6].

A second independent plantar-pressure dataset with PD participants, **WearGait-PD**, became openly available in
2026 [1]. It uses a different insole (Moticon OpenGo), a different walking protocol (four short passes over a
16-ft pressure walkway), and different clinical sites. This makes a genuine external test of gaitpdb-trained
models possible for the first time.

### 1.3 Aim and research question

**Research question.** How well do gaitpdb-trained plantar-pressure models transport to WearGait-PD, and which
components of the shift explain any loss?

The study was designed so the external result could not be tuned after seeing it:
- **Pre-registered, frozen external test.** The model, training data, analysis code and statistical tests were
  fixed and fingerprinted (git tag `model-frozen`) before any WearGait-PD diagnosis was read. WearGait-PD files
  were renamed to random codes and the diagnosis key kept apart until the freeze.
- **Same signal pipeline in both datasets.** Total vertical force per foot, the same contact detection and the
  same stride-validity rules. Interpretable stride-timing features and logistic regression, so each feature's
  role can be inspected.
- **A device-agreement check before the test.** The insole events were compared with the walkway on
  2,947 strides. Features the insoles could not measure reliably were removed by a pre-declared rule.
- **Separating walk length from the rest of the shift.** WearGait-PD walks are short: about 17 valid
  strides per foot, against 90+ in gaitpdb. The internal performance expected under WearGait-PD's walk
  structure was estimated by cutting gaitpdb walks into matching segments, so the *residual* gap could be
  tested.
- **Three confirmatory hypotheses with multiplicity control:**
  - H1: the frozen model discriminates above chance externally.
  - H2: it still discriminates at equal walking speed.
  - H3: a gap remains after matching walk structure.
- **Exploratory sensitivity analyses after unblinding,** reported separately.

**Contributions**
1. The first pre-registered, frozen external validation of a gaitpdb-trained plantar-pressure PD model on an
   independent dataset collected with a different device and protocol.
2. A label-free insole-versus-walkway agreement analysis that decides which gait features are transportable.
3. A walk-structure emulation that separates the effect of short walks from the residual cross-dataset gap.
4. A transparent account of a negative result, with sensitivity analyses and a case-mix and site explanation
   grounded in the dataset's design.

---

## 2. Related work

### 2.1 Gait markers of Parkinson's disease

- **Stride-time variability.** Higher stride-to-stride variability is a well-replicated feature of PD gait. It
  is linked to falls and partly independent of walking speed [8].
- **Speed.** Frenkel-Toledo et al. varied treadmill speed in 36 people with PD (H&Y 2–2.5) and 30 controls [7].
  Faster walking lowered *stride-time* variability in both groups, while *swing-time* variability did not
  change with speed. Some timing markers are therefore entangled with speed, and others less so. This
  motivated our speed-adjusted test (H2).
- **Age.** Gait variability also rises with age. In a population-based study of 412 people aged 60–86, every
  variability measure increased with age, and much of the effect on temporal variability was explained by
  slower speed [2]. A marker of PD can therefore look like a marker of age when the groups differ in age.
- **Task conditions.** Dual-tasking [22] and rhythmic auditory cueing [12] change gait variability by
  design. gaitpdb contains such walks for some participants. We therefore use only each participant's
  usual-walking recording.

### 2.2 Machine learning on plantar pressure and gaitpdb

- **The dataset.** gaitpdb [5] combines three studies (Ga, Ju, Si) recorded with 8-sensor insoles per foot at
  100 Hz: 93 people with PD and 73 controls walking at their usual pace on level ground.
- **High-claim window-level track.**
  - Deep models on 1–2-s windows report accuracies near 99% [11, 23]. One 2026 CNN-GRU-KAN reports 98.43%,
    with leave-one-dataset-out accuracy (≈98%) *higher* than its within-cohort accuracy [11]. That pattern
    is unusual for a true out-of-distribution test.
  - The 2026 review notes that K-fold and hold-out splits dominate and that external generalisation is rarely
    tested [20].
- **Rigorous participant-level track (2026).**
  - Sifat et al. use repeated, fully nested participant-level CV on the `_01` walks only and report AUC 0.894
    with 10 centre-of-pressure features (preprint [18]).
  - Jin reports a gait-only hold-out AUC of 0.844, with leave-one-study-out AUCs ranging from 0.500 to 1.000
    [14].
  - Choi et al. report AUC 0.934–0.940 with repeated subject-wise splits [15].
  - These studies show that honest *internal* performance on gaitpdb is roughly AUC 0.8–0.9. All of them
    stay within gaitpdb: the same insole type and the same research group's protocols.
- **Our position.** Our internal results sit in this range for the full model (AUC 0.82) and lower for the
  two-feature stride-time model (0.67). The question we ask is what happens *outside* gaitpdb.

### 2.3 Evaluation pitfalls

- **Identity leakage.** Saeb et al. showed that record-wise splits can grossly overestimate accuracy when
  subjects recur across folds [16]. Horst et al. showed with explainable deep learning that individuals can be
  identified from their gait [9].
- **Leakage in general.** Kapoor and Narayanan catalogue leakage as a widespread cause of irreproducible
  ML-based science [10].
- **Reporting.** TRIPOD+AI asks for discrimination *and* calibration, a clear description of the
  validation data and transparent reporting of model updating [4]. We follow these points: AUC with CIs,
  calibration intercept and slope, and recalibration reported separately from the frozen result.

### 2.4 External validation, transportability and dataset shift

- **What external validation means.** External validation tests a frozen model on data from a different
  setting. Its interpretation depends on how similar the case-mix is [3].
- **How models fail under shift.** Finlayson et al. describe how changes in population, measurement or
  practice degrade clinical AI models [6]. In gait specifically, a cross-dataset study of freezing-of-gait
  detectors found AUCs falling from 0.77–0.94 within datasets to 0.65–0.84 across datasets [17].
- **Leave-one-study-out is not external.** Leave-one-study-out within gaitpdb holds out a study but keeps
  the same insole system and closely related protocols. It is a weaker test than transport to an
  independently collected dataset.

### 2.5 WearGait-PD

- **The dataset.** Anderson et al. released WearGait-PD in 2026 [1]: 100 people with PD (age 67 ± 8; modified
  H&Y 2.15 ± 0.61) and 85 controls (age 74 ± 9).
  - Recording: Moticon OpenGo 16-sensor insoles, full-body IMUs, a 16-ft ProtoKinetics Zeno walkway and
    video.
  - The SelfPace task is four passes across the walkway, starting and ending about 5 ft off the mat.
  - Medication state was not standardised.
  - Recruitment ran at three sites, with subject-ID prefixes NLS = Johns Hopkins Outpatient Center,
    HC = Johns Hopkins Bayview and WPD/WHC = VA Puget Sound.
  - The authors note site-specific, consistent delays in insole timing.
- **Site confounding.** Using the walkway's own gait metrics, Sifat et al. (preprint [19]) show that recruitment
  site alone predicts diagnosis almost perfectly in WearGait-PD (AUC 0.955). The Johns Hopkins Outpatient
  Center contributed only PD participants and Bayview only controls. A model trained on two sites and tested
  on the third reached only AUC 0.576. They conclude that participant-disjoint validation within WearGait-PD
  does not establish disease-related generalisation.
- **This applies directly to our external set.** The WearGait-PD release we used contains NLS and HC
  participants only, so **every PD participant comes from one site and every control from another.**
- **Other WearGait-PD work.** Other studies train and test within WearGait-PD, for example an IMU-based
  spatio-temporal graph model (balanced accuracy 0.834) [21]. A dual-task severity model reports an
  "external validation" on gaitpdb, but how a 66-channel model was applied to gaitpdb's 16 channels is not
  explained [24]. A public course project plans a gaitpdb → WearGait-PD test of timing features but has
  not reported a result (unpublished).

### 2.6 Gap addressed by this study

No published study has tested a gaitpdb-trained plantar-pressure model on WearGait-PD with:
- a frozen, pre-registered protocol;
- a device-agreement check deciding which features are transportable;
- an estimate of how much of the loss is explained by walk length alone.

This study does all three. It also reports the result, a failure to transport, together with the
sensitivity, case-mix and site evidence needed to interpret it.

### 2.7 Statistical methods used

- DeLong's method for the AUC and its variance [13].
- Holm's step-down procedure for multiplicity [25].
- The Janes and Pepe covariate-adjusted ROC/AUC for speed and age adjustment [26].
- Bland and Altman limits of agreement for repeated measurements per person, for the device check [27].

---

## References

1. Anderson AJ, Eguren D, Gonzalez MA, et al. WearGait-PD: an open-access wearables dataset for gait in Parkinson's disease and age-matched controls. *Sci Data.* 2026;13:440. doi:10.1038/s41597-026-06806-2
2. Callisaya ML, Blizzard L, Schmidt MD, McGinley JL, Srikanth VK. Ageing and gait variability—a population-based study of older people. *Age Ageing.* 2010;39(2):191–197. doi:10.1093/ageing/afp250
3. Debray TPA, Vergouwe Y, Koffijberg H, Nieboer D, Steyerberg EW, Moons KGM. A new framework to enhance the interpretation of external validation studies of clinical prediction models. *J Clin Epidemiol.* 2015;68(3):279–289.
4. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ.* 2024;385:e078378.
5. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet. *Circulation.* 2000;101(23):e215–e220. (Gait in Parkinson's Disease database, v1.0.0.)
6. Finlayson SG, Subbaswamy A, Singh K, et al. The clinician and dataset shift in artificial intelligence. *N Engl J Med.* 2021;385(3):283–286.
7. Frenkel-Toledo S, Giladi N, Peretz C, Herman T, Gruendlinger L, Hausdorff JM. Effect of gait speed on gait rhythmicity in Parkinson's disease: variability of stride time and swing time respond differently. *J Neuroeng Rehabil.* 2005;2:23. doi:10.1186/1743-0003-2-23
8. Hausdorff JM. Gait dynamics in Parkinson's disease: common and distinct behavior among stride length, gait variability, and fractal-like scaling. *Chaos.* 2009;19(2):026113.
9. Horst F, Lapuschkin S, Samek W, Müller KR, Schöllhorn WI. Explaining the unique nature of individual gait patterns with deep learning. *Sci Rep.* 2019;9:2391. doi:10.1038/s41598-019-38748-8
10. Kapoor S, Narayanan A. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns.* 2023;4(9):100804.
11. Fu, Jiang, Al-qaness. CNN-GRU-KAN for Parkinson's disease detection from vertical ground reaction force. *Bioengineering.* 2026;13(8):905. doi:10.3390/bioengineering13080905
12. Hausdorff JM, Lowenthal J, Herman T, Gruendlinger L, Peretz C, Giladi N. Rhythmic auditory stimulation modulates gait variability in Parkinson's disease. *Eur J Neurosci.* 2007;26(8):2369–2375.
13. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics.* 1988;44(3):837–845.
14. Jin. Identifying Parkinson's disease from gait biomechanics using a participant-level machine learning analysis pipeline. *Appl Sci.* 2026;16(13):6296. doi:10.3390/app16136296
15. Choi, Jo, Jeong. Digital gait biomarkers for Parkinson's disease: a subject-wise validated explainable AI framework using vertical ground reaction force. *Bioengineering.* 2026;13(3):360. doi:10.3390/bioengineering13030360
16. Saeb S, Lonini L, Jayaraman A, Mohr DC, Kording KP. The need to approximate the use-case in clinical machine learning. *GigaScience.* 2017;6(5):gix019.
17. Sigcha L, Borzì L, Olmo G. Deep learning for freezing of gait detection: a cross-dataset study. *Expert Syst Appl.* 2024.
18. Sifat M, Akter S, Islam A, Hamid ME. A compact stance-indexed AP-COP representation for Parkinson's disease classification from plantar VGRF. arXiv:2609.22956, 2026 (preprint).
19. Sifat M, Akter S, Islam A, Hamid ME. When participant disjoint validation is not enough to establish generalization in Parkinson's gait classification. arXiv:2610.16220, 2026 (preprint).
20. Wang X, Zhao Z, et al. Comprehensive survey on diagnosis and assessment of Parkinson's disease via plantar pressure analysis. *npj Parkinsons Dis.* 2026;12:229. doi:10.1038/s41531-026-01416-6
21. Multi-task wearable Parkinson's disease detection with a pretrained ST-GCN encoder. *Bioengineering.* 2026;13(8):860. doi:10.3390/bioengineering13080860
22. Yogev G, Giladi N, Peretz C, Springer S, Simon ES, Hausdorff JM. Dual tasking, gait rhythmicity, and Parkinson's disease: which aspects of gait are attention demanding? *Eur J Neurosci.* 2005;22(5):1248–1256.
23. Yuan, Wang, Zhou, et al. Foot pressure-based abnormal gait recognition with multi-scale cross-attention fusion. *IEEE Trans Neural Syst Rehabil Eng.* 2025;33:3146–3159. doi:10.1109/TNSRE.2025.3597639
24. Yu, et al. Dual-task gait fusion framework for classifying Parkinson's disease severity. *Diagnostics.* 2026;16(15):2386. doi:10.3390/diagnostics16152386
25. Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat.* 1979;6(2):65–70.
26. Janes H, Pepe MS. Adjusting for covariate effects on classification accuracy using the covariate-adjusted receiver operating characteristic curve. *Biometrika.* 2009;96(2):371–382.
27. Bland JM, Altman DG. Agreement between methods of measurement with multiple observations per individual. *J Biopharm Stat.* 2007;17(4):571–582.

*Reference notes.* [18], [19] are preprints. [11], [14], [15], [21], [24] were checked from abstracts or full
text as recorded in `RESEARCH_EXTENSION_PLAN.md`. First names and full author lists marked with "et al." or
initials should be completed from the publisher pages before submission.
