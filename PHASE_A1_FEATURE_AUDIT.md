# Phase A1 — Feature-Comparability Audit and Proposed Protocol

**Study:** How well do gaitpdb-trained plantar-pressure models transport to WearGait-PD, and which components of the shift explain any loss in predictive performance?

**Phase:** A1 only (desk audit). No code was modified, and no WearGait-PD data was downloaded or accessed. · **Date:** 18 Nov 2026 · **Status:** for review. Implementation waits for your approval.

**Evidence tags**
| Tag | Meaning |
|---|---|
| **[C]** | Confirmed by reading the repository code (`file:line`), or computed directly from existing result files |
| **[M]** | Measured on the downloaded gaitpdb data during the feasibility review (`CASSNET_FEASIBILITY_REVIEW.md` §5). That scratch copy has since been cleared, so these numbers must be re-measured in A2 with committed scripts |
| **[P]** | Stated in a published source (WearGait-PD paper, PMC13009270; PhysioNet gaitpdb page) |
| **[A]** | Assumption, not yet verified |
| **[U]** | Unresolved: must be verified before the relevant phase |

---

## 0. How the existing pipeline works, and what A1 found

### 0.1 Data flow (as implemented)
1. **File discovery.** `main.load_subject_info()` ([main.py:515-571](main.py#L515-L571)) loads **every** file matching `^(Ga|Ju|Si)(Co|Pt)\d+_\d+\.txt$` ([main.py:512](main.py#L512)). That is all 306 recordings, including Ga dual-task (`_10`) and Ju cued walks [C][M]. Labels come from the filename (`Co` → 0, otherwise 1) ([main.py:543-546](main.py#L543-L546)).
2. **Stride detection** per foot, using the total-force columns 17/18 ([gait_common.py:147](gait_common.py#L147), [220-223](gait_common.py#L220-L223)):
   - Smooth with a **5-sample** moving average ([gait_common.py:149](gait_common.py#L149), [177](gait_common.py#L177)).
   - Contact = smoothed force > p5 + 0.10 × (p95 − p5) of *that recording* ([gait_common.py:178-179](gait_common.py#L178-L179)).
   - Debounce contacts and gaps shorter than 100 ms ([gait_common.py:151](gait_common.py#L151), [164-172](gait_common.py#L164-L172), [180](gait_common.py#L180)).
   - Heel strike (HS) = contact onset, toe-off (TO) = contact end. A stride is HS → TO → next HS with exactly one TO in between ([gait_common.py:192-206](gait_common.py#L192-L206)).
   - Valid strides last 0.5–3.0 s and lie within 3 robust SDs (MAD) of the median, which removes turns ([gait_common.py:152-153](gait_common.py#L152-L153), [211-216](gait_common.py#L211-L216)).
3. **Features per recording.** `gait_features()` ([gait_common.py:274-320](gait_common.py#L274-L320)) computes 13 features from the valid strides, requiring at least 3 strides per foot ([gait_common.py:263](gait_common.py#L263), [287](gait_common.py#L287)).
4. **Features per subject.** `subject_feature_table()` **averages the recording-level features over all of a subject's recordings**, whatever the walking condition ([gait_common.py:323-351](gait_common.py#L323-L351)) [C].
5. **Models.** RF/XGBoost in a `Pipeline` with median imputation fitted on training folds ([gait_common.py:389-413](gait_common.py#L389-L413)). Folds come from `make_cv_folds`, stratified on **label only** ([main.py:865-870](main.py#L865-L870)) [C].
6. **MSCAF** uses all 16 sensor channels, z-scored per sensor using the mean and SD of the **whole recording** ([main.py:427-433](main.py#L427-L433)). It takes 10-s windows with 50% overlap over **all recordings, all conditions** ([main.py:436-446](main.py#L436-L446)), and aggregates per subject by majority vote ([main.py:705](main.py#L705)) [C].

### 0.2 Issues found in the existing code
These are recorded for A2. None has been fixed.

| # | Location | Issue | Impact on the new study |
|---|---|---|---|
| I1 | [gait_common.py:323-351](gait_common.py#L323-L351) | Features are averaged over usual, **dual-task (Ga `_10`) and cued (Ju) walks** | Mixes conditions that change variability by design. The primary analysis needs usual walking only |
| I2 | [gait_common.py:330-333](gait_common.py#L330-L333) | The feature cache is reused whenever the **set of subject IDs** matches. Restricting to usual walks keeps all 165 subjects, so the **old all-walk cache would be returned silently** | A real risk of producing wrong results without any error. A2 must use a new cache key or location |
| I3 | [gait_common.py:149](gait_common.py#L149), [177](gait_common.py#L177) | The smoothing window is a sample count (5), not a time | At 50 Hz it would become 100 ms. Must be defined in seconds if WearGait-PD's insole rate is not 100 Hz |
| I4 | [gait_common.py:296-299](gait_common.py#L296-L299), [311](gait_common.py#L311) | Double support is computed over the whole span from the first to the last valid stride. That span includes **invalid strides, turns and pauses**, and the contact masks are recomputed on the segment with new percentiles | Inflated by non-walking periods. It is the **only** existing feature that depends on left/right time synchronisation |
| I5 | [gait_common.py:301-303](gait_common.py#L301-L303), [318](gait_common.py#L318) | The "body-weight proxy" is the mean of L + R force over the same span, so it includes non-walking periods. The label "× body wt" ([gait_common.py:260](gait_common.py#L260)) is misleading | `peak_force_norm` is peak ÷ mean *measured* force, not ÷ body weight |
| I6 | [gait_common.py:310](gait_common.py#L310), [312](gait_common.py#L312) | `swing_pct = 100 − stance_pct`, and `cadence = 120 / stride_time_mean` | Exact redundancy (Spearman ρ = −1.000 in both cases, from [results_new/gait_features_subject.csv](results_new/gait_features_subject.csv)) [C] |
| I7 | [main.py:479-506](main.py#L479-L506), [553-557](main.py#L553-L557) | Demographics are read only from `.xls`/`.xlsx`. If that file is missing, UPDRS and H&Y become **random synthetic values** | Irrelevant to PD detection, but these columns must never be used. The tab-separated demographics file has a malformed row (line 161 has 30 fields) [M], so A2 needs a robust parser for speed, age and sex |
| I8 | [main.py:865-870](main.py#L865-L870) | Folds are stratified by label, not by label × study | Fold composition by study varies. Matters for study-related shift |
| I9 | [cross_study.py:33-40](cross_study.py#L33-L40) | The "per-study z" variant standardises using the **held-out study's own** mean and SD | Label-free but transductive. Under the new protocol this kind of normalisation is allowed only in exploratory analyses |
| I10 | [main.py:705](main.py#L705) | The majority vote breaks ties towards class 0 (`argmax` of `bincount`) | Minor bias towards "control" for subjects with an even split |
| I11 | [main.py:436](main.py#L436) | MSCAF windows include standing and turn segments | Adds non-gait content. Matters for any external MSCAF comparison |

### 0.3 Univariate signal in the existing (all-walk) feature table [C]
Computed from [results_new/gait_features_subject.csv](results_new/gait_features_subject.csv) (n = 165). These values are indicative only, because the features mix walking conditions (I1).

| Feature | Univariate AUC (PD = 1) | Feature | Univariate AUC |
|---|---|---|---|
| swing_time_cv | 0.778 | asym_peak | 0.624 |
| double_support_pct | 0.755 | asym_stride | 0.579 |
| asym_swing | 0.736 | stride_time_mean | 0.566 |
| stance_pct | 0.723 | cadence | 0.435 (= 0.565 inverted) |
| stride_time_cv | 0.699 | peak_force_L / R | 0.484 / 0.485 |
| peak_force_norm | 0.352 (= 0.648 inverted) | swing_pct | 0.277 (= 0.723 inverted) |

Other observations from the same table:
- Other strong correlations: stance % vs double support, ρ = 0.947; peak force L vs R, ρ = 0.904; stride CV vs swing CV, ρ = 0.756.
- The 13 features span roughly **9 independent dimensions**.
- The SHAP top three (asym_swing, swing_time_cv, double_support_pct; [results_new/shap_importance.json](results_new/shap_importance.json)) are all timing features.

---

## Deliverable 1 — Feature Comparability Report

**Event definitions used by all features:**
- HS and TO come from a relative threshold on per-foot total vertical force.
- In gaitpdb, the total force is **exactly the sum of the 8 sensors** (maximum difference 0.000 N) [M].
- In WearGait-PD, the Moticon OpenGo export is described as including a total-force value in newtons computed by the vendor [A, from Moticon documentation]. Its exact definition is [U].

**Comparability scale:**
- **High:** same construct and estimator; insensitive to amplitude and geometry.
- **Moderate:** same construct, but sensitive to event-detection bias or protocol.
- **Low:** depends on the device's amplitude, coverage or geometry.
- **Not directly comparable:** no defensible mapping without new assumptions.

| # | Feature (code) | Definition | Signals and events | Depends on amplitude? | Depends on geometry / placement? | Depends on L/R sync? | Computable from WearGait-PD? | Main error sources | Comparability | Recommended tier / use |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `stride_time_mean` ([gait_common.py:291](gait_common.py#L291), [306](gait_common.py#L306)) | Mean of (next HS − HS) / fs over valid strides, both feet pooled | Total force per foot; **HS only** | No (relative threshold) | Weak: heel-onset detection, but a constant offset cancels HS → HS | No | Yes, if total force and sampling rate are adequate [U] | Sampling resolution; quality-control differences; turn-adjacent strides; few strides | **High** | **T1-core** |
| 2 | `stride_time_cv` ([gait_common.py:307](gait_common.py#L307), [266-267](gait_common.py#L266-L267)) | Mean over feet of 100·SD/mean of valid stride times (population SD) over the whole recording | HS only | No | Weak | No | Yes | **The estimator depends on the number of strides (K) and on how steady they are.** Gaitpdb gives about 74–98 strides per foot per walk [M]; WearGait-PD's self-paced task gives an estimated 16–28 [A]. Acceleration and turn strides inflate it | **Moderate** (protocol-sensitive, not device-sensitive) | **T1-core**, with stride counts matched to the target protocol |
| 3 | `swing_time_cv` ([gait_common.py:308](gait_common.py#L308)) | Mean over feet of the CV of swing durations (TO → next HS) | HS **and TO** | No | **Yes:** TO timing depends on forefoot/toe sensor coverage and the threshold | No | Yes | TO detection bias and jitter; small K; turn strides | **Moderate** | **T1-core**, if TO agreement with the walkway is acceptable (Phase B) |
| 4 | `stance_pct` ([gait_common.py:292](gait_common.py#L292), [309](gait_common.py#L309)) | Mean over strides of 100·stance/stride | HS and TO | No | **Yes:** a systematic TO bias shifts stance % one way (e.g., early TO if toe-region coverage is weaker) | No | Yes | Systematic TO bias; threshold choice | **Moderate** | **T1-core**, conditional on a bias check against walkway contact times |
| 5 | `swing_pct` ([gait_common.py:310](gait_common.py#L310)) | 100 − stance_pct | — | — | — | — | — | **Exactly redundant** (ρ = −1.000) [C] | n/a | **Exclude** (keep stance_pct) |
| 6 | `double_support_pct` ([gait_common.py:296-299](gait_common.py#L296-L299), [311](gait_common.py#L311)) | % of samples between the first valid HS and the last valid next HS (either foot) in which both contact masks are on | HS and TO of **both feet on a common clock** | No | Yes (TO) | **Yes (critical)** | Only if the two insoles are well synchronised. WearGait-PD corrected desynchronisation by shifting/trimming [P]; residual offsets are unknown [U] | L/R clock offset or drift; the span includes turns and pauses (I4); ρ = 0.947 with stance % | **Low as implemented**; Moderate if redefined and validated | **T1-extended (conditional)**: redefine per valid stride pair, check against walkway double support. Otherwise exclude |
| 7 | `cadence` ([gait_common.py:312](gait_common.py#L312)) | 120 / stride_time_mean (steps/min) | HS only | No | Weak | No | Yes | **Deterministic transform of #1** (ρ = −1.000) [C] | High, but redundant | **Exclude from models**; report as descriptive |
| 8 | `asym_stride` ([gait_common.py:313](gait_common.py#L313)) | 100·\|mean stride L − mean stride R\| / mean | HS of both feet (separately) | No | Weak | No | Yes | In continuous walking L and R stride times are equal on average. Observed medians are 0.11–0.16% [C], so the value mostly reflects **which strides passed quality control on each side** | **Low (poor construct validity)** | **Exclude** |
| 9 | `asym_swing` ([gait_common.py:293](gait_common.py#L293), [314](gait_common.py#L314)) | 100·\|ln(mean swing R / mean swing L)\|, a gait-asymmetry index in the style of Plotnik et al. 2007 [K] | HS and TO, each foot separately | No | Yes (TO), but a **common** TO bias largely cancels in the ratio | No | Yes | Unequal fit or calibration of the two insoles; small K | **Moderate** | **T1-core** |
| 10 | `asym_peak` ([gait_common.py:294](gait_common.py#L294), [315](gait_common.py#L315)) | 100·\|peak L − peak R\| / mean peak (per-stride maximum force) | Force amplitude per foot | **Yes (ratio)** | **Yes:** per-insole calibration, coverage, fit | No | Computable, but meaning differs | Insole-specific calibration; coverage of the loaded region. WearGait-PD's walkway gives only relative pressure, so there is **no amplitude reference** to validate against [P] | **Low** | **T2-exploratory** |
| 11 | `peak_force_L` ([gait_common.py:316](gait_common.py#L316)) | Mean per-stride maximum of left total force (N) | Absolute amplitude | **Yes (absolute)** | **Yes** | No | Values differ in meaning: Moticon reports pressure at 16 discrete sensors with partial coverage; gaitpdb sums 8 force sensors | Device calibration, coverage, body weight | **Not directly comparable** | **Exclude externally** (internal reference only) |
| 12 | `peak_force_R` ([gait_common.py:317](gait_common.py#L317)) | Same, right foot | Same | Yes | Yes | No | Same | Same; ρ = 0.904 with #11 | **Not directly comparable** | **Exclude externally** |
| 13 | `peak_force_norm` ([gait_common.py:301-303](gait_common.py#L301-L303), [318](gait_common.py#L318)) | Mean(peak L, peak R) ÷ mean(L + R force over the stride span) | Amplitude ratio | Scale cancels, but the **shape of the force curve** matters | **Yes:** the force profile depends on which regions are sensed | No | Computable | Coverage-dependent profile shape; denominator includes non-walking periods (I5) | **Low** | **T2-exploratory** |

**Resulting tiers (proposed)**

| Tier | Contents | Status |
|---|---|---|
| **T1-core** | stride_time_mean, stride_time_cv, swing_time_cv, stance_pct, asym_swing (5 features; univariate AUC 0.57–0.78) | Proposed **primary** feature set. Free of amplitude and L/R synchronisation dependence |
| **T1-extended** | T1-core + **redefined** double_support_pct | Only if Phase B shows agreement with walkway double support |
| **T1-fallback** | stride_time_mean + stride_time_cv (HS-only features) | Pre-declared fallback if Phase B shows unacceptable TO bias |
| **T2-exploratory (existing)** | asym_peak, peak_force_norm | External analysis exploratory only |
| **T2 candidates (new, not implemented)** | (a) per-stride, amplitude-normalised force-profile shape (relative timing of first peak, valley and second peak; peak-to-valley ratio); (b) stance-indexed AP-COP descriptors (as in Sifat et al.); (c) fraction of load in heel / midfoot / forefoot / toe regions | **Low to uncertain** comparability. Gaitpdb has 8 sensors at 5 heel-to-toe levels and 2 side-to-side columns [P, format.txt]. Moticon has 16 sensors with a different layout, and its COP is reported in % of insole length [A]. The reference frames differ. Include externally only if Phase B's label-free distribution checks pass; otherwise exploratory |
| **Excluded** | swing_pct, cadence (redundant); asym_stride (construct validity); peak_force_L, peak_force_R (absolute amplitude) | — |
| **T3 raw** | 16 sensor channels (gaitpdb, N); MSCAF input = 16 channels z-scored per recording × 1,000 samples ([main.py:428-441](main.py#L428-L441)) | **Not directly comparable** with WearGait-PD's 32 Moticon channels: different counts, positions, sensing areas and units (N vs N/cm²). Internal reference only. External use only through a **pre-declared** 16 → 8 per-foot spatial mapping, labelled exploratory |

**Validation needed before any feature enters the external experiment**

| Check | Applies to | Proposed acceptance criterion (provisional, to approve) |
|---|---|---|
| HS timing vs walkway initial contact | All T1 | Median absolute error ≤ 20 ms, after a label-free cross-correlation alignment of the two contact sequences |
| TO timing and stance-duration bias vs walkway | Features 3, 4, 9 | Stance % bias ≤ 2 percentage points; Bland–Altman limits reported |
| Double support vs walkway double support | Feature 6 (redefined) | Bias ≤ 2 percentage points and no drift across the trial; otherwise exclude |
| Stride time vs walkway stride time | Features 1, 2 | Bias ≤ 10 ms |
| Label-free distribution overlap (all participants pooled, gaitpdb vs WearGait-PD) | T2 | Pre-declared overlap statistic (e.g., standardised mean difference < 0.5); used only to decide inclusion, never labels |

*Note:* If a timing bias is found, it must **not** be corrected using WearGait-PD data in the primary analysis. That would mean fitting preprocessing on external data. The pre-declared fallback set is used instead, and the bias is reported.

---

## Deliverable 2 — Proposed Experimental Protocol

### 2.1 Hypotheses (at most three confirmatory, tested in a fixed sequence)

**Primary model M1:** class-balanced L2-penalised logistic regression on T1-core, trained on gaitpdb (see §2.6).

| ID | Type | Hypothesis | Estimand and test |
|---|---|---|---|
| **H1** | Confirmatory (primary) | M1, frozen, carries transportable signal on WearGait-PD | External ROC-AUC on the primary WearGait-PD cohort. Test: lower bound of the 95% CI > 0.50. **Pre-declared interpretation bands:** ≥ 0.80 strong; 0.70–0.80 moderate; 0.60–0.70 weak; < 0.60 failure of transport |
| **H2** | Confirmatory | Plantar-pressure timing adds value beyond walking speed in the external cohort | Paired external ΔAUC: frozen [speed + T1-core] LR minus frozen [speed-only] LR, both trained on gaitpdb. Test: one-sided, ΔAUC > 0 (DeLong; paired bootstrap as a check) |
| **H3** | Confirmatory (estimation) | Part of the external loss is **not** explained by protocol (walk length/structure) | Residual gap = AUC(M1, internal, *protocol-emulated* gaitpdb) − AUC(M1, external). Test: lower bound of the 95% CI of the gap > 0 (bootstrap over both samples) |

**Multiplicity:** fixed-sequence testing H1 → H2 → H3 at α = 0.05. A hypothesis is tested only if the previous one was rejected.

**Precision (approximate, Hanley–McNeil):**
- With about 100 PD and 85 controls and AUC ≈ 0.75, the 95% CI is roughly ±0.07. With about 150 participants after quality control it is roughly ±0.08.
- Only large differences can be confirmed. Effect estimates and CIs will be reported regardless of test outcomes.

**Secondary and exploratory (no confirmatory claims):**
- XGBoost-T1 vs M1; T1-extended; T2 tiers; T1-fallback
- Sensitivity, specificity and balanced accuracy at the pre-declared threshold; calibration
- Covariate-adjusted AUC (age, sex)
- Analyses restricted to the age-overlap range, and age-matched analyses
- Common site (VA-Seattle) and other sites; sex strata
- DBS-excluded sensitivity analysis
- Hurried-pace condition; the self-paced on-mat condition
- Marker-level transportability (direction and size of each feature's PD effect in each dataset)
- Recalibration (separate experiment; §2.11)
- Reverse direction (train on WearGait-PD, test on gaitpdb)
- MSCAF with a pre-declared mapping
- Stride-set model (only if the gates in the feasibility review pass)

### 2.2 Datasets, inclusion and exclusion

**gaitpdb (development)**
- **Version:** 1.0.0; record the checksums in `SHA256SUMS.txt`.
- **Include:** all subjects with at least one usual-walking recording yielding ≥ K_min valid strides per foot. K_min is fixed in A3 (provisionally 10).
- **Usual walking (provisional; verify, [U]):**
  - Ga `_01`, `_02`; Ga `_10` is dual-task.
  - Ju `_01`; Ju `_02`–`_07` are cued or follow cueing (protocol semantics to be checked against Hausdorff et al. 2007).
  - Si `_01`, treadmill.
- **Training cohort:** decision **D3**. Option A: all 165 subjects, primary. Option B: overground only (Ga + Ju, about 101 subjects), sensitivity analysis.

**WearGait-PD (external test; Version 1 cross-sectional)**
- **Include:** participants with a SelfPace (SP) insole trial in which both insoles are present and ≥ K_min valid strides per foot remain after quality control.
- **Exclude (decided label-blind):**
  - missing insole data
  - a dropout or sensor loss affecting total force during the analysed segment
  - a sampling rate below the pre-declared minimum
- **DBS participants:** included in the primary analysis. A DBS-excluded sensitivity analysis is applied by the custodian, because DBS status reveals the label.
- **Not used:** the Version 2 longitudinal data (repeat visits of the same people).

### 2.3 Walking-condition selection
| Dataset | Primary | Secondary / exploratory | Excluded |
|---|---|---|---|
| gaitpdb | Usual walking (above) | Ga dual-task and Ju cued, for describing condition sensitivity | — |
| WearGait-PD | **SelfPace (SP)**: overground, 4 passes, turns off the mat [P] | SelfPace_mat (on-mat with stops); HurriedPace (within-subject speed manipulation) | FreeWalk (site-specific route with stairs and chairs [P]), TUG, Balance, TandemGait, turning tasks |

### 2.4 Feature extraction and harmonisation
1. **One code path for both datasets.** The input is per-foot total force at a common sampling rate.
   - gaitpdb: columns 17/18.
   - WearGait-PD: the vendor total force, or Σ(pressure × sensor area). The choice is fixed in Phase B, label-blind [U].
2. **Time-based parameters only.** Smoothing 50 ms, debounce 100 ms, stride range 0.5–3.0 s, 3-MAD filter. If the WearGait-PD rate is not 100 Hz, resample to 100 Hz by linear interpolation; this adds no information.
3. **Per-recording relative contact threshold** (10% of p5–p95). This is a within-recording operation, so it is part of the frozen pipeline and pools nothing across external participants.
4. **Features from valid strides only.** Double support is redefined per stride (I4) as a new, versioned feature; the old definition is kept for reproducing the existing results.
5. **Subject level:** average the walk-level features over usual walks, weighted by valid-stride count.
6. **No dataset-level standardisation or bias correction using WearGait-PD data** in the primary analysis. Model standardisation is fitted on gaitpdb training data only.
7. **Protocol emulation on gaitpdb (A3).** Build pseudo-SP walks of 4 segments × about 6–7 consecutive valid strides, dropping turn-adjacent strides. Repeat with seeded random draws (e.g., 20) and average.

### 2.5 Models
| Model | Role |
|---|---|
| **M1: L2 logistic, T1-core** (standardisation + logistic in one pipeline; class-balanced; C chosen by inner-CV log-loss) | Primary (H1, H3) |
| M2: L2 logistic, speed + T1-core; M0: L2 logistic, speed only | H2 |
| Age + sex logistic | Baseline (secondary) |
| XGBoost-T1 (same fixed hyperparameters as [gait_common.py:406-410](gait_common.py#L406-L410), refit on usual-walking T1) | Secondary |
| XGBoost-13 (existing; refit on usual walking for reference) | **Internal reference only** |
| MSCAF-Gait (existing checkpoints and code) | Internal reference. External use only through a pre-declared mapping (exploratory), or not at all |

**Model strengths, weaknesses and risks (Deliverable C)**

| Model | Strengths | Weaknesses / risks |
|---|---|---|
| **M1 (LR-T1)** | Few parameters (6) for about 93 events (≈ 18 events per variable); monotone and interpretable; coefficients can be checked for sign consistency across datasets; logistic calibration moves more predictably under shift | Linear, so it may miss interactions. A systematic device bias in stance % or swing CV shifts the logit, mainly affecting calibration-in-the-large. Collinearity is controlled by excluding the redundant features |
| **XGBoost-T1** | Captures nonlinearities; already used in the project | Split points learned on gaitpdb are **brittle under feature offsets** (step functions); flat extrapolation; poorer calibration; higher variance at n = 165 |
| **XGBoost-13** | Best existing internal model (AUC 0.86, all-walk features) | Includes absolute-amplitude features (11, 12), redundant features (5, 7), an invalid construct (8) and condition-mixed inputs (I1). **Not defensible externally** |
| **MSCAF-Gait** | The base architecture; deep reference | Weak internally (AUC 0.73); trained on all conditions (I1/I11); its input layout (8 Ultraflex sensors per foot) does not exist in WearGait-PD. Any 16 → 8 mapping is an untestable assumption; at best exploratory |

**Recommendation:** M1 as primary is **supported by the audit**. The 5 T1-core features carry most of the univariate signal (AUC 0.57–0.78) and are the only features with high or moderate comparability that are also free of synchronisation dependence. The final choice is decision **D2**.

### 2.6 Participant-level validation (internal, gaitpdb)
- **Repeated nested CV:** 20 repeats × 5 outer folds, stratified by **label × study**, with subject-disjoint folds asserted.
  - Inner 5-fold CV chooses C (log-loss).
  - Pooled out-of-fold predictions per repeat; summary = median across repeats, with a bootstrap CI on the pooled OOF of a fixed reference repeat.
- **Leave-one-study-out:** Ga / Ju / Si.
- **Protocol emulation:** the same CV with emulated pseudo-SP walks, which is what H3 uses.
- **Final frozen models:** refit on all eligible gaitpdb subjects with the hyperparameter chosen by the full-data inner CV. Save the model files with SHA-256 hashes.

### 2.7 Decision threshold
- **Fixed in advance at 0.5** on the predicted probability of the class-balanced model. No data-driven threshold is used.
- **Secondary:** the threshold maximising Youden's index on gaitpdb pooled OOF predictions, also fixed before external testing.
- No threshold is ever chosen on WearGait-PD.

### 2.8 Metrics
- **Primary:** ROC-AUC.
- **Secondary:**
  - sensitivity, specificity and balanced accuracy at the fixed threshold
  - Brier score
  - calibration intercept and slope, and a smoothed calibration curve
  - for completeness, PPV and NPV at the observed prevalence (no clinical interpretation)
- **Reported alongside:** subgroup n, number of events, and exclusions by stage.

### 2.9 Confidence intervals
- **External:** stratified bootstrap over participants (2,000 resamples; percentile CI; BCa as a check). The DeLong CI is also given for AUC.
- **Paired comparisons:** DeLong for correlated AUCs, plus a paired bootstrap.
- **H3:** independent bootstrap of the internal and external samples, combined.
- **Internal CV CIs** are approximate; this will be stated.

### 2.10 Missing data
- **Features:** a participant with fewer than K_min valid strides per foot is excluded from the analysis (complete case). Counts are reported by dataset, site and (after unblinding) group.
- **Covariates (speed, age, sex):** complete case within the baseline and H2 models, with counts reported. gaitpdb is missing `Speed_01` for 1%, weight for 1–2% [M].
- **Differential exclusion.** If exclusions differ by group (e.g., strides removed around freezing), this will be reported and a sensitivity analysis run with a relaxed K_min.
- **No imputation using WearGait-PD statistics.** Median imputation is fitted on gaitpdb inside the pipeline.

### 2.11 Recalibration (Refinement 1)
- The primary external result is the **unmodified frozen model**.
- Recalibration is a **separate exploratory experiment**. Logistic recalibration (intercept only, then intercept + slope) is estimated by subject-level 5-fold CV inside WearGait-PD and reported in a separate table, clearly labelled.
- It never replaces or merges with the primary result.

### 2.12 Leakage prevention
- **Participant disjointness:** subject-disjoint folds asserted ([main.py:889-898](main.py#L889-L898) is the existing pattern).
- **Fitting within training data only:** all fitted transforms (scaler, imputer, C selection) sit inside the training pipeline.
- **No external information:**
  - WearGait-PD is never used for feature selection, hyperparameters, thresholds, K_min, quality-control rules or model choice.
  - Phase B decisions use label-blind data only.
- **Separate outputs:** new outputs go in a new directory, so existing `results_*` folders are never overwritten. The cache issue (I2) is avoided by a new cache key.
- **Freeze record:** git tag + commit hash + model hashes + protocol PDF, created **before** unblinding.

### 2.13 Label-blinding procedure (Refinement 2)
1. **The custodian** (a named teammate who does no harmonisation work) downloads WearGait-PD and keeps the original copy and the ID ↔ diagnosis mapping offline.
2. **The custodian creates the blinded copy:**
   - a random participant code (e.g., `P001`–`P185`)
   - merge `PD PARTICIPANTS/` and `CONTROL PARTICIPANTS/` into one folder; randomise file order
   - strip site/group-revealing ID prefixes (NLS, HC, WPD, WHC) from filenames **and file contents/headers**
   - normalise file timestamps
   - remove all demographic and clinical CSVs (diagnosis, MDS-UPDRS, H&Y, medication, DBS, age, sex, height, weight) and site identifiers
   - recode the walkway-metrics files and remove any group or clinical columns
3. **Analysts receive only:** insole files for SP / SP_mat / HP, walkway footfall timing (for event validation only), and a task code.
   - **No covariates before the freeze:** speed, age and sex are not needed for harmonisation and partly reveal group (controls are older).
4. **Aggregate counts:** the custodian may report only aggregate counts (site × group n) for planning.
5. **Freeze:** protocol, inclusion list by blinded code, code commit and model hashes are frozen and timestamped (git tag, plus OSF if approved).
6. **Unblinding:** the custodian releases the mapping and covariates; the pre-written evaluation script runs **once**. Any deviation is logged in a deviations section.

*Limitation:* Blinding cannot stop analysts inferring group from gait patterns. It does stop label-driven decisions.

### 2.14 Statistical analysis plan (summary)
| Level | Analyses |
|---|---|
| **Primary** | H1 (M1 external AUC with CI and interpretation band) |
| **Confirmatory** | H2, then H3 (fixed sequence) |
| **Secondary** | Sensitivity, specificity, balanced accuracy, Brier, calibration for M1, M2, M0, age + sex, XGBoost-T1. Internal repeated CV and leave-one-study-out. Curve of AUC against number of strides (K-curve) |
| **Exploratory** | Covariate-adjusted AUC (age, sex); age-overlap and age-matched subsets; site and sex strata; DBS exclusion; SP_mat and HP conditions; marker-level effect sizes (Hedges' g by dataset); label-free shift diagnostics (dataset-membership classifier, standardised mean differences); recalibration; reverse direction; T2 tiers; MSCAF with mapping; stride-set model |
| **Reproducibility** | Fixed seeds (existing `CONFIG['seed']` = 42; repeat r uses 42 + r); `pip freeze` lockfile; dataset versions and checksums; Synapse entity versions; a single script per table and figure |

**Software (current local environment) [C]:** Python 3.13, torch 2.14.1, numpy 2.5.3, scipy 1.18.1, pandas 3.0.6. The scikit-learn and xgboost versions, and the Kaggle environment used for the existing results, are [U] and must be recorded in A2.

---

## Deliverable 3 — Risk and Feasibility Assessment

### 3.1 Methodological problems: what preprocessing can address vs what must be a stated limitation

| Issue | What we know | Addressable in preprocessing? | Remaining limitation |
|---|---|---|---|
| **Sampling rate** | gaitpdb 100 Hz [M]; WearGait-PD insole rate not stated in the main text [U]; OpenGo supports 10–100 Hz [A] | Yes: time-based parameters (I3) and resampling | If below 100 Hz, timing resolution limits swing CV and stance % |
| **Sensor geometry / placement** | 8 vs 16 sensors; different layouts and units [P][A] | Partly: T1 avoids amplitude and geometry except through TO timing | TO bias from coverage; T2/T3 comparability may be unattainable |
| **L/R synchronisation** | WearGait-PD desynchronisation was corrected by shifting/trimming [P] | **Yes for T1-core:** all 5 features are within-foot [C]. Double support only if validated | Residual offset affects double support only |
| **Event-detection reliability** | Insole–walkway alignment is "less consistent" [P] | Yes: label-free alignment, then agreement statistics | The walkway covers only the on-mat strides |
| **Protocol and trial length** | gaitpdb about 2-min walks with 4–6 runs, longest run about 21–28 strides [M]; WearGait-PD SP is 4 passes with turns [P] | Partly: protocol emulation, K-matching, excluding turn-adjacent strides | Cannot emulate surface (mat), site, IMU suit or footwear |
| **Limited strides in WearGait-PD** | Estimate 16–28 per foot [A] | Partly: K_min; K-curve | Variability features are noisier, which lowers achievable AUC. This is a real deployment constraint, not only a nuisance |
| **Missing / incomplete recordings** | 83% of trials complete; 259 files with significant loss [P] | Yes: label-blind inclusion rules | Possible differential exclusion by group |
| **Walking-speed measurement** | gaitpdb `Speed_01` method per study [U] (Si probably treadmill belt speed [A]); WearGait-PD speed from walkway software [P] | No | H2 compares models whose speed input is measured differently. This must be stated |
| **Age and sex imbalance** | WearGait-PD controls 74 ± 9 vs PD 67 ± 8; controls 44% male vs PD 65% [P]. gaitpdb medians 64 vs 68 [M]; gaitpdb sex distribution [U] | Analysis only: covariate-adjusted AUC, age-overlap and matched subsets (exploratory) | No retraining. Case-mix shift is part of the transport result |
| **Disease severity and medication** | gaitpdb H&Y 2 / 2.5 / 3 only [M]; WearGait-PD modified H&Y 2.15 ± 0.61, medication state uncontrolled, DBS included [P] | Descriptive strata (exploratory) | Spectrum differences; ON/OFF state unknown |
| **Inclusion criteria** | gaitpdb idiopathic PD [P]; WearGait-PD physician-diagnosed PD with broad exclusions [P] | No | Different control and PD populations |
| **Condition handling in the existing code** | All conditions averaged (I1); cache risk (I2) | Yes (A2) | — |

### 3.2 Feasibility
| Status | Items |
|---|---|
| **Feasible now (gaitpdb)** | Usual-walking T1-core features; speed and age + sex baselines; repeated nested CV; leave-one-study-out; K-curve; protocol emulation; MSCAF internal reference (existing) |
| **Feasible with verification (WearGait-PD)** | SP insole total force → T1-core; walkway-based event validation; blinded inventory |
| **Uncertain** | Redefined double support (synchronisation); all T2 features (geometry); MSCAF external (mapping) |
| **Not feasible / excluded** | Absolute-force features externally; IMU inputs in the primary analysis; staging claims |

### 3.3 What could invalidate the external experiment (stop conditions)
1. **The WearGait-PD insole total force or sampling rate is unusable** (e.g., no reliable total force; rate below 50 Hz) → T1 is not computable.
2. **Event agreement with the walkway fails** the pre-declared criteria for HS → no defensible T1 features.
3. **Too few participants reach K_min** in SP → insufficient sample. Consider adding SP_mat **before** unblinding.
4. **Blinding breach** (labels seen before the freeze) → the external test is no longer independent. Report it as such.
5. **Internal gate A2:** T1 adds nothing over speed on gaitpdb → H2 is moot. Reframe before Phase B.
6. **Internal gate A3:** AUC under protocol emulation collapses (e.g., < 0.65) → the external test is expected to fail for protocol reasons alone. Reconsider the condition choice before Phase B.
7. **Large differential exclusion by group** → biased external estimate. Report it, with a sensitivity analysis.

---

## Deliverable 4 — Decision Checklist

### Ready to implement in A2 (after approval)
- [ ] A new module and outputs directory (no edits to existing files), reusing `gait_common` functions.
- [ ] Usual-walking filter, with a **new cache key** (I2).
- [ ] T1-core features from the existing `gait_features()` outputs (no new feature code needed). Versioned per-stride double support as an addition.
- [ ] Robust parser for the demographics text file (I7); speed-only and age + sex baselines.
- [ ] Repeated nested CV stratified by label × study; M1, M2, M0 and XGBoost-T1; bootstrap CIs; calibration.
- [ ] Re-measure the [M] items with committed scripts (stride counts, bandwidth, sum = total check).

### Needs verification
- [ ] gaitpdb walk-number semantics: Ga `_02`; Ju `_02`–`_07`. Check the original protocols and PhysioNet notes.
- [ ] How gaitpdb `Speed_01` was measured per study (especially Si).
- [ ] gaitpdb `Gender` coding and sex distribution by group.
- [ ] Whether `demographics.xls` exists in gaitpdb 1.0.0 (I7).
- [ ] WearGait-PD (Phase B, blinded):
  - insole sampling rate, units, total-force definition, COP export
  - sensor coordinates and insole sizes
  - SP files per participant and valid strides per foot
  - insole–walkway alignment
  - completeness
  - aggregate site × group counts
  - availability of walkway speed for SP

### Exclude
- swing_pct and cadence (redundant); asym_stride (construct validity); peak_force_L and peak_force_R (absolute amplitude) from external models.
- IMU inputs; transductive normalisation or bias correction in the primary analysis; recalibration in the primary result; thresholds or K_min chosen on WearGait-PD.
- FreeWalk, TUG, Balance and turning tasks from the primary analysis.

### Decisions requiring your approval before A2
| ID | Decision | Recommendation |
|---|---|---|
| D1 | Primary feature set | **T1-core (5 features)**. Double support conditional (T1-extended). T1-fallback pre-declared |
| D2 | Primary model | **M1: class-balanced L2 logistic, C by inner-CV log-loss** |
| D3 | gaitpdb training cohort | **All 165 usual-walking subjects (primary)**; overground-only (Ga + Ju) as sensitivity. *Trade-off:* B matches the target protocol better but gives about 101 subjects; Si's treadmill walking reduces stride variability |
| D4 | Usual-walk definition per study | Provisional Ga `_01`/`_02`, Ju `_01`, Si `_01`, pending verification |
| D5 | Redefine double support per stride (versioned; old definition kept) | Yes |
| D6 | Hypotheses H1–H3, fixed-sequence testing, interpretation bands | As in §2.1. *Alternative for H1:* test against 0.70 instead of 0.50. Stronger, but likely underpowered |
| D7 | K_min (provisionally 10) and protocol-emulation design | Finalise in A3 from gaitpdb only |
| D8 | Event-agreement acceptance criteria (§1 validation table) | Approve the provisional thresholds, or set your own |
| D9 | Decision threshold | Fixed 0.5 (primary); gaitpdb-Youden (secondary) |
| D10 | Blinding custodian and procedure; pre-registration venue (git tag vs OSF) | Name the custodian. OSF + git tag |
| D11 | DBS handling; age-overlap rule for exploratory subsets | DBS included (primary), excluded (sensitivity). Age-overlap = the range shared by both groups in WearGait-PD, defined by the custodian before unblinding |
| D12 | Role of MSCAF externally | Internal reference only, unless you approve a pre-declared 16 → 8 mapping as exploratory |
