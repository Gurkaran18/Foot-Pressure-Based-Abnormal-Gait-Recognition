# A1.1 — Methods Revisions

**Supersedes parts of:** [PHASE_A1_FEATURE_AUDIT.md](../../PHASE_A1_FEATURE_AUDIT.md), which is kept unchanged as the original audit record
**Date:** 18 Nov 2026 · **Status:** for review. No code changed; no models trained; WearGait-PD not accessed.

**Evidence tags**
| Tag | Meaning |
|---|---|
| **[C]** | Repository code or existing result files |
| **[D]** | Verified in official gaitpdb documentation (PhysioNet page, `demographics.html`, `SHA256SUMS.txt`, retrieved 18 Nov 2026) or in a source paper's text |
| **[P]** | Published WearGait-PD paper (PMC13009270) |
| **[M]** | Measured earlier on downloaded gaitpdb; must be re-measured in A2 |
| **[A]** | Assumption |
| **[U]** | Unresolved |
| **[K]** | Standard methods reference |

---

## R0. New facts established in A1.1 that change the protocol

| Fact | Source | Consequence |
|---|---|---|
| All gaitpdb recordings are described as walking "at their usual, self-selected pace for approximately 2 minutes **on level ground**". Signals are sampled at 100 samples/s, and the total columns are "the sum of the 8 sensor outputs for each foot" | PhysioNet gaitpdb 1.0.0 page [D] | Sampling rate and the total = sum relation are now documentation-verified, not only measured |
| The Si source study (Frenkel-Toledo et al., *J NeuroEng Rehabil* 2005;2:23): 2-minute tests on a **35 m level walkway**, plus separate treadmill tests. "Average gait speed on level ground was determined using a stopwatch by measuring the average time the subject walked the **middle 10 meters** of the 35 meter walkway". Treadmill speed was derived from the level-ground speed | Full text, PMC1188069 [D]; Mov Disord 2005;20:1109 abstract [D] | **Si `_01` is very likely overground walking, not treadmill.** Supporting evidence: Si walks show 4 [3–7] uninterrupted stride runs per walk, i.e. turns [M], consistent with ≈ 130 m on a 35 m walkway. The claim in `PROJECT_OVERVIEW.md` and [cross_study.py:8](../../cross_study.py#L8) that Si is treadmill walking is probably wrong (erratum; not edited) |
| The Ju source study (Hausdorff et al., *Eur J Neurosci* 2007;26:2369): **100 m walks**; rhythmic auditory stimulation (RAS) at 100% and 110% of usual cadence; carryover walks 2 and 15 min later | Abstract [D] | Ju walks are distance-based, consistent with Ju's variable recording lengths (50–134 s [M]). Ju `_02`–`_07` are RAS or carryover walks [A: exact mapping undocumented] |
| The Ga source study (Yogev et al., *Eur J Neurosci* 2005;22:1248): usual walking plus "different types of dual tasking" | Abstract [D] | The meaning of Ga `_02` is **undocumented** [U]. `demographics.html` gives speed only for `_01` and `_10` [D] |
| Every one of the 165 subjects has a `_01` recording. Walk patterns: Ga {01}: 7, {01,02}: 13, {01,02,10}: 26, {01,10}: 1; Ju {01}: 40, multi-walk: 14; Si {01}: 64 | PhysioNet file listing [D] | A uniform **`_01`-only** primary definition keeps all 165 subjects |
| `Gender`: 1 = male, 2 = female (TXT vs HTML cross-check). PD 58 M / 35 F; controls 40 M / 32 F | `demographics.html` vs `demographics.txt` [D] | Sex can be used as a covariate |
| **Ju heights are in centimetres** (150–185), while Ga and Si use metres (1.45–1.95), despite the header "Height (meters)" | `demographics.html` [D] | A2 must harmonise units. Height has no role in the primary models |
| `demographics.xls` **exists** in gaitpdb 1.0.0. `main.py` reads it with `pd.read_excel`, which for `.xls` needs `xlrd`; `xlrd` is not in `requirements.txt`. If the read fails, UPDRS/H&Y become random ([main.py:553-557](../../main.py#L553-L557)). H&Y is cast with `int()`, so 2.5 becomes 2 ([main.py:497](../../main.py#L497)) | File listing [D]; code [C] | The A2 pipeline must never use `main.load_subject_info()` for covariates. Use a dedicated parser that never produces synthetic values |

---

## R1. Corrected H3: residual transportability gap (not a causal decomposition)

**Revised H3.** *A measurable transportability gap remains between protocol-matched internal validation and independent external validation.*

**Estimand.** Gap = AUC(M1 procedure, gaitpdb, protocol-emulated walks, internal CV) − AUC(frozen M1, WearGait-PD SelfPace).

**What protocol emulation can establish**
- The discrimination expected when the **walk structure** of the external protocol is reproduced in the source data, while device, site, population and clinical labelling stay fixed. Walk structure here means number of strides, short straight segments, and removal of turn-adjacent strides.
- So emulation can show whether walk length and segmentation *alone* are enough to explain a loss of the observed size.

**What it cannot establish**
1. **Attribution of the remaining gap.** Device, site, population, disease spectrum, medication state, and unemulated protocol aspects all change together. The residual is a **cross-dataset performance gap**, not an estimate of any single shift component.
2. **Faithful emulation of everything.** Pseudo-walks cut from long, steady walks do not reproduce:
   - gait initiation and acceleration over the 5-ft lead-in
   - walking on a pressure mat
   - wearing 13 IMUs
   - different instructions or footwear

   Some protocol effect may therefore remain in the residual.
3. **Estimand equivalence.** Internal CV estimates the performance of models trained on about 4/5 of gaitpdb. The frozen model is trained on all of it. The internal term is therefore slightly pessimistic, which biases the gap towards zero.
4. **Causal mechanism.** Neither term explains *why* performance differs. Mechanisms are addressed descriptively in Phase D: device agreement, case-mix diagnostics, marker-level effects.

**Reporting language (fixed):** "residual cross-dataset gap after matching walk structure". Never "device effect" or "population effect".

---

## R2. Walking-speed comparability and a revised H2

### R2.1 What is known about `Speed_01`

| Study | How `Speed_01` was obtained | Status |
|---|---|---|
| **Si** | Stopwatch: average time over the **middle 10 m of a 35 m walkway**, averaged over the 2-min level-ground test at comfortable speed (Frenkel-Toledo et al., JNER 2005, Methods) | **[D] for the source study.** That the PhysioNet Si cohort *is* this cohort is a strong inference [A]: PhysioNet cites both Frenkel-Toledo 2005 papers; that cohort was 36 PD / 30 controls vs Si 35 / 29 |
| **Si: treadmill belt speed?** | Not for `Speed_01`. The treadmill speed was **set from** the level-ground speed. The PhysioNet recordings are described as level ground | **[D]** (protocol); [A] (file-level mapping) |
| **Ju** | Walks were 100 m (Hausdorff 2007). The speed method (stopwatch over a segment, or 100 m ÷ walk time including turns) is **not stated in the abstract**. The full text is not open access | **[U]** |
| **Ga** | Gait speed was measured (the abstract reports speed decreasing under dual task). The method is not stated in the abstract. The full text is not open access | **[U]** |
| **WearGait-PD** | ProtoKinetics Zeno walkway (16 ft × 2 ft); SelfPace passes start and end 5 ft off the mat [P]. Walkway-derived gait metrics exist for SelfPace and HurriedPace (`PKMAS Walkway Gait Metrics - HP+SP.csv`, seen in a third-party repository) [A]. The exact velocity definition is [U] | Verify in Phase B |

**Construct comparison.**
- All sources aim at the same construct: **steady-state comfortable straight-line walking speed** (m/s).
- The measurements differ:
  - stopwatch timing over 10 m, averaged over a 2-min walk (Si)
  - unknown methods (Ga, Ju); possibly whole-walk average speed including turns for Ju [A]
  - instrumented mat speed over ≤ 4.9 m per pass with a 1.5 m lead-in (WearGait-PD)
- Distance, start condition, number of passes, and timing method (human vs instrumented) all differ.
- **Conclusion: speed comparability across the datasets is not established.** It is plausible for Si; unknown for Ga and Ju.
- *Evidence needed to resolve:* the Methods sections of Yogev 2005 and Hausdorff 2007 (institutional access), and the WearGait-PD Table S5 velocity definition.

### R2.2 Revised H2 (no cross-dataset speed harmonisation required)

Transporting a **speed coefficient** fitted on gaitpdb's mixed speed measurements to WearGait-PD's walkway speed would silently combine incompatible measurements. The original M2-vs-M0 design is therefore **demoted to descriptive**.

**Revised H2 (confirmatory).** *Among WearGait-PD participants with similar walking speed, the frozen M1 score still discriminates PD from controls.*

- **Estimand:** covariate-adjusted AUC (AAUC) of the frozen M1 linear predictor, adjusted for WearGait-PD walkway speed (Janes & Pepe, *Biometrika* 2009;96:371–382 [K]).
  1. Model the control distribution of the score given speed: a linear regression of score on speed in WearGait-PD controls, with the empirical distribution of standardised residuals.
  2. Compute each PD participant's placement value.
  3. AAUC = 1 − mean placement value.
- **Test:** H0: AAUC = 0.5 vs AAUC > 0.5. Stratified participant bootstrap (B = 2,000), one-sided p.
- **Why it is defensible:**
  - Speed is used **only within one dataset and one measurement method**, as a conditioning variable at evaluation.
  - The classifier is not refitted.
  - No speed coefficient is transported.
  - Using labels to *estimate* an accuracy measure is part of evaluation, exactly as computing an AUC is.
- **Precondition:** walkway speed is available for the SelfPace trials and the speed ranges of the two groups overlap. Report the overlap. If it fails, H2 is reported as **not testable**, not as failed.
- **Internal analogue (gaitpdb, secondary):** AAUC of M1 adjusted for `Speed_01` **within each study**, since measurement is consistent within a study. Studies are pooled by placement values.
- **Descriptive only:**
  - frozen M2 (speed + T1) vs frozen M0 (speed only) ΔAUC on WearGait-PD, with the measurement-incompatibility caveat
  - exploratory within-WearGait-PD cross-validated incremental-value analysis (refits on external data, so never confirmatory)

---

## R3. Sampling-rate strategy (no automatic resampling)

1. **Parameters in seconds:** smoothing 0.05 s, debounce 0.10 s, stride range 0.5–3.0 s.
   - Convert to samples with the recording's **actual** sampling frequency: `n = max(1, round(seconds × fs))`.
   - fs is inferred from the time column, checked against metadata, and the file is rejected if non-uniform (tolerance pre-declared: 1%).
2. **Event times kept in seconds** (sample index / fs). All durations are computed in seconds.
3. **Native resolution preserved.** No resampling for event detection or T1 features.
4. **Resample only where a common grid is genuinely required:**
   - T3/MSCAF windows, which need a fixed number of samples per window
   - per-stride shape normalisation (T2 candidates), which resamples each stride to a fixed number of points independently of fs
5. **Limits of upsampling a lower-rate signal:**
   - it adds **no information**
   - interpolation cannot place an event more precisely than the original sample spacing (about ±1/(2·fs) quantisation)
   - smoothing kernels defined in samples change their time span
   - interpolated contact thresholds can create apparent sub-sample precision that is not real

   If WearGait-PD insoles are sampled below 100 Hz, timing jitter increases. This inflates variability features (see R6.3) and is reported as a limitation, not hidden by resampling.
6. **Known code issue:** [gait_common.py:149](../../gait_common.py#L149) defines smoothing in samples. The new module defines all parameters in seconds. At fs = 100 Hz it must reproduce the legacy events exactly (equivalence test).

---

## R4. Stride-count (K) experiment — required, gaitpdb only

**Levels.** K = 5, 8, 10, 15, 20, 25 valid strides **per foot**, plus "all" as reference.
- The data support this range. The minimum usual-walk stride count per foot is Ju ≈ 31 [M], so every level ≤ 25 is reachable for all, or nearly all, subjects.
- Levels may be trimmed if A2's eligibility counts fall below 80% of subjects at a level.

**Sampling schemes**
- **S1 (primary): contiguous time window.** Pick a random start heel strike, using a seeded generator. Take the shortest time window that contains K valid strides on *each* foot. Invalid strides inside the window are dropped, as in a real short recording.
- **S2 (sensitivity): random strides.** Draw K valid strides per foot without replacement, regardless of position.

**Independence and repetition**
- Each subject contributes **exactly one** feature vector per draw.
- Draws are **never pooled** as extra samples (that would be pseudo-replication).
- R_K = 20 draws per level, with seed = hash(base seed, K, draw, subject).
- Each draw is analysed separately with the full subject-level CV design (R5). Results are summarised across draws.

**Training regimes, both reported**
- **Train-full / test-K:** the model is fit on full-walk features and evaluated on K-stride features. This mirrors applying a gaitpdb model to short external walks.
- **Train-K / test-K:** features at the same K in training and test.

**Reported per K**
- eligible n by group and study
- AUC: median and 2.5–97.5 percentiles across draws, plus a participant-level CI (R5)
- **feature stability for each T1 feature:**
  - ICC(2,1) between the K-sample value and the full-walk value
  - within-subject SD across draws
  - change in univariate AUC
  - bias relative to the full-walk value (CV features are expected to be biased low at small K)

**Rules**
- K is never chosen from WearGait-PD labels or external performance.
- Variability features use the sample SD (ddof = 1; decision D17). With the population SD (current code), CV is underestimated by a factor √((K−1)/K), i.e. 10.6% at K = 5.

---

## R5. Uncertainty: what each interval means

| Source | Reported as | Method | Limitation |
|---|---|---|---|
| **CV partition variability** | Distribution across repeats: median, 2.5–97.5 percentiles over R = 20 repeated 5-fold splits | Repeated nested CV | Not a confidence interval. Describes sensitivity to fold assignment only |
| **Participant sampling + model fitting + hyperparameter selection (internal)** | 95% CI for the expected CV-AUC of the *procedure* | **Bootstrap of the whole nested-CV procedure**: resample participants (B = 500), rerun outer and inner CV including C selection. **Duplicated participants are kept in the same fold** (grouped). Cross-check: influence-curve CI for cross-validated AUC (LeDell, Petersen & van der Laan, *Electron J Stat* 2015;9:1583–1607 [K]) | Approximate. CV-based intervals can under- or over-cover (Bates, Hastie & Tibshirani, *JASA* 2024 [K]). Wide at n = 165 |
| **Model-fitting variability** | Coefficient stability | Coefficients across bootstrap refits: median, IQR, % sign agreement | Descriptive |
| **External sampling uncertainty (primary external)** | 95% CI for the **frozen model's** performance in the WearGait-PD population | Stratified participant bootstrap (B = 2,000; percentile, with BCa check). DeLong CI for AUC | Conditional on the frozen model. **Does not include training-sample uncertainty.** This is the correct estimand for "how well does this model transport" |
| **External uncertainty including training (exploratory)** | Procedure-level external interval | Two-stage bootstrap: resample gaitpdb and refit, then resample WearGait-PD and evaluate | Different estimand. Reported separately |
| **H3 gap** | 95% CI | Independent combination of the internal bootstrap-of-CV distribution and the external bootstrap distribution (the samples are independent) | Mixes a procedure-level and a model-level estimand (R1, point 3) |

**Rule:** a bootstrap of one fixed out-of-fold prediction set is never presented as capturing CV uncertainty. If shown at all, it is labelled "conditional on this partition".

---

## R6. Insole–walkway agreement validation (Phase B) — specification

### R6.1 Principle
The two systems have **independent clocks**, so absolute event latency (e.g., a constant heel-strike detection delay) **cannot be separated from clock offset**. Validation therefore centres on **offset-free quantities**: durations measured within each system (stride, stance, swing, double support). Alignment is used **only to pair strides**, never to "remove" error in those metrics.

### R6.2 Procedure

1. **Allowed data.** Blinded WearGait-PD insole files and walkway footfall event times (initial contact and, if exported, final contact) for SelfPace trials. No labels and no covariates.
2. **Unit of alignment: one offset per trial**, not per participant. Each trial is a separate recording start, so offsets can differ between trials.
3. **One shared offset for left and right.**
   - It is estimated from the cross-correlation of the binary contact signals of both feet (insole vs walkway), searched over ±5 s.
   - Separate left/right offsets are **not** used, because they would absorb inter-insole desynchronisation, which is exactly what corrupts double support.
   - Per-foot offsets are estimated **diagnostically only**. Their difference is reported as the inter-insole desynchronisation estimate.
4. **Raw (pre-alignment) report:** distribution of clock offsets per trial. This describes synchronisation quality, not detection error.
5. **Stride pairing after alignment.**
   - Each walkway initial contact is matched to the nearest insole heel strike of the same foot within ±150 ms.
   - Report match rate, unmatched walkway contacts, and unmatched insole strides.
   - Only on-mat strides can be validated.
6. **Residual event errors after alignment** (insole HS − walkway IC; insole TO − walkway final contact).
   - These are reported, with the caveat that the mean HS residual partly reflects how the offset was estimated.
   - Their **SD (jitter)** is informative; their mean is only partly identifiable.
7. **Primary agreement metrics (offset-free, per paired stride):** differences in stride duration, stance duration, swing duration (where consecutive walkway contacts exist) and per-trial double support %.
   - Report bias and 95% Bland–Altman limits.
   - Stride-level limits use the repeated-measures correction (Bland & Altman, *J Biopharm Stat* 2007 [K]); participant-mean limits are also reported.
8. **Drift detection.** For each trial, regress the post-alignment HS residual on time within the trial; report the slope (ms/s) and compare first vs last pass.
   - A trial is flagged if |slope| × trial duration exceeds the pre-declared jitter tolerance.
   - Inter-insole drift is checked the same way, using the left-minus-right residual difference.

### R6.3 Tolerances: provisional engineering choices, not clinical standards

Why event **jitter** matters more than bias for T1:
- With independent timing error of SD σ_e on each event, a duration's error SD is √2·σ_e.
- The observed CV becomes approximately √(CV_true² + (√2·σ_e / μ)²), where μ is the mean duration.

| Example | Result |
|---|---|
| **Swing** (μ ≈ 0.40 s), σ_e = 5 ms | Adds ≈ 1.8 CV points: a true 3.0% reads as ≈ 3.5% |
| **Swing**, σ_e = 10 ms | Adds ≈ 3.5 points: a true 3.0% reads as ≈ 4.6% |
| **Stride** (μ ≈ 1.1 s), σ_e = 10 ms | Adds ≈ 1.3 points |

These are illustrative calculations from the formula [K-derived], not measured values.

**Provisional tolerances (to approve, D8)**

| Metric | Tolerance | Rationale |
|---|---|---|
| Stride-duration bias | ≤ 10 ms | About 1 sample at 100 Hz; about 1% of stride time |
| Stance-duration bias | ≤ 20 ms (≈ 2 percentage points of stance % at a 1.1 s stride) | Comparable to the threshold- and coverage-related bias expected between systems; small relative to the PD–control stance % difference observed internally (to be quantified in A2) |
| **SD of per-stride swing-duration differences** | ≤ 15 ms | Keeps the added swing-CV inflation within about 1.5–2 CV points. This is an upper bound, because walkway jitter is included |
| Double support % bias | ≤ 2 percentage points, with no drift flag | — |

- These numbers are **engineering tolerances tied to sampling resolution and their expected effect on features**. They are not published clinical standards.
- A2 will quantify the internal PD–control difference in each feature, so the tolerances can be expressed as a fraction of the effect size before Phase B.
- **Failure rule** (pre-declared): if a feature's tolerance fails, that feature is removed and the pre-declared fallback set is used. **No correction using WearGait-PD data** is applied in the primary analysis.

---

## R7. T1-core: what it does and does not eliminate

- **T1-core** (stride_time_mean, stride_time_cv, swing_time_cv, stance_pct, asym_swing) **minimises** dependence on force amplitude and on left/right synchronisation. All five are computed within each foot's own event sequence [C].
- It does **not eliminate device-dependent event-detection bias or jitter**:
  - **Toe-off timing** depends on sensor coverage and the contact threshold. This affects stance_pct, swing_time_cv and asym_swing.
  - **Timing jitter** inflates both CV features.
- **T1-fallback** (stride_time_mean, stride_time_cv) depends on heel-strike events only. It is the pre-declared reduction if toe-off agreement fails.
- **Double support** remains conditional and outside the primary model.

---

## R8. Unweighted vs class-balanced logistic regression

**Effects of class weighting**
- For logistic regression, class weighting mainly **shifts the intercept**. Under correct specification, the shift is about log(n₀/n₁) relative to the unweighted fit. Slopes change only under misspecification.
- **AUC is essentially unaffected**, because ranking is invariant to the intercept.
- Predicted probabilities from the balanced model describe a hypothetical 50% prevalence. They are **miscalibrated** for the training prevalence (93/165 = 0.564) and for the external prevalence (WearGait-PD: 100/185 = 0.541 before exclusions [P]).
- **Interpreting 0.5:**
  - In the balanced model, a 0.5 threshold roughly balances sensitivity and specificity.
  - In the unweighted model, 0.5 favours the majority class (PD). The roughly equivalent threshold is the training prevalence, 0.564.
  - With a mild class ratio (1.29), the intercept shift is small, about log(1.29) ≈ 0.26 on the logit scale.

**Comparison in A2 (internal only; pre-declared selection rule)**
- **Criteria:**
  - AUC (expected difference < 0.01)
  - Brier score
  - calibration intercept and slope
  - coefficient stability (bootstrap IQR, sign agreement)
  - sensitivity/specificity balance at each model's natural threshold (0.5 balanced; 0.564 unweighted)
- **Rule:**
  - Choose **unweighted** unless the balanced model has a lower Brier score **and** an equal or better calibration slope.
  - In a tie, choose unweighted, because its probabilities are interpretable at the training prevalence.
  - The threshold follows the chosen model: unweighted uses 0.564; balanced uses 0.5.
- WearGait-PD performance is never used.

---

## R9. Training cohort and walking-condition definition

| Item | Revision |
|---|---|
| **Usual walking (primary)** | **`_01` for all three studies.** Uniform, documented as usual/comfortable level-ground walking, and keeps all 165 subjects [D] |
| Ga `_02` (39 subjects) | **Excluded from the primary analysis.** Meaning undocumented [U]. Becomes a sensitivity analysis only if verified as usual walking (Yogev 2005 full text, or the data authors) |
| Ga `_10` (27) | Excluded: dual task |
| Ju `_02`–`_07` (75 files) | Excluded: RAS or carryover walks (Hausdorff 2007) |
| **Si compatibility** | Compatible: level-ground comfortable walking on a 35 m walkway (R0). "All 165" is therefore a coherent overground cohort |
| Primary training cohort | All 165 subjects (`_01`) |
| Sensitivity | Ga + Ju only (101), as you requested. Rationale now: speed method and protocol details differ by study, and Si's verified protocol differs (35 m walkway, stopwatch speed) |
| Leave-one-study-out | Retained (internal heterogeneity) |

---

## R10. Multiplicity: review of fixed-sequence testing

- **Fixed sequence H1 → H2 → H3** stops testing at the first non-rejection.
- That is appropriate for **H2 after H1**: speed-adjusted discrimination is meaningless if there is no discrimination at all.
- It is **not** appropriate for **H3**. A residual gap is scientifically important *especially* when H1 fails, and fixed sequence would block it.

**Recommendation (D14)**
- Use **Holm–Bonferroni across H1, H2 and H3** (family-wise α = 0.05, one-sided tests). It controls error without ordering assumptions.
- Add a **logical restriction**: an H2 rejection is interpreted only if H1 is also rejected.
- *Cost:* slightly lower power than fixed sequence for H1.
- *Alternative:* a graphical gatekeeping procedure (Bretz et al., *Stat Med* 2009 [K]) giving H1 and H3 α = 0.025 each, with H2 inheriting H1's α. More complex and no clear gain at this sample size.

**Tests**
- **H1:** one-sided DeLong z-test of AUC = 0.5 (bootstrap check). The 0.70 value is an **interpretation benchmark** only.
- **H2:** bootstrap one-sided p for AAUC = 0.5.
- **H3:** bootstrap one-sided p for gap ≤ 0.

---

## References verified in A1.1

- **PhysioNet gaitpdb 1.0.0:** https://physionet.org/content/gaitpdb/1.0.0/; `demographics.html` and `SHA256SUMS.txt` in the same project.
- **Frenkel-Toledo S, et al.** Effect of gait speed on gait rhythmicity in Parkinson's disease: variability of stride time and swing time respond differently. *J NeuroEng Rehabil* 2005;2:23. doi:10.1186/1743-0003-2-23 (PMC1188069; full text read).
- **Frenkel-Toledo S, et al.** Treadmill walking as an external pacemaker to improve gait rhythm and stability in Parkinson's disease. *Mov Disord* 2005;20:1109–1114. doi:10.1002/mds.20507 (abstract read).
- **Hausdorff JM, et al.** Rhythmic auditory stimulation modulates gait variability in Parkinson's disease. *Eur J Neurosci* 2007;26:2369–2375. doi:10.1111/j.1460-9568.2007.05810.x (abstract read).
- **Yogev G, et al.** Dual tasking, gait rhythmicity, and Parkinson's disease: which aspects of gait are attention demanding? *Eur J Neurosci* 2005;22:1248–1256. doi:10.1111/j.1460-9568.2005.04298.x (abstract read).
- **Anderson AJ, et al.** WearGait-PD. *Sci Data* 2026;13:440. doi:10.1038/s41597-026-06806-2 (PMC13009270).
- **Standard methods [K], not re-fetched:**
  - Janes & Pepe, *Biometrika* 2009 (covariate-adjusted ROC)
  - LeDell et al., *Electron J Stat* 2015 (CV-AUC CI)
  - Bates, Hastie & Tibshirani, *JASA* 2024 (cross-validation)
  - Bland & Altman, *J Biopharm Stat* 2007 (repeated-measures agreement)
  - Bretz et al., *Stat Med* 2009 (graphical multiplicity)
  - DeLong et al., *Biometrics* 1988
