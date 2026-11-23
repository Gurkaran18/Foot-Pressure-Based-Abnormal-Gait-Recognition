# Viva notes: likely questions and short answers

For the two presenters. Every number is from a logged run (`PROJECT_LOG.md`). Answer in one or two
sentences, then stop. If asked for more, go to the slide or table named in brackets.

---

## 0. The 30-second version (say this if asked "summarise your project")

> We trained an honest, subject-level model on gaitpdb and tested it, frozen and pre-registered, on
> WearGait-PD: a different insole, protocol and hospital. Inside gaitpdb it reaches AUC 0.67. On
> WearGait-PD it reaches 0.47, which is chance. Short walks don't explain it: the emulated internal AUC is
> 0.673. Measurement doesn't explain it either: the insoles match the walkway on stride time to 0.4 ms.
> The leading explanation is case-mix. WearGait-PD's controls are 11.5 years older and come from a
> different hospital site than its patients, so disease, age and site can't be separated.

## 1. Numbers to know by heart

| What | Number |
|---|---|
| gaitpdb cohort | 165 (93 PD, 72 controls), `_01` usual walks only |
| WearGait-PD cohort | 123 (72 PD, 51 controls). 113 scored (67 PD, 46 controls). 10 without insoles: 5 PD, 5 controls |
| Internal AUC | M1 (5 features) **0.824**; M1-fb (2 features) **0.665**; speed alone 0.770 |
| Emulated internal AUC (WearGait-PD walk structure) | **0.673** (H3 reference) |
| External AUC, frozen M1-fb (primary) | **0.470** [0.357, 0.582] |
| External AUC, frozen M1 (secondary) | **0.385** [0.281, 0.490] |
| H1 / H2 / H3 p | 0.703 / 0.171 / **0.0065** (Holm thresholds 0.05 / 0.025 / 0.0167) |
| Gap (H3) | **0.204** [0.034, 0.342] |
| Device check | stride +0.4 ms (pass); stance −35.3 ms, swing SD 39.5 ms (fail); 2,947 strides |
| Model trained inside WearGait-PD | **0.449** |
| Ages (median) | gaitpdb controls 64, PD 68; **WearGait-PD controls 78.5, PD 67** |
| Site | all WearGait-PD PD from Johns Hopkins Outpatient Center, all controls from Bayview |

---

## 2. Question and motivation

**Q: Why is this question important?**
Most published gaitpdb models report 97–99% accuracy but are never tested on another dataset; only 6% of 102
studies do. External validation is the standard test of a clinical model. Without it we don't know if the
model learned Parkinson's or learned gaitpdb. [slide 3]

**Q: How is this different from your earlier MSCAF-Gait work?**
That phase showed leakage inflates accuracy: the base paper itself drops from 99.6% to 82.3% on new people.
This phase asks the next question: does an honest model survive a new dataset? [slide 3]

**Q: Hasn't someone done this already?**
Not as a frozen, pre-registered gaitpdb → WearGait-PD test. A 2026 preprint shows site confounding *within*
WearGait-PD, and a course project plans a similar test but hasn't reported one. [INTRODUCTION_RELATED_WORK §2.5]

## 3. Data

**Q: Why only the `_01` walks of gaitpdb?**
The other walks are cued (Ju `_02`–`_07`) or dual-task (Ga `_10`). Those conditions change gait variability
by design, and only some participants have them. `_01` is usual walking for everyone. [RESULTS §1]

**Q: Why are only 113 of 123 WearGait-PD people scored?**
Ten trials have no insole data. We excluded them before unblinding (decision B-D1). After unblinding they
turned out to be 5 PD and 5 controls, so the exclusion is balanced.

**Q: WearGait-PD walks are short. How many strides?**
Median 17 valid strides per foot (four passes over a 16-ft walkway), against about 90 in gaitpdb. That's why
we built the walk-structure emulation. [slide 8]

**Q: How did you handle missing insole samples?**
Gaps up to 20 ms (1–2 samples, packet drops) are linearly interpolated. Any stride touching a longer gap is
invalid. All 113 trials passed QC (≥ 10 valid strides per foot).

**Q: How did you get walking speed for WearGait-PD?**
From the walkway footprints for all 123 people: pressure-weighted foot centroids, stride length over stride
time. The official speed file covers only 94 people, and which ones are missing depends on group, which would
leak the diagnosis. Footprint speed agrees with it at r = 0.990.

**Q: How did you keep the diagnosis hidden?**
Files were renamed to random codes at download, and the key went to a git-ignored folder that no analysis
code read until the freeze tag existed. The only exceptions were the download and covariate scripts, approved
and logged, which never output a group.

## 4. Method

**Q: Why logistic regression and not deep learning?**
Interpretability, and the question is transport, not peak accuracy. With 165 people a simple model is less
likely to overfit gaitpdb quirks. If a two-feature linear model doesn't transport, the failure is easy to
diagnose: we can look at each feature directly.

**Q: Why only two features in the primary model?**
The device check, done before the freeze, showed the insoles detect toe-off about 35 ms differently from the
walkway, varying by person. So stance, swing and asymmetry can't be trusted on WearGait-PD. Our pre-declared
rule said to use the heel-strike-only model, M1-fb. [slide 7]

**Q: Isn't that just making the model weaker so it fails?**
No. The rule was written before any result, and the five-feature model did *worse* externally (0.385,
reversed). The weaker model was the more honest one.

**Q: What is the walk-structure emulation?**
We cut gaitpdb walks into 4 segments of 4–5 consecutive straight strides, like WearGait-PD's four short
passes, and re-ran CV 20 times. M1-fb still gets 0.673. So short walks alone should not cost much, and the
H3 gap is measured against that. [slide 8]

**Q: What is D15?**
A pre-declared rule: train the frozen model on short pseudo-walks only if that beats full-walk training by
≥ 0.02. It gained +0.005, so the frozen model is trained on full walks.

**Q: How do you avoid leakage?**
All splits are by subject. Imputation, scaling and the regularisation strength are fitted inside training
folds only. The external data never touched training.

## 5. Freeze and statistics

**Q: What does "frozen" mean exactly?**
Before unblinding we pushed a git tag (`model-frozen`, commit 3cfa8ac). It records the code and the SHA-256
fingerprints of the external table, the training table and the emulation results. We scored WearGait-PD,
logged the score fingerprints, and only then read the diagnoses. [slide 5]

**Q: Why AUC and not accuracy?**
AUC doesn't depend on a threshold or on the PD/control ratio. The two datasets have different ratios and a
shifted score scale, so accuracy at a gaitpdb threshold would mix calibration failure with ranking failure.

**Q: Why one-sided tests?**
Each hypothesis has a direction: better than chance (H1, H2) and a positive gap (H3). That was pre-registered.

**Q: Why Holm and not Bonferroni or a fixed order?**
Holm controls the family error at 0.05 like Bonferroni, but with more power. A fixed order would stop at H1
and never test H3, and H3 matters most exactly when H1 fails.

**Q: What is AAUC (H2)?**
A covariate-adjusted AUC (Janes & Pepe): it compares each patient only with controls of similar walking speed.
It asks whether the score separates PD from controls at equal speed. We use it within WearGait-PD only,
because speed is measured differently in the two datasets.

**Q: How is the H3 interval computed?**
By combining, independently, the bootstrap distribution of the internal emulated AUC (2,000 replicates) with the
bootstrap distribution of the external AUC. p = the share of gap replicates ≤ 0.

**Q: Isn't the internal estimate optimistic?**
If anything it is slightly pessimistic. Internal CV trains on four-fifths of gaitpdb, while the frozen model uses
all of it. That biases the gap towards zero, so the real gap is not smaller than we report.

## 6. Results

**Q: Your M1 AUC is 0.385. Isn't that a sign error?**
No. We checked the label mapping (HC = control, NLS = PD) and recomputed the AUC independently. The cause is
real: in WearGait-PD the gaitpdb markers point the other way. For example, stance % has an AUC of 0.28 there
against 0.72 in gaitpdb.

**Q: Why not just recalibrate?**
Recalibration fixes probabilities, not ranking. M1-fb's calibration slope is 0.006, so its scores carry no
ranking information. Recalibration lowers the Brier score only by predicting the base rate. We report it
separately, as pre-registered.

**Q: Is 113 people enough?**
The AUC interval is about ±0.11. That's wide, but its upper end (0.58) is below the internal 0.67, which is
why the gap test is significant. We can rule out "it works as well as inside gaitpdb".

## 7. Robustness ("is it a bug?")

**Q: How do you know it isn't your code?**
1. **Same answer from a different device.** Stride features from the walkway, a different measurement system, give 0.39.
2. **No signal even when trained on WearGait-PD.** A model trained inside WearGait-PD itself gets 0.449, so the stride-timing signal is absent there.
3. **Other trials, stricter QC, DBS excluded, medication on or off:** all near chance.
4. **All tests pass,** and the E9 numbers reproduced exactly on a second machine.

[slide 10]

**Q: Any analysis where it does work?**
Age-matched (n = 52): 0.57 [0.41, 0.73]. More advanced PD (H&Y ≥ 2.5): 0.575. Both are higher but uncertain,
and both were added after unblinding, so they are exploratory. Walkway strides in trials with turning on the mat
give 0.61–0.66, but those measure turning, not straight walking. That's a hypothesis for future work.

## 8. Interpretation

**Q: So why did it fail?**
Case-mix and site are the leading explanation.
- In gaitpdb the patients are slightly older and slower than the controls.
- In WearGait-PD the controls are 11.5 years older (78.5 vs 67), slower and mostly women, and older people walk more irregularly. So old controls look "Parkinson-like": the model's score rises with age among the controls (ρ = 0.29).
- Every WearGait-PD patient comes from one Johns Hopkins site and every control from another, so disease, age and site can't be separated.

[slide 11]

**Q: Could it be the insoles?**
Unlikely for the primary model. Stride time agrees with the walkway to 0.4 ms, and walkway-based features fail too.

**Q: Can you say the gap is caused by age?**
No. We say "a residual cross-dataset gap after matching walk structure", the fixed wording from our protocol.
Age and site are consistent with it but not proven: the evidence is exploratory and the age-matched subset is small.

**Q: What is the clinical take-away?**
Simple stride-timing markers learned from one cohort don't identify mild, treated Parkinson's disease in an
independent cohort with a different age and recruitment structure. A good internal AUC is not evidence of a
usable tool.

**Q: What would you do differently?**
Choose an external set where patients and controls come from the same site. WearGait-PD's VA Puget Sound
site has both, but it isn't in the release we used. We would also pre-register an age-adjusted analysis.

## 9. Process and integrity

**Q: Did anything change after the freeze?**
No code under `experiments/` changed before the confirmatory run. Everything after it (sensitivity
analyses, figures) is labelled exploratory and doesn't replace the frozen result.

**Q: Any deviations from the plan?**
- **Emulation tuning:** the emulation used a fixed regularisation setting (C = 1.0) while the frozen model tunes C. This is documented and has a small effect.
- **Review order:** one pre-freeze fix PR was merged before the teammate's review, because of the deadline. This is logged.
- **Primary model change:** M1-fb became primary under the pre-declared device-check rule.

**Q: Who did what?**
- **Lead (Gurkaranbir):** data pipeline, WearGait-PD download and blinding, features, device check, external table, freeze and scoring, sensitivity analyses, write-up.
- **Teammate (Janak):** splits, models, metrics, uncertainty, demographics, the K-curve and emulation evaluation, and the first versions of the scoring and hypothesis-test scripts.

All of it is in the shared project log, which records every decision with its date.

## 10. Future work

- A within-site test (VA Puget Sound: both PD and controls).
- Centre-of-pressure and per-sensor features, which beat timing features inside gaitpdb in 2026 work.
- Turning analysis on the walkway.
- The same frozen protocol for every new claim.
