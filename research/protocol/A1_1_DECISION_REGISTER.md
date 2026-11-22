# A1.1 — Decision Register

**Scope:**
- D1–D12 come from [PHASE_A1_FEATURE_AUDIT.md](../../PHASE_A1_FEATURE_AUDIT.md).
- D13–D17 are new decisions raised in [A1_1_METHODS_REVISIONS.md](A1_1_METHODS_REVISIONS.md) ("R" numbers refer to that document).

**Status values:**
- **Approved** — already approved by you
- **Proposed** — needs your approval
- **Blocked** — needs evidence first

**Tags:** as in the methods-revisions document ([C], [D], [P], [M], [A], [U], [K]).

| ID | Decision | Recommended choice | Evidence | Remaining uncertainty | Approval required? | Status |
|---|---|---|---|---|---|---|
| **D1** | Primary feature set | **T1-core**: stride_time_mean, stride_time_cv, swing_time_cv, stance_pct, asym_swing. **T1-fallback**: stride_time_mean, stride_time_cv. Double support conditional, outside the primary model | All five are computed within each foot (no L/R sync dependence) and independent of amplitude [C]. Univariate AUC 0.57–0.78 [C]. Two existing features are exact duplicates (ρ = −1) [C] | Toe-off bias and jitter on WearGait-PD (R6, R7); the univariate AUCs came from condition-mixed features | No (approved). Fallback trigger rule in D8 | Approved |
| **D2** | Primary model | **L2 logistic regression.** Weighted vs unweighted decided by the pre-declared internal rule in R8 (default: **unweighted**) | Class weighting mainly shifts the intercept; AUC is essentially unchanged; balanced probabilities are miscalibrated at the real prevalence (R8) | Internal calibration and coefficient-stability results (A2) | **Yes:** approve the selection rule in R8 | Approved 2026-11-19 |
| **D3** | gaitpdb training cohort | **All 165 subjects (`_01`)** as primary. Ga + Ju (101) as sensitivity. Leave-one-study-out retained | Si `_01` is very likely level-ground walking, not treadmill (PhysioNet statement; Frenkel-Toledo protocol; turn-interrupted stride runs) [D][M] | File-level mapping of Si `_01` to the "comfortable speed, level ground" condition [A] | **Yes:** confirm after reading R0/R9 | Approved 2026-11-19 |
| **D4** | Usual-walking definition | **`_01` only, for all studies.** Exclude Ga `_02` (undocumented), Ga `_10` (dual task), Ju `_02`–`_07` (RAS/carryover) | All 165 subjects have `_01` [D]. Ju protocol [D] | Meaning of Ga `_02` [U]. Ju walk-number mapping [A]. Evidence needed: Yogev 2005 and Hausdorff 2007 full texts, or the data authors | **Yes** | Approved 2026-11-19 |
| **D5** | Feature version v2 | Versioned features **v2**: double support redefined per valid stride; event times in seconds; legacy v1 kept for reproduction | Issues I3, I4 [C] | None for v1 reproduction (an equivalence test is required) | **Yes** (with D17) | Approved 2026-11-19 |
| **D6** | Confirmatory hypotheses | **H1:** frozen M1 external AUC > 0.5 (0.70 is only an interpretation benchmark). **H2 (revised):** speed-adjusted external AAUC of frozen M1 > 0.5, using WearGait-PD's own walkway speed. **H3 (revised):** a residual cross-dataset gap remains after protocol matching (not causal) | R1, R2 | H2 depends on walkway speed being available for SelfPace (Phase B) | **Yes:** approve the revised H2/H3 wording | Approved 2026-11-19 |
| **D7** | K levels and emulation | K-curve **required**: K = 5, 8, 10, 15, 20, 25 + all. Primary scheme S1 (contiguous window); S2 (random strides) as sensitivity. 20 draws per level; both training regimes reported. Emulation = 4 segments of about 6–7 consecutive valid strides, turn-adjacent strides removed | The minimum usual-walk stride count per foot is about 31 [M]. WearGait-PD SelfPace geometry [P] | Final segment length and K_target (see D16) | **Yes** | Approved 2026-11-19 |
| **D8** | Insole–walkway agreement and tolerances | The R6 procedure: one shared L/R offset per trial; offset-free duration metrics; drift regression. **Provisional tolerances:** stride bias ≤ 10 ms; stance bias ≤ 20 ms; SD of swing differences ≤ 15 ms; double support bias ≤ 2 pp. If a tolerance fails, drop the feature and use the pre-declared fallback; no correction | Formula for jitter-induced CV inflation (R6.3); sampling resolution | Tolerances are engineering choices, not clinical standards. They will be restated relative to internal effect sizes in A2 | **Yes** | Approved 2026-11-19 |
| **D9** | Decision threshold | Follows D2: **unweighted → training prevalence (0.564)**; balanced → 0.5. Secondary: gaitpdb out-of-fold Youden threshold. Never chosen on WearGait-PD | R8 | — | **Yes** | Approved 2026-11-19 |
| **D10** | Blinding and pre-registration | Custodian procedure from A1 §2.13 (approved as Refinement 2). Pre-registration: **git tag + OSF** before unblinding | — | **Name the custodian** | **Yes:** name the custodian; confirm OSF | Approved (modified 2026-11-19: no custodian; freeze = `model-frozen` git tag, `labels/` unread before tag; OSF optional) |
| **D11** | DBS and age-overlap subsets | DBS included in the primary analysis; DBS-excluded sensitivity applied by the custodian. Age-overlap range defined by the custodian from aggregate data before unblinding | DBS status reveals the label [P] | Size of the subsets (Phase B aggregate counts) | **Yes** | Approved 2026-11-19 |
| **D12** | External role of MSCAF | **Internal reference only.** No external evaluation unless a pre-declared 16 → 8 mapping is separately approved (exploratory) | Different sensor count, layout and units [P][D] | — | **Yes** | Approved 2026-11-19 |
| **D13** | How H2 handles speed | **Covariate-adjusted AUC within WearGait-PD** (R2.2). Frozen M2-vs-M0 ΔAUC only descriptive. Internal analogue adjusted for `Speed_01` within each study | Speed methods: Si verified (stopwatch, middle 10 m) [D]; Ga and Ju unknown [U]; WearGait-PD is walkway-based [P] | Whether WearGait-PD SelfPace speed exists, and its definition (Phase B) | **Yes** | Approved 2026-11-19 |
| **D14** | Multiplicity procedure | **Holm–Bonferroni over H1–H3** (one-sided, family α = 0.05). H2 interpreted only if H1 is rejected | Fixed sequence would block H3 exactly when it matters most (R10) | — | **Yes** | Approved 2026-11-19 |
| **D15** | Train on full-walk or K-matched features | Pre-declared internal rule: if the emulated **train-K/test-K** AUC exceeds **train-full/test-K** by ≥ 0.02 (median over draws), train the frozen model on K-matched pseudo-walks; otherwise train on full walks | K-curve design (R4) | A2/A3 results | **Yes** | Approved 2026-11-19 |
| **D16** | Source of K_target (stride count expected in WearGait-PD) | **Option A (recommended):** fix from the published protocol geometry now (candidates 10/15/20; final pick from the label-blind inventory median, rounded **down** to a grid level). Option B: fix from geometry only | Protocol [P]. Using label-free stride counts is a design choice, not fitting a parameter, but it is data-dependent | Whether you accept label-blind external stride counts as design input | **Yes** | Approved 2026-11-19 |
| **D17** | SD convention for CV features | **Sample SD (ddof = 1)** in v2 (the current code uses ddof = 0) | ddof = 0 understates CV by √((K−1)/K), i.e. −10.6% at K = 5 (R4) | — | **Yes** (with D5) | Approved 2026-11-19 |

## Phase B decisions (label-blind, before the freeze)

All made from blinded WearGait-PD data with no labels read. Details and numbers are in `PROJECT_LOG.md` (2026-11-22). The config keys are in `experiments/external_transport/config/a2.yaml`.

| # | Decision | Basis | Status |
|---|---|---|---|
| **B-D1** | Trials without insole columns excluded (10/123 → 113) | Equipment; group split to be reported after unblinding | Approved 2026-11-22 |
| **B-D2** | `ClinicalEvent` (FoG/Stagger) never read, never used to drop strides | Reveals diagnosis | Approved 2026-11-22 |
| **B-D3** | Gaps ≤ 20 ms linearly interpolated; a stride touching a longer gap is invalid | 1,771/1,780 in-walk gaps are 1 sample | Approved 2026-11-22 (revised the same day) |
| **B-D4** | Only strides fully inside one `Walk` pass; MAD rule on those | Mirrors gaitpdb straight walking | Approved 2026-11-22 |
| **B-D5** | **K_target = 15** (resolves D16) | Inventory median 17, rounded down to the grid | Approved 2026-11-22 |
| **B-D6** | Emulation 4 segments × **4–5** strides (resolves D7 "final segment length") | Inventory ≈ 4.25 strides/pass | Approved 2026-11-22 |
| **B-D7** | `min_strides_per_foot` stays 10 | All 113 pass | Approved 2026-11-22 |
| **B-D8** | **D8 fallback triggered: primary external model = M1-fb** (stride_time_mean, stride_time_cv). M1 (T1-core) reported as a clearly labelled secondary analysis | R6: stride bias +0.4 ms (pass); stance bias −35 ms (fail, tol. 20); swing SD 39.5 ms RM / 20.6 within-subject (fail, tol. 15). Tolerances unchanged after seeing the result | Approved 2026-11-22 |

**Effect on H1–H3:** H1, H2 and H3 are evaluated with the frozen **M1-fb**. Holm over H1–H3 (D14) is unchanged. The H3 internal reference is the M1-fb emulated AUC. The M1 results (same H1–H3 procedures) are secondary, not part of the Holm family.

## Approval record

**2026-11-19:** the lead approved all recommended defaults (D2–D17). D10 was modified: no custodian; the freeze is done with the `model-frozen` git tag; no code reads `labels/` before that tag. Ga `_02` stays excluded. Ga/Ju speed methods stay "unverified", since only online sources are available.

## Questions raised before A2 (now resolved by the approval above)

1. **Ga `_02`:**
   - Accept excluding it from the primary analysis (D4)?
   - Can you obtain the Yogev 2005 and Hausdorff 2007 Methods sections through the NIT library? These would also settle the Ga/Ju speed methods.
2. **H2:** approve the speed-adjusted AUC formulation (D6/D13) in place of transporting a speed coefficient.
3. **Multiplicity:** Holm (recommended) or fixed sequence (D14).
4. **K_target source** (D16): Option A or B.
5. **Model weighting rule and threshold** (D2/D9).
6. **Custodian name** (D10).
7. **Errata:** may I later correct the "Si = treadmill" statements in `PROJECT_OVERVIEW.md` and the [cross_study.py:8](../../cross_study.py#L8) docstring? This is a separate, documentation-only change. No change will be made without approval.
