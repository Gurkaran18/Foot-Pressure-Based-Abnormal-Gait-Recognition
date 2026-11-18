# A2 — Implementation Specification (design only; not executed)

**Depends on:** [A1_1_METHODS_REVISIONS.md](A1_1_METHODS_REVISIONS.md) ("R" numbers below) and [A1_1_DECISION_REGISTER.md](A1_1_DECISION_REGISTER.md). Proposed decisions are marked ⟨D#⟩.

**Scope of A2:** gaitpdb only. No WearGait-PD data. No external evaluation. Nothing in the existing pipeline is modified.

---

## 1. Design principles

1. **Isolation.**
   - All new code lives in `experiments/external_transport/`.
   - Legacy modules (`main.py`, `gait_common.py`, …) are **not imported at runtime**. Importing `gait_common` pulls in `main.py`, with torch, global `CONFIG` and `set_seed` side effects.
   - Legacy code is imported **only in tests**, to prove equivalence.
2. **Protected paths are read-only.** A guard refuses to write to them and checks their hashes before and after every run (§9):
   - `results_new/`, `results_pd/`, `checkpoints_pd/`, `_old/`
   - every root-level `*.py`, `*.md`, `*.png`, `*.pdf`, `*.pptx`
3. **One configuration file per run** (`config/a2.yaml`).
   - Every output directory is named by a run ID: the hash of the config plus the code version.
4. **Time in seconds** throughout. Native sampling rate is kept (R3).
5. **Every learned transform sits inside a scikit-learn `Pipeline`** fitted on training indices only.
6. **No synthetic data in any model input**; missing values stay missing.

---

## 2. Directory layout

```
experiments/external_transport/
├── README.md                     # how to run, what each script produces
├── config/
│   └── a2.yaml                   # all parameters (below)
├── etx/                          # package (no side effects on import)
│   ├── __init__.py
│   ├── paths.py                  # data root, outputs root, protected-path list
│   ├── guards.py                 # protected-path snapshot/verify; write guard
│   ├── manifest.py               # recording manifest + SHA-256 verification
│   ├── conditions.py             # walking-condition rules (versioned)
│   ├── demographics.py           # robust parser, unit harmonisation, audit
│   ├── signal.py                 # load, fs inference, smoothing/debounce in s, contact mask
│   ├── events.py                 # HS/TO detection, stride table (seconds), validity
│   ├── features.py               # T1 v1 (legacy-equivalent) and v2 definitions
│   ├── qc.py                     # stride/feature QC flags, eligibility
│   ├── cache.py                  # content-addressed cache
│   ├── splits.py                 # repeated stratified subject-level outer/inner folds
│   ├── models.py                 # pipeline factories (LR, XGB-T1, baselines)
│   ├── metrics.py                # AUC, DeLong, Brier, calibration, sens/spec/BAcc, AAUC
│   ├── uncertainty.py            # repeat aggregation, bootstrap-of-CV, LeDell IC CI
│   ├── kcurve.py                 # stride sampling S1/S2, training regimes, stability stats
│   ├── emulation.py              # pseudo-SelfPace walk construction
│   └── reporting.py              # environment capture, run manifest, tables
├── scripts/                      # thin CLI entry points, one per step (§10)
│   ├── a2_00_env_and_guard.py
│   ├── a2_01_manifest.py
│   ├── a2_02_demographics_audit.py
│   ├── a2_03_events_features.py
│   ├── a2_04_legacy_equivalence.py
│   ├── a2_05_weighting_comparison.py
│   ├── a2_06_internal_primary.py
│   ├── a2_07_sensitivity_loso.py
│   ├── a2_08_kcurve.py
│   ├── a2_09_emulation.py
│   └── a2_10_report.py
├── tests/                        # pytest; synthetic fixtures + legacy equivalence
└── outputs/<run_id>/             # git-ignored; never inside results_new/
```

---

## 3. Configuration (`config/a2.yaml`, proposed values)

```yaml
data:
  gaitpdb_root: ${GAITPDB_ROOT}          # env var; never hard-coded
  checksums_file: SHA256SUMS.txt
signal:
  smooth_s: 0.05
  debounce_s: 0.10
  contact_fraction: 0.10                 # of p5..p95 range, per recording
  percentiles: [5, 95]
  stride_range_s: [0.5, 3.0]
  outlier_mads: 3.0
  fs_uniformity_tol: 0.01
conditions:
  version: c1
  primary_walks: ["01"]                  # ⟨D4⟩
  excluded: {Ga: ["02", "10"], Ju: ["02","03","04","05","06","07"]}
features:
  version: v2                            # ⟨D5, D17⟩ (v1 = legacy, for equivalence only)
  t1_core: [stride_time_mean, stride_time_cv, swing_time_cv, stance_pct, asym_swing]
  t1_fallback: [stride_time_mean, stride_time_cv]
  cv_ddof: 1
  min_strides_per_foot: 10               # K_min provisional ⟨D7⟩
cv:
  outer_folds: 5
  repeats: 20
  inner_folds: 5
  strata: [label, study]
  base_seed: 20261118
models:
  lr_C_grid: logspace(-3, 2, 11)
  lr_selection: neg_log_loss
  lr_weighting: [none, balanced]         # compared in step 05 ⟨D2⟩
  xgb_t1: {n_estimators: 300, max_depth: 3, learning_rate: 0.05,
           subsample: 0.8, colsample_bytree: 0.8, min_child_weight: 2}
uncertainty:
  bootstrap_cv_B: 500
  ledell_ic: true
kcurve:
  levels: [5, 8, 10, 15, 20, 25, all]
  draws: 20
  schemes: [S1_contiguous, S2_random]
emulation:
  segments: 4
  strides_per_segment: [6, 7]
  drop_turn_adjacent: true
  draws: 20
```

---

## 4. Modules and responsibilities

| Module | Responsibility | Key rules |
|---|---|---|
| `paths`, `guards` | Resolve the data root from an environment variable. Hold the protected-path list. Snapshot their SHA-256 before a run and verify after. Provide `safe_write(path)`, which raises on a protected path | The run aborts if any protected hash changes |
| `manifest` | Scan the data root with the regex `^(Ga\|Ju\|Si)(Co\|Pt)(\d+)_(\d+)\.txt$`. Parse fields. Verify the SHA-256 of each recording against the official `SHA256SUMS.txt` | 306 recordings and 165 subjects expected [D]. Fail on any hash mismatch |
| `conditions` | Assign each recording: `usual_primary`, `ga02_unverified`, `dual_task`, `ju_ras_or_carryover`. Set `include_primary` | Rules are versioned (`c1`). No other inclusion path exists |
| `demographics` | Parse `demographics.txt` (tolerant of the malformed row) and cross-check against `demographics.html`. Map `Gender` 1 → M, 2 → F. Convert Ju heights cm → m. Keep `-`/`NaN` as missing. Write an audit of missingness by study × group | **Never** fills values. **Never** calls `main.load_subject_info`. UPDRS/H&Y are exported for description only and are blocked from model inputs |
| `signal` | Load a trial; infer fs from the time column and check uniformity; smoothing and debounce from seconds via `n = max(1, round(s·fs))`; per-recording relative contact threshold | Within-recording operations only |
| `events` | HS/TO detection identical in logic to [gait_common.py:183-217](../../gait_common.py#L183-L217), but in seconds. Stride table with validity and `invalid_reason` | At fs = 100 Hz must reproduce the legacy sample indices exactly |
| `features` | **v1** reproduces [gait_common.py:274-320](../../gait_common.py#L274-L320) exactly (equivalence only). **v2:** T1-core with ddof = 1, plus double support v2 | Formulas in §6 |
| `qc` | Per-walk QC flags; eligibility for each K; per-subject aggregation | Eligibility depends only on the signal |
| `cache` | Content-addressed cache (§7) | The legacy cache `results_new/gait_features_subject.csv` is never read |
| `splits` | Repeated stratified subject-level folds (strata = label × study). Inner folds inside each outer-training set. Assertions | One row per subject. Seeds derived deterministically |
| `models` | Pipeline factories (§8) | All learned transforms inside the pipeline |
| `metrics` | AUC; DeLong variance and paired test (fast DeLong, Sun & Xu 2014 [K]); Brier; calibration intercept and slope; sensitivity/specificity/BAcc at a threshold; AAUC (Janes & Pepe [K]) | Tested against reference values (§11) |
| `uncertainty` | Summaries across repeats; bootstrap-of-CV with grouped duplicates; LeDell influence-curve CI; coefficient stability | Implements R5 |
| `kcurve` | S1 and S2 stride sampling; both training regimes; stability statistics (ICC(2,1), within-subject SD, bias, change in univariate AUC) | One vector per subject per draw. No pooling of draws |
| `emulation` | Pseudo-SelfPace walks: 4 segments of 6–7 consecutive valid strides per foot; strides adjacent to invalid (turn) strides dropped | Seeded; reproducible |
| `reporting` | Environment capture (python, platform, `pip freeze`, git commit, dirty flag); dataset checksums; run manifest; tables and figures | Written to `outputs/<run_id>/` |

---

## 5. Input / output schemas

| File (`outputs/<run_id>/…`) | Columns |
|---|---|
| `manifest/recordings.csv` | recording_id, subject_id, study, group_code, label, walk, condition, include_primary, rel_path, sha256, sha256_official_match, n_samples, duration_s, fs_hz, fs_uniform |
| `manifest/excluded.csv` | recording_id, condition, exclusion_reason |
| `demographics/clean.csv` | subject_id, study, label, sex, age_y, height_m, weight_kg, speed_01_mps, speed_method_status ∈ {verified_stopwatch_mid10m (Si), unverified (Ga/Ju)}, tuag_s, hy_desc_only, updrs_desc_only, source_txt_row, html_match |
| `demographics/audit.csv` | variable, study, group, n, n_missing, pct_missing, unit_note |
| `events/strides.parquet` | recording_id, foot, stride_idx, hs_s, to_s, next_hs_s, stride_s, stance_s, swing_s, peak_n (v1 only), valid, invalid_reason |
| `features/walk_v2.csv` | recording_id, subject_id, feature_version, n_valid_L, n_valid_R, t1 features…, double_support_v2_pct, qc_pass, qc_flags |
| `features/subject_v2.csv` | subject_id, label, study, sex, age_y, speed_01_mps, t1 features…, n_valid_L, n_valid_R, source_recordings |
| `cv/predictions.parquet` | experiment, model_id, repeat, outer_fold, subject_id, y, p, linear_predictor, C_selected |
| `cv/coefficients.parquet` | model_id, repeat, outer_fold \| bootstrap_b, feature, coef |
| `metrics/*.json` | estimand, value, interval_type, lower, upper, n, n_events, method, seeds |
| `kcurve/results.parquet` | scheme, K, draw, regime, model_id, n_eligible_PD, n_eligible_Co, auc, interval fields |
| `kcurve/stability.parquet` | scheme, K, feature, icc21, within_sd, bias_vs_full, univariate_auc |
| `emulation/results.parquet` | draw, model_id, regime, auc, … |
| `run_manifest.json` | run_id, config hash, git commit, dirty flag, python, packages, dataset checksums, protected-path snapshot hashes (before/after), start/end time |

---

## 6. Feature definitions (v2)

**Notation.** For foot f ∈ {L, R}, valid strides i have heel strike `hs_i`, toe-off `to_i` and next heel strike `nhs_i` (seconds), with:
- stride Tᵢ = nhs_i − hs_i
- stance Sᵢ = to_i − hs_i
- swing Wᵢ = nhs_i − to_i

`CV(x) = 100 · sd(x, ddof=1) / mean(x)`, computed only if n ≥ 2.

| Feature | Definition (v2) | Difference from v1 |
|---|---|---|
| stride_time_mean | Mean of T over the valid strides of both feet (pooled) | None (seconds instead of samples) |
| stride_time_cv | ½ [CV(T_L) + CV(T_R)] | ddof = 1 |
| swing_time_cv | ½ [CV(W_L) + CV(W_R)] | ddof = 1 |
| stance_pct | 100 · mean(Sᵢ / Tᵢ) over the valid strides of both feet | None |
| asym_swing | 100 · \|ln(mean W_R / mean W_L)\| | None |
| double_support_v2_pct *(conditional)* | For each valid left stride i: the total overlap time between the left contact interval [hs_i, to_i] and right contact intervals belonging to valid right strides, within [hs_i, nhs_i], divided by Tᵢ. Averaged over left strides; symmetric for right; mean of both | v1 used a whole-span mask including invalid strides and pauses (I4) |

**Subject level (primary):** features from the single `_01` walk.

**Eligibility:** ≥ `min_strides_per_foot` valid strides on each foot.

---

## 7. Cache-key design

- **Key** = SHA-256 of canonical JSON (sorted keys) of:
  1. sorted list of (recording_id, file_sha256) actually used
  2. the `signal`, `conditions` and `features` config blocks
  3. module versions: the SHA-256 of the source text of `signal.py`, `events.py`, `features.py`, `conditions.py`
  4. the cache schema version
- **Location:** `outputs/cache/<key>/`. Entries are immutable.
- A **new key on any change** to recordings, file contents, parameters or code.
- No path exists to the legacy cache.

---

## 8. Model pipelines

| ID | Inputs | Pipeline |
|---|---|---|
| **M1** | T1-core | `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=5000, class_weight=⟨D2⟩)`. C from the inner CV (neg log-loss) |
| M1-fb | T1-fallback | Same as M1 |
| M0 | speed_01 | Same as M1 (descriptive internal baseline) |
| M2 | speed_01 + T1-core | Same as M1 (descriptive) |
| AS | age, sex | Same as M1 (descriptive baseline) |
| X1 | T1-core | `SimpleImputer(median)` → `XGBClassifier` (fixed parameters, `random_state` from the seed) |
| X13-ref | Legacy 13 features (v1), recomputed on `_01` | Legacy `make_model('xgboost')` parameters; internal reference only |

**Final candidate models** (A2 produces *candidates*; the frozen external models are fixed only after Phase B decisions):
- Inner CV on all 165 subjects selects C; refit on all 165.
- Save with joblib, record the SHA-256, and save the coefficients.

---

## 9. CV design and statistical methods

1. **Outer CV:** R = 20 repeats of stratified 5-fold over subjects (strata = label × study). Seed for repeat r = base_seed + r.
2. **Inner CV:** stratified 5-fold within each outer-training set (strata = label × study if every stratum has ≥ 5 subjects in the training set; otherwise label only). Seed = hash(base_seed, r, outer_fold).
3. **Point estimate:** pooled out-of-fold AUC per repeat. Report the mean of the repeats and their 2.5–97.5 percentile range (partition variability).
4. **Participant-level CI:** bootstrap-of-CV, B = 500, with duplicates grouped to the same fold (R5). LeDell CI as a cross-check.
5. **Secondary metrics per repeat:**
   - Brier score
   - calibration intercept (logistic recalibration with the model logit as offset)
   - calibration slope (coefficient on the logit)
   - sensitivity/specificity/BAcc at the ⟨D9⟩ threshold
6. **Paired model comparisons:** DeLong on the out-of-fold predictions within each repeat (summarised across repeats) plus the bootstrap-of-CV difference.
7. **Internal AAUC:** M1 adjusted for `Speed_01` within each study (placement values pooled across studies).
8. **Weighting comparison (step 05):** apply the pre-declared rule in R8.
9. **Sensitivity analyses:** Ga + Ju only; leave-one-study-out (train on 2 studies, test on the third; no transductive standardisation).

---

## 10. Execution order (after approval)

| Step | Script | Produces | Gate |
|---|---|---|---|
| E0 | `a2_00_env_and_guard` | Environment lockfile; protected-path snapshot | — |
| E1 | `a2_01_manifest` | Recording manifest; checksum verification | Tests T1, T2 pass; counts 306 / 165 / 165 primary |
| E2 | `a2_02_demographics_audit` | Clean demographics; missingness audit | Test T8 passes |
| E3 | `a2_03_events_features` | Strides; walk/subject features v1 and v2; QC | Tests T4, T5 pass |
| E4 | `a2_04_legacy_equivalence` | v1 reproduces the legacy all-walk table | **Equivalence gate** (§12) |
| — | Full test suite | — | **All tests T1–T10 pass before any model training** |
| E5 | `a2_05_weighting_comparison` | Unweighted vs balanced LR (internal) | The R8 rule is applied; the result is recorded |
| E6 | `a2_06_internal_primary` | M1, M1-fb, M0, M2, AS, X1, X13-ref; uncertainty layers; internal AAUC | **Internal gate:** if internal speed-adjusted AAUC of M1 has a CI including 0.5, report to you before continuing |
| E7 | `a2_07_sensitivity_loso` | Ga + Ju; leave-one-study-out | — |
| E8 | `a2_08_kcurve` | K-curve (S1, S2; both regimes); stability | **K gate:** report eligibility and AUC at K ≤ 15 |
| E9 | `a2_09_emulation` | Emulated internal performance (for H3 and D15) | **Emulation gate:** if emulated AUC < 0.65, report before Phase B |
| E10 | `a2_10_report` | Reproducibility report; candidate models with hashes | Protected-path hashes unchanged |

---

## 11. Test plan (pytest; all must pass before any model training)

| # | Test | Method | Pass criterion |
|---|---|---|---|
| **T1** | Usual-walking filter correctness | (a) A synthetic list of filenames covering every pattern (Ga 01/02/10, Ju 01–07, Si 01, non-trial files, malformed names). (b) The real gaitpdb manifest | (a) Only `_01` is included; non-trial files are ignored. (b) 165 primary recordings, 165 subjects; Ga 47, Ju 54, Si 64 |
| **T2** | No cued or dual-task recordings in the primary manifest | Assert on the real manifest | 0 primary rows with walk ≠ `01`. Excluded counts: Ga `_02` = 39, Ga `_10` = 27, Ju `_02`–`_07` = 75; total 306 |
| **T3** | Cache invalidation | Compute keys for a baseline; then change one recording ID, one file hash, each signal/feature/condition parameter, and the source text of a feature module | Identical inputs give an identical key. Every single change gives a new key. A spy shows the legacy cache path is never opened |
| **T4** | Sampling-rate conversion | Synthetic contact waveforms with known HS/TO times generated at fs = 50, 100, 120 and 200 Hz, including an 80 ms glitch and a 120 ms true contact | Events within ±1 sample (±1/fs s) of the truth at every fs. The 80 ms glitch is removed and the 120 ms contact kept at every fs. Durations agree across fs within 1/fs |
| **T5** | Stride-event and feature calculations | Synthetic bilateral gait: L strides alternating 1.00/1.10 s, R 1.05 s; stance 0.62 s; swing L 0.40 s vs R 0.44 s | Features equal closed-form values to within 1e-9: stride mean; CV with ddof = 1; stance %; swing CV; asym_swing = 100·ln(1.1) = 9.531. **Plus legacy equivalence:** the v1 events and features on all 306 real files equal [gait_common.py](../../gait_common.py) outputs (exact sample indices; features to within 1e-9) |
| **T6** | Subject-disjoint folds | All repeats × outer × inner folds | No overlap between any train/test (or inner train/validation) subject sets. Each subject is tested exactly once per repeat. Strata proportions within ±1 subject of expected |
| **T7** | No preprocessing fitted on validation folds | (a) A spy transformer records the row indices it was fitted on. (b) **Poison test:** set test-fold feature values to extreme constants and recompute | (a) Fitted indices ⊆ training indices for every fold, including C selection. (b) Training-fold scaler and imputer parameters and C are unchanged |
| **T8** | No synthetic demographic data | Parse the demographics and compare with `demographics.html` values for every subject. Static check: no `random` use in `etx/`. Model-input check | All non-missing values match the HTML (after the documented Ju height conversion). Missing values stay missing. UPDRS/H&Y columns never reach a pipeline. `main.load_subject_info` is never called |
| **T9** | Determinism | Run E1–E9 on a reduced configuration twice with the same seeds, then once with a different seed | Identical SHA-256 for every output table on the identical runs. Different folds and draws with the different seed |
| **T10** | No modification of existing results or checkpoints | Snapshot protected paths before the full test session and verify after. Attempt `safe_write` into `results_new/` | Identical hashes. The write attempt raises |

**Additional tests**
- Checksum verification against `SHA256SUMS.txt`.
- Gender mapping (1 → M, 2 → F) for all 165 subjects against the HTML.
- K-sampling: one vector per subject per draw; sampled strides belong to that subject's walk; the S1 window contains ≥ K valid strides per foot.
- Emulation segments consist of consecutive valid strides with no turn-adjacent strides.
- Metric implementations:
  - AUC equals scikit-learn
  - DeLong variance matches a published worked example or a simulation
  - calibration slope ≈ 1 on simulated well-calibrated data
  - AAUC = 0.5 when the score depends only on the covariate
- Bootstrap-of-CV keeps duplicated subjects in the same fold.

---

## 12. Acceptance criteria for approving A2 results

1. Tests T1–T10 and the additional tests all pass. The pytest log is stored in `outputs/<run_id>/tests/`.
2. **Legacy equivalence:**
   - v1 events on 100 Hz gaitpdb match legacy sample indices for 100% of strides.
   - The v1 all-walk subject table reproduces `results_new/gait_features_subject.csv` to within 1e-9 for all 13 features (read-only comparison).
   - Any intentional v2 differences are listed.
3. **Re-measured facts:** sampling rate 100 Hz; total = sum of the 8 sensors; per-walk stride counts. Agreement with earlier values is reported, and any discrepancy explained.
4. The manifest and demographic counts match the documentation (306 recordings / 165 subjects / 165 primary; sex counts PD 58/35, control 40/32).
5. **Protected-path hashes are unchanged** before and after the whole of A2.
6. **Complete reproducibility record:** environment, git commit (clean tree), config hash and dataset checksums.
7. **All internal results report the R5 interval types with their labels.** No external data was touched.

**Stop after A2 results are produced and wait for review before Phase B.**
