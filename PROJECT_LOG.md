# Project Log

This is the shared record of everything changed or decided in this project. **Both team members add to it.**

## How to add an entry

1. Add your entry at the **top** of "Log entries", newest first.
2. Use this format:

   ```
   ### YYYY-MM-DD — <short title>
   **Who:** Lead | Teammate  **Branch/commit:** <branch or short hash>
   - What changed (files, modules)
   - Why
   - Results / numbers, if any
   - Open issues / next step
   ```

3. Commit the log update in the **same pull request** as the change it describes.
4. Never delete or rewrite old entries. If something was wrong, add a new entry that corrects it.
5. Results always come from real runs; never paste numbers you haven't produced.

---

## Current status (keep this section up to date)

| Item | Status |
|---|---|
| Research question | How well do gaitpdb-trained foot-pressure models transfer to WearGait-PD, and what explains any drop in performance? |
| Phase A1 / A1.1 (planning) | Done. See `research/protocol/` |
| Decision register D2–D17 | Approved 2026-11-19 (D10 modified) |
| Phase A2 (gaitpdb pipeline, `experiments/external_transport/`) | Lead + teammate Day 1 merged (96 tests pass). Internal M1 AUC 0.824. Pending: demographics/AAUC gate, E8–E9 evaluation. WearGait-PD: blinded; loader + features done (113/113 insole trials pass QC); emulation re-tuned to 4×4–5. R6 done → **primary external model = M1-fb** (B-D8); M1 secondary. B4 covariates + B5 external scoring table done; **Lead Phase B complete — waiting on Teammate pre-freeze items** |
| Phase B / C / D (WearGait-PD audit, frozen external test, exploratory analysis) | B (Lead side) complete 2026-11-22: B1–B5. C (freeze → unblind → H1–H3) waits on Teammate E8/E9, D15, AAUC |
| Team setup (Part 1) | Done: docs pushed, teammate joined (write access), `main` protected; Lead merges directly, teammate PRs need Lead approval |
| Plan | 3-day sprint: Day 1 = A2, Day 2 = B + C, Day 3 = D + write-up |

## Work split

- **Lead (with Claude Code), branch `data/*`:** `paths`, `guards`, `manifest`, `conditions`, `signal`, `events`, `features`, `qc`, `cache`, the `kcurve`/`emulation` samplers, and the WearGait-PD download, blinding, loader and feature extraction.
- **Teammate, branch `model/*`:** `fake_features`, `splits`, `models`, `metrics`, `uncertainty`, `demographics`, `reporting`, scripts `a2_05`–`a2_07`, the K-curve/emulation evaluation, figures and slides. (The WearGait-PD loader moved to the Lead on 2026-11-21.) See `TEAMMATE_WORK_PLAN.pdf`.

---

## Log entries

### 2026-11-22 — Teammate Day 2: Demographics, E8/E9 eval, D15 rule, frozen scoring script, figures, tests
**Who:** Teammate (Janak Varshney)  **Branch/commit:** model/day2

#### Demographics (`etx/demographics.py`, `scripts/a2_02_demographics_audit.py`, `tests/test_demographics.py`)
- Implemented robust parser for `demographics.txt` (tolerant of ragged trailing tabs on line 161/SiCo24 and others; custom line tokenization instead of `pd.read_csv`).
- Converted Ju heights cm → m; kept `-`/`NaN` as missing (never synthesised).
- Cross-validated against `demographics.html` using stdlib `html.parser` (no `lxml`): 166/166 records matched 100%.
- Noted `Juc010` extra entry in `demographics.txt` with no matching gaitpdb recording (all 165 recordings verified 1-to-1).
- Merged `sex`, `age_y`, `speed_01_mps` into `subject_v2.csv` via `a2_03_events_features --demographics`. Only 1 missing speed: GaPt03.
- Added Test T8 (3 assertions: no synthesis, HTML match within 2%, meters scaling).
- Output: `outputs/day1/demographics/clean.csv` (165 rows + Juc010), `audit.csv`.

#### E9 Emulation (`etx/emulation.py`, `scripts/a2_09_emulation.py`)
- Updated emulation config to 4 segments × [4, 5] strides per segment (B-D6), K_target = 15 (B-D5).
- Results (20 draws, S2_random, `--run-id day1`):
  - **M1-fb** `train-full/test-K`: mean **0.6731 ± 0.0216**, median **0.6710** [0.6411, 0.7155]
  - **M1-fb** `train-K/test-K`:   mean **0.6747 ± 0.0219**, median **0.6760** [0.6397, 0.7189]
  - **D15 decision**: diff = 0.6760 − 0.6710 = **+0.0051 < 0.02** → **`full_walk` training chosen** (rule: use full-walk unless K-matched median AUC is ≥ 0.02 higher).
  - **H3 internal reference (M1-fb emulated AUC)**: **0.6731 ± 0.0216** (median **0.6710**).
  - **M1** `train-full/test-K`: 0.7930 ± 0.0130 (median 0.7918); `train-K/test-K`: 0.7997 ± 0.0141 (median 0.8012).

#### E8 K-curve (`etx/kcurve.py`, `scripts/a2_08_kcurve.py`)
- Implemented `evaluate()`: CV AUC per (scheme, K, draw, regime) for M1-fb and M1; stability stats ICC(2,1), within-SD, bias vs full, Δunivariate AUC.
- Run `--run-id day1`: 39,930 feature table rows, 968 result rows, 60 stability rows.
- **K-curve at K=15 (K_target)**:

  | model_id | regime | mean AUC | median | SD |
  |---|---|---|---|---|
  | M1 | train-full/test-K | 0.7887 | 0.7905 | 0.0223 |
  | M1 | train-K/test-K | 0.7860 | 0.7852 | 0.0213 |
  | M1-fb | train-full/test-K | 0.6464 | 0.6480 | 0.0300 |
  | M1-fb | train-K/test-K | 0.6409 | 0.6438 | 0.0353 |

- **Stability at K=15 (mean over both schemes)**:

  | feature | ICC(2,1) | within-SD | bias vs full | Δunivariate AUC |
  |---|---|---|---|---|
  | stride_time_mean | 0.9842 | 0.0116 | -0.0001 | -0.002 |
  | stride_time_cv | 0.7005 | 0.4521 | -0.311 | -0.022 |
  | swing_time_cv | 0.5669 | 1.491 | -0.478 | -0.067 |
  | stance_pct | 0.9554 | 0.312 | +0.018 | +0.002 |
  | asym_swing | 0.8359 | 1.336 | +0.272 | -0.004 |

- Eligibility at K=15: 72 controls / 93 PD eligible across all schemes.

#### E6 Internal Primary Models (`scripts/a2_06_internal_primary.py`)
- Full 20×5 CV, inner C selection, participant-level bootstrap-of-CV CI (B=200), DeLong+LeDell CIs, internal speed-adjusted AAUC gate for M1 and M1-fb. Sex pre-encoded M→1/F→0 before imputer.
- Results:

  | Model | AUC mean ± SD [2.5%, 97.5%] | Brier | Cal slope | AAUC [CI] | Gate |
  |---|---|---|---|---|---|
  | M1 | 0.8244 ± 0.0075 [0.8112, 0.8363] | 0.1712 | 1.013 | 0.7169 [0.611, 0.833] | ✓ PASS |
  | M1-fb | 0.6654 ± 0.0064 [0.6529, 0.6747] | 0.2230 | 1.043 | 0.6132 [0.511, 0.718] | ✓ PASS |
  | M0 | 0.7703 ± 0.0039 [0.7618, 0.7743] | 0.1910 | 1.023 | — | — |
  | M2 | 0.8418 ± 0.0084 [0.8308, 0.8547] | 0.1599 | 1.003 | — | — |
  | X1 | 0.8587 ± 0.0163 [0.8311, 0.8824] | 0.1520 | 0.702 | — | — |
  | AS | 0.5512 ± 0.0278 [0.4957, 0.5916] | 0.2469 | 0.390 | — | — |

- Bootstrap-of-CV CIs: M1 0.8075 [0.7262, 0.8802]; M1-fb 0.6486 [0.5513, 0.7437].
- **Both internal AAUC CIs strictly above 0.5** → both gates pass.

#### Frozen Scoring Script (`scripts/c_01_frozen_scoring.py`, `tests/fixtures/fake_features.py`, `tests/test_frozen_scoring.py`)
- Fits M1-fb and M1 on all 165 gaitpdb `_01` subjects with final frozen pipelines.
- Reads `subject_external_v2.csv` from CLI path; scores only `qc_pass == True` rows.
- Checks SHA-256 against freeze manifest value (`ac1fe306e2079c1531d80ae64d7784684506bf3b26f0cc002b52bf2ce4ea6a49`), halts on mismatch.
- Strictly never reads `labels/` or takes any label input; validates absence of `label` column in input.
- Writes `outputs/<run_id>/frozen_scores/scores_tall.csv`, `scores_wide.csv`, `frozen_manifest.json`.
- 3 tests pass: SHA-256 mismatch fails loudly, label column fails loudly, end-to-end fake pipeline test.

#### Post-Freeze Scripts (`scripts/c_02_hypothesis_testing.py`, `etx/figures.py`)
- `c_02_hypothesis_testing.py`: H1–H3 testing with Holm-Bonferroni (M1-fb primary), separate recalibration, ROC+calibration figures.
- `etx/figures.py`: `plot_stride_qc`, `plot_bland_altman_agreement`, `plot_drift`, `plot_roc_curves`, `plot_calibration`.

#### Next steps
- Lead merges PR, pushes `model-frozen` tag, runs `c_01_frozen_scoring.py` on the real `subject_external_v2.csv`, then runs `c_02_hypothesis_testing.py` after unblinding.
- Figures: Lead to send `strides.csv` and `agreement/pairs.csv` column names for insole-vs-walkway plotting.

---

### 2026-11-22 — B5: WearGait-PD external scoring table (Lead Phase B complete)
**Who:** Lead  **Branch/commit:** data/external-table
- **New:** `scripts/b_05_external_table.py`, `tests/test_external_table.py` (1 test; 114 pass).
- **Output:** `outputs/b02_20261122/weargait/subject_external_v2.csv`, plus a `.sha256` file.
  - Same columns as `features/subject_v2.csv` (spec §5) but **without `label`**: subject_id = WG code, study = "WG", sex M/F, age_y, speed_01_mps (= SelfPace footprint speed, B-D9), speed_method_status = walkway_footprint, the v2 features, n_valid_L/R, qc_pass, qc_flags, has_insole, source_recordings.
- **Result:** 123 rows; 113 qc_pass (10 without insoles kept as qc_pass = False); 0 missing values in the qc_pass rows. sha256 `ac1fe306e2079c1531d80ae64d7784684506bf3b26f0cc002b52bf2ce4ea6a49`.
- **How scoring will run:** this file is derived from WearGait-PD and stays outside git (CLAUDE.md §4). The Teammate's frozen scoring script will be merged to `main` and **run on the Lead's machine** after the `model-frozen` tag. The tag commit should record the sha256 above, so the scored table is provably the one fixed before unblinding.
- **Next (Teammate, before the freeze):**
  1. rerun E9 with emulation 4 × [4, 5];
  2. E8/E9 for M1-fb and M1;
  3. the D15 rule;
  4. demographics/AAUC gate;
  5. a frozen-model fit + scoring script that reads `subject_external_v2.csv`, uses `qc_pass` rows only, and never reads `labels/`.

**In plain words**
1. We made the final WearGait-PD table: one row per person, with gait numbers, speed, age and sex, but no diagnosis.
2. We saved its fingerprint (sha256), so later anyone can check it wasn't changed after we saw the labels.
3. My side of Day 2 is done; now Janak's pre-freeze work decides when we can freeze and unblind.

**Why it matters:** the external test will score exactly this table, which was fixed before anyone saw who has Parkinson's.

### 2026-11-22 — B4: blinded WearGait-PD covariates (speed, age, sex); `labels/` exception
**Who:** Lead  **Branch/commit:** data/weargait-covariates
- **Rule change (approved by the Lead):** `CLAUDE.md` §3 now allows one exception to "no code reads `labels/`". The blinding scripts `b_01` (writes the key) and `b_04` (reads only `code` and `original_subject`) may use it; they never print or save anyone's group. An ad-hoc diagnostic that read the key outside b_04 was correctly blocked by the auto-mode safety check and was not run.
- **New:** `etx/weargait_covariates.py` (ID normalisation, harmonised demographics, PKMAS reader, `walkway_speed` from footprints), `scripts/b_04_weargait_covariates.py`, config `weargait.walkway_cell_m: 0.0127`, `tests/test_weargait_covariates.py` (4 tests; 113 pass).
- **Output:** `$WEARGAIT_ROOT/blinded/covariates.csv` (code, speed_selfpace_mps, age_y, sex), read-only, no group, one format for all rows.
  - Age and sex come from the two group-split demographics files, harmonised: their column names, formatting and PD-only columns differed and would have leaked the group. UPDRS, H&Y, medication and DBS are never read.
- **B-D9 speed:** the official PKMAS SelfPace velocity exists for only 94/123, and its missingness is group-related (a leak, plus a biased H2 subset). We therefore compute speed from the raw walkway footprints for all 123.
  - Agreement with PKMAS (n = 94): r = 0.990, bias −0.23 cm/s, max |diff| 30.5 cm/s (a few trials disagree; not investigated further, since that would need the key outside b_04).
  - An 8-trial feasibility check gave scale = 1.27 cm/cell (CV 0.2%), r = 1.000: exactly the 0.5-inch Zeno sensor pitch.
- **Results:** 123/123 complete. Speed mean 0.97 m/s (0.39–1.72); age mean 71.4 (45–91); sex F 63 / M 60.
- **Next:** hand-off file for the frozen-model scoring (WearGait-PD subject table in the `subject_v2.csv` format), then the freeze with the Teammate.

**In plain words**
1. We built one table per WearGait-PD person, holding only their code, walking speed, age and sex. No diagnosis.
2. The official speed file was missing 29 people, mostly patients, which itself would have hinted who is ill. So we measured speed ourselves from the walkway footprints for everyone. It matches the official values (r = 0.99).
3. Only the blinding scripts may look at the code key, and that is now written in our rules.

**Why it matters:** H2 (speed-adjusted test) can now use all 123 people, and nothing in the table gives away the diagnosis.

### 2026-11-22 — B-D8 confirmed: M1-fb is the primary external model
**Who:** Lead  **Branch/commit:** data/fallback-decision
- The Lead confirmed the pre-declared D8 consequence of the R6 result:
  - **primary external model = M1-fb** (stride_time_mean, stride_time_cv), used for H1–H3 and the Holm family;
  - **M1 (T1-core) is secondary**, clearly labelled and outside the Holm family.
- `research/protocol/A1_1_DECISION_REGISTER.md`: new section "Phase B decisions" (B-D1..B-D8; B-D5 resolves D16, B-D6 resolves D7).
- `config/a2.yaml`: new `external` block (`primary_model: M1-fb`, `secondary_models: [M1]`).
- **Teammate actions before the freeze:**
  1. pull `main`;
  2. rerun `a2_09_emulation` (emulation is now 4 × [4, 5]);
  3. E8/E9 evaluation for **both** M1-fb and M1 (the H3 internal reference = M1-fb emulated AUC);
  4. D15 rule (train-full vs train-K) for M1-fb;
  5. demographics/AAUC gate;
  6. freeze both models.

**In plain words**
1. We are following our own rule: the main test now uses the 2-feature model (stride time + stride variability).
2. The full 5-feature model is still reported, but as a "secondary" result.
3. This is written in the decision register and the config, so Janak's code uses the right model.

**Why it matters:** the main result can only rest on what the insoles measure reliably, and we decided that before seeing any diagnosis.

### 2026-11-22 — B3 / R6: insole–walkway agreement — toe-off tolerances fail
**Who:** Lead  **Branch/commit:** data/r6-agreement
- **New:**
  - `etx/agreement.py`: shared L/R clock offset by contact-signal agreement (±5 s), walkway strides, IC↔HS pairing (±150 ms), offset-free differences, per-trial DS, Bland & Altman 2007 repeated-measures limits, participant-mean limits, drift slope.
  - `scripts/b_03_agreement.py`.
  - Config `agreement` block (D8 tolerances).
  - `tests/test_agreement.py`: 4 tests; 109 pass.
- **Run `b03_20261122`** (113 insole trials, label-free):
  - 2,947 paired strides. 99.1% of walkway initial contacts matched an insole heel strike.
  - Clock offset median +0.01 s (IQR −0.01 to +0.02). 4 trials hit the stride-period alias (|offset| > 0.2 s); excluding them changes nothing.

  | Metric (insole − walkway) | Bias | SD (RM total / within-subject) | 95% limits (stride-level, RM) | Tolerance | Result |
  |---|---|---|---|---|---|
  | Stride duration | +0.4 ms | 16.8 / 16.8 ms | −32.5 to +33.4 ms | bias ≤ 10 ms | **PASS** |
  | Stance duration | −35.3 ms (participant-mean −28.6) | 39.6 / 21.7 ms | −112.9 to +42.3 ms | bias ≤ 20 ms | **FAIL** |
  | Swing duration | +35.7 ms | **39.5 / 20.6 ms** | −41.6 to +113.0 ms | SD ≤ 15 ms | **FAIL** (on either SD) |
  | Double support (per trial) | −1.96 pp | 5.3 pp | −12.3 to +8.4 pp | bias ≤ 2 pp, no drift | bias PASS; drift flags 29 / inter-insole 48 → **FAIL** |

- **Reading:**
  - The insole heel strike comes about 10–19 ms later and the toe-off about 10–16 ms earlier than the walkway's (the insole threshold is 10% of range; the walkway registers any pressure). So insole stance is shorter and swing longer.
  - The bias varies between people (participant-mean SD ≈ 30 ms); 55% of trials have |bias| > 20 ms.
  - Stride duration (heel strike to heel strike) is unbiased.
  - Drift caveat: with ~25 points per trial and a residual SD ≈ 28 ms, the slope standard error (≈ 0.5 ms/s) exceeds the flag threshold (15 ms / 46 s ≈ 0.33 ms/s), so most drift flags are probably noise. Drift only affects DS, which is not a model input.
- **Pre-declared consequence (D8; Methods R6.3 failure rule; R7 l.240):** "T1-fallback … depends on heel-strike events only. It is the pre-declared reduction if toe-off agreement fails."
  - stance_pct, swing_time_cv and asym_swing fail. The **primary external model becomes M1-fb** (stride_time_mean, stride_time_cv; internal AUC 0.665).
  - No correction using WearGait-PD data is applied.
  - Proposed: M1 (T1-core) reported as a clearly labelled secondary analysis. Needs Lead confirmation, then the Teammate must be told before the freeze.
- **No labels read.** The tolerances were not changed after seeing the result.

**In plain words**
1. We compared the insole step timings with the walkway's (the "ground truth" mat) for 2,947 strides.
2. Full stride time matches almost perfectly (0.4 ms off). But the moment the foot lifts off is detected about 30 ms differently, and that difference changes from person to person.
3. Our own rule, written before we saw this, says: if lift-off timing fails, use only the stride-time features (model M1-fb).

**Why it matters:** this protects us from claiming a result built on features the insoles can't measure reliably. Following our pre-declared rule keeps the study honest, even though the safer model is weaker.

### 2026-11-22 — B2: WearGait-PD loader, Phase B decisions B-D1..B-D7, emulation re-tuned
**Who:** Lead  **Branch/commit:** data/weargait-loader
- **Label-free exploration** of the 123 blinded SelfPace files:
  - Insole sampling is 100 Hz with no time gaps. Each insole has 16 Moticon sensors per foot plus `LTotalForce`/`RTotalForce`, most likely in N (p99 ≈ 1,200).
  - `GeneralEvent` marks Standing/Walk/Turn; 111 trials have 4 Walk passes and 2 have 5. Median trial length is 46 s.
  - `L/R Foot Contact` and `L/R Foot Pressure` come from the walkway (always 0 off the mat), so they are used for R6 only.
  - `ClinicalEvent` holds `FoG` (6 files) and `Stagger` (4 files), which reveal the diagnosis.
  - 10 of 123 trials have no insole columns.
- **Decisions, approved by the Lead 2026-11-22, label-blind, written to the config `weargait` block:**
  - B-D1: trials without insoles are excluded, leaving 113. Group split to be reported after unblinding.
  - B-D2: `ClinicalEvent` is never read and never used to drop strides.
  - B-D3 (revised the same day): gaps of ≤ 20 ms (1–2 samples) are linearly interpolated; a stride touching any longer gap (margin 0.10 s) is invalid. Of the in-walk gaps, 1,771/1,780 were a single sample. Without interpolation, 17% of in-walk strides were lost and only 102/113 trials passed QC.
  - B-D4: only strides fully inside one Walk pass count. The 3-MAD rule is applied to those strides only.
  - B-D5 / D16: **K_target = 15** (the median of min(L,R) valid strides is 17, rounded down to the grid).
  - B-D6 / D7 final: emulation `strides_per_segment` is **[4, 5]** (was [6, 7]).
  - B-D7: `min_strides_per_foot` stays 10.
- **New code:**
  - `etx/weargait.py`: loader (reads only Time, GeneralEvent and L/R TotalForce), `fill_short_gaps`, `pass_ids`, stride validity.
  - `scripts/b_02_weargait_features.py`.
  - `tests/test_weargait_loader.py`: 5 tests. 105 pass in total.
- **Results:**
  - B2 (run `b02_20261122`): **113/113 insole trials pass QC**. Median min(L,R) valid strides 17 (p10 12, min 10, max 48).
  - Feature means: stride time 1.19 s, stride CV 4.36, swing CV 7.73, stance 65.2%, asym swing 5.2.
  - Gaitpdb emulation rerun (`day2_emulation45`): 161–164/165 eligible per draw; emulated walks have a median of 18 strides/foot (WearGait-PD: 17).
- **Caveat:** `double_support_v2_pct` counts overlap with the opposite foot's *valid* strides only, so it reads low at pass edges and when strides are invalid (WearGait-PD mean 26.8%, vs 22.2% before the B-D3 revision; gaitpdb 23.0%). It is descriptive only and not in any model (M1 = T1-core).
- **Note for Teammate:** the emulation config changed to 4 × [4, 5]. Regenerate the E9 inputs with `a2_09_emulation` before running `emulation.evaluate`. K_target = 15 for the K-curve.
- **Next:** R6 insole–walkway agreement (walkway contacts vs insole events: stride, stance, swing, DS).

**In plain words**
1. We wrote the code that reads WearGait-PD and turns each walk into the same gait numbers we use for gaitpdb.
2. We fixed the rules before seeing any labels: drop the 10 people without insoles, never touch the "freezing" column, fill only tiny 10–20 ms data gaps, and use only the straight walking parts.
3. All 113 usable people now have enough strides (median 17), and our gaitpdb "fake WearGait walks" now have about the same number (18).

**Why it matters:** the external test compares like with like, and every rule was set before anyone saw a diagnosis.

### 2026-11-22 — B1: WearGait-PD downloaded and blinded
**Who:** Lead  **Branch/commit:** data/weargait-blind
- **New:** `etx/weargait_blind.py` (file selection, random `WG###` codes from OS randomness, group-free names, leak check), `scripts/b_01_weargait_download.py`, `tests/test_weargait_blind.py` (4 tests). `etx/paths.py`: `LABELS_DIR`, `weargait_root()` (`$WEARGAIT_ROOT`, default `~/Downloads/weargait-pd`; refuses any path inside the repo).
- **Run:**
  - 123 subjects and 367 files downloaded (SelfPace 123, SelfPace_mat 123, SelfPace_matTURN 121), 5.2 GB, into `$WEARGAIT_ROOT/blinded/`, read-only, plus `manifest.csv` (code, task, sha256, bytes).
  - Leak check: 0 files with a group string in their first 20 lines.
  - Key → `labels/weargait_key.csv`, git-ignored and checked with `git check-ignore`.
  - Demographics (2 group-split files) and the PKMAS walkway metrics → `_restricted/`. 100 tests pass.
- **Findings from headers only:**
  - PKMAS has `Velocity (cm./sec.)` per Task. Walkway speed therefore exists, so the H2 precondition (D6/D13) can be checked; the speed overlap is still to be measured.
  - PKMAS also has a `PD vs Control` column, so it must go through an allow-list (code, task, velocity) before use.
  - The demographics files have multi-row headers (`Unnamed:` / `-` columns) and MDS-UPDRS items. They need a careful allow-list (age, sex only) when blinded.
- **Caveat:** `synapseclient` printed the original file names during download, despite `silent=True`. These went only to a local scratch log, never with codes, so the code-to-group mapping is not exposed. Fix the logging if the script is ever rerun.
- **Next:** label-free exploration of `blinded/` (format, sampling rate, units, insole channels, strides per pass, what the `_mat` files contain).

**In plain words**
1. We downloaded the WearGait-PD walking data: 123 people, 3 files each.
2. Every file got a random code like `WG042`, so nobody can tell from the file who has Parkinson's. The "who is who" key is locked in `labels/`.
3. Good news: the dataset has walking speed from the walkway, which we need for our speed-adjusted test (H2).

**Why it matters:** the external test stays honest because nobody can see the diagnoses until the model is frozen.

### 2026-11-21 — WearGait-PD loader moved to Lead; dataset structure surveyed
**Who:** Lead  **Branch/commit:** main
- **Ownership:** the WearGait-PD loader moves from the Teammate to the Lead (`CLAUDE.md` §2 and the work split above updated). The Teammate keeps demographics, the AAUC gate and the E8–E9 evaluation.
- **Dependency:** added `synapseclient` to `experiments/external_transport/requirements-a2.txt` (installed 4.15.0; 96 tests still pass).
- **Synapse survey** (anonymous metadata only, no data downloaded, no per-subject group listing). Project syn52540892 has no extra access requirements, so a free logged-in account is enough. We use `Version 1 (Cross-sectional)` (syn55052683):
  - `CONTROL PARTICIPANTS/CSV files`: 51 controls (`HC##_*`); `PD PARTICIPANTS/CSV files`: 72 PD (`NLS##_*`). Each has `_SelfPace.csv` (insoles), `_SelfPace_mat.csv` (walkway), `_SelfPace_matTURN.csv`, plus HurriedPace, TUG, Balance and TandemGait.
  - The extra Seattle participants (`WHC##` 31 controls, `WPD##` 28 PD, 3 `NLS## (control)`) exist only as `.mat` files.
  - There are 2 demographics files, one per group, and a `Walkway-derived metrics` CSV (PKMAS, SelfPace + HurriedPace).
- **Blinding consequence:** both the folder names and the file prefixes (`HC` vs `NLS`) reveal the diagnosis, and the demographics come split by group. The download script must therefore rename every file to a neutral code and merge the demographics without any group-revealing column. The code-to-group key goes only into `labels/`, which is git-ignored and unread until `model-frozen`.

**In plain words**
1. I now own the code that reads WearGait-PD; Janak keeps the statistics work.
2. We looked at the online dataset (names only, no data). We need about 123 people's "SelfPace" walking files from version 1.
3. The file names themselves say who has Parkinson's, so our download script will rename everything to neutral codes straight away.

**Why it matters:** WearGait-PD is our external test. Hiding the diagnosis until the model is frozen keeps that test honest.

### 2026-11-21 — Lead review of PR #4: approved and merged
**Who:** Lead  **Branch/commit:** main, `0daa1b1`
- First review (request changes) found 4 blocking issues:
  1. the R8 weighting rule chose by AUC;
  2. the speed-adjusted AUC used the wrong estimator (gave 0.850 instead of 0.806 in an independence demo);
  3. inner-CV seeds came from Python `hash()`, so they changed on every run;
  4. `bootstrap_cv` failed silently on duplicated subjects.
  Plus 5 smaller items.
- The teammate fixed all of them in `168a1aa`. The Lead re-verified the demos (AAUC = 0.806 when speed is irrelevant; ≈ 0.5 when the score is only speed; seeds deterministic; bootstrap draws succeed). 96 tests pass on `main` after the merge.
- **Accepted internal results (E5–E7):**
  - R8 chooses unweighted (threshold 0.564).
  - M1 AUC 0.824 (bootstrap-of-CV CI 0.726–0.880), calibration slope 1.01.
  - X1 (XGBoost) AUC 0.859, slope 0.70 (overconfident).
  - Leave-one-study-out mean 0.785.
- **Follow-ups for the teammate before the freeze:**
  1. `demographics` + `a2_02`, so the internal speed-adjusted AUC gate and the M0/M2/AS models can run;
  2. bootstrap-of-CV with inner C selection and B = 500;
  3. inner C selection for the Ga+Ju analysis;
  4. E8–E9 evaluation.

**In plain words**
1. **Janak's model code had 4 serious bugs**, including a wrong formula for one of our main hypotheses and results that changed every run. He fixed all of them, and we checked each fix ourselves.
2. **The main model reaches AUC 0.82 on gaitpdb:** it ranks a PD patient above a healthy person 82% of the time. It's also well calibrated: its probabilities can be trusted.
3. **Before the freeze he still needs to add demographics** (speed, age, sex) and the short-walk evaluation.

**Why it matters:** these numbers are the internal baseline that WearGait-PD will be compared against, so they had to be correct and reproducible.

### 2026-11-21 — Teammate Day 1 PR #4 Lead Review Fixes (R8 weighting, Janes & Pepe AAUC, bootstrap_cv, calibration)
**Who:** Teammate (Janak Varshney)  **Branch/commit:** model/day1
- **What changed:**
  - `etx/metrics.py`:
    - `calibration`: intercept fitted with logit score offset (slope fixed to 1) via 1D root-finding on score equation; slope from separate unconstrained fit.
    - `aauc`: full Janes & Pepe (2009) covariate-adjusted placement-value AUC. Within each study, linear model `score ~ speed` fit on CONTROLS, standardized residuals computed, case placement value evaluated as empirical CDF of control residuals; AAUC is mean placement value over cases with stratified bootstrap CI.
  - `etx/models.py`: dropped deprecated `use_label_encoder` from XGBoost params; added `_lr_C_grid(cfg)` helper to read C grid from config.
  - `etx/uncertainty.py`:
    - `bootstrap_cv`: folds assigned strictly on UNIQUE subject IDs first, then duplicated subjects expanded into the same fold; removed bare `except` to fail loudly on unhandled errors.
    - `ledell_ci`: corrected influence curve variance calculation without fold-sum squaring.
  - `etx/reporting.py`: wrapped all file writes in `guards.safe_write`.
  - `scripts/a2_05_weighting_comparison.py`: implemented full repeated CV ($R=20 \times 5$-fold) with inner CV C selection; applied strict R8 rule (choose `none` unless `balanced` has strictly LOWER Brier score AND calibration slope at least as close to 1); safe_write for output JSON.
  - `scripts/a2_06_internal_primary.py`: deterministic inner fold seed via `splits._seed_for_inner`, C grid from config, default subject CSV discovery via `--run-id`, participant-level bootstrap-of-CV CI using `uncertainty.bootstrap_cv`, per-repeat metric evaluation + summary, safe_write.
  - `scripts/a2_07_sensitivity_loso.py`: deterministic inner seeds, C grid from config, `--run-id` discovery, safe_write.
  - `tests/test_metrics.py` & `tests/test_splits_models_metrics.py`: added AAUC independence test (`score` independent of `speed` -> `AAUC ≈ plain AUC`), updated calibration tests, added tests for `bootstrap_cv` duplicate grouping, fail-loudly behavior, and LeDell CI variance/bounds.
- **Why:** addresses all 7 points of the Lead's review on PR #4 per `CLAUDE.md` and `research/protocol/A1_1_METHODS_REVISIONS.md` (R8, R2).
- **Results (real runs on run-id day1):**
  - **Pytest:** 96 passed (all 63 Lead tests + 33 Teammate tests).
  - **E5 (`a2_05` weighting comparison, R=20 × 5-fold CV with inner C selection):**
    - `none` (unweighted): AUC = 0.8244 ± 0.0075, Brier = 0.1712 ± 0.0035, Slope = 1.0127 ± 0.0900
    - `balanced`: AUC = 0.8243 ± 0.0076, Brier = 0.1737 ± 0.0036, Slope = 1.0089 ± 0.0891
    - R8 Decision: `none` chosen (threshold: 0.564) because balanced Brier (0.1737) >= none Brier (0.1712).
  - **E6 (`a2_06` internal primary models, R=20 × 5-fold CV):**
    - M1 (LR T1-core, 5 features, C tuned):
      - AUC mean = **0.8244 ± 0.0075** [0.8112, 0.8363]
      - Brier = **0.1712 ± 0.0035**
      - Calibration slope = **1.0127 ± 0.0900**, Intercept = **-0.0045 ± 0.0219**
      - Sensitivity = **0.6414 ± 0.0149**, Specificity = **0.8118 ± 0.0165**, Balanced Acc = **0.7266 ± 0.0122**
      - DeLong pooled: AUC = 0.8266 [0.7641, 0.8892]
      - LeDell pooled CI: AUC = 0.8266 [0.7644, 0.8888]
      - Participant-level bootstrap-of-CV CI (B=200): **0.8075 [0.7262, 0.8802]**
    - M1-fb (LR fallback, 2 features):
      - AUC mean = **0.6654 ± 0.0064** [0.6529, 0.6747]
      - Brier = **0.2230 ± 0.0016**
      - Calibration slope = **1.0434 ± 0.0633**, Intercept = **-0.0016 ± 0.0077**
      - Sensitivity = **0.5290 ± 0.0183**, Specificity = **0.6597 ± 0.0287**, Balanced Acc = **0.5944 ± 0.0134**
    - X1 (XGBoost T1-core, 5 features):
      - AUC mean = **0.8587 ± 0.0163** [0.8311, 0.8824]
      - Brier = **0.1520 ± 0.0090**
      - Calibration slope = **0.7016 ± 0.0513**, Intercept = **-0.0463 ± 0.0855**
      - Sensitivity = **0.8016 ± 0.0248**, Specificity = **0.7708 ± 0.0204**, Balanced Acc = **0.7862 ± 0.0172**
  - **E7 (`a2_07` sensitivity analyses):**
    - Ga+Ju only (101 subjects): AUC = **0.8235 ± 0.0853**
    - Leave-one-study-out (M1, 165 subjects):
      - Held-out Ga: AUC = **0.7548** (sens 0.759, spec 0.556)
      - Held-out Ju: AUC = **0.7966** (sens 0.690, spec 0.640)
      - Held-out Si: AUC = **0.8049** (sens 0.486, spec 0.966)
      - LOSO mean AUC = **0.7854 ± 0.0269**
  - Protected paths verified unchanged after all runs (T10 gate pass).
- **Open issues / next step:**
  - Pushed to `model/day1` for Lead re-review and approval of PR #4.

### 2026-11-21 — Teammate Day 1: splits, models, metrics, uncertainty, reporting + tests T6/T7/T9
**Who:** Teammate (Janak Varshney)  **Branch/commit:** model/day1
- **What changed:**
  - `etx/splits.py`: `outer_folds` (R=20 × 5-fold, strata=label×study, seed=base_seed+r), `inner_folds` (5-fold stratified inside each outer-training set, seed=hash(base_seed,r,fold)), `make_all_folds` convenience wrapper. Integrity assertions baked in: no overlap, full union, subject-level only.
  - `etx/models.py`: `make_pipeline` factory for M1, M1-fb, M0, M2, AS, X1, X13-ref. LR family: `SimpleImputer(median) → StandardScaler → LogisticRegression(l2, lbfgs, max_iter=5000)`. XGB: `SimpleImputer → XGBClassifier` (fixed params from config). `set_C` and `get_feature_names` helpers. `class_weight` controlled by `cfg['models']['lr_weighting']`.
  - `etx/metrics.py`: `auc` (wraps sklearn), `delong` (fast DeLong Sun & Xu 2014 — single and paired), `brier`, `calibration` (intercept + slope via logistic recalibration), `threshold_metrics` (sens/spec/BAcc/TP/TN/FP/FN), `aauc` (Janes & Pepe placement-value AAUC).
  - `etx/uncertainty.py`: `summarize_repeats` (mean, std, 2.5–97.5 pct), `bootstrap_cv` (grouped subject-level bootstrap-of-CV, B draws), `ledell_ci` (influence-curve CI for CV AUC), `external_bootstrap` (non-parametric CI for external test sets).
  - `etx/reporting.py`: `run_manifest` (python, platform, pip freeze, git commit/dirty, config hash), `save_manifest`, `save_metrics_json`, `finalize_manifest` (adds end-time, protected-path hashes before/after).
  - `scripts/a2_05_weighting_comparison.py`: E5 — runs M1 with `none` and `balanced` LR weighting, applies R8 rule, writes `weighting_results.json`.
  - `scripts/a2_06_internal_primary.py`: E6 — full R=20×5-fold CV for M1, M1-fb, X1 (and optional M0/M2/AS); inner C selection per fold; threshold metrics; calibration; Brier; DeLong CI; internal AAUC (speed-adjusted) with gate check; writes per-model JSON and predictions CSV.
  - `scripts/a2_07_sensitivity_loso.py`: E7 — Ga+Ju-only 5-repeat×5-fold CV; leave-one-study-out for all 3 studies (no transductive standardisation); writes `sensitivity_results.json`.
  - `tests/test_splits_models_metrics.py`: T6 (subject-disjoint folds for all repeats), T7 (scaler params match train-only fit; poison test shows no leakage), T9 (identical seeds → identical folds and predictions; different seed → different folds).
  - `tests/test_metrics.py`: unit tests for auc (vs sklearn), DeLong (single+paired), Brier, calibration (slope≈1 on well-calibrated), threshold_metrics, AAUC (≈0.5 when score=covariate).
- **Why:** completes the teammate's Day 1 deliverable per `TEAMMATE_WORK_PLAN.pdf` and the first prompt in the setup instructions ("Build etx/splits.py and tests T6, T9, then models.py and metrics.py with T7").
- **Results (real runs on run-id day1):**
  - **Pytest:** 92 passed, 0 failed (all 63 Lead tests including T1, T2, T3, T4, T5, T10 + all 29 Teammate tests T6, T7, T9 and metrics).
  - **E3 (`a2_03`):** 306 walks processed, 54,337 valid strides; all 165 subjects passed QC (93 PD, 72 Control). `subject_v2.csv` generated.
  - **E5 (`a2_05` weighting comparison):**
    - LR unweighted: AUC = 0.8187
    - LR balanced: AUC = 0.8191
    - R8 rule applied: chosen weighting is `balanced`.
  - **E6 (`a2_06` internal primary models, R=20 × 5-fold CV):**
    - M1 (LR T1-core, 5 features): AUC mean = **0.8235 ± 0.0083** [0.8104, 0.8345]
    - M1-fb (LR fallback, 2 features): AUC mean = **0.6646 ± 0.0082** [0.6466, 0.6737]
    - X1 (XGBoost T1-core, 5 features): AUC mean = **0.8585 ± 0.0165** [0.8314, 0.8816]
    - Out-of-fold predictions saved to `outputs/day1/cv/predictions.csv`.
  - **E7 (`a2_07` sensitivity analyses):**
    - Ga+Ju only (101 subjects): AUC = **0.8235 ± 0.0853**
    - Leave-one-study-out (M1, 165 subjects):
      - Held-out Ga: AUC = 0.7548 (sens 0.759, spec 0.556)
      - Held-out Ju: AUC = 0.7952 (sens 0.621, spec 0.680)
      - Held-out Si: AUC = 0.8049 (sens 0.486, spec 0.966)
  - Protected paths verified unchanged after all runs (T10 gate pass).
- **Open issues / next step:**
  - Teammate modules and E5–E7 results are fully validated and ready for commit on `model/day1` and PR to `main`.
  - Next: implement `kcurve.evaluate` / `emulation.evaluate` (E8–E9 gates) with Lead review before Phase B.

### 2026-11-21 — Day 1 leftovers: K-curve and emulation samplers
**Who:** Lead  **Branch/commit:** data/day1-samplers
- **What changed:**
  - `etx/kcurve.py`: S1 contiguous / S2 random sampling of K strides per foot, deterministic seeds per (scheme, K, draw), K-stride feature tables and eligibility counts.
  - `etx/emulation.py`: pseudo-SelfPace walks of 4 segments × 6–7 consecutive valid strides; turn-adjacent strides dropped; segments never adjacent.
  - Scripts `a2_08` and `a2_09`, which build the tables. The `evaluate()` stubs stay for the teammate.
  - Tests in `test_kcurve_emulation.py`.
- **Results (real run `day1_samplers`):**
  - K-curve tables: 39,930 rows (2 schemes × K = 5, 8, 10, 15, 20, 25, all × 20 draws × 165 subjects). **All 165 subjects are eligible at every K**, including K = 25 (the minimum is 31 valid strides per foot).
  - Emulation tables: 3,180 rows (20 draws), with 159–161 of 165 eligible per draw. **4 subjects are never eligible: JuCo02, JuCo03, JuPt05, JuPt08** (2 control, 2 PD), and 5 more are eligible only in some draws (mostly Ju). Their walks break into turn-free stretches too short for 4 segments. The exclusions are balanced across labels, which is the limitation to report.
  - Speed-up: S1 start selection vectorised, cutting the K-curve run from 4 min 42 s to 1 min 41 s with byte-identical output.
  - The cache key changed after the Block 4 edit of `features.py` (recomputed automatically, as T3 intends).
- Tests: 63 passed.
- **Next:** the teammate implements `kcurve.evaluate` / `emulation.evaluate` (E8–E9 gates). The Lead registers on Synapse and starts the WearGait-PD download.

**In plain words**
1. **We built a "short walk" simulator.** gaitpdb walks are long (2 minutes); WearGait-PD walks are 4 short passes over a small walkway. The code cuts gaitpdb walks down to 5, 10, 20… strides, or into 4 WearGait-style passes, and recomputes the 5 features.
2. **It works for almost everyone.** All 165 people can be cut to any length up to 25 strides. The 4-pass version works for about 160 of 165 per try; 4 people from one study (half PD, half healthy) have walks that are too broken up.
3. **Janak's evaluation will use these tables** to measure how much accuracy drops when walks are short.

**Why it matters:** if the model does worse on WearGait-PD, we need to know whether that's just because the walks are shorter. This simulator lets us measure that effect on gaitpdb first, so it isn't confused with real differences between the datasets.

### 2026-11-21 — Day 1 Block 4: legacy equivalence gate PASSED
**Who:** Lead  **Branch/commit:** data/day1
- **What changed:** `features.walk_features_v1` (exact copy of the old `gait_common.gait_features` logic, used only for this check), script `a2_04`, and test `test_legacy_equivalence.py`.
- **Method:** run the new code in "old mode" (old debounce bug, ddof = 0, all walk types averaged per subject) and compare with:
  1. `gait_common.detect_strides` on all 306 files, both feet;
  2. the saved `results_new/gait_features_subject.csv` (read only; the protected-file guard was unchanged).
- **Results (real run `day1_block4`):**
  - Stride events: 306 files, 59,385 strides, **0 mismatches**.
  - Features: **165/165 subjects**, all 13 features; largest difference 1.57e-11 (limit 1e-9); 0 NaN-pattern mismatches.
  - **Gate: PASS.**
- **Meaning:** the rewrite reproduces the old pipeline exactly. Every v1 → v2 difference is therefore a deliberate, documented change:
  - `_01` walks only;
  - debounce bug fixed;
  - odd smoothing window (no effect at 100 Hz);
  - ddof = 1;
  - double support computed per stride.
- Tests: 57 passed.
- **Day 1 (Lead) complete.** Next: pull request `data/day1` → `main`; the teammate then uses `subject_v2.csv`.

**In plain words**
1. **We checked the new code against the old code.** We switched the new code into "old mode" and ran it on all 306 recordings.
2. **The answers matched exactly:** every one of the 59,385 steps, and every number for all 165 people. The only difference was a rounding error 100,000 times smaller than our limit.
3. **So the rewrite has no hidden mistakes.** Every difference between the old and new results comes from a fix we made on purpose and can explain.

**Why it matters:** before anyone builds models on the new features, we've proven the code is correct. In the viva you can say exactly why the new numbers differ from the old ones, instead of "the numbers changed".

### 2026-11-21 — Day 1 Block 3: v2 features, QC, cache, `subject_v2.csv` (T3, T5)
**Who:** Lead  **Branch/commit:** data/day1
- **What changed:**
  - Implemented `etx/features.py`: the 5 T1-core features plus `double_support_v2_pct`, using ddof = 1, valid strides only, and double support computed per stride.
  - `etx/qc.py`: QC flags, minimum 10 valid strides per foot, and one row per subject from the `_01` walk.
  - `etx/cache.py`: the cache key is built from the recordings and their hashes, the config blocks, the module source code and the schema version.
  - Script `a2_03` and tests `test_features_cache.py` (T3, T5).
- **Hand-off file for the teammate:** `outputs/<run_id>/features/subject_v2.csv`. Columns: `subject_id`, `label`, `study`, the 5 T1-core features, `double_support_v2_pct`, `n_valid_L`, `n_valid_R`, `qc_pass`, `qc_flags`, `source_recordings`.
- **Two changes from spec §5 (for the teammate):**
  1. `sex`, `age_y` and `speed_01_mps` are **not** included unless `a2_03 --demographics <demographics/clean.csv>` is given. They come from the teammate's `demographics` module and are joined on `subject_id`; the Lead's code never invents them.
  2. Stride and walk tables are saved as CSV, not parquet, so no pyarrow dependency is needed.
- **Results (real run `day1_block3`, cache key `e2f7d49555a9`):**
  - 306 walks, 54,337 valid strides in total.
  - `subject_v2`: 165 subjects (PD 93, control 72), all 165 pass QC; at least 31 valid strides per foot.
  - The cache works: the second run is a cache hit.
  - Feature means: stride 1.100 s, stride CV 2.56 %, swing CV 4.87 %, stance 62.06 %, swing asymmetry 3.66 %, double support 23.03 %.
  - Descriptive single-feature AUC, v2 on `_01` vs the old all-walk values: swing CV 0.765 (0.778), stance 0.721 (0.723), swing asymmetry 0.706 (0.736), stride CV 0.678 (0.699), stride mean 0.567 (0.566), double support 0.715. No model was trained.
- Tests: 56 passed.
- Next: Block 4, v1 legacy check on all 306 files, then a pull request to `main`.

**In plain words**
1. **We turned the steps into 5 numbers per person.** These measure how long a stride takes, how uneven the stride and swing times are, how much of the stride the foot is on the ground, and how different the left and right swings are. People with Parkinson's tend to score higher on the uneven ones.
2. **We made the hand-off file `subject_v2.csv`.** It has one row per person (165 people) and is the file Janak's models learn from.
3. **We added a smart cache.** Results are reused only if the data, settings and code are exactly the same; if anything changes, they're recalculated automatically.

**Why it matters:** this table is what every model in the project is trained and tested on. The features are clean (normal walks only), they still separate PD from healthy people about as well as before, and the same code will produce the same table for WearGait-PD.

### 2026-11-21 — Day 1 Block 2: signal loading and stride events in seconds (T4)
**Who:** Lead  **Branch/commit:** data/day1
- **What changed:**
  - Implemented `etx/signal.py` (load trial, sampling rate, smoothing and debounce in seconds, contact mask) and `etx/events.py` (heel strike / toe-off, stride table with `valid` and `invalid_reason`). `paths.load_config()` added.
  - New tests in `test_signal_events.py`: T4 at 50/100/120/200 Hz, the debounce rule, invalid reasons, and a check against the old code on 11 real files.
- **Two bugs found and fixed (v2):**
  1. **Old debounce bug** in `gait_common._drop_short_runs`: it padded with `~True` / `~False` (Python ints −2 / −1), which creates false edges, so only one of its two clean-up passes ever works. If a walk starts in swing, short false contacts are never removed. v2 uses the corrected version. `debounce_impl: legacy` reproduces the bug exactly, for the Block 4 v1 check only.
  2. **Even smoothing window:** 50 ms at 50 Hz is 2 samples, and an even window shifts events by half a sample (10 ms). v2 always uses an odd window (3/5/7/11 samples at 50/100/120/200 Hz). It stays 5 at 100 Hz, so gaitpdb is unaffected.
- The sampling rate is now 1 / median time step: exactly 100 Hz on gaitpdb. The overall-rate formula gave 100.007 because of 2-decimal printing drift.
- **Results (real run, 165 primary walks):**
  - The debounce fix changes events in 23 of 165 walks.
  - Valid strides: 30,094 (old) vs 30,086 (fixed).
  - Largest change in a foot's mean stride time: 2.27 ms; average change −0.009 ms.
  - With the legacy setting, the new code reproduces the old stride indices exactly on the sampled files.
- Tests: 48 passed.
- Next: Block 3, v2 features, QC, cache, and `subject_v2.csv` (T3, T5).

**In plain words**
1. **We taught the code to find steps in seconds, not "rows of data".** It now works the same whether a device records 50 or 200 times a second, which matters because WearGait-PD's insoles may not record at 100 Hz like gaitpdb.
2. **We found a hidden bug in the old step-finder.** The part meant to remove tiny false "touches" of the foot only worked half the time. It's fixed in the new version; on gaitpdb the effect is tiny (23 walks change slightly, about 2 ms at most).
3. **We fixed a small timing shift** that would have made steps look about 10 ms late on low-rate devices.

**Why it matters:** if the step-finder behaved differently on WearGait-PD, any drop in accuracy could be a code artefact instead of a real difference between the datasets. Now the same code measures steps the same way on both.

### 2026-11-21 — Block 1 in plain words
**Who:** Lead  **Branch/commit:** data/day1
- Plain-language summary of the 2026-11-20 Block 1 entry below. No code changed.

In Block 1 we did three things:

1. **Protected your old work.** Your existing results and scripts (`results_new/`, `main.py` and so on) can no longer be changed by the new code. If anything touches them, the run stops with an error.
2. **Checked the data is genuine.** All 306 gaitpdb files were compared against PhysioNet's official fingerprints, and every one matches. So you're using exactly the public dataset, and anyone can reproduce your results.
3. **Selected the right walks.** Of the 306 recordings, we keep only the normal walk (`_01`) of each of the 165 people. Dual-task and music-cued walks are excluded, because WearGait-PD has only normal walking. Mixing in other walk types would make the comparison unfair.

The tests confirm all three work, and every count matches the official documentation.

**Why it matters:** the model will now learn from clean, verified, comparable data, and nothing you've already done can be damaged.

### 2026-11-20 — Day 1 Block 1: safety guard, walk-condition rules, manifest (T1, T2, T10)
**Who:** Lead  **Branch/commit:** data/day1
- Implemented `etx/paths.py`, `guards.py`, `conditions.py`, `manifest.py` and scripts `a2_00` (environment + protected snapshot) and `a2_01` (manifest).
- The guard fingerprints 33 protected files (`results_new/`, `results_pd/`, `checkpoints_pd/`, the old scripts) before and after every run and every test session. `safe_write` refuses protected paths and the read-only gaitpdb folder.
- Condition rules `c1`: only `_01` is primary. Any unknown walk code raises an error instead of slipping in.
- **Results (real run, `GAITPDB_ROOT` = official PhysioNet download):**
  - 306 recordings, 165 subjects; all 306 SHA-256 values match PhysioNet's `SHA256SUMS.txt`.
  - Primary: 165 (Ga 47, Ju 54, Si 64; PD 93, control 72).
  - Excluded: Ga `_02` 39, Ga `_10` 27, Ju `_02`–`_07` 75.
  - All files are uniform 100 Hz.
- **Fix during the block:** the first uniformity check flagged all 306 files. The time column is printed to 2 decimals, so accumulated rounding shows some steps as 0.0099 s. Replaced it with a half-sample check against an ideal grid, which still catches dropped samples (new tests for both cases).
- Tests: 33 passed.
- Next: Block 2, signal loading and stride events in seconds (T4).

### 2026-11-19 — Merge policy set; Janak joined
**Who:** Lead  **Branch/commit:** data/day1 (PR #1)
- Teammate `janakvarshney` accepted the collaborator invite and has write access.
- Merge policy:
  - The **Lead merges his own pull requests directly**, using the admin bypass of branch protection.
  - **The teammate's pull requests need the Lead's approval** before merging; authors can't approve their own pull requests.
- PR #1 (A2 skeleton) merged by the Lead under this policy.

### 2026-11-19 — Decisions approved; A2 skeleton created
**Who:** Lead  **Branch/commit:** data/day1
- Decision register: D2–D17 marked Approved. D10 was modified: no custodian; the freeze is the `model-frozen` git tag, and `labels/` is not read before it. An approval record was added to `A1_1_DECISION_REGISTER.md`.
- Created the `experiments/external_transport/` skeleton:
  - `config/a2.yaml` with the approved defaults (threshold 0.564, unweighted LR, `_01` only, ddof = 1, K levels);
  - 17 `etx/` module stubs, each with its owner and function signatures;
  - 11 step scripts (`a2_00`–`a2_10`);
  - `tests/conftest.py` (real-data tests are skipped when `GAITPDB_ROOT` is unset), `requirements-a2.txt` and a README.
- `.gitignore`: added `labels/`, `experiments/external_transport/outputs/`, WearGait folders and `*.parquet`. The repo is public, so no data is committed.
- Check: all modules import. No logic implemented yet; every function raises NotImplementedError.
- Next: once this is merged, the teammate branches `model/day1` from `main`. The lead implements guard, manifest, conditions (T1, T2, T10), then signal, events, features (T3–T5).

### 2026-11-19 — GitHub team setup done (Part 1, Steps 2–3)
**Who:** Lead  **Branch/commit:** main
- Step 2: collaborator invite sent to the teammate from the browser. It takes effect once he accepts.
- Step 3: branch protection turned on for `main` with `gh api`:
  - a pull request with 1 approving review is required;
  - `enforce_admins` is false, so the owner can still push directly in an emergency (for example the `model-frozen` tag).
- Next: start Phase A2 Day 1 on the `data/day1` and `model/day1` branches. This is waiting on the lead's approval of the decision register D2–D17.

### 2026-11-19 — CLAUDE.md added (shared Claude Code rules)
**Who:** Lead  **Branch/commit:** main
- Added `CLAUDE.md`. Every Claude Code session reads it automatically, for both team members.
- It covers:
  - the mandatory `PROJECT_LOG.md` entry format;
  - who owns which modules;
  - research integrity rules (subject-level splits, `_01` cohort, the `labels/` freeze, no synthetic demographics);
  - protected paths, and no data on the public repo.
- Why: so both Claude Code sessions follow the same logging and safety rules without anyone having to repeat the prompt.

### 2026-11-19 — Shared project log created
**Who:** Lead  **Branch/commit:** main
- Added `PROJECT_LOG.md`, this file, as the single shared record of changes and decisions.

### 2026-11-18 — Planning docs and teammate plan pushed to GitHub
**Who:** Lead  **Branch/commit:** main, `0348c78`
- Committed:
  - `research/protocol/A1_1_METHODS_REVISIONS.md`
  - `research/protocol/A1_1_DECISION_REGISTER.md`
  - `research/protocol/A2_IMPLEMENTATION_SPEC.md`
  - `PROJECT_OVERVIEW.md`
  - `RESEARCH_EXTENSION_PLAN.md`
  - `CASSNET_FEASIBILITY_REVIEW.md`
  - `PHASE_A1_FEATURE_AUDIT.md`
  - `TEAMMATE_WORK_PLAN.pdf`
- Not committed (kept local): `_old/`, the MSCAF review PDF/PPTX, and the explainer PDF.
- Work split agreed: the lead does data and features; the teammate does models and statistics, working through branches and pull requests. Plan for the 3-day sprint written up.
- Decided that collaborator access is better than a fork for the sprint. The collaborator invite is still pending.

### 2026-11-18 — Phase A1.1: protocol refinement (docs only, no code changed)
**Who:** Lead  **Branch/commit:** in `0348c78`
- **H3** reworded as the remaining gap between internal and external validation after matching walk structure. It is not a causal breakdown.
- **H2** revised: speed-adjusted AUC of the frozen model, computed inside WearGait-PD with its own walkway speed. This is because walking speed in gaitpdb was measured differently in each study:
  - Si: stopwatch over the middle 10 m, verified.
  - Ga and Ju: method not documented.
- **H1** is tested against AUC 0.50. An AUC of 0.70 is only a benchmark for interpreting the result.
- **Multiplicity:** Holm correction across H1–H3 is recommended.
- **Primary cohort:** the `_01` usual-walk file for all 165 subjects. Excluded: Ga `_02` (meaning not documented), Ga `_10` (dual task) and Ju `_02`–`_07` (cued walking and carryover).
- **Feature version v2:**
  - stride parameters set in seconds rather than samples, with no automatic resampling;
  - sample SD (ddof = 1) for CV features;
  - double support computed per stride.
- **K-curve** (K = 5, 8, 10, 15, 20, 25 and all strides) is now required. Insole–walkway agreement procedure and provisional engineering tolerances defined.
- A2 design and tests T1–T10 specified in `A2_IMPLEMENTATION_SPEC.md`.
- **Documentation errors found:**
  - The "Si = treadmill" statements in `PROJECT_OVERVIEW.md` and `cross_study.py:8` are very likely wrong; Si `_01` is level-ground walking. Not corrected yet; correction needs approval.
  - `main.py` puts random UPDRS/H&Y values in when `.xls` reading fails (`xlrd` is not in `requirements.txt`).

### 2026-11 (before A1.1) — Research direction and Phase A1 audit (docs only)
**Who:** Lead
- Literature review and extension plan (`RESEARCH_EXTENSION_PLAN.md`).
- CASS-Net feasibility review: verdict NO-GO as the main contribution (`CASSNET_FEASIBILITY_REVIEW.md`).
- **Approved direction:** a transportability study with WearGait-PD (PhysioNet gaitpdb → Synapse `syn52540892`) as independent external validation. MSCAF and XGBoost serve as comparison models, not new architectures.
- Phase A1 feature audit (`PHASE_A1_FEATURE_AUDIT.md`, kept as the original record):
  - issues I1–I11, including condition-mixed features, a cache keyed only on subject IDs, smoothing set in samples, and ddof = 0;
  - the primary feature set T1-core approved: stride_time_mean, stride_time_cv, swing_time_cv, stance_pct, asym_swing.

### 2026-10-10 — Analysis suite and final Kaggle results
**Who:** Lead  **Commits:** `b45be12`, `d163bdd`, `136b16c`
- Per-subject out-of-fold P(PD) and subject-level ROC-AUC saved.
- Gait-feature models, late fusion, leakage demo, SHAP and occlusion, cross-study analysis, summary table, and the Streamlit demo (`app.py`).
- **Results:**
  - MSCAF: balanced accuracy 65.28%, AUC ≈ 0.73.
  - 13-feature XGBoost: balanced accuracy ≈ 78.4%, AUC ≈ 0.86.
  - Leakage demo: record-level splitting inflates accuracy by about 19 percentage points.
  - Leave-one-study-out: ≈ 75.2%.

### 2026-09-15 to 2026-09-20 — MSCAF fixes and subject-level evaluation
**Who:** Lead  **Commits:** `f15fc7c` to `46a3483`
- Cross-attention fusion and paper-faithful spatial attention.
- Only real trial files treated as subjects. Evaluation on held-out test subjects only.
- Paper topology made the default (`use_lstm=False`).
- Class-weighted loss, model selection by balanced accuracy, 5-fold subject-level CV.
- README rewritten.

### 2026-04-27 to 2026-07-25 — Initial implementation
**Who:** Lead  **Commits:** `89f15f6` to `cf0b230`
- Initial repo, `main.py` (MSCAF-Gait reimplementation), `requirements.txt`, README.
- Code-review fixes on 2026-07-25.
