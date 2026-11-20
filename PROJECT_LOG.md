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
| Phase A2 (gaitpdb pipeline, `experiments/external_transport/`) | Block 1 done (guard, manifest, conditions; 33 tests pass). Block 2 (signal/events) next |
| Phase B / C / D (WearGait-PD audit, frozen external test, exploratory analysis) | Not started |
| Team setup (Part 1) | Done: docs pushed, teammate joined (write access), `main` protected; Lead merges directly, teammate PRs need Lead approval |
| Plan | 3-day sprint: Day 1 = A2, Day 2 = B + C, Day 3 = D + write-up |

## Work split

- **Lead (with Claude Code), branch `data/*`:** `paths`, `guards`, `manifest`, `conditions`, `signal`, `events`, `features`, `qc`, `cache`, the `kcurve`/`emulation` samplers, and the WearGait-PD feature extraction.
- **Teammate, branch `model/*`:** `fake_features`, `splits`, `models`, `metrics`, `uncertainty`, `demographics`, `reporting`, scripts `a2_05`–`a2_07`, the K-curve/emulation evaluation, the WearGait-PD loader, figures and slides. See `TEAMMATE_WORK_PLAN.pdf`.

---

## Log entries

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
