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
| Decision register D2–D17 | Waiting for the lead's approval |
| Phase A2 (gaitpdb pipeline, `experiments/external_transport/`) | Not started |
| Phase B / C / D (WearGait-PD audit, frozen external test, exploratory analysis) | Not started |
| Team setup (Part 1) | Step 1 done; Step 2 (collaborator) and Step 3 (protect `main`) pending |
| Plan | 3-day sprint: Day 1 = A2, Day 2 = B + C, Day 3 = D + write-up |

## Work split

- **Lead (with Claude Code), branch `data/*`:** `paths`, `guards`, `manifest`, `conditions`, `signal`, `events`, `features`, `qc`, `cache`, the `kcurve`/`emulation` samplers, and the WearGait-PD feature extraction.
- **Teammate, branch `model/*`:** `fake_features`, `splits`, `models`, `metrics`, `uncertainty`, `demographics`, `reporting`, scripts `a2_05`–`a2_07`, the K-curve/emulation evaluation, the WearGait-PD loader, figures and slides. See `TEAMMATE_WORK_PLAN.pdf`.

---

## Log entries

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
