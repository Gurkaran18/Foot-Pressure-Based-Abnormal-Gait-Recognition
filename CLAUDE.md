# CLAUDE.md — rules for every Claude Code session in this repo

Project: detecting Parkinson's disease from foot pressure. The main question is how well gaitpdb-trained models transfer to WearGait-PD.
Protocol and specs: `research/protocol/` (A2 design: `A2_IMPLEMENTATION_SPEC.md`). Teammate tasks: `TEAMMATE_WORK_PLAN.pdf`.

## 1. Project log (mandatory)

For **every** code change, documentation change or decision, add an entry to `PROJECT_LOG.md` in the **same commit / pull request** as the change.

Add new entries at the **top** of "Log entries" (newest first):

```
### YYYY-MM-DD — <short title>
**Who:** Lead | Teammate  **Branch/commit:** <branch or short hash>
- What changed (files, modules)
- Why
- Results / numbers, if any
- Open issues / next step
```

- Update the "Current status" table whenever a phase or task changes status.
- Never delete or rewrite old entries. If something was wrong, add a new entry that corrects it.
- Only record numbers from real runs; never invent results.
- If `PROJECT_LOG.md` has a merge conflict, keep **both** entries, ordered by date.

## 2. Who owns what (do not edit the other person's modules)

- **Lead**, `data/*` branches: `paths`, `guards`, `manifest`, `conditions`, `signal`, `events`, `features`, `qc`, `cache`, the `kcurve`/`emulation` samplers, and the WearGait-PD feature extraction.
- **Teammate**, `model/*` branches: `fake_features`, `splits`, `models`, `metrics`, `uncertainty`, `demographics`, `reporting`, scripts `a2_05`–`a2_07`, the K-curve/emulation evaluation, the WearGait-PD loader, and figures.
- New code goes in `experiments/external_transport/`. All merges into `main` go through a pull request reviewed by the other person.
- Changing any output file format in section 5 of the A2 spec requires telling the other person first.

## 3. Research integrity rules

- Splits are by **subject** only. Never split randomly by window or recording.
- All learned preprocessing (scaling, imputation) is fitted inside the training folds only.
- The primary gaitpdb cohort is the `_01` walk of all 165 subjects. Cued (Ju `_02`–`_07`), dual-task (Ga `_10`) and Ga `_02` recordings are excluded.
- No code reads `labels/` (WearGait-PD diagnosis) until the `model-frozen` git tag has been pushed. Recalibration is reported separately and never replaces the primary external result.
- Never use synthetic or random demographic values. UPDRS and H&Y are for description only and are never model inputs.

## 4. Protected paths and data

- Don't modify or overwrite `results_new/`, `results_pd/`, `checkpoints_pd/` or the existing scripts (`main.py`, `gait_common.py`, `gait_features.py`, `cross_study.py`, `explain.py`, `leakage_demo.py`, `app.py`) unless explicitly asked.
- New outputs go to `experiments/external_transport/outputs/<run_id>/`.
- **The repo is public.** Never commit datasets (gaitpdb, WearGait-PD), `labels/`, `outputs/` or credentials. Never re-upload WearGait-PD data, because its licence forbids it.
