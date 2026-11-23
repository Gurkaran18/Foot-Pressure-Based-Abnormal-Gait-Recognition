# Freeze record: `model-frozen`

Frozen on 2026-11-23 by the Lead, before any code or person has read the WearGait-PD diagnosis (`labels/`).
The git tag `model-frozen` points at the merge commit that adds this file. Everything that decides the
external result is fixed by that commit plus the input fingerprints below.

## What is frozen

| Item | Frozen value |
|---|---|
| Primary external model | **M1-fb**: L2 logistic regression on `stride_time_mean`, `stride_time_cv` (B-D8) |
| Secondary model | M1: T1-core (5 features), outside the Holm family |
| Training data | All 165 gaitpdb `_01` walks, **full-walk features** (D15: K-matched gain +0.0051 < 0.02) |
| Training procedure | `scripts/c_01_frozen_scoring.py`: median imputation, standardisation and C (inner 5-fold CV, log loss, label × study strata) fitted on the 165 subjects only |
| External cohort | 113 WearGait-PD SelfPace participants with `qc_pass` (B-D1: 10 without insoles not scored; group split reported after unblinding) |
| Confirmatory tests | `scripts/c_02_hypothesis_testing.py`: H1 one-sided DeLong; H2 speed-adjusted AAUC of the linear predictor, bootstrap B = 2,000, overlap precondition; H3 gap vs internal emulated AUC, bootstrap combination; Holm over H1–H3, α = 0.05 one-sided; H2 interpreted only if H1 rejected (D6, D13, D14, A1.1 R2.2/R5/R10) |
| H3 internal reference | M1-fb emulated AUC **0.6731** (train-full/test-K, 20 draws, 4 × [4, 5] strides); internal bootstrap-of-CV B = 2,000 |
| Recalibration | Reported separately; never replaces the primary result |

## Input fingerprints (SHA-256)

| File (git-ignored, Lead machine) | SHA-256 |
|---|---|
| `outputs/b02_20261122/weargait/subject_external_v2.csv` (external table, no labels) | `ac1fe306e2079c1531d80ae64d7784684506bf3b26f0cc002b52bf2ce4ea6a49` |
| `outputs/c_prefreeze/features/subject_v2.csv` (gaitpdb training table) | `f382c26c402ea4fed4f6cee0a087f3dd9e415cc6031813afd438dc705b0b05a6` |
| `outputs/c_prefreeze/emulation/summary.json` (D15 + H3 reference) | `7d21eb9e60786c8f90bf3102acf795e328f115477c345b3202488c489e3ff1d6` |
| `outputs/c_prefreeze/emulation/h3_internal_bootstrap.csv` (H3 internal draws) | `01159da48130b61c5be3cfe1c17c1fde1c296c4ba24be9970019574c6e0b79ba` |
| `experiments/external_transport/config/a2.yaml` (in git) | `cf188d3512a26f06911698990bd6c94701c8693ecf08f488033b5575001cc687` |

## Commands run after the tag (in this order)

```
python -m experiments.external_transport.scripts.c_01_frozen_scoring \
    --external-csv experiments/external_transport/outputs/b02_20261122/weargait/subject_external_v2.csv \
    --train-csv experiments/external_transport/outputs/c_prefreeze/features/subject_v2.csv \
    --expected-train-sha256 f382c26c402ea4fed4f6cee0a087f3dd9e415cc6031813afd438dc705b0b05a6 \
    --run-id c_frozen
# then unblinding (labels/ readable from here on):
python -m experiments.external_transport.scripts.c_02_hypothesis_testing \
    --scores-csv experiments/external_transport/outputs/c_frozen/frozen_scores/external_scores.csv \
    --key-csv labels/weargait_key.csv \
    --external-table experiments/external_transport/outputs/b02_20261122/weargait/subject_external_v2.csv \
    --emulation-summary experiments/external_transport/outputs/c_prefreeze/emulation/summary.json \
    --run-id c_frozen
```

Any change after the tag (code, inputs or tests) is a deviation. It must be logged as such and reported
separately; it never replaces the frozen result.
