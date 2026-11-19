# External transport pipeline (Phase A2+)

Can gaitpdb-trained foot-pressure models predict PD on WearGait-PD?
Design: `research/protocol/A2_IMPLEMENTATION_SPEC.md`. Rules: repo-root `CLAUDE.md`.

## Setup
```bash
pip install -r requirements.txt -r experiments/external_transport/requirements-a2.txt
export GAITPDB_ROOT=/path/to/gaitpdb/folder   # holds GaCo01_01.txt, ..., SHA256SUMS.txt
pytest experiments/external_transport/tests -q
```

## Ownership
| Lead (`data/*`) | Teammate (`model/*`) |
|---|---|
| paths, guards, manifest, conditions, signal, events, features, qc, cache, kcurve/emulation samplers | tests/fixtures/fake_features, demographics, splits, models, metrics, uncertainty, reporting, kcurve/emulation evaluation |

**Hand-off file:** `outputs/<run_id>/features/subject_v2.csv` (columns: spec §5). Until it exists,
the teammate develops on `tests/fixtures/fake_features.py` output with the same columns.

## Steps
`scripts/a2_00` ... `a2_10` = execution order E0–E10 (spec §10). Outputs go to `outputs/<run_id>/` (git-ignored).
