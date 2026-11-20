"""Walking-condition rules (version c1). The ONLY path to include_primary.

Owner: Lead (data/*). Spec §4; decision D4 (approved 2026-11-19).
  _01 (all studies)      -> usual_primary        (included)
  Ga _02                 -> ga02_unverified      (excluded: meaning undocumented)
  Ga _10                 -> dual_task            (excluded)
  Ju _02.._07            -> ju_ras_or_carryover  (excluded: cued walking / carryover)
Anything else raises, so an unexpected file can never slip into the primary cohort.
"""
VERSION = "c1"


def classify(study: str, walk: str) -> tuple:
    """Return (condition, include_primary)."""
    if walk == "01":
        return "usual_primary", True
    if study == "Ga" and walk == "02":
        return "ga02_unverified", False
    if study == "Ga" and walk == "10":
        return "dual_task", False
    if study == "Ju" and walk in {"02", "03", "04", "05", "06", "07"}:
        return "ju_ras_or_carryover", False
    raise ValueError(f"Unknown walking condition: study={study!r} walk={walk!r} (rules {VERSION})")
