"""Per-walk QC flags, K-eligibility, subject aggregation -> features/subject_v2.csv (the hand-off file).

Owner: Lead (data/*). Spec §5, §6.

subject_v2.csv = one row per subject from the single primary (_01) walk:
  subject_id, label, study, <T1-core>, double_support_v2_pct, n_valid_L, n_valid_R, qc_pass, qc_flags,
  source_recordings
Demographic columns (sex, age_y, speed_01_mps) are joined from the teammate's
demographics/clean.csv on subject_id when that file is supplied; this module never invents them.
"""
import pandas as pd

from .features import T1_CORE, WALK_V2_FEATURES


def qc_flags(row: dict, min_strides: int) -> list:
    flags = []
    if row["n_valid_L"] < min_strides:
        flags.append(f"few_strides_L<{min_strides}")
    if row["n_valid_R"] < min_strides:
        flags.append(f"few_strides_R<{min_strides}")
    if any(pd.isna(row.get(k)) for k in T1_CORE):
        flags.append("missing_feature")
    return flags


def subject_table(walk_df: pd.DataFrame, demographics_df: pd.DataFrame = None) -> pd.DataFrame:
    """One row per subject from primary walks. Raises if a subject has != 1 primary walk."""
    prim = walk_df[walk_df["include_primary"]].copy()
    dup = prim["subject_id"].value_counts()
    if (dup != 1).any():
        raise RuntimeError(f"subjects with !=1 primary walk: {dup[dup != 1].to_dict()}")
    cols = ["subject_id", "label", "study"] + WALK_V2_FEATURES + ["n_valid_L", "n_valid_R", "qc_pass",
                                                                  "qc_flags", "recording_id"]
    sub = prim[cols].rename(columns={"recording_id": "source_recordings"})
    if demographics_df is not None:
        demo = demographics_df[["subject_id"] + [c for c in ("sex", "age_y", "speed_01_mps")
                                                 if c in demographics_df.columns]]
        sub = sub.merge(demo, on="subject_id", how="left", validate="one_to_one")
    return sub.sort_values("subject_id").reset_index(drop=True)
