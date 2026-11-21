"""Repeated stratified subject-level outer/inner folds (strata label x study), deterministic seeds.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §9
"""
import hashlib
from pathlib import Path
from typing import Generator, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _stratum_col(df: pd.DataFrame, strata: list) -> pd.Series:
    """Combine strata columns into a single string column for stratification."""
    return df[strata[0]].astype(str).str.cat(
        [df[s].astype(str) for s in strata[1:]], sep="_"
    )


def _seed_for_inner(base_seed: int, repeat: int, outer_fold: int) -> int:
    """Deterministic inner-fold seed derived from base_seed, repeat and outer_fold."""
    raw = f"{base_seed}_{repeat}_{outer_fold}".encode()
    return int(hashlib.sha256(raw).hexdigest(), 16) % (2 ** 31)


def outer_folds(
    subjects_df: pd.DataFrame,
    repeats: int,
    k: int,
    base_seed: int,
    strata: list = None,
) -> Generator[Tuple[int, int, List[str], List[str]], None, None]:
    """Yield (repeat, fold_idx, train_ids, test_ids) for every outer fold.

    subjects_df must have columns: subject_id, label, study.
    strata defaults to ['label', 'study']; if any stratum cell has fewer than k subjects
    it falls back to ['label'] only.

    Each subject_id appears in exactly one test_ids per repeat.
    Assertions:
      - no overlap between train_ids and test_ids
      - train_ids ∪ test_ids == all subject_ids
    """
    if strata is None:
        strata = ["label", "study"]

    ids = subjects_df["subject_id"].values
    n = len(ids)

    for r in range(repeats):
        seed = base_seed + r
        # Try strata = [label, study]; fall back to [label] if any cell < k subjects
        try:
            stratum = _stratum_col(subjects_df, strata)
            min_cell = stratum.value_counts().min()
            if min_cell < k:
                stratum = subjects_df["label"].astype(str)
        except KeyError:
            stratum = subjects_df["label"].astype(str)

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(ids, stratum)):
            train_ids = list(ids[train_idx])
            test_ids = list(ids[test_idx])

            # Integrity assertions
            assert set(train_ids).isdisjoint(set(test_ids)), \
                f"r={r} fold={fold_idx}: overlap between train and test"
            assert set(train_ids) | set(test_ids) == set(ids), \
                f"r={r} fold={fold_idx}: train ∪ test ≠ all subjects"

            yield (r, fold_idx, train_ids, test_ids)


def inner_folds(
    train_df: pd.DataFrame,
    k: int,
    seed: int,
    strata: list = None,
) -> Generator[Tuple[List[str], List[str]], None, None]:
    """Yield (inner_train_ids, inner_val_ids) for each inner fold.

    train_df must have columns: subject_id, label, study.
    Falls back to label-only strata if any stratum cell has < k subjects.
    Assertions: no overlap and union equals all train subjects.
    """
    if strata is None:
        strata = ["label", "study"]

    ids = train_df["subject_id"].values

    try:
        stratum = _stratum_col(train_df, strata)
        min_cell = stratum.value_counts().min()
        if min_cell < k:
            stratum = train_df["label"].astype(str)
    except KeyError:
        stratum = train_df["label"].astype(str)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for inner_train_idx, inner_val_idx in skf.split(ids, stratum):
        inner_train_ids = list(ids[inner_train_idx])
        inner_val_ids = list(ids[inner_val_idx])

        assert set(inner_train_ids).isdisjoint(set(inner_val_ids)), \
            "Inner fold: overlap between inner_train and inner_val"
        assert set(inner_train_ids) | set(inner_val_ids) == set(ids), \
            "Inner fold: inner_train ∪ inner_val ≠ all training subjects"

        yield (inner_train_ids, inner_val_ids)


def make_all_folds(
    subjects_df: pd.DataFrame,
    cfg: dict,
) -> list:
    """Return a flat list of all outer fold dicts (for scripts that want them all upfront).

    Each dict: {repeat, fold, train_ids, test_ids, inner: list of {inner_train, inner_val}}.
    """
    cv_cfg = cfg["cv"]
    result = []
    for repeat, fold, train_ids, test_ids in outer_folds(
        subjects_df,
        repeats=cv_cfg["repeats"],
        k=cv_cfg["outer_folds"],
        base_seed=cv_cfg["base_seed"],
        strata=cv_cfg.get("strata"),
    ):
        inner_seed = _seed_for_inner(cv_cfg["base_seed"], repeat, fold)
        train_df = subjects_df[subjects_df["subject_id"].isin(train_ids)].reset_index(drop=True)
        inner = [
            {"inner_train": it, "inner_val": iv}
            for it, iv in inner_folds(
                train_df,
                k=cv_cfg["inner_folds"],
                seed=inner_seed,
                strata=cv_cfg.get("strata"),
            )
        ]
        result.append({
            "repeat": repeat,
            "fold": fold,
            "train_ids": train_ids,
            "test_ids": test_ids,
            "inner": inner,
        })
    return result
