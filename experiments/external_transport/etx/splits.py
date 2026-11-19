"""Repeated stratified subject-level outer/inner folds (strata label x study), deterministic seeds.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def outer_folds(subjects_df, repeats, k, base_seed):
    """Yield (repeat, fold, train_ids, test_ids)."""
    raise NotImplementedError

def inner_folds(train_df, k, seed):
    """Yield (inner_train_ids, inner_val_ids)."""
    raise NotImplementedError
