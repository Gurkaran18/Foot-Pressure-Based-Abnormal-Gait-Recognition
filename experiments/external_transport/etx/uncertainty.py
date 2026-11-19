"""Repeat summaries, bootstrap-of-CV (grouped duplicates), LeDell CI, external bootstrap, coef stability.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def summarize_repeats(values): raise NotImplementedError
def bootstrap_cv(run_cv, subjects_df, B, seed): raise NotImplementedError
def ledell_ci(y, p, folds): raise NotImplementedError
def external_bootstrap(y, p, B, seed): raise NotImplementedError
