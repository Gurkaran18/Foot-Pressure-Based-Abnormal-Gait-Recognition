"""AUC, fast DeLong, Brier, calibration intercept/slope, sens/spec/BAcc, AAUC.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md
"""
from pathlib import Path

def auc(y, p): raise NotImplementedError
def delong(y, p1, p2=None): raise NotImplementedError
def brier(y, p): raise NotImplementedError
def calibration(y, p): raise NotImplementedError
def threshold_metrics(y, p, thr): raise NotImplementedError
def aauc(y, score, covariate, strata=None): raise NotImplementedError
