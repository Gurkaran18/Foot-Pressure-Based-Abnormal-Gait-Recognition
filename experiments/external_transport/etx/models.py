"""Pipeline factories M1, M1-fb, M0, M2, AS, X1, X13-ref (spec §8). All learned transforms inside the pipeline.

Owner: Teammate (model/*). Spec: research/protocol/A2_IMPLEMENTATION_SPEC.md §8
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

T1_CORE = ["stride_time_mean", "stride_time_cv", "swing_time_cv", "stance_pct", "asym_swing"]
T1_FALLBACK = ["stride_time_mean", "stride_time_cv"]
T1_LEGACY_13 = [
    "stride_time_mean", "stride_time_cv", "swing_time_cv", "stance_pct", "swing_pct",
    "double_support_pct", "cadence", "asym_stride", "asym_swing", "asym_peak",
    "peak_force_L", "peak_force_R", "peak_force_norm",
]

FEATURE_SETS = {
    "M1": T1_CORE,
    "M1-fb": T1_FALLBACK,
    "M0": ["speed_01_mps"],
    "M2": ["speed_01_mps"] + T1_CORE,
    "AS": ["age_y", "sex"],       # sex should be pre-encoded (0/1)
    "X1": T1_CORE,
    "X13-ref": T1_LEGACY_13,
}


# ---------------------------------------------------------------------------
# LR C grid
# ---------------------------------------------------------------------------

def _lr_C_grid(cfg: dict) -> np.ndarray:
    """Parse models.lr_C_grid from config (supports {logspace: [start, stop, num]})."""
    grid = cfg["models"]["lr_C_grid"]
    if isinstance(grid, dict) and "logspace" in grid:
        start, stop, num = grid["logspace"]
        return np.logspace(start, stop, int(num))
    return np.asarray(grid, float)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def make_pipeline(model_id: str, cfg: dict, seed: int = 42) -> Pipeline:
    """Return an unfitted sklearn Pipeline for the given model ID.

    For LR models (M1, M1-fb, M0, M2, AS): the pipeline is
        SimpleImputer(median) → StandardScaler → LogisticRegression(l2, lbfgs)
    C is selected via inner CV in the scripts; here we default to C=1.0.
    class_weight is controlled by cfg['models']['lr_weighting'].

    For XGB models (X1, X13-ref): SimpleImputer(median) → XGBClassifier.
    """
    if model_id not in FEATURE_SETS:
        raise ValueError(f"Unknown model_id {model_id!r}. Known: {list(FEATURE_SETS)}")

    xgb_ids = {"X1", "X13-ref"}

    if model_id in xgb_ids:
        if not _HAS_XGB:
            raise ImportError("xgboost is required for X1 / X13-ref models. pip install xgboost")
        xgb_params = dict(cfg["models"].get("xgb_t1", {}))
        xgb_params["random_state"] = seed
        xgb_params["use_label_encoder"] = False
        xgb_params["eval_metric"] = "logloss"
        clf = XGBClassifier(**xgb_params)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])
    else:
        # LR family
        weighting = cfg["models"].get("lr_weighting", "none")
        class_weight = "balanced" if weighting == "balanced" else None
        clf = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            class_weight=class_weight,
            random_state=seed,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])


def set_C(pipeline: Pipeline, C: float) -> Pipeline:
    """Set the C hyperparameter on the LogisticRegression step (mutates in place)."""
    pipeline.named_steps["clf"].C = C
    return pipeline


def get_feature_names(model_id: str) -> list:
    """Return the ordered list of feature names for a model."""
    return list(FEATURE_SETS[model_id])
