"""
zorix/models/ensemble.py

Stacking ensemble (Section 7): replaces the old manual
"0.55*XGBoost + 0.45*LSTM" averaging with a trained meta-model over
out-of-fold base-model probabilities plus context features (regime,
sentiment, volatility).

Leakage control: base models are trained on TimeSeriesSplit folds and the
meta-model only ever sees each row's prediction from a model that was
NOT trained on that row (classic stacking OOF discipline), extended to
respect time order (no shuffling — Section 45).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class StackingResult:
    meta_model: object
    oof_proba: np.ndarray               # out-of-fold base predictions used to train meta-model
    base_model_names: List[str]
    context_feature_names: List[str]


def default_base_model_factories(random_state: int = 42) -> Dict[str, callable]:
    """Standard base-model set for OOF stacking: LogReg + RF always, XGBoost/
    LightGBM only if installed — mirrors ModelTrainer's availability checks
    so the ensemble never claims a model participated that didn't actually
    run (Section 32 / Section 50 #10).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from zorix.models.trainer import XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE

    factories: Dict[str, callable] = {
        "LogReg": lambda: LogisticRegression(max_iter=2000, C=0.5, random_state=random_state),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10, n_jobs=-1, random_state=random_state
        ),
    }
    if XGBOOST_AVAILABLE:
        import xgboost as xgb
        factories["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3, eval_metric="mlogloss",
            random_state=random_state, n_jobs=-1,
        )
    if LIGHTGBM_AVAILABLE:
        import lightgbm as lgb
        factories["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, objective="multiclass", num_class=3,
            random_state=random_state, n_jobs=-1, verbosity=-1,
        )
    return factories


def build_oof_base_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    base_model_factories: Dict[str, callable],
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    base_model_factories: dict of name -> zero-arg callable returning an
    UNFIT sklearn-compatible classifier (so a fresh one is made per fold).

    Returns a DataFrame with columns like "XGBoost_SELL", "XGBoost_HOLD",
    "XGBoost_BUY", ... — one probability column per (model, class) pair —
    aligned to X's index. Rows in the first fold's training-only region
    (which never appears in any test fold) will be NaN and must be dropped
    before meta-model training.
    """
    n = len(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof = {name: np.full((n, 3), np.nan) for name in base_model_factories}

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        for name, factory in base_model_factories.items():
            model = factory()
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)
            # Some folds may not see all 3 classes in y_tr; align columns defensively.
            full_proba = np.zeros((len(X_te), 3))
            for j, c in enumerate(getattr(model, "classes_", [0, 1, 2])):
                full_proba[:, int(c)] = proba[:, j]
            oof[name][test_idx] = full_proba

    frames = []
    for name, arr in oof.items():
        cols = [f"{name}_SELL", f"{name}_HOLD", f"{name}_BUY"]
        frames.append(pd.DataFrame(arr, columns=cols, index=X.index))
    return pd.concat(frames, axis=1)


def train_meta_model(
    oof_base_proba: pd.DataFrame,
    context_features: Optional[pd.DataFrame],
    y: pd.Series,
    random_state: int = 42,
) -> StackingResult:
    """Trains a Logistic Regression meta-model (interpretable, per Section 7's
    'avoid overengineering') on OOF base probabilities + optional context
    features (regime one-hots, sentiment, volatility, ...).
    """
    meta_X = oof_base_proba.copy()
    if context_features is not None:
        meta_X = meta_X.join(context_features, how="left")

    valid = meta_X.dropna().index
    meta_X_valid = meta_X.loc[valid]
    y_valid = y.loc[valid]

    # Note: sklearn >=1.5 removed the multi_class kwarg (multinomial is automatic).
    meta = LogisticRegression(max_iter=2000, random_state=random_state)
    meta.fit(meta_X_valid, y_valid)

    return StackingResult(
        meta_model=meta,
        oof_proba=meta_X_valid.values,
        base_model_names=list({c.rsplit("_", 1)[0] for c in oof_base_proba.columns}),
        context_feature_names=list(context_features.columns) if context_features is not None else [],
    )


def predict_stacked(stacking_result: StackingResult, base_proba_row: pd.DataFrame,
                     context_row: Optional[pd.DataFrame] = None) -> np.ndarray:
    """base_proba_row / context_row: single-row DataFrames with matching
    column names to what train_meta_model saw."""
    X = base_proba_row.copy()
    if context_row is not None:
        X = X.join(context_row, how="left")
    return stacking_result.meta_model.predict_proba(X)
