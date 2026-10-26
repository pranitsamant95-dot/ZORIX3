"""
zorix/features/regime.py

Market regime detection (Section 18): BULL/BEAR x LOW/HIGH VOL.

Preference order, all graceful (Section 6 / Section 43):
    1. hmmlearn.GaussianHMM   (preferred — proper Hidden Markov Model)
    2. sklearn GaussianMixture (unsupervised fallback if hmmlearn missing)
    3. Quantile-based regime rule (always available, no extra dependency)

Whichever engine actually ran is recorded on the result so the UI never
claims "HMM regime detection" when it silently fell back to something else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    labels: pd.Series          # per-row regime label, e.g. "BULL / HIGH VOL"
    confidence: pd.Series      # per-row confidence in [0, 1]
    engine: str                # "hmm" | "gmm" | "quantile"
    current_regime: Optional[str]
    current_confidence: Optional[float]


_REGIME_NAMES = ["BEAR / LOW VOL", "BEAR / HIGH VOL", "BULL / LOW VOL", "BULL / HIGH VOL"]


def _build_regime_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    ret = df["Close"].pct_change()
    trend = df["Close"].pct_change(lookback)
    vol = ret.rolling(lookback).std() * np.sqrt(252)
    feats = pd.DataFrame({"trend": trend, "vol": vol}, index=df.index)
    return feats.dropna()


def _label_from_trend_vol(trend: float, vol: float, vol_median: float) -> str:
    bull = trend >= 0
    high_vol = vol >= vol_median
    if bull and not high_vol:
        return "BULL / LOW VOL"
    if bull and high_vol:
        return "BULL / HIGH VOL"
    if not bull and not high_vol:
        return "BEAR / LOW VOL"
    return "BEAR / HIGH VOL"


def detect_regime(df: pd.DataFrame, lookback: int = 20, n_regimes: int = 4) -> RegimeResult:
    feats = _build_regime_features(df, lookback)
    if feats.empty:
        empty = pd.Series(dtype=object)
        return RegimeResult(empty, empty, "unavailable", None, None)

    vol_median = feats["vol"].median()

    # ---- 1. Try hmmlearn ----
    try:
        from hmmlearn.hmm import GaussianHMM
        X = feats.values
        model = GaussianHMM(n_components=n_regimes, covariance_type="diag",
                             n_iter=200, random_state=42)
        model.fit(X)
        state_seq = model.predict(X)
        post = model.predict_proba(X)
        confidence = post.max(axis=1)

        # Map each HMM state to a human label using that state's mean trend/vol
        state_means = pd.DataFrame(X, columns=["trend", "vol"])
        state_means["state"] = state_seq
        means = state_means.groupby("state")[["trend", "vol"]].mean()
        label_map = {
            s: _label_from_trend_vol(row["trend"], row["vol"], vol_median)
            for s, row in means.iterrows()
        }
        labels = pd.Series([label_map[s] for s in state_seq], index=feats.index)
        conf_series = pd.Series(confidence, index=feats.index)
        engine = "hmm"
    except ImportError:
        # ---- 2. Try sklearn GaussianMixture ----
        try:
            from sklearn.mixture import GaussianMixture
            X = feats.values
            model = GaussianMixture(n_components=n_regimes, random_state=42, n_init=3)
            model.fit(X)
            state_seq = model.predict(X)
            post = model.predict_proba(X)
            confidence = post.max(axis=1)

            state_means = pd.DataFrame(X, columns=["trend", "vol"])
            state_means["state"] = state_seq
            means = state_means.groupby("state")[["trend", "vol"]].mean()
            label_map = {
                s: _label_from_trend_vol(row["trend"], row["vol"], vol_median)
                for s, row in means.iterrows()
            }
            labels = pd.Series([label_map[s] for s in state_seq], index=feats.index)
            conf_series = pd.Series(confidence, index=feats.index)
            engine = "gmm"
        except Exception:
            # ---- 3. Quantile fallback (always works) ----
            labels = feats.apply(
                lambda r: _label_from_trend_vol(r["trend"], r["vol"], vol_median), axis=1
            )
            conf_series = pd.Series(0.5, index=feats.index)  # no probabilistic confidence available
            engine = "quantile"

    current_regime = labels.iloc[-1] if len(labels) else None
    current_confidence = float(conf_series.iloc[-1]) if len(conf_series) else None

    return RegimeResult(labels, conf_series, engine, current_regime, current_confidence)


def attach_regime_features(df: pd.DataFrame, regime: RegimeResult) -> pd.DataFrame:
    """Adds regime as one-hot columns + confidence, aligned to df's index.
    Rows before the regime model had enough lookback data get NaN (dropped
    later during training, same as every other rolling feature)."""
    out = df.copy()
    if regime.engine == "unavailable":
        return out

    aligned_labels = regime.labels.reindex(out.index)
    aligned_conf = regime.confidence.reindex(out.index)

    for name in _REGIME_NAMES:
        out[f"Regime_{name.replace(' / ', '_').replace(' ', '')}"] = (aligned_labels == name).astype(float)
    out["Regime_Confidence"] = aligned_conf
    out["Regime_Engine"] = regime.engine
    return out
