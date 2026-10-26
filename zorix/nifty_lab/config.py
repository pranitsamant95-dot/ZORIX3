"""
zorix/nifty_lab/config.py

Config for Mode 1 — NIFTY Lab. Deliberately its own dataclass, not an
extension of ``zorix.config.settings.Config`` — Mode 1 has no shared
mutable state with Standard Zorix, by design (see mission decisions).
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class NiftyLabConfig:
    # ---- Data ----
    csv_path: str = "data/NIFTY_spot_1m_extended.csv"

    # Volume is excluded project-wide: 100% zero in 2023-2024, 83.5% zero
    # in 2025, only populated in 2026 — not a real feed, not a feature.
    use_volume: bool = False

    # The 9 "bad OHLC" rows are float-precision artifacts
    # (e.g. 22532.05078125 vs 22532.05), not corrupted data — kept, not
    # dropped, just rounded during cleaning.
    ohlc_round_dp: int = 2

    # ---- Intraday model ----
    intraday_horizons_min: List[int] = field(default_factory=lambda: [5, 15, 30])

    # Capped to ~3-4 weeks of 1-min bars. Training a RandomForest on the
    # full ~270K-row history was too slow for an interactive per-request
    # retrain in this environment; this cap keeps a live forecast call to
    # a few seconds. Callers that explicitly want a longer window can pass
    # a larger value to the training functions.
    intraday_train_window: int = 8_000

    intraday_n_estimators: int = 100
    intraday_max_depth: int = 5
    # -1 matches the rest of the codebase's convention and is correct for
    # normal multi-core deployment hardware; this sandbox's 2 cores are
    # just an unusually slow test environment, not a reason to pin to 1.
    intraday_n_jobs: int = -1

    # ---- Next-day model ----
    nextday_n_estimators: int = 200
    nextday_max_depth: int = 6
    nextday_n_jobs: int = -1

    # ---- Historical simulation / backtest ----
    walkforward_n_folds: int = 3
    walkforward_test_days: int = 20      # trading days per fold
    one_month_ahead_days: int = 21       # ≈ 1 trading month

    # ---- Misc ----
    random_state: int = 42


DEFAULT_NIFTYLAB_CONFIG = NiftyLabConfig()
