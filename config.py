"""
config.py — All project parameters in one place.
Change anything here and the whole pipeline updates.
"""

# ─── Asset Universe ───────────────────────────────────────────────
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EFA", "VNQ"]

# ─── Sample Period ────────────────────────────────────────────────
TRAIN_START = "2010-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-03-31"

# ─── Kalman Filter Parameters ─────────────────────────────────────
# F = Identity (random walk assumption for latent expected returns)
# H = Identity (we directly observe returns)
# Q = process noise — how fast we believe true returns drift
# R = observation noise — estimated from training data
Q_SCALE = 1e-5  # multiplied by identity matrix
# R is computed from data in kalman_filter.py

# ─── Portfolio Optimization ───────────────────────────────────────
RISK_AVERSION = 2.0       # lambda in mean-variance objective
LONG_ONLY = True          # w_i >= 0
FULLY_INVESTED = True     # sum(w_i) = 1

# ─── Backtest Settings ────────────────────────────────────────────
REBALANCE_FREQ = "ME"      # M = monthly, W = weekly
COV_WINDOW = 60           # rolling window for covariance estimation (trading days)
RETURN_WINDOW = 60        # rolling window for benchmark return estimation

# ─── Regime Periods (for analysis) ────────────────────────────────
REGIMES = {
    "Pre-COVID":   ("2020-01-01", "2020-02-19"),
    "COVID Crash":  ("2020-02-20", "2020-06-30"),
    "Recovery":     ("2020-07-01", "2021-12-31"),
    "Rate Hikes":   ("2022-01-01", "2023-12-31"),
    "Recent":       ("2024-01-01", "2026-03-31"),
}

# ─── Output ───────────────────────────────────────────────────────
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
