# Asset Universe
TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EFA", "VNQ"]

# Sample Period
TRAIN_START = "2010-01-01"
TRAIN_END   = "2024-12-31"
TEST_START  = "2025-01-01"
TEST_END    = "2026-03-31"

# Kalman Filter Parameters
# Q_SCALE is a fallback only — at runtime, main.py estimates Q_SCALE
# via EM (Shumway & Stoffer, 2000) on training data and overrides this value.
# All strategy functions accept q_scale as an argument so EM value propagates.
Q_SCALE = 1e-5   # fallback default, overridden by EM in main.py
Q_REGIME_ALPHAS = {}  # populated at runtime by calibrate_regime_alphas
TURNOVER_GAMMA = 0.0  # overridden at runtime via CV

# Portfolio Optimization
RISK_AVERSION   = 2.0    # lambda in mean-variance objective
LONG_ONLY       = True   # w_i >= 0
FULLY_INVESTED  = True   # sum(w_i) = 1
MAX_WEIGHT      = 0.4   # maximum weight per asset

# Backtest Settings
REBALANCE_FREQ  = "ME"   # ME = month-end, W = weekly
COV_WINDOW      = 60     # rolling window for covariance estimation (trading days)
RETURN_WINDOW   = 60     # rolling window for benchmark return estimation

# Regime Periods (for sub-period analysis)
REGIMES = {
    "Pre-COVID":   ("2020-01-01", "2020-02-19"),
    "COVID Crash": ("2020-02-20", "2020-06-30"),
    "Recovery":    ("2020-07-01", "2021-12-31"),
    "Rate Hikes":  ("2022-01-01", "2023-12-31"),
    "Recent":      ("2024-01-01", "2026-03-31"),
}

# Output
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
