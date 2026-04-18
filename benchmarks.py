"""
benchmarks.py — Strategy definitions for backtesting.

Each strategy is a function with signature:
    strategy(date, returns_so_far) -> np.ndarray of weights

Usage:
    from benchmarks import (
        equal_weight_strategy,
        rolling_mv_strategy,
        static_mv_strategy,
        kalman_strategy,
    )
"""

import numpy as np
import pandas as pd
from optimizer import optimize_portfolio
from kalman_filter import KalmanReturnFilter
from config import COV_WINDOW, RETURN_WINDOW, Q_SCALE


# 1. Equal Weight

def equal_weight_strategy(date, returns_so_far):
    """1/N equal weight across all assets."""
    n = returns_so_far.shape[1]
    return np.ones(n) / n


# 2. Rolling Mean-Variance 

def make_rolling_mv_strategy(cov_window=COV_WINDOW, ret_window=RETURN_WINDOW, turnover_gamma=0.0):
    n_assets = [None]
    w_prev = [None]

    def strategy(date, returns_so_far):
        if len(returns_so_far) < cov_window:
            n = returns_so_far.shape[1]
            w_prev[0] = np.ones(n) / n
            return w_prev[0]

        recent = returns_so_far.iloc[-cov_window:]
        mu = recent.mean().values
        sigma = recent.cov().values

        w = optimize_portfolio(mu, sigma, w_prev=w_prev[0], turnover_gamma=turnover_gamma)
        w_prev[0] = w.copy()
        return w

    return strategy


# 3. Static Mean-Variance 

def make_static_mv_strategy(train_returns: pd.DataFrame):
    """
    Factory: creates a static strategy using estimates computed once
    from the training period. Never updates.
    """
    mu_static = train_returns.mean().values
    sigma_static = train_returns.cov().values
    weights_static = optimize_portfolio(mu_static, sigma_static)

    def strategy(date, returns_so_far):
        return weights_static

    return strategy


# 4. Kalman Filter Strategy 

def make_kalman_strategy(
    train_returns: pd.DataFrame,
    q_scale: float = Q_SCALE,
    cov_window: int = COV_WINDOW,
    regime_labels: pd.Series = None,
    regime_alphas: dict = None,
    turnover_gamma: float = 0.0,       # add this
):
    n_assets = train_returns.shape[1]
    R = train_returns.cov().values
    kf = KalmanReturnFilter(n_assets=n_assets, q_scale=q_scale, R=R)
    mu_0 = train_returns.mean().values
    kf.initialize(mu_0=mu_0)
    last_processed_idx = 0
    w_prev = [None]                    # add this

    def strategy(date, returns_so_far):
        nonlocal last_processed_idx

        current_len = len(returns_so_far)
        for i in range(last_processed_idx, current_len):
            r_t = returns_so_far.iloc[i].values
            date_i = returns_so_far.index[i]

            q_override = None
            if (regime_labels is not None
                    and regime_alphas is not None
                    and date_i in regime_labels.index):
                alpha = regime_alphas.get(regime_labels.loc[date_i], 1.0)
                if alpha != 1.0:
                    q_override = alpha * kf.Q

            kf.predict(q_override=q_override)
            kf.update(r_t)

        last_processed_idx = current_len
        mu_kf = kf.mu_hat.copy()

        if current_len < cov_window:
            n = returns_so_far.shape[1]
            w_prev[0] = np.ones(n) / n
            return w_prev[0]

        recent = returns_so_far.iloc[-cov_window:]
        sigma = recent.cov().values

        w = optimize_portfolio(mu_kf, sigma, w_prev=w_prev[0], turnover_gamma=turnover_gamma)
        w_prev[0] = w.copy()           # add this
        return w

    return strategy