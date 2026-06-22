"""
covariance.py — Rolling covariance estimation for portfolio optimization.

Simple rolling sample covariance used as the baseline estimator.
Ledoit-Wolf shrinkage is implemented separately in benchmarks.py
(make_ledoit_wolf_strategy). Filtered covariance is implemented via
the Kalman-Sigma MV and Kalman-Full MV variants, also in benchmarks.py.

Usage:
    from covariance import rolling_covariance
"""

import numpy as np
import pandas as pd
from config import COV_WINDOW


def rolling_covariance(
    returns: pd.DataFrame,
    window: int = COV_WINDOW,
) -> dict[pd.Timestamp, np.ndarray]:
    """
    Compute rolling sample covariance matrices.

    Parameters
    ----------
    returns : DataFrame of daily returns (T x n_assets)
    window  : lookback window in trading days

    Returns
    -------
    Dictionary mapping each date to its covariance matrix (n x n).
    Only dates where a full window is available are included.
    """
    cov_dict = {}

    for i in range(window, len(returns)):
        date = returns.index[i]
        window_returns = returns.iloc[i - window:i]
        cov_dict[date] = window_returns.cov().values

    return cov_dict


# Quick test
if __name__ == "__main__":
    from data_loader import load_prices, compute_returns

    prices = load_prices()
    returns = compute_returns(prices, method="simple")

    cov_dict = rolling_covariance(returns)
    last_date = list(cov_dict.keys())[-1]
    print(f"\nCovariance matrix on {last_date.date()}:")
    print(pd.DataFrame(
        cov_dict[last_date] * 252,  # annualized
        index=returns.columns,
        columns=returns.columns,
    ).round(4))