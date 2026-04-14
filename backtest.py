"""
backtest.py — Backtest engine for portfolio strategies.

Runs a monthly-rebalanced portfolio backtest given a return series
and a strategy function that produces weights at each rebalance date.

Usage:
    from backtest import run_backtest
"""

import numpy as np
import pandas as pd
from config import REBALANCE_FREQ


def get_rebalance_dates(returns: pd.DataFrame, freq: str = REBALANCE_FREQ) -> list[pd.Timestamp]:
    """
    Get the last trading day of each period (month/week) within the return series.
    These are the dates on which the portfolio is rebalanced.
    """
    return list(returns.resample(freq).last().index)


def run_backtest(
    returns: pd.DataFrame,
    weight_func,
    rebalance_freq: str = REBALANCE_FREQ,
    name: str = "Strategy",
) -> dict:
    """
    Run a backtest for a portfolio strategy.

    Parameters
    ----------
    returns : DataFrame of daily returns (T x n_assets)
    weight_func : callable(date, returns_up_to_date) -> np.ndarray
        Given a rebalance date and all returns up to that date,
        returns a weight vector.
    rebalance_freq : "M" for monthly, "W" for weekly
    name : label for the strategy

    Returns
    -------
    Dictionary with:
        - "daily_returns": Series of daily portfolio returns
        - "cumulative": Series of cumulative wealth (starts at 1.0)
        - "weights_history": DataFrame of weights at each rebalance
        - "name": strategy name
    """
    rebalance_dates = get_rebalance_dates(returns, rebalance_freq)
    n_assets = returns.shape[1]

    daily_port_returns = []
    weights_history = []
    current_weights = np.ones(n_assets) / n_assets  # start equal weight

    for i, date in enumerate(returns.index):
        # Check if we should rebalance
        if date in rebalance_dates:
            returns_so_far = returns.loc[:date]
            try:
                new_weights = weight_func(date, returns_so_far)
                if new_weights is not None and not np.any(np.isnan(new_weights)):
                    current_weights = new_weights
            except Exception:
                pass  # keep current weights if strategy fails

            weights_history.append({
                "date": date,
                **{col: current_weights[j] for j, col in enumerate(returns.columns)},
            })

        # Daily portfolio return = weighted sum of asset returns
        r_t = returns.iloc[i].values
        port_return = current_weights @ r_t
        daily_port_returns.append({"date": date, "return": port_return})

    # Build output
    daily_df = pd.DataFrame(daily_port_returns).set_index("date")["return"]
    cumulative = (1 + daily_df).cumprod()
    weights_df = pd.DataFrame(weights_history).set_index("date")

    print(f"  [{name}] Backtest complete: {len(daily_df)} days, "
          f"{len(weights_df)} rebalances")

    return {
        "daily_returns": daily_df,
        "cumulative": cumulative,
        "weights_history": weights_df,
        "name": name,
    }
