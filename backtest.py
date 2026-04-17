"""
backtest.py — Backtest engine for portfolio strategies.
"""

import numpy as np
import pandas as pd
from config import REBALANCE_FREQ


def get_rebalance_dates(returns: pd.DataFrame, freq: str = REBALANCE_FREQ) -> list:
    return list(returns.resample(freq).last().index)


def run_backtest(
    returns: pd.DataFrame,
    weight_func,
    rebalance_freq: str = REBALANCE_FREQ,
    name: str = "Strategy",
    cost_bps: float = 0.0,
) -> dict:
    """
    Run a backtest with optional transaction costs.

    Cost model: on each rebalance date, deduct
        (cost_bps / 10_000) * sum(|w_new - w_old|)
    from that day's return. This is one-way turnover cost —
    sum(|Δw|) ranges from 0 (no change) to 2.0 (full portfolio flip).
    """
    rebalance_dates = set(get_rebalance_dates(returns, rebalance_freq))
    n_assets = returns.shape[1]
    cost_rate = cost_bps / 10_000.0

    daily_port_returns = []
    weights_history = []
    turnover_records = []
    current_weights = np.ones(n_assets) / n_assets

    for i, date in enumerate(returns.index):
        cost_drag = 0.0

        if date in rebalance_dates:
            returns_so_far = returns.loc[:date]
            try:
                new_weights = weight_func(date, returns_so_far)
                if new_weights is not None and not np.any(np.isnan(new_weights)):
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    cost_drag = cost_rate * turnover
                    turnover_records.append({"date": date, "turnover": turnover})
                    current_weights = new_weights
            except Exception:
                pass

            weights_history.append({
                "date": date,
                **{col: current_weights[j] for j, col in enumerate(returns.columns)},
            })

        r_t = returns.iloc[i].values
        port_return = (current_weights @ r_t) - cost_drag
        daily_port_returns.append({"date": date, "return": port_return})

    daily_df = pd.DataFrame(daily_port_returns).set_index("date")["return"]
    cumulative = (1 + daily_df).cumprod()
    weights_df = pd.DataFrame(weights_history).set_index("date")
    turnover_s = pd.DataFrame(turnover_records).set_index("date")["turnover"] if turnover_records else pd.Series(dtype=float)

    return {
        "daily_returns": daily_df,
        "cumulative": cumulative,
        "weights_history": weights_df,
        "turnover_series": turnover_s,
        "name": name,
        "cost_bps": cost_bps,
    }