"""
evaluation.py — Performance metrics for portfolio backtests.

Usage:
    from evaluation import compute_metrics, compare_strategies, print_metrics
"""

import numpy as np
import pandas as pd


def compute_metrics(result: dict, trading_days: int = 252) -> dict:
    """
    Compute standard portfolio performance metrics.

    Parameters
    ----------
    result : dict from run_backtest()
    trading_days : annualization factor

    Returns
    -------
    Dictionary of metrics.
    """
    daily_ret = result["daily_returns"]
    cumulative = result["cumulative"]

    # Total growth multiplier — cumulative.iloc[-1] already equals prod(1+r_t).
    # Do NOT divide by cumulative.iloc[0]: that's (1+r_0), not 1, so dividing
    # by it silently drops the first day's return from every metric below.
    total_growth = cumulative.iloc[-1]
    n_years = len(daily_ret) / trading_days
    ann_return = total_growth ** (1 / n_years) - 1

    # Annualized volatility
    ann_vol = daily_ret.std() * np.sqrt(trading_days)

    # Sharpe ratio (assumes risk-free rate ~ 0 for simplicity)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Maximum drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Turnover — use backtest turnover_series for consistency with cost_analysis.py
    turnover_s = result.get("turnover_series", pd.Series(dtype=float))
    avg_turnover = turnover_s.mean() if len(turnover_s) > 0 else 0.0

    return {
        "Strategy": result["name"],
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.3f}",
        "Max Drawdown": f"{max_drawdown:.2%}",
        "Avg Turnover": f"{avg_turnover:.4f}",
        "Total Return": f"{(total_growth - 1):.2%}",
    }


def compare_strategies(results: list[dict], trading_days: int = 252) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.

    Parameters
    ----------
    results : list of dicts from run_backtest()

    Returns
    -------
    DataFrame with one row per strategy and columns for each metric.
    """
    metrics = [compute_metrics(r, trading_days) for r in results]
    return pd.DataFrame(metrics).set_index("Strategy")


def regime_analysis(
    results: list[dict],
    regimes: dict[str, tuple[str, str]],
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute Sharpe ratio for each strategy within each regime period.

    Parameters
    ----------
    results : list of dicts from run_backtest()
    regimes : dict mapping regime name -> (start_date, end_date)

    Returns
    -------
    DataFrame with strategies as columns and regimes as rows.
    """
    rows = []
    for regime_name, (start, end) in regimes.items():
        row = {"Regime": regime_name}
        for result in results:
            daily_ret = result["daily_returns"]
            regime_ret = daily_ret.loc[start:end]

            if len(regime_ret) > 5:
                ann_ret = regime_ret.mean() * trading_days
                ann_vol = regime_ret.std() * np.sqrt(trading_days)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                row[result["name"]] = round(sharpe, 3)
            else:
                row[result["name"]] = None

        rows.append(row)

    return pd.DataFrame(rows).set_index("Regime")


def print_metrics(comparison_df: pd.DataFrame):
    """Pretty-print the comparison table."""
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string())
    print("=" * 70)