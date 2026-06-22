"""
data_loader.py — Download ETF prices and compute daily returns.

Usage:
    from data_loader import load_prices, compute_returns, split_data
"""

import pandas as pd
import yfinance as yf
from config import TICKERS, TRAIN_START, TRAIN_END, TEST_START, TEST_END


def load_prices(
    tickers: list[str] = TICKERS,
    start: str = TRAIN_START,
    end: str = TEST_END,
) -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    # yfinance treats end as exclusive; add one day so TEST_END is included.
    yf_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading {len(tickers)} ETFs from {start} to {end}...")
    data = yf.download(tickers, start=start, end=yf_end, auto_adjust=True)

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers

    prices = prices.dropna()
    print(f"  Loaded {len(prices)} trading days, {prices.shape[1]} assets.")
    return prices


def compute_returns(prices: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
    """
    Compute daily returns from price data.

    Parameters
    ----------
    prices : DataFrame of adjusted close prices
    method : "simple" (default) for arithmetic returns, "log" for log returns.

    Returns
    -------
    DataFrame of daily returns (first row is NaN, dropped)

    Notes
    -----
    Default is "simple", not "log", and this isn't interchangeable with the
    rest of the pipeline. backtest.py aggregates a portfolio's daily return
    as a weighted sum across assets (`weights @ r_t`) -- that's only exact
    for simple returns (dollar values combine linearly, so simple returns
    do too). For log returns, the true portfolio log return is
    ln(1 + weights @ r_simple), which by Jensen's inequality (log is
    concave) is always >= weights @ r_log -- a weighted sum of log returns
    silently understates the true portfolio return, with the gap growing
    with cross-sectional dispersion across assets (i.e. worst exactly when
    assets are moving most differently from each other, such as during
    CRISIS/HIGH_VOL regimes). Separately, every cumulative-growth
    calculation downstream (`(1 + daily_returns).cumprod()` in
    backtest.py/evaluation.py/cost_analysis.py) is also only the correct
    compounding formula for simple returns. Both issues are resolved at
    once by using simple returns as the base series feeding the pipeline --
    nothing downstream needs to change to accommodate it.
    """
    if method == "log":
        import numpy as np
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()

    return returns.dropna()


def split_data(
    returns: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split returns into training and testing periods.

    Returns
    -------
    (train_returns, test_returns) tuple of DataFrames
    """
    train = returns.loc[:train_end]
    test = returns.loc[test_start:]

    print(f"  Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} days)")
    print(f"  Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} days)")

    return train, test


# Quick test
if __name__ == "__main__":
    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    print("\nTraining set summary:")
    print(train.describe().round(4))

    print("\nTest set summary:")
    print(test.describe().round(4))