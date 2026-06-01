"""
data_loader.py — Download ETF prices and compute daily log returns.

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


def compute_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    """
    Compute daily returns from price data.

    Parameters
    ----------
    prices : DataFrame of adjusted close prices
    method : "log" for log returns, "simple" for arithmetic returns

    Returns
    -------
    DataFrame of daily returns (first row is NaN, dropped)
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
