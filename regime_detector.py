"""
regime_detector.py — Data-driven market regime classification.

Replaces the manual date-range approach in config.py with a method
that classifies each day into a regime based on realized volatility.
This is methodologically stronger because:
  1. Regimes are defined by market conditions, not by hindsight date-picking
  2. Results are reproducible and not dependent on the researcher's
     choice of regime boundaries
  3. It directly tests the hypothesis: "Kalman MV outperforms during
     high-volatility / transitioning regimes"

Regime Classification:
    LOW_VOL    : realized vol in bottom tercile
    MED_VOL    : realized vol in middle tercile
    HIGH_VOL   : realized vol in top tercile
    CRISIS     : realized vol > 2 standard deviations above mean
                 (COVID crash, 2008 GFC, etc.)

All volatility is computed on a rolling 21-day window (1 trading month)
to avoid look-ahead bias.

Usage:
    from regime_detector import classify_regimes, regime_conditional_performance
"""

import numpy as np
import pandas as pd


# Regime classification

def compute_realized_volatility(
    returns: pd.DataFrame,
    window: int = 21,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute rolling realized volatility as the average across assets.
    """
    per_asset_vol = returns.rolling(window=window).std()
    market_vol = per_asset_vol.mean(axis=1)  # average across assets

    if annualize:
        market_vol = market_vol * np.sqrt(trading_days)

    return market_vol.dropna()


def classify_regimes(
    returns: pd.DataFrame,
    window: int = 21,
    crisis_threshold: float = 2.0,
    reference_returns: pd.DataFrame = None,
) -> pd.Series:
    """
    Classify each date into a volatility regime.
    """
    vol = compute_realized_volatility(returns, window=window)

    # Use reference period for thresholds if provided, otherwise use full series
    ref_vol = compute_realized_volatility(reference_returns, window=window) \
              if reference_returns is not None else vol

    # Thresholds computed on reference period only (training data)
    vol_mean = ref_vol.mean()
    vol_std  = ref_vol.std()
    q33      = ref_vol.quantile(0.33)
    q67      = ref_vol.quantile(0.67)

    # Z-score using reference statistics
    vol_zscore = (vol - vol_mean) / vol_std

    # Assign regimes to full series using reference thresholds
    regimes = pd.Series(index=vol.index, dtype=str)
    regimes[vol <= q33]                    = "LOW_VOL"
    regimes[(vol > q33) & (vol <= q67)]   = "MED_VOL"
    regimes[vol > q67]                     = "HIGH_VOL"
    regimes[vol_zscore > crisis_threshold] = "CRISIS"   # override tercile

    return regimes


def regime_summary(regimes: pd.Series) -> pd.DataFrame:
    """
    Print a summary of how many days fall into each regime.
    """
    counts = regimes.value_counts()
    pct    = (counts / len(regimes) * 100).round(1)
    df     = pd.DataFrame({"Days": counts, "Pct (%)": pct})
    df.index.name = "Regime"
    return df.sort_index()


# Regime-conditional performance

def regime_conditional_performance(
    results: list[dict],
    regimes: pd.Series,
    trading_days: int = 252,
    min_days: int = 20,
) -> pd.DataFrame:
    """
    Compute performance metrics for each strategy within each regime.
    """
    regime_labels = regimes.unique()
    order         = ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]
    regime_labels = [r for r in order if r in regime_labels]

    rows = []
    for regime in regime_labels:
        regime_dates = regimes[regimes == regime].index

        for result in results:
            daily_ret  = result["daily_returns"]
            regime_ret = daily_ret.loc[daily_ret.index.intersection(regime_dates)]

            if len(regime_ret) < min_days:
                continue

            ann_ret = regime_ret.mean() * trading_days
            ann_vol = regime_ret.std() * np.sqrt(trading_days)
            sharpe  = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0

            cum      = (1 + regime_ret).cumprod()
            roll_max = cum.cummax()
            max_dd   = ((cum - roll_max) / roll_max).min()

            ew_result = next((r for r in results if r["name"] == "Equal Weight"), None)
            if ew_result is not None:
                ew_ret   = ew_result["daily_returns"].loc[
                    ew_result["daily_returns"].index.intersection(regime_dates)]
                common   = regime_ret.index.intersection(ew_ret.index)
                win_rate = (regime_ret.loc[common] > ew_ret.loc[common]).mean()
            else:
                win_rate = np.nan

            rows.append({
                "Regime":              regime,
                "Strategy":            result["name"],
                "N Days":              len(regime_ret),
                "Ann. Return":         ann_ret,
                "Ann. Vol":            ann_vol,
                "Sharpe":              sharpe,
                "Max Drawdown":        max_dd,
                "Daily Win Rate vs EW": win_rate,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.pivot_table(
        index   = "Regime",
        columns = "Strategy",
        values  = ["Ann. Return", "Ann. Vol", "Sharpe",
                   "Max Drawdown", "Daily Win Rate vs EW"],
        aggfunc = "first",
    ).round(4)


def regime_sharpe_table(
    results: list[dict],
    regimes: pd.Series,
    trading_days: int = 252,
    min_days: int = 20,
) -> pd.DataFrame:
    """
    Simplified version: just Sharpe ratio per regime per strategy.
    """
    regime_labels = regimes.unique()
    order         = ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]
    regime_labels = [r for r in order if r in regime_labels]

    rows = []
    for regime in regime_labels:
        regime_dates = regimes[regimes == regime].index
        row          = {"Regime": regime, "N Days": len(regime_dates)}

        for result in results:
            daily_ret  = result["daily_returns"]
            regime_ret = daily_ret.loc[daily_ret.index.intersection(regime_dates)]

            if len(regime_ret) < min_days:
                row[result["name"]] = np.nan
            else:
                ann_ret = regime_ret.mean() * trading_days
                ann_vol = regime_ret.std() * np.sqrt(trading_days)
                row[result["name"]] = round(ann_ret / ann_vol, 3) \
                                      if ann_vol > 1e-10 else 0.0

        rows.append(row)

    return pd.DataFrame(rows).set_index("Regime")


def kalman_outperformance_by_regime(
    results: list[dict],
    regimes: pd.Series,
    trading_days: int = 252,
    min_days: int = 20,
    baseline: str = "Rolling MV",
    strategy: str = "Kalman MV",
) -> pd.DataFrame:
    """
    Compute `strategy` Sharpe MINUS `baseline` Sharpe for each regime.

    Parameters
    ----------
    strategy : name of the strategy column to compare against baseline.
               Generalized from a hardcoded "Kalman MV" so this can be
               called once per Kalman variant (Kalman-Mu MV, Kalman-Sigma MV,
               Kalman-Full MV, etc.) without code changes.

    Returns
    -------
    DataFrame: N Days, baseline Sharpe, strategy Sharpe, strategy Advantage
    """
    sharpe_table = regime_sharpe_table(results, regimes, trading_days, min_days)

    if strategy not in sharpe_table.columns or baseline not in sharpe_table.columns:
        raise ValueError(f"Need '{strategy}' and '{baseline}' in results.")

    out = pd.DataFrame({
        "N Days":                  sharpe_table["N Days"],
        f"{baseline} Sharpe":      sharpe_table[baseline],
        f"{strategy} Sharpe":      sharpe_table[strategy],
        f"{strategy} Advantage":   sharpe_table[strategy] - sharpe_table[baseline],
    })

    return out.round(4)