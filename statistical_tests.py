"""
statistical_tests.py — Rigorous statistical testing for portfolio comparison.

Two tests implemented:

1. Diebold-Mariano (DM) Test
   Tests whether a Kalman variant's filtered return-signal forecast is
   significantly more accurate than the rolling-window forecast. Operates
   on forecast errors, not portfolio returns or covariance forecasts. This
   is the right test for comparing *predictive accuracy* of the mean
   estimate each variant actually used during its own backtest (regime-
   switching included) -- not a generic, shared "Kalman Filter" series.

   H0: The two methods have equal forecast accuracy (E[d_t] = 0)
   H1: The Kalman variant's filter is more accurate (E[d_t] < 0)

   Run once per variant (Kalman-Mu MV, Kalman-Sigma MV, Kalman-Full MV),
   since each has its own calibrated Q* and regime alphas and therefore its
   own filtered mean series.

   Reference: Diebold & Mariano (1995), JBES.

2. Block Bootstrap Sharpe Ratio Confidence Intervals
   The standard Sharpe ratio has no closed-form CI when returns are
   autocorrelated (they always are). Block bootstrap preserves the
   autocorrelation structure by resampling in contiguous blocks.

   I compute 95% CIs for each strategy's Sharpe ratio and check
   whether each Kalman variant's CI lies above the Rolling MV CI.

   Reference: Ledoit & Wolf (2008), Journal of Econometrics.

Usage:
    from statistical_tests import diebold_mariano_test, bootstrap_sharpe_ci, run_all_tests
"""

import numpy as np
import pandas as pd
from scipy import stats


# 1. Diebold-Mariano Test

def _newey_west_variance(d: np.ndarray, max_lag: int = None) -> float:
    """
    Compute Newey-West HAC variance of the loss differential series d.

    This is needed because d_t may be autocorrelated (forecast errors
    at adjacent time steps are not independent). Using OLS variance
    would give incorrect standard errors.

    Parameters
    ----------
    d        : loss differential series (T,)
    max_lag  : number of lags for HAC correction (default: T^(1/3))

    Returns
    -------
    HAC-corrected variance estimate (scalar)
    """
    T = len(d)
    if max_lag is None:
        max_lag = int(np.floor(T ** (1 / 3)))  # standard rule of thumb

    d_demeaned = d - d.mean()
    gamma_0 = np.dot(d_demeaned, d_demeaned) / T  # variance at lag 0

    hac_var = gamma_0
    for lag in range(1, max_lag + 1):
        weight = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.dot(d_demeaned[lag:], d_demeaned[:-lag]) / T
        hac_var += 2 * weight * gamma_lag

    return max(hac_var, 1e-12)  # floor at near-zero to avoid division errors


def diebold_mariano_test(
    forecast_errors_kalman: pd.Series,
    forecast_errors_rolling: pd.Series,
    loss: str = "squared",
    alternative: str = "less",
) -> dict:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Compares whether a Kalman variant's filtered mean has significantly
    lower forecast error than the rolling-window estimator. Pass in the
    forecast-error series for the SPECIFIC variant being evaluated -- this
    function doesn't know or care which variant it is, that's determined
    entirely by what filtered_mu series produced forecast_errors_kalman.

    Parameters
    ----------
    forecast_errors_kalman  : Series of (actual - kalman_forecast) at each step
    forecast_errors_rolling : Series of (actual - rolling_forecast) at each step
    loss       : "squared" (MSE-based) or "absolute" (MAE-based)
    alternative: "less" = test KF < Rolling (one-sided, KF is better)
                 "two-sided" = test KF != Rolling

    Returns
    -------
    dict with keys: DM_statistic, p_value, conclusion, loss_differential_mean
    """
    # Align on common dates
    common_idx = forecast_errors_kalman.index.intersection(forecast_errors_rolling.index)
    e_kf = forecast_errors_kalman.loc[common_idx].values
    e_roll = forecast_errors_rolling.loc[common_idx].values

    # Compute loss differential: d_t = L(e_kalman_t) - L(e_rolling_t)
    # Negative d_t means KF made smaller error at time t (KF is better)
    if loss == "squared":
        d = e_kf ** 2 - e_roll ** 2
    elif loss == "absolute":
        d = np.abs(e_kf) - np.abs(e_roll)
    else:
        raise ValueError(f"Unknown loss: {loss}. Use 'squared' or 'absolute'.")

    T = len(d)
    d_bar = d.mean()  # mean loss differential

    # HAC variance for the test statistic
    hac_var = _newey_west_variance(d)
    dm_stat = d_bar / np.sqrt(hac_var / T)

    # p-value
    if alternative == "less":
        # One-sided: KF has lower loss (d_bar < 0 is good)
        p_value = stats.norm.cdf(dm_stat)
    elif alternative == "two-sided":
        p_value = 2 * stats.norm.cdf(-abs(dm_stat))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Interpretation
    if alternative == "less":
        if p_value < 0.05:
            conclusion = "Reject H0 at 5%: Kalman Filter has significantly lower forecast error."
        elif p_value < 0.10:
            conclusion = "Reject H0 at 10%: Kalman Filter has marginally lower forecast error."
        else:
            conclusion = "Fail to reject H0: No significant difference in forecast accuracy."
    else:
        conclusion = "Reject H0." if p_value < 0.05 else "Fail to reject H0."

    return {
        "DM_statistic": round(dm_stat, 4),
        "p_value": round(p_value, 4),
        "mean_loss_differential": round(d_bar, 8),
        "n_observations": T,
        "conclusion": conclusion,
        "loss_function": loss,
        "alternative": alternative,
    }


def compute_forecast_errors(
    filtered_mu: pd.DataFrame,
    returns: pd.DataFrame,
    window: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute one-step-ahead forecast errors for both methods.

    At each time t, the forecast is made at t-1 and evaluated at t.
    - Kalman forecast: filtered_mu at t-1 (already computed elsewhere --
      pass in the specific variant's real, regime-alpha-aware trajectory,
      e.g. read directly off that variant's strategy closure, not a
      separately-rebuilt constant-Q filter)
    - Rolling forecast: mean of returns[t-window:t-1]

    Parameters
    ----------
    filtered_mu : DataFrame of one Kalman variant's filtered expected
                  returns (T x N) -- this function is variant-agnostic,
                  it just evaluates whatever series it's given
    returns     : DataFrame of actual daily returns (T x N)
    window      : lookback window for rolling mean forecast

    Returns
    -------
    (errors_kalman, errors_rolling) — both DataFrames (T x N)
    """
    common_assets = filtered_mu.columns.intersection(returns.columns)
    common_idx = filtered_mu.index.intersection(returns.index)

    mu_kf = filtered_mu.loc[common_idx, common_assets]
    ret = returns.loc[common_idx, common_assets]

    # Kalman forecast error: actual_t - mu_kf_{t-1}
    errors_kf = ret - mu_kf.shift(1)

    # Rolling forecast error: actual_t - rolling_mean_{t-1}
    rolling_mu = ret.rolling(window=window, min_periods=window // 2).mean().shift(1)
    errors_roll = ret - rolling_mu

    return errors_kf.dropna(), errors_roll.dropna()


# 2. Block Bootstrap Sharpe Ratio Confidence Intervals

def block_bootstrap_sharpe(
    daily_returns: pd.Series,
    block_size: int = 20,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    trading_days: int = 252,
    random_state: int = 42,
) -> dict:
    """
    Block bootstrap confidence interval for the annualized Sharpe ratio.

    Standard Sharpe ratio CIs assume i.i.d. returns, which is wrong —
    financial returns have autocorrelation (momentum, mean-reversion).
    Block bootstrap resamples contiguous blocks of length `block_size`
    to preserve this autocorrelation structure.

    Parameters
    ----------
    daily_returns : Series of daily portfolio returns
    block_size    : length of each bootstrap block (default: 20 trading days ≈ 1 month)
    n_bootstrap   : number of bootstrap resamples
    confidence    : confidence level (default: 0.95 for 95% CI)
    trading_days  : annualization factor

    Returns
    -------
    dict with: sharpe, ci_lower, ci_upper, std_error, n_bootstrap
    """
    rng = np.random.default_rng(random_state)
    returns = daily_returns.dropna().values
    T = len(returns)

    def sharpe_from_array(r):
        mean = r.mean() * trading_days
        std = r.std() * np.sqrt(trading_days)
        return mean / std if std > 1e-10 else 0.0

    observed_sharpe = sharpe_from_array(returns)

    # Generate bootstrap samples
    bootstrap_sharpes = np.zeros(n_bootstrap)
    n_blocks = int(np.ceil(T / block_size))

    for b in range(n_bootstrap):
        # Sample random starting points for blocks
        starts = rng.integers(0, T - block_size + 1, size=n_blocks)
        # Build bootstrap sample by concatenating blocks
        blocks = [returns[s : s + block_size] for s in starts]
        boot_sample = np.concatenate(blocks)[:T]  # trim to original length
        bootstrap_sharpes[b] = sharpe_from_array(boot_sample)

    # Percentile bootstrap CI (BCa would be more accurate but this is standard)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_sharpes, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2))

    return {
        "sharpe": round(observed_sharpe, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std_error": round(bootstrap_sharpes.std(), 4),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "confidence": confidence,
    }


def bootstrap_sharpe_comparison(
    results: list[dict],
    block_size: int = 20,
    n_bootstrap: int = 1000,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Run block bootstrap Sharpe CIs for all strategies and return a comparison table.

    Parameters
    ----------
    results     : list of dicts from run_backtest()
    block_size  : bootstrap block length in trading days
    n_bootstrap : number of bootstrap samples

    Returns
    -------
    DataFrame with one row per strategy: Sharpe, 95% CI lower, 95% CI upper, Std Error
    """
    rows = []
    for result in results:
        ci = block_bootstrap_sharpe(
            result["daily_returns"],
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            trading_days=trading_days,
        )
        rows.append({
            "Strategy": result["name"],
            "Sharpe": ci["sharpe"],
            "95% CI Lower": ci["ci_lower"],
            "95% CI Upper": ci["ci_upper"],
            "Std Error": ci["std_error"],
        })

    df = pd.DataFrame(rows).set_index("Strategy")
    return df


# 3. Convenience wrapper — run everything
#
# NOTE: not used by main.py, which calls compute_forecast_errors /
# diebold_mariano_test / bootstrap_sharpe_comparison directly, once per
# Kalman variant, since each variant needs its own filtered_mu series (see
# module docstring). This wrapper predates the 3-variant design and only
# knows about a single "Kalman MV" -- kept for reference / potential reuse
# in a single-variant context, but would need a variant-aware rewrite
# (a loop over KALMAN_VARIANTS, like main.py's Step 6) before being usable
# in the current pipeline.

def run_all_tests(
    results: list[dict],
    filtered_mu: pd.DataFrame,
    returns: pd.DataFrame,
    rolling_window: int = 60,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Run the full statistical testing suite for a single Kalman variant.

    Parameters
    ----------
    results      : list of dicts from run_backtest()
    filtered_mu  : one variant's filtered expected returns (T x N) -- its
                   own real, regime-alpha-aware trajectory, not a generic
                   constant-Q stand-in
    returns      : actual daily returns (T x N)
    rolling_window : window used in rolling MV strategy
    n_bootstrap  : bootstrap resamples for Sharpe CIs

    Returns
    -------
    dict with keys: 'dm_results', 'sharpe_cis'
    """
    print("\n=== STATISTICAL TESTS ===")

    # Diebold-Mariano test (per asset, then averaged)
    print("\n[1] Diebold-Mariano Forecast Accuracy Test")
    print("    H0: this variant's filter and Rolling have equal forecast accuracy")
    print("    H1: this variant's filter has lower forecast error (one-sided)")

    errors_kf, errors_roll = compute_forecast_errors(filtered_mu, returns, window=rolling_window)

    dm_results = {}
    for asset in errors_kf.columns:
        dm = diebold_mariano_test(
            errors_kf[asset],
            errors_roll[asset],
            loss="squared",
            alternative="less",
        )
        dm_results[asset] = dm
        print(f"    {asset}: DM={dm['DM_statistic']:+.3f}, p={dm['p_value']:.3f} : {dm['conclusion'][:50]}")

    # Overall DM test on portfolio-aggregated errors (mean across assets)
    e_kf_agg = errors_kf.mean(axis=1)
    e_roll_agg = errors_roll.mean(axis=1)
    dm_overall = diebold_mariano_test(e_kf_agg, e_roll_agg, loss="squared", alternative="less")
    dm_results["OVERALL"] = dm_overall
    print(f"\n    OVERALL: DM={dm_overall['DM_statistic']:+.3f}, p={dm_overall['p_value']:.3f}")
    print(f"    {dm_overall['conclusion']}")

    # Block Bootstrap Sharpe CIs
    print(f"\n[2] Block Bootstrap Sharpe Ratio Confidence Intervals")
    print(f"    (block_size=20 days, n_bootstrap={n_bootstrap})")
    sharpe_cis = bootstrap_sharpe_comparison(results, n_bootstrap=n_bootstrap)
    print(f"\n{sharpe_cis.to_string()}")

    # Assess overlap -- caller must pass the variant's actual strategy name
    # (this wrapper has no way to know which of the 3 Kalman variants
    # `filtered_mu` belongs to; it only checks the literal name "Kalman MV",
    # which no longer exists as a strategy label and will simply skip this
    # block in the current 3-variant pipeline)
    kalman_row = sharpe_cis.loc["Kalman MV"] if "Kalman MV" in sharpe_cis.index else None
    rolling_row = sharpe_cis.loc["Rolling MV"] if "Rolling MV" in sharpe_cis.index else None

    if kalman_row is not None and rolling_row is not None:
        no_overlap = kalman_row["95% CI Lower"] > rolling_row["95% CI Upper"]
        partial_overlap = kalman_row["95% CI Upper"] > rolling_row["95% CI Upper"]
        print("\n    CI Interpretation:")
        if no_overlap:
            print("    CIs do NOT overlap: this variant's Sharpe is statistically superior to Rolling MV.")
        elif partial_overlap:
            print("    CIs partially overlap: this variant shows higher Sharpe but difference is not statistically conclusive.")
        else:
            print("    CIs fully overlap: No statistically significant difference in Sharpe ratios.")

    return {
        "dm_results": dm_results,
        "sharpe_cis": sharpe_cis,
        "forecast_errors": {"kalman": errors_kf, "rolling": errors_roll},
    }