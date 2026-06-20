"""
diagnostic_analysis.py — Collect diagnostic data for Section 6.2 plots.

Runs a separate forward pass through the data to collect:
1. Kalman gain (average across assets) over time
2. Frobenius norm divergence between Kalman and Rolling covariance estimates

These are used for the Discussion section plots and are never fed back
into the backtest or portfolio construction.
"""

import numpy as np
import pandas as pd
from kalman_filter import KalmanReturnFilter
from config import Q_SCALE, COV_WINDOW


def collect_kalman_gain(
    returns: pd.DataFrame,
    train: pd.DataFrame,
    q_scale: float = Q_SCALE,
    regime_labels: pd.Series = None,
    regime_alphas: dict = None,
) -> pd.Series:
    """
    Run the Kalman filter forward and record the average Kalman gain
    across assets at each date.

    Returns
    -------
    pd.Series indexed by date, values = mean Kalman gain across assets
    """
    n_assets = returns.shape[1]
    R = train.cov().values + np.eye(n_assets) * 1e-6
    Q_base = q_scale * np.eye(n_assets)

    kf = KalmanReturnFilter(n_assets=n_assets, q_scale=q_scale, R=R)
    kf.initialize(mu_0=train.mean().values)

    gains = {}

    for t in range(len(returns)):
        date_t = returns.index[t]
        r_t = returns.iloc[t].values

        # Regime-conditional Q override
        q_override = None
        if (regime_labels is not None
                and regime_alphas is not None
                and date_t in regime_labels.index):
            alpha = regime_alphas.get(regime_labels.loc[date_t], 1.0)
            if alpha != 1.0:
                q_override = alpha * kf.Q

        # Predict
        Q_use = q_override if q_override is not None else kf.Q
        kf.mu_hat = kf.F @ kf.mu_hat
        kf.P = kf.F @ kf.P @ kf.F.T + Q_use

        # Compute Kalman gain before update
        S = kf.H @ kf.P @ kf.H.T + R
        K = kf.P @ kf.H.T @ np.linalg.inv(S)

        # Store mean diagonal gain (average responsiveness across assets)
        gains[date_t] = np.diag(K).mean()

        # Update
        y_tilde = r_t - kf.H @ kf.mu_hat
        kf.mu_hat = kf.mu_hat + K @ y_tilde
        kf.P = (np.eye(n_assets) - K @ kf.H) @ kf.P

    return pd.Series(gains)


def collect_covariance_divergence(
    returns: pd.DataFrame,
    train: pd.DataFrame,
    q_scale: float = Q_SCALE,
    cov_window: int = COV_WINDOW,
    regime_labels: pd.Series = None,
    regime_alphas: dict = None,
) -> pd.Series:
    """
    At each rebalancing date, compute the Frobenius norm of the difference
    between the Kalman-filtered covariance and the rolling-window covariance.

    Returns
    -------
    pd.Series indexed by rebalancing dates, values = Frobenius norm difference
    """
    n_assets = returns.shape[1]
    R = train.cov().values + np.eye(n_assets) * 1e-6

    kf = KalmanReturnFilter(n_assets=n_assets, q_scale=q_scale, R=R)
    kf.initialize(mu_0=train.mean().values)

    # Run filter to get filtered returns
    filtered_mu = np.zeros((len(returns), n_assets))

    for t in range(len(returns)):
        date_t = returns.index[t]
        r_t = returns.iloc[t].values

        q_override = None
        if (regime_labels is not None
                and regime_alphas is not None
                and date_t in regime_labels.index):
            alpha = regime_alphas.get(regime_labels.loc[date_t], 1.0)
            if alpha != 1.0:
                q_override = alpha * kf.Q

        kf.predict(q_override=q_override)
        kf.update(r_t)
        filtered_mu[t] = kf.mu_hat.copy()

    filtered_df = pd.DataFrame(
        filtered_mu, index=returns.index, columns=returns.columns
    )

    # Compute Frobenius norm at each rebalancing date
    divergence = {}
    rebal_dates = returns.resample("ME").last().index

    for date in rebal_dates:
        if date not in returns.index:
            continue
        idx = returns.index.get_loc(date)
        if idx < cov_window:
            continue

        window_slice = slice(idx - cov_window + 1, idx + 1)

        # Rolling covariance from raw returns
        sigma_rolling = returns.iloc[window_slice].cov().values

        # Kalman covariance from filtered returns
        sigma_kalman = filtered_df.iloc[window_slice].cov().values

        # Frobenius norm of difference
        divergence[date] = np.linalg.norm(sigma_kalman - sigma_rolling, "fro")

    return pd.Series(divergence)