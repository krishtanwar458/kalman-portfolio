"""
kalman_filter.py — Kalman Filter for time-varying expected return estimation.

The state-space model:
    State:       mu_t = F * mu_{t-1} + w_t,   w_t ~ N(0, Q)
    Observation: r_t  = H * mu_t    + v_t,   v_t ~ N(0, R)

Where:
    mu_t = latent expected return vector (what we're estimating)
    r_t  = observed return vector
    F    = state transition (Identity = random walk)
    H    = observation matrix (Identity = direct observation)
    Q    = process noise covariance (how fast true returns drift)
    R    = observation noise covariance (how noisy observed returns are)

Usage:
    from kalman_filter import KalmanReturnFilter
"""

import numpy as np
import pandas as pd
from config import Q_SCALE


class KalmanReturnFilter:
    """
    Multivariate Kalman Filter for estimating time-varying expected returns.

    The filter recursively updates its estimate of the latent expected return vector 
    as new observations arrive, balancing prior beliefs against new evidence through 
    the Kalman gain.

    Parameters
    ----------
    n_assets : int
        Number of assets in the universe.
    q_scale : float
        Scalar multiplier for process noise covariance Q = q_scale * I.
        Controls how fast the filter believes true returns change.
        Small q_scale = slow adaptation (trusts history more).
        Large q_scale = fast adaptation (trusts recent data more).
    R : np.ndarray or None
        Observation noise covariance matrix (n_assets x n_assets).
        If None, must be set before running the filter.
    """

    def __init__(self, n_assets: int, q_scale: float = Q_SCALE, R: np.ndarray = None):
        self.n = n_assets
        self.q_scale = q_scale

        # State transition: Identity (random walk model)
        self.F = np.eye(self.n)

        # Observation matrix: Identity (we directly observe returns)
        self.H = np.eye(self.n)

        # Process noise covariance
        self.Q = q_scale * np.eye(self.n)

        # Observation noise covariance (set from training data)
        self.R = R if R is not None else np.eye(self.n) * 0.01

        # State estimate and covariance (initialized in .initialize())
        self.mu_hat = None  # filtered expected return estimate
        self.P = None       # state estimation covariance

    def initialize(self, mu_0: np.ndarray, P_0: np.ndarray = None):
        """
        Set initial state estimate and covariance.

        Parameters
        ----------
        mu_0 : initial expected return estimate (n_assets,)
        P_0  : initial state covariance (n_assets x n_assets), default = I
        """
        self.mu_hat = mu_0.copy()
        self.P = P_0 if P_0 is not None else np.eye(self.n) * 1e-4

    def predict(self, q_override=None):
        """
        Prediction step: propagate state estimate forward.
        q_override: if provided, use this matrix instead of self.Q for this step only.
        """
        Q_use = q_override if q_override is not None else self.Q
        self.mu_hat = self.F @ self.mu_hat
        self.P = self.F @ self.P @ self.F.T + Q_use

    def update(self, r_observed: np.ndarray):
        """
        Update step: incorporate new observation.

        innovation:  y_tilde = r_t - H * mu_{t|t-1}
        innovation cov: S = H * P_{t|t-1} * H' + R
        Kalman gain: K = P_{t|t-1} * H' * S^{-1}
        updated state: mu_{t|t} = mu_{t|t-1} + K * y_tilde
        updated cov:   P_{t|t} = (I - K * H) * P_{t|t-1}

        Parameters
        ----------
        r_observed : observed return vector (n_assets,)
        """
        # Innovation
        y_tilde = r_observed - self.H @ self.mu_hat

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state estimate
        self.mu_hat = self.mu_hat + K @ y_tilde

        # Updated covariance
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

    def filter_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full Kalman Filter over a return series.

        Parameters
        ----------
        returns : DataFrame of observed daily returns (T x n_assets)

        Returns
        -------
        DataFrame of filtered expected return estimates (T x n_assets),
        same index and columns as input.
        """
        T = len(returns)
        assets = returns.columns
        filtered_mu = np.zeros((T, self.n))

        for t in range(T):
            r_t = returns.iloc[t].values

            # Predict
            self.predict()

            # Update with new observation
            self.update(r_t)

            # Store filtered estimate
            filtered_mu[t] = self.mu_hat.copy()

        return pd.DataFrame(filtered_mu, index=returns.index, columns=assets)


def build_filter_from_training(train_returns: pd.DataFrame, q_scale: float = Q_SCALE) -> KalmanReturnFilter:
    """
    Convenience function: build and initialize a KalmanReturnFilter
    from training data.

    - R is estimated as the sample colvariance of training returns
    - Initial mu_0 is the mean of training returns
    - Initial P_0 is a small diagonal matrix

    Parameters
    ----------
    train_returns : DataFrame of training period returns
    q_scale : process noise scale

    Returns
    -------
    Initialized KalmanReturnFilter ready to run on test data
    """
    n_assets = train_returns.shape[1]

    # Estimate observation noise from training data
    R = train_returns.cov().values + np.eye(n_assets) * 1e-6

    # Initialize filter
    kf = KalmanReturnFilter(n_assets=n_assets, q_scale=q_scale, R=R)

    # Initial state: mean return from training period
    mu_0 = train_returns.mean().values

    # Initial covariance: small (we're fairly confident in training estimate)
    P_0 = np.eye(n_assets) * 1e-4

    kf.initialize(mu_0=mu_0, P_0=P_0)

    return kf


# Quick test
if __name__ == "__main__":
    from data_loader import load_prices, compute_returns, split_data

    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    # Build filter from training data
    kf = build_filter_from_training(train)

    # Run filter on full dataset (train + test)
    filtered = kf.filter_returns(returns)

    print("\nFiltered expected returns (last 5 days):")
    print((filtered.tail() * 252).round(4))  # annualized
    print("\nRolling mean returns (last 5 days):")
    print((returns.rolling(60).mean().tail() * 252).round(4))  # annualized
