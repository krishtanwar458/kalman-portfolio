"""
benchmarks.py — Strategy definitions for backtesting.

Each strategy is a function with signature:
    strategy(date, returns_so_far) -> np.ndarray of weights
"""

import numpy as np
import pandas as pd
from optimizer import optimize_portfolio
from kalman_filter import KalmanReturnFilter
from config import COV_WINDOW, RETURN_WINDOW, Q_SCALE


# 1. Equal Weight

def equal_weight_strategy(date, returns_so_far):
    """1/N equal weight across all assets."""
    n = returns_so_far.shape[1]
    return np.ones(n) / n


# 2. Rolling Mean-Variance

def make_rolling_mv_strategy(cov_window=COV_WINDOW, ret_window=RETURN_WINDOW, turnover_gamma=0.0):
    w_prev = [None]

    def strategy(date, returns_so_far):
        if len(returns_so_far) < cov_window:
            n = returns_so_far.shape[1]
            w_prev[0] = np.ones(n) / n
            return w_prev[0]

        recent = returns_so_far.iloc[-cov_window:]
        mu = recent.mean().values
        sigma = recent.cov().values

        w = optimize_portfolio(mu, sigma, w_prev=w_prev[0], turnover_gamma=turnover_gamma)
        w_prev[0] = w.copy()
        return w

    return strategy


# 3. Static Mean-Variance

def make_static_mv_strategy(train_returns: pd.DataFrame):
    mu_static = train_returns.mean().values
    sigma_static = train_returns.cov().values
    weights_static = optimize_portfolio(mu_static, sigma_static)

    def strategy(date, returns_so_far):
        return weights_static

    return strategy


# 4. Kalman Filter Strategy — generalized to 3 variants via flags
#
#    use_filtered_mu=False, use_filtered_sigma=True   -> Kalman-Sigma MV (covariance-isolation design)
#    use_filtered_mu=True,  use_filtered_sigma=False  -> Kalman-Mu MV    (mean-isolation design)
#    use_filtered_mu=True,  use_filtered_sigma=True   -> Kalman-Full MV  (both adaptive)

def make_kalman_strategy(
    train_returns: pd.DataFrame,
    q_scale: float = Q_SCALE,
    cov_window: int = COV_WINDOW,
    ret_window: int = RETURN_WINDOW,
    regime_labels: pd.Series = None,
    regime_alphas: dict = None,
    turnover_gamma: float = 0.0,
    use_filtered_mu: bool = False,
    use_filtered_sigma: bool = True,
):
    n_assets = train_returns.shape[1]
    R = train_returns.cov().values + np.eye(n_assets) * 1e-6
    kf = KalmanReturnFilter(n_assets=n_assets, q_scale=q_scale, R=R)
    mu_0 = train_returns.mean().values
    kf.initialize(mu_0=mu_0)
    last_processed_idx = 0
    w_prev = [None]
    filtered_history = []   # running record of kf.mu_hat, one row per day
    filtered_index = []

    def strategy(date, returns_so_far):
        nonlocal last_processed_idx

        current_len = len(returns_so_far)
        for i in range(last_processed_idx, current_len):
            r_t = returns_so_far.iloc[i].values
            date_i = returns_so_far.index[i]

            q_override = None
            if (regime_labels is not None
                    and regime_alphas is not None
                    and date_i in regime_labels.index):
                alpha = regime_alphas.get(regime_labels.loc[date_i], 1.0)
                if alpha != 1.0:
                    q_override = alpha * kf.Q

            kf.predict(q_override=q_override)
            kf.update(r_t)

            filtered_history.append(kf.mu_hat.copy())
            filtered_index.append(date_i)

        last_processed_idx = current_len

        if current_len < cov_window:
            n = returns_so_far.shape[1]
            w_prev[0] = np.ones(n) / n
            return w_prev[0]

        # Return estimate
        if use_filtered_mu:
            mu = kf.mu_hat.copy()
        else:
            mu = returns_so_far.iloc[-ret_window:].mean().values

        # Covariance estimate
        if use_filtered_sigma:
            filtered_recent = pd.DataFrame(
                filtered_history[-cov_window:],
                index=filtered_index[-cov_window:],
                columns=returns_so_far.columns,
            )
            filt_cov = filtered_recent.cov().values
            raw_cov = returns_so_far.iloc[-cov_window:].cov().values
            # Rescale the filtered covariance's overall magnitude to match the
            # raw covariance over the same window. Filtering intentionally
            # suppresses variance (more aggressively at low Q) — that's
            # meaningful as far as *shape* (which assets it currently
            # believes co-move), but it leaves the matrix on a completely
            # different absolute scale than RISK_AVERSION was calibrated
            # against (raw-magnitude covariance, same as Rolling MV /
            # Ledoit-Wolf MV). Left unscaled, very small Q produces a
            # near-zero-magnitude covariance the optimizer's risk term can't
            # see past the constant return term, so it collapses into an
            # unconstrained-return chase regardless of Q. Matching the trace
            # preserves the filter's relative/correlation structure (the part
            # that's actually Q-sensitive) while keeping the comparison to
            # the other strategies apples-to-apples.
            filt_trace = np.trace(filt_cov)
            raw_trace = np.trace(raw_cov)
            if filt_trace > 1e-12 and raw_trace > 0:
                sigma = filt_cov * (raw_trace / filt_trace)
            else:
                # Filtered covariance has collapsed to near-zero -- rescaling
                # by a near-infinite factor would be numerically unstable AND
                # uninformative (there's no real signal left to preserve).
                # Fall back to the raw covariance directly, rather than
                # silently passing the unscaled near-zero matrix through
                # (which would recreate the exact optimizer-saturation
                # problem the rescaling exists to fix).
                sigma = raw_cov.copy()
            sigma = 0.5 * (sigma + sigma.T)
            sigma = sigma + np.eye(n_assets) * 1e-8
        else:
            sigma = returns_so_far.iloc[-cov_window:].cov().values

        w = optimize_portfolio(mu, sigma, w_prev=w_prev[0], turnover_gamma=turnover_gamma)
        w_prev[0] = w.copy()
        return w

    # Expose the real, regime-alpha-aware mu_hat trajectory this strategy
    # actually used, so callers (e.g. the DM test) can evaluate the filter
    # this variant really ran, instead of rebuilding a separate constant-Q
    # filter that silently ignores regime_alphas. filtered_history/index are
    # mutated in place during the backtest, so these references stay valid
    # (and complete) once run_backtest() has finished calling strategy().
    strategy.filtered_history = filtered_history
    strategy.filtered_index = filtered_index

    return strategy


from sklearn.covariance import LedoitWolf

# 5. Ledoit-Wolf Shrinkage Mean-Variance — unaffected by the mu/sigma ablation above

def make_ledoit_wolf_strategy(cov_window=COV_WINDOW, ret_window=RETURN_WINDOW, turnover_gamma=0.0):
    w_prev = [None]

    def strategy(date, returns_so_far):
        if len(returns_so_far) < cov_window:
            n = returns_so_far.shape[1]
            w_prev[0] = np.ones(n) / n
            return w_prev[0]

        recent = returns_so_far.iloc[-cov_window:]
        mu = recent.mean().values
        sigma = LedoitWolf().fit(recent.values).covariance_

        w = optimize_portfolio(mu, sigma, w_prev=w_prev[0], turnover_gamma=turnover_gamma)
        w_prev[0] = w.copy()
        return w

    return strategy