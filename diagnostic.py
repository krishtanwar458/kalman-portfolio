"""
diagnose_q_degeneracy.py — One-off, read-only diagnostic for the Kalman-Sigma
Q-grid plateau. Does not modify any pipeline file or write any results.

For each candidate Q, prints:
  - the RAW (no-ridge) trace of cov(filtered_mean) over a 60-day window
  - the ridge term's trace, for direct comparison
  - the ratio ridge/raw — >> 1 means the ridge is dominating the signal at
    that Q, which is the suspected mechanism behind the flat plateau

Run from the repo root: python diagnose_q_degeneracy.py
"""
import numpy as np
from kalman_filter import KalmanReturnFilter
from data_loader import load_prices, compute_returns, split_data
from q_calibration import Q_GRID

RIDGE = 1e-8
COV_WINDOW = 60


def diagnose(train_returns, q_grid, cov_window=COV_WINDOW, ridge=RIDGE):
    Y = train_returns.values
    T, n = Y.shape
    R = np.cov(Y.T) + np.eye(n) * 1e-6

    raw_returns_cov_trace = np.trace(np.cov(Y[-cov_window:].T))
    print(f"Reference — raw returns cov trace (last {cov_window}d of train): {raw_returns_cov_trace:.4e}")
    print(f"Current ridge term: {ridge:.1e} * I  ->  ridge trace = {ridge * n:.4e}\n")

    print(f"{'Q_SCALE':>12}  {'Raw filt-cov trace':>20}  {'Ridge trace':>14}  {'Ridge/Raw ratio':>16}")
    print("-" * 68)

    for q in q_grid:
        kf = KalmanReturnFilter(n_assets=n, q_scale=q, R=R)
        kf.initialize(mu_0=Y[:cov_window].mean(axis=0))

        filtered_history = []
        for t in range(T):
            kf.predict()
            kf.update(Y[t])
            filtered_history.append(kf.mu_hat.copy())

        filt_window = np.array(filtered_history[-cov_window:])
        raw_filt_trace = np.trace(np.cov(filt_window.T))
        ridge_trace = ridge * n
        ratio = ridge_trace / raw_filt_trace if raw_filt_trace > 0 else float("inf")

        flag = "  <- ridge-dominated" if ratio > 10 else ""
        print(f"{q:12.2e}  {raw_filt_trace:20.4e}  {ridge_trace:14.4e}  {ratio:16.1f}{flag}")


if __name__ == "__main__":
    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    diagnose(train, Q_GRID)