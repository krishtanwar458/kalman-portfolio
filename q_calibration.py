"""
q_calibration.py — Walk-forward cross-validation to select Q_SCALE.

Evaluates a grid of Q values using one-step-ahead forecast RMSE
on the last 30% of the training period (walk-forward within training).
No test data is ever touched.

Reference: time-series cross-validation following Hyndman & Athanasopoulos (2018).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ── Grid of Q values to evaluate ─────────────────────────────────────────────
Q_GRID = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


def _run_kalman(Y: np.ndarray, q: float, R: np.ndarray, mu0: np.ndarray) -> np.ndarray:
    """
    Run forward Kalman filter and return one-step-ahead forecast errors.
    F = H = I (local-level / random-walk model).
    """
    T, n = Y.shape
    Q = q * np.eye(n)
    P = R.copy()        # diffuse initialisation
    mu = mu0.copy()
    errors = np.zeros((T, n))

    for t in range(T):
        # Predict
        mu_pred = mu
        P_pred  = P + Q

        # Innovation (one-step-ahead forecast error)
        errors[t] = Y[t] - mu_pred

        # Update
        S = P_pred + R
        K = P_pred @ np.linalg.solve(S.T, np.eye(n)).T
        mu = mu_pred + K @ errors[t]
        P  = (np.eye(n) - K) @ P_pred

    return errors


def select_q_cv(
    train_returns: pd.DataFrame,
    q_grid: list = Q_GRID,
    val_fraction: float = 0.30,
    verbose: bool = True,
) -> tuple:
    """
    Select Q_SCALE via walk-forward CV on training data.
    Criterion: annualised Sharpe of Kalman MV portfolio on validation window.
    Warmup: first 70% of training. Validation: last 30%.
    No test data is ever touched.
    """
    from optimizer import optimize_portfolio

    Y    = train_returns.values
    T, n = Y.shape
    assets = train_returns.columns

    split  = int(T * (1 - val_fraction))
    Y_warm = Y[:split]
    Y_val  = Y[split:]

    R   = np.cov(Y.T) + np.eye(n) * 1e-6
    mu0 = Y_warm.mean(axis=0)

    rows = []
    for q in q_grid:
        Q_mat = q * np.eye(n)

        # ── Warmup pass ──────────────────────────────────────
        P  = R.copy()
        mu = mu0.copy()
        for t in range(len(Y_warm)):
            mu_p = mu
            P_p  = P + Q_mat
            S    = P_p + R
            K    = P_p @ np.linalg.solve(S.T, np.eye(n)).T
            mu   = mu_p + K @ (Y_warm[t] - mu_p)
            P    = (np.eye(n) - K) @ P_p

        # ── Validation pass: run filter + MV optimizer ───────
        mu_val = mu.copy()
        P_val  = P.copy()

        port_returns = []
        current_weights = np.ones(n) / n
        rebal_counter = 0

        for t in range(len(Y_val)):
            # Rebalance monthly (~21 trading days)
            if rebal_counter % 21 == 0:
                try:
                    w = optimize_portfolio(mu_val, R)
                    if w is not None and not np.any(np.isnan(w)):
                        current_weights = w
                except Exception:
                    pass

            port_returns.append(current_weights @ Y_val[t])
            rebal_counter += 1

            # Update filter
            mu_p   = mu_val
            P_p    = P_val + Q_mat
            err    = Y_val[t] - mu_p
            S      = P_p + R
            K      = P_p @ np.linalg.solve(S.T, np.eye(n)).T
            mu_val = mu_p + K @ err
            P_val  = (np.eye(n) - K) @ P_p

        r = np.array(port_returns)
        sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
        rows.append({"Q_SCALE": q, "Val_Sharpe": sharpe})

    results = pd.DataFrame(rows).set_index("Q_SCALE")
    best_q  = float(results["Val_Sharpe"].idxmax())   # maximise Sharpe, not minimise RMSE

    if verbose:
        print("\n── Q Grid Search Results (training CV, criterion = Sharpe) ──")
        print(f"{'Q_SCALE':>12}  {'Val Sharpe':>12}")
        for q, row in results.iterrows():
            marker = "  ← best" if q == best_q else ""
            print(f"  {q:12.2e}  {row['Val_Sharpe']:12.6f}{marker}")
        print(f"\n  Selected Q_SCALE = {best_q:.4e}")

    return best_q, results


def plot_q_selection(results: pd.DataFrame, best_q: float,
                     save_path: str = "plots/q_selection.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(len(results)), results["Val_Sharpe"].values,
            color="#1f77b4", linewidth=2, marker="o", markersize=6)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([f"{q:.0e}" for q in results.index], rotation=45, fontsize=9)
    ax.set_xlabel("Q_SCALE", fontsize=12)
    ax.set_ylabel("Validation Sharpe (portfolio)", fontsize=12)
    ax.set_title("Q Selection via Walk-Forward CV — Training Data Only\n(Criterion: Portfolio Sharpe on Validation Window)",
                 fontsize=12, fontweight="bold")

    best_idx = list(results.index).index(best_q)
    ax.axvline(best_idx, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate(f"Best Q = {best_q:.2e}",
                xy=(best_idx, results["Val_Sharpe"].iloc[best_idx]),
                xytext=(best_idx + 0.4, results["Val_Sharpe"].min() * 1.01),
                fontsize=10, color="red")

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()
