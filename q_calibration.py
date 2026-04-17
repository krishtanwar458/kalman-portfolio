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


# Grid of Q values to evaluate
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
    Criterion: annualized Sharpe of Kalman MV portfolio on validation window.
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

        # Warmup pass
        P  = R.copy()
        mu = mu0.copy()
        for t in range(len(Y_warm)):
            mu_p = mu
            P_p  = P + Q_mat
            S    = P_p + R
            K    = P_p @ np.linalg.solve(S.T, np.eye(n)).T
            mu   = mu_p + K @ (Y_warm[t] - mu_p)
            P    = (np.eye(n) - K) @ P_p

        # Validation pass: run filter + MV optimizer
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

# Regime-specific alpha grid
ALPHA_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
REGIMES_TO_TUNE = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]  # CRISIS fixed at base Q


def _run_regime_validation(
    Y_warm: np.ndarray,
    Y_val: np.ndarray,
    dates_val: pd.DatetimeIndex,
    regime_labels: pd.Series,
    R: np.ndarray,
    Q_base: np.ndarray,
    regime_alphas: dict,
) -> float:
    """
    Run one validation pass with a given set of regime alphas.
    Returns annualized Sharpe on the validation window.
    regime_alphas: dict mapping regime name -> alpha (e.g. {'LOW_VOL': 0.1, ...})
    CRISIS always uses 1.0 (base Q).
    """
    from optimizer import optimize_portfolio

    n = Y_warm.shape[1]

    # Build per-regime Q matrices
    Q_map = {regime: regime_alphas.get(regime, 1.0) * Q_base
             for regime in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]}

    # Warmup pass — use base Q throughout (no switching during warmup)
    P = R.copy()
    mu = Y_warm.mean(axis=0)
    for t in range(len(Y_warm)):
        mu_p = mu
        P_p = P + Q_base
        S = P_p + R
        K = P_p @ np.linalg.solve(S.T, np.eye(n)).T
        mu = mu_p + K @ (Y_warm[t] - mu_p)
        P = (np.eye(n) - K) @ P_p

    # Validation pass — regime-switching Q
    mu_val = mu.copy()
    P_val = P.copy()
    port_returns = []
    current_weights = np.ones(n) / n
    rebal_counter = 0

    for t in range(len(Y_val)):
        date_t = dates_val[t]

        # Look up regime, default to base Q if date not in labels (warmup edge)
        regime_t = regime_labels.get(date_t, "MED_VOL") if hasattr(regime_labels, 'get') \
                   else (regime_labels.loc[date_t] if date_t in regime_labels.index else "MED_VOL")
        Q_t = Q_map.get(regime_t, Q_base)

        # Monthly rebalance
        if rebal_counter % 21 == 0:
            try:
                w = optimize_portfolio(mu_val, R)
                if w is not None and not np.any(np.isnan(w)):
                    current_weights = w
            except Exception:
                pass

        port_returns.append(current_weights @ Y_val[t])
        rebal_counter += 1

        # Filter update
        mu_p = mu_val
        P_p = P_val + Q_t
        err = Y_val[t] - mu_p
        S = P_p + R
        K = P_p @ np.linalg.solve(S.T, np.eye(n)).T
        mu_val = mu_p + K @ err
        P_val = (np.eye(n) - K) @ P_p

    r = np.array(port_returns)
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def calibrate_regime_alphas(
    train_returns: pd.DataFrame,
    regime_labels: pd.Series,
    q_best: float,
    alpha_grid: list = ALPHA_GRID,
    val_fraction: float = 0.30,
    verbose: bool = True,
) -> tuple:
    """
    Calibrate per-regime Q attenuation factors for LOW_VOL, MED_VOL, HIGH_VOL.
    CRISIS is always fixed at base Q (alpha = 1.0) — too few days to calibrate.

    Strategy: sequential greedy search.
      1. Start with all alphas = 1.0 (base Q everywhere).
      2. For each regime in [LOW_VOL, MED_VOL, HIGH_VOL]:
           Grid search alpha for this regime, holding others fixed at current best.
           Keep whichever alpha maximises validation Sharpe.
    This avoids a full 3D grid search (6^3 = 216 combinations) while still
    finding a good joint solution.

    Returns
    -------
    best_alphas : dict  e.g. {'LOW_VOL': 0.2, 'MED_VOL': 1.0, 'HIGH_VOL': 0.05, 'CRISIS': 1.0}
    all_results : dict of DataFrames, one per regime
    """
    Y = train_returns.values
    T, n = Y.shape

    split = int(T * (1 - val_fraction))
    Y_warm = Y[:split]
    Y_val = Y[split:]
    dates_val = train_returns.index[split:]

    R = np.cov(Y.T) + np.eye(n) * 1e-6
    Q_base = q_best * np.eye(n)

    # Start: all regimes at base Q
    current_alphas = {r: 1.0 for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]}
    all_results = {}

    if verbose:
        print("\n── Per-Regime Alpha Calibration (greedy sequential, criterion = Val Sharpe) ──")

    for regime in REGIMES_TO_TUNE:
        rows = []
        for alpha in alpha_grid:
            # Hold all other regimes fixed, try this alpha for current regime
            trial_alphas = {**current_alphas, regime: alpha}
            sharpe = _run_regime_validation(
                Y_warm, Y_val, dates_val, regime_labels, R, Q_base, trial_alphas
            )
            rows.append({"alpha": alpha, "Val_Sharpe": sharpe})

        results = pd.DataFrame(rows).set_index("alpha")
        best_alpha = float(results["Val_Sharpe"].idxmax())
        current_alphas[regime] = best_alpha
        all_results[regime] = results

        if verbose:
            print(f"\n  {regime}:")
            print(f"  {'Alpha':>8}  {'Val Sharpe':>12}")
            for alpha, row in results.iterrows():
                marker = "  ← best" if alpha == best_alpha else ""
                print(f"  {alpha:8.3f}  {row['Val_Sharpe']:12.6f}{marker}")
            print(f"  → Selected alpha = {best_alpha:.3f}  (Q = {best_alpha:.3f} × Q_base)")

    if verbose:
        print("\n── Final Regime Alphas ──")
        for regime, alpha in current_alphas.items():
            tag = "(fixed)" if regime == "CRISIS" else ""
            print(f"  {regime:<12} alpha = {alpha:.3f}  {tag}")

    return current_alphas, all_results

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
