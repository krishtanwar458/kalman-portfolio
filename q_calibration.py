"""
q_calibration.py — Walk-forward cross-validation to select Q_SCALE and regime alphas.

Generalized with use_filtered_mu / use_filtered_sigma flags so that each of the
three Kalman variants (Kalman-Mu, Kalman-Sigma, Kalman-Full) gets its Q* and
regime alphas selected against its OWN mechanism, not a shared proxy.

No test data is ever touched.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import COV_WINDOW, RETURN_WINDOW


# Grid of Q values to evaluate
Q_GRID = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]


def select_q_cv(
    train_returns: pd.DataFrame,
    q_grid: list = Q_GRID,
    val_fraction: float = 0.30,
    cov_window: int = COV_WINDOW,
    ret_window: int = RETURN_WINDOW,
    use_filtered_mu: bool = False,
    use_filtered_sigma: bool = True,
    verbose: bool = True,
    label: str = "",
) -> tuple:
    """
    Select Q_SCALE via walk-forward CV on training data.
    Criterion: annualized Sharpe of the portfolio on the validation window,
    scored using the SAME mu/sigma mechanism as the target strategy variant.
    """
    from optimizer import optimize_portfolio

    Y    = train_returns.values
    T, n = Y.shape

    split  = int(T * (1 - val_fraction))
    Y_warm = Y[:split]
    Y_val  = Y[split:]

    R   = np.cov(Y.T) + np.eye(n) * 1e-6
    mu0 = Y_warm.mean(axis=0)
    buf_len = max(cov_window, ret_window)

    rows = []
    for q in q_grid:
        Q_mat = q * np.eye(n)

        # Warmup pass — collect filtered history regardless of which mechanism is used
        P  = R.copy()
        mu = mu0.copy()
        filtered_warm = np.zeros((len(Y_warm), n))
        for t in range(len(Y_warm)):
            mu_p = mu
            P_p  = P + Q_mat
            S    = P_p + R
            K    = P_p @ np.linalg.solve(S.T, np.eye(n)).T
            mu   = mu_p + K @ (Y_warm[t] - mu_p)
            P    = (np.eye(n) - K) @ P_p
            filtered_warm[t] = mu

        # Validation pass
        mu_val = mu.copy()
        P_val  = P.copy()

        raw_buffer  = list(Y_warm[-buf_len:])
        filt_buffer = list(filtered_warm[-buf_len:])

        port_returns = []
        current_weights = np.ones(n) / n
        rebal_counter = 0

        for t in range(len(Y_val)):
            if rebal_counter % 21 == 0 and len(filt_buffer) >= buf_len:
                try:
                    if use_filtered_mu:
                        mu_t = mu_val.copy()
                    else:
                        mu_t = np.mean(raw_buffer[-ret_window:], axis=0)

                    if use_filtered_sigma:
                        sigma_t = np.cov(np.array(filt_buffer[-cov_window:]).T) + np.eye(n) * 1e-8
                    else:
                        sigma_t = np.cov(np.array(raw_buffer[-cov_window:]).T) + np.eye(n) * 1e-8

                    w = optimize_portfolio(mu_t, sigma_t)
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

            raw_buffer.append(Y_val[t])
            filt_buffer.append(mu_val.copy())

        r = np.array(port_returns)
        sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
        rows.append({"Q_SCALE": q, "Val_Sharpe": sharpe})

    results = pd.DataFrame(rows).set_index("Q_SCALE")
    best_q  = float(results["Val_Sharpe"].idxmax())

    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"\n── Q Grid Search Results{tag} (training CV, criterion = Sharpe) ──")
        print(f"{'Q_SCALE':>12}  {'Val Sharpe':>12}")
        for q, row in results.iterrows():
            marker = "  ← best" if q == best_q else ""
            print(f"  {q:12.2e}  {row['Val_Sharpe']:12.6f}{marker}")
        print(f"\n  Selected Q_SCALE{tag} = {best_q:.4e}")

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
    cov_window: int = COV_WINDOW,
    ret_window: int = RETURN_WINDOW,
    use_filtered_mu: bool = False,
    use_filtered_sigma: bool = True,
) -> float:
    """
    Run one validation pass with a given set of regime alphas, scored using
    the same mu/sigma mechanism as the target strategy variant.
    CRISIS always uses 1.0 (base Q).
    """
    from optimizer import optimize_portfolio

    n = Y_warm.shape[1]
    buf_len = max(cov_window, ret_window)

    Q_map = {regime: regime_alphas.get(regime, 1.0) * Q_base
             for regime in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]}

    # Warmup pass — base Q throughout, collect filtered history
    P = R.copy()
    mu = Y_warm.mean(axis=0)
    filtered_warm = np.zeros((len(Y_warm), n))
    for t in range(len(Y_warm)):
        mu_p = mu
        P_p = P + Q_base
        S = P_p + R
        K = P_p @ np.linalg.solve(S.T, np.eye(n)).T
        mu = mu_p + K @ (Y_warm[t] - mu_p)
        P = (np.eye(n) - K) @ P_p
        filtered_warm[t] = mu

    # Validation pass — regime-switching Q
    mu_val = mu.copy()
    P_val = P.copy()
    raw_buffer  = list(Y_warm[-buf_len:])
    filt_buffer = list(filtered_warm[-buf_len:])
    port_returns = []
    current_weights = np.ones(n) / n
    rebal_counter = 0

    for t in range(len(Y_val)):
        date_t = dates_val[t]
        regime_t = regime_labels.get(date_t, "MED_VOL") if hasattr(regime_labels, 'get') \
                   else (regime_labels.loc[date_t] if date_t in regime_labels.index else "MED_VOL")
        Q_t = Q_map.get(regime_t, Q_base)

        if rebal_counter % 21 == 0 and len(filt_buffer) >= buf_len:
            try:
                if use_filtered_mu:
                    mu_t = mu_val.copy()
                else:
                    mu_t = np.mean(raw_buffer[-ret_window:], axis=0)

                if use_filtered_sigma:
                    sigma_t = np.cov(np.array(filt_buffer[-cov_window:]).T) + np.eye(n) * 1e-8
                else:
                    sigma_t = np.cov(np.array(raw_buffer[-cov_window:]).T) + np.eye(n) * 1e-8

                w = optimize_portfolio(mu_t, sigma_t)
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

        raw_buffer.append(Y_val[t])
        filt_buffer.append(mu_val.copy())

    r = np.array(port_returns)
    return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def calibrate_regime_alphas(
    train_returns: pd.DataFrame,
    regime_labels: pd.Series,
    q_best: float,
    alpha_grid: list = ALPHA_GRID,
    val_fraction: float = 0.30,
    use_filtered_mu: bool = False,
    use_filtered_sigma: bool = True,
    verbose: bool = True,
    label: str = "",
) -> tuple:
    """
    Calibrate per-regime Q attenuation factors for LOW_VOL, MED_VOL, HIGH_VOL.
    CRISIS is always fixed at base Q (alpha = 1.0) — too few days to calibrate.
    Greedy sequential search; scoring mechanism set by use_filtered_mu/use_filtered_sigma.
    """
    Y = train_returns.values
    T, n = Y.shape

    split = int(T * (1 - val_fraction))
    Y_warm = Y[:split]
    Y_val = Y[split:]
    dates_val = train_returns.index[split:]

    R = np.cov(Y.T) + np.eye(n) * 1e-6
    Q_base = q_best * np.eye(n)

    current_alphas = {r: 1.0 for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]}
    all_results = {}

    if verbose:
        tag = f" [{label}]" if label else ""
        print(f"\n── Per-Regime Alpha Calibration{tag} (greedy sequential, criterion = Val Sharpe) ──")

    for regime in REGIMES_TO_TUNE:
        rows = []
        for alpha in alpha_grid:
            trial_alphas = {**current_alphas, regime: alpha}
            sharpe = _run_regime_validation(
                Y_warm, Y_val, dates_val, regime_labels, R, Q_base, trial_alphas,
                use_filtered_mu=use_filtered_mu, use_filtered_sigma=use_filtered_sigma,
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
        tag = f" [{label}]" if label else ""
        print(f"\n── Final Regime Alphas{tag} ──")
        for regime, alpha in current_alphas.items():
            ctag = "(fixed)" if regime == "CRISIS" else ""
            print(f"  {regime:<12} alpha = {alpha:.3f}  {ctag}")

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