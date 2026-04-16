"""
optimizer.py — Mean-variance portfolio optimization with constraints.

Solves:  max  w'mu - (lambda/2) * w'Sigma*w
         s.t. sum(w) = 1
              w >= 0  (long only)
              w <= MAX_WEIGHT  (position cap)

"""

import numpy as np
import cvxpy as cp
from config import RISK_AVERSION, LONG_ONLY, FULLY_INVESTED, MAX_WEIGHT


def optimize_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float = RISK_AVERSION,
    long_only: bool = LONG_ONLY,
    max_weight: float = MAX_WEIGHT,
) -> np.ndarray:
    """
    Solve the mean-variance optimization problem.

    Parameters
    ----------
    mu    : expected return vector (n_assets,)
    sigma : covariance matrix (n_assets x n_assets)
    risk_aversion : lambda parameter (higher = more conservative)
    long_only : if True, enforce w >= 0
    max_weight : maximum weight per asset (position cap)

    Returns
    -------
    Optimal weight vector (n_assets,)
    Returns equal-weight if optimization fails.
    """
    n = len(mu)
    w = cp.Variable(n)

    # Objective: maximize expected return minus risk penalty
    ret = mu @ w
    risk = cp.quad_form(w, sigma)
    objective = cp.Maximize(ret - (risk_aversion / 2) * risk)

    # Constraints
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    constraints.append(w <= max_weight)  # position cap

    # Solve
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, warm_start=True)

        if w.value is not None and problem.status == "optimal":
            weights = w.value
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
            return weights
    except Exception:
        pass

    # Fallback: equal weight
    return np.ones(n) / n