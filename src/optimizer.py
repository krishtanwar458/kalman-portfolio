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
    w_prev: np.ndarray = None,
    turnover_gamma: float = 0.0,
) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)

    ret = mu @ w
    risk = cp.quad_form(w, sigma)
    objective_expr = ret - (risk_aversion / 2) * risk

    # Turnover penalty: penalise L1 deviation from previous weights
    if w_prev is not None and turnover_gamma > 0.0:
        turnover = cp.norm1(w - w_prev)
        objective_expr = objective_expr - turnover_gamma * turnover

    objective = cp.Maximize(objective_expr)

    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    constraints.append(w <= max_weight)

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

    return np.ones(n) / n