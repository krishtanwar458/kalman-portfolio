"""
plots.py — Generate all figures for the paper.

Usage:
    from plots import (
        plot_cumulative_returns,
        plot_drawdowns,
        plot_weights_over_time,
        plot_filtered_vs_rolling_mu,
    )
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import PLOTS_DIR, REGIMES

# Style
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_cumulative_returns(results: list[dict], title: str = "Cumulative Returns"):
    """Plot cumulative wealth curves for all strategies."""
    fig, ax = plt.subplots()

    for r in results:
        ax.plot(r["cumulative"], label=r["name"], linewidth=1.5)

    # Shade regime periods
    colors = ["#FAEEDA","#FCEBEB","#E1F5EE","#EEEDFE","#E6F1FB"]
    for i, (regime_name, (start, end)) in enumerate(REGIMES.items()):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=colors[i % len(colors)], label=regime_name)

    ax.set_title(title, pad=12)
    ax.set_ylabel("Cumulative Wealth ($1 invested)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(pad=1.5)
    fig.savefig(f"{PLOTS_DIR}/cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {PLOTS_DIR}/cumulative_returns.png")


def plot_drawdowns(results: list[dict], title: str = "Drawdowns"):
    """Plot drawdown curves for all strategies."""
    fig, ax = plt.subplots()

    for r in results:
        cum = r["cumulative"]
        drawdown = (cum - cum.cummax()) / cum.cummax()
        ax.plot(drawdown, label=r["name"], linewidth=1.2)

    ax.set_title(title, pad=12)
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(pad=1.5)
    fig.savefig(f"{PLOTS_DIR}/drawdowns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {PLOTS_DIR}/drawdowns.png")


def plot_weights_over_time(result: dict, title: str = None):
    """Plot stacked area chart of portfolio weights for one strategy."""
    weights = result["weights_history"]
    if weights.empty:
        return

    name = result["name"]
    fig, ax = plt.subplots()
    ax.stackplot(weights.index, weights.values.T, labels=weights.columns, alpha=0.8)
    ax.set_title(title or f"Portfolio Weights — {name}", pad=12)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout(pad=1.5)

    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(f"{PLOTS_DIR}/weights_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {PLOTS_DIR}/weights_{safe_name}.png")


def plot_filtered_vs_rolling_mu(
    filtered_mu: pd.DataFrame,
    returns: pd.DataFrame,
    asset: str = "SPY",
    window: int = 60,
    smooth: int = 21,
):
    """
    Compare Kalman-filtered expected return vs rolling mean for one asset.
    Shows:
      - Rolling 60-day mean (blue) — the naive benchmark
      - Raw Kalman daily estimate (light orange, transparent) — shows reactivity
      - Smoothed Kalman trend (dark orange) — shows regime adaptation
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    rolling_mu = returns[asset].rolling(window).mean() * 252
    kf_mu      = filtered_mu[asset] * 252
    kf_smooth  = kf_mu.rolling(smooth).mean()

    # Raw Kalman in background — shows it's responsive but noisy
    ax.plot(kf_mu, color="orange", linewidth=0.6, alpha=0.25, label="_nolegend_")

    # Smoothed Kalman — the key line
    ax.plot(kf_smooth, color="orange", linewidth=1.8,
            label=f"Kalman estimate ({smooth}-day smoothed)")

    # Rolling mean — the benchmark
    ax.plot(rolling_mu, color="steelblue", linewidth=1.5, alpha=0.85,
            label=f"Rolling {window}-day mean")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title(f"Expected Return Estimate — {asset}", pad=12)
    ax.set_ylabel("Annualized Expected Return")
    ax.legend(fontsize=9)
    fig.tight_layout(pad=1.5)
    fig.savefig(f"{PLOTS_DIR}/filtered_vs_rolling_{asset}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {PLOTS_DIR}/filtered_vs_rolling_{asset}.png")