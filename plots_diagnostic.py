"""
plots_diagnostic.py — Diagnostic plots for Discussion section (Section 6.2).

Plot 1: Average Kalman gain over time with regime shading
Plot 2: Frobenius norm divergence between Kalman and Rolling covariance estimates

These plots illustrate the Kalman-Sigma MV variant's behaviour (the
covariance-isolation design), which is what main.py's Step 8b passes in.
Colors are imported from plots_extended.py — the single source of truth
for strategy colors across all figures in the paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from config import PLOTS_DIR
from plots_extended import COLORS

os.makedirs(PLOTS_DIR, exist_ok=True)

REGIME_COLORS = {
    "LOW_VOL":  "#66BB6A",
    "MED_VOL":  "#FFA726",
    "HIGH_VOL": "#EF5350",
    "CRISIS":   "#7B1FA2",
}


def _add_regime_shading(ax, regimes: pd.Series):
    """Add background regime shading to an axes object."""
    prev_regime = None
    start_date = None
    for date, regime in regimes.items():
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                ax.axvspan(
                    start_date, date,
                    alpha=0.20,
                    color=REGIME_COLORS.get(prev_regime, "#ccc"),
                    linewidth=0,
                )
            start_date = date
            prev_regime = regime
    if prev_regime is not None and start_date is not None:
        ax.axvspan(
            start_date, regimes.index[-1],
            alpha=0.20,
            color=REGIME_COLORS.get(prev_regime, "#ccc"),
            linewidth=0,
        )


def plot_kalman_gain(
    gain_series: pd.Series,
    regimes: pd.Series,
    title: str = "Average Kalman Gain Over Time",
    save: bool = True,
):
    """
    Plot the average Kalman gain across assets over time with regime shading.
    Higher gain = filter is placing more weight on new observations.
    Colored using Kalman-Sigma MV (the variant driving Step 8b diagnostics).
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    common_idx = gain_series.index.intersection(regimes.index)
    _add_regime_shading(ax, regimes.loc[common_idx])

    gain_smooth = gain_series.rolling(21, min_periods=1).mean()
    ax.plot(
        gain_series.index,
        gain_smooth.values,
        color=COLORS["Kalman-Sigma MV"],
        linewidth=1.2,
        alpha=0.9,
        label="Kalman Gain (21-day smoothed)",
    )

    ax.set_ylabel("Average Kalman Gain $K_t$", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r)
        for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]
        if r in regimes.values
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, fontsize=9, loc="upper right", ncol=2)

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, "kalman_gain_over_time.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


def plot_covariance_divergence(
    divergence_series: pd.Series,
    regimes: pd.Series,
    title: str = "Covariance Estimate Divergence: Kalman vs Rolling Window",
    save: bool = True,
):
    """
    Plot the Frobenius norm of (Kalman covariance - Rolling covariance)
    at each rebalancing date, with regime shading.
    Higher values = the two estimators disagree more.
    Colored using Kalman-Sigma MV (the variant driving Step 8b diagnostics).
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    common_idx = divergence_series.index.intersection(regimes.index)
    if len(common_idx) > 0:
        _add_regime_shading(ax, regimes.loc[common_idx])

    ax.bar(
        divergence_series.index,
        divergence_series.values,
        color=COLORS["Kalman-Sigma MV"],
        alpha=0.7,
        width=20,
        label="Frobenius norm $\\|\\hat{\\Sigma}^{\\mathrm{Kalman}}_t - \\hat{\\Sigma}^{\\mathrm{Rolling}}_t\\|_F$",
    )

    ax.set_ylabel("Frobenius Norm Difference", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    patches = [
        mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r)
        for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]
        if r in regimes.values
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, fontsize=9, loc="upper right", ncol=2)

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, "covariance_divergence.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()