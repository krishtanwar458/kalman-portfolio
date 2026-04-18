import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from config import PLOTS_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)

# Color palette — consistent across all plots
COLORS = {
    "Kalman MV":  "#2196F3",   # blue
    "Rolling MV": "#FF5722",   # red-orange
    "Static MV":  "#9E9E9E",   # grey
    "Equal Weight": "#4CAF50", # green
}

REGIME_COLORS = {
    "LOW_VOL":  "#66BB6A",   # green
    "MED_VOL":  "#FFA726",   # amber
    "HIGH_VOL": "#EF5350",   # red
    "CRISIS":   "#7B1FA2",   # purple
}


# ─────────────────────────────────────────────────────────────
# Figure 1: Sharpe Ratio Confidence Intervals
# ─────────────────────────────────────────────────────────────

def plot_sharpe_confidence_intervals(
    sharpe_ci_df: pd.DataFrame,
    title: str = "Sharpe Ratio Estimates with 95% Block Bootstrap CIs",
    save: bool = True,
):
    """
    Forest plot of Sharpe ratios with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = sharpe_ci_df.index.tolist()
    y_pos = np.arange(len(strategies))

    for i, strat in enumerate(strategies):
        row = sharpe_ci_df.loc[strat]
        color = COLORS.get(strat, "#607D8B")

        ax.errorbar(
            x=row["Sharpe"],
            y=i,
            xerr=[[row["Sharpe"] - row["95% CI Lower"]],
                  [row["95% CI Upper"] - row["Sharpe"]]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=6,
            capthick=2,
            markersize=10,
            label=strat,
            zorder=3,
        )

        # Bottom strategy: label above point to avoid x-axis overlap
        # All others: label below point
        if i == 0:
            y_text, va = i + 0.18, "bottom"
        else:
            y_text, va = i - 0.18, "top"

        ax.text(
            row["Sharpe"],
            y_text,
            f"{row['Sharpe']:.3f}",
            ha="center",
            va=va,
            fontsize=9,
            color=color,
        )

    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="Sharpe = 0")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies, fontsize=11)
    ax.set_xlabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, "sharpe_confidence_intervals.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Figure 2: Regime-conditional Sharpe bar chart
# ─────────────────────────────────────────────────────────────

def plot_regime_performance(
    regime_sharpe_df: pd.DataFrame,
    title: str = "Sharpe Ratio by Market Regime",
    save: bool = True,
):
    """
    Grouped bar chart: x-axis = regimes, bars = strategies.
    """
    plot_df = regime_sharpe_df.drop(columns=["N Days"], errors="ignore")
    strategies = [c for c in plot_df.columns if c in COLORS]
    regimes = plot_df.index.tolist()

    n_strategies = len(strategies)
    n_regimes = len(regimes)
    bar_width = 0.7 / n_strategies
    x = np.arange(n_regimes)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, strat in enumerate(strategies):
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        values = plot_df[strat].values
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=strat,
            color=COLORS.get(strat, "#607D8B"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ypos = bar.get_height() + 0.03
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    ypos,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=COLORS.get(strat, "#333"),
                )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)

    xlabels = []
    for regime in regimes:
        n_days = int(regime_sharpe_df.loc[regime, "N Days"]) if "N Days" in regime_sharpe_df.columns else ""
        xlabels.append(f"{regime}\n(n={n_days}d)")
    ax.set_xticklabels(xlabels, fontsize=11)

    # Add headroom above tallest bar so labels don't collide with title
    current_top = ax.get_ylim()[1]
    ax.set_ylim(top=current_top * 1.18)

    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, "regime_sharpe_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Figure 3: Kalman advantage by regime
# ─────────────────────────────────────────────────────────────

def plot_kalman_advantage_by_regime(
    advantage_df: pd.DataFrame,
    title: str = "Kalman MV Sharpe Advantage over Rolling MV by Regime",
    save: bool = True,
):
    """
    Bar chart showing Kalman MV Sharpe MINUS Rolling MV Sharpe per regime.
    """
    regimes = advantage_df.index.tolist()
    advantage = advantage_df["Kalman Advantage"].values

    colors = [REGIME_COLORS.get(r, "#607D8B") for r in regimes]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(regimes, advantage, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, advantage):
        if not np.isnan(val):
            if abs(val) > 0.02:
                # Large enough — center label inside bar with white text
                ypos = bar.get_height() / 2
                va = "center"
                text_color = "white"
            elif val >= 0:
                # Small positive — label above bar
                ypos = bar.get_height() + 0.008
                va = "bottom"
                text_color = "black"
            else:
                # Small negative — label below bar
                ypos = bar.get_height() - 0.008
                va = "top"
                text_color = "black"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:+.3f}",
                ha="center",
                va=va,
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_ylabel("Sharpe Advantage (Kalman MV − Rolling MV)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Market Regime (by realized volatility)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add headroom so bar labels don't clip into title
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.20)

    # Add regime day counts at the bottom
    if "N Days" in advantage_df.columns:
        for i, (regime, row) in enumerate(advantage_df.iterrows()):
            ax.text(i, ymin * 0.85, f"n={int(row['N Days'])}d",
                    ha="center", fontsize=8, color="grey")

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, "kalman_advantage_by_regime.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Figure 4: Realized volatility timeline with regime shading
# ─────────────────────────────────────────────────────────────

def plot_realized_volatility_with_regimes(
    vol: pd.Series,
    regimes: pd.Series,
    title: str = "Realized Market Volatility with Regime Classification",
    save: bool = True,
):
    """
    Line chart of realized vol with background shading by regime.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    common_idx = vol.index.intersection(regimes.index)
    vol_plot = vol.loc[common_idx]
    reg_plot = regimes.loc[common_idx]

    prev_regime = None
    start_date = None
    for date, regime in reg_plot.items():
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                color = REGIME_COLORS.get(prev_regime, "#ccc")
                ax.axvspan(start_date, date, alpha=0.12, color=color, linewidth=0)
            start_date = date
            prev_regime = regime

    if prev_regime is not None and start_date is not None:
        ax.axvspan(start_date, reg_plot.index[-1], alpha=0.12,
                   color=REGIME_COLORS.get(prev_regime, "#ccc"), linewidth=0)

    ax.plot(vol_plot.index, vol_plot.values, color="#1565C0", linewidth=1.2,
            alpha=0.85, label="Realized Vol (annualized)")

    vol_mean = vol_plot.mean()
    vol_std = vol_plot.std()
    ax.axhline(vol_mean, color="grey", linestyle="--", linewidth=0.8,
               alpha=0.6, label=f"Mean ({vol_mean:.1%})")
    ax.axhline(vol_mean + 2 * vol_std, color="#7B1FA2", linestyle=":", linewidth=1,
               alpha=0.7, label=f"Crisis threshold (+2σ = {vol_mean + 2*vol_std:.1%})")

    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r)
               for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"] if r in reg_plot.values]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + patches, fontsize=9, loc="upper right", ncol=2)

    ax.set_ylabel("Annualized Realized Volatility", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, "realized_volatility_regimes.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────
# Figure 5: Forecast error comparison (for DM test section)
# ─────────────────────────────────────────────────────────────

def plot_forecast_error_comparison(
    errors_kalman: pd.DataFrame,
    errors_rolling: pd.DataFrame,
    asset: str = "SPY",
    title: str = None,
    save: bool = True,
):
    """
    Rolling RMSE comparison between Kalman and rolling-window forecasts.
    """
    if asset not in errors_kalman.columns:
        asset = errors_kalman.columns[0]
        print(f"  [Warning] Requested asset not found, using {asset}")

    if title is None:
        title = f"Rolling 60-Day Forecast RMSE: Kalman vs Rolling Mean ({asset})"

    window = 60
    rmse_kf = errors_kalman[asset].pow(2).rolling(window).mean().pow(0.5)
    rmse_roll = errors_rolling[asset].pow(2).rolling(window).mean().pow(0.5)

    common = rmse_kf.dropna().index.intersection(rmse_roll.dropna().index)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(common, rmse_kf.loc[common], color=COLORS["Kalman MV"], linewidth=1.2,
            alpha=0.9, label="Kalman Filter RMSE")
    ax.plot(common, rmse_roll.loc[common], color=COLORS["Rolling MV"], linewidth=1.2,
            alpha=0.9, label="Rolling Mean RMSE", linestyle="--")

    diff = rmse_kf.loc[common] - rmse_roll.loc[common]
    ax.fill_between(common, 0, 1, where=diff < 0,
                    transform=ax.get_xaxis_transform(),
                    alpha=0.08, color=COLORS["Kalman MV"],
                    label="Kalman lower error")

    ax.set_ylabel("Rolling RMSE (60-day)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.5)
    if save:
        path = os.path.join(PLOTS_DIR, f"forecast_error_{asset}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()