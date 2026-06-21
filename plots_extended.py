import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from config import PLOTS_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)

# Color palette — single source of truth, imported by plots.py and
# cost_analysis.py too, so every figure in the paper uses the same colors
# for the same strategy.
COLORS = {
    "Equal Weight":     "#4CAF50",   # green
    "Rolling MV":       "#FF5722",   # red-orange
    "Static MV":        "#9E9E9E",   # grey
    "Ledoit-Wolf MV":   "#00BCD4",   # cyan
    "Kalman-Mu MV":     "#1565C0",   # deep blue     — mean-isolation
    "Kalman-Sigma MV":  "#8E24AA",   # purple        — covariance-isolation
    "Kalman-Full MV":   "#D81B60",   # magenta/pink  — both adaptive
}

REGIME_COLORS = {
    "LOW_VOL":  "#66BB6A",   # green
    "MED_VOL":  "#FFA726",   # amber
    "HIGH_VOL": "#EF5350",   # red
    "CRISIS":   "#7B1FA2",   # purple
}

REGIME_ORDER = ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]


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

    Includes EVERY strategy column present in regime_sharpe_df — strategies
    not in COLORS get a safe fallback color rather than being silently
    dropped from the plot (this is what caused the Kalman bars to vanish
    after the 3-variant rename: the old version filtered to `c in COLORS`).
    """
    plot_df = regime_sharpe_df.drop(columns=["N Days"], errors="ignore")
    strategies = list(plot_df.columns)
    regimes = plot_df.index.tolist()

    n_strategies = len(strategies)
    n_regimes = len(regimes)
    bar_width = 0.7 / n_strategies
    x = np.arange(n_regimes)

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, strat in enumerate(strategies):
        offset = (i - n_strategies / 2 + 0.5) * bar_width
        values = plot_df[strat].values
        color = COLORS.get(strat, "#607D8B")
        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=strat,
            color=color,
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
                    fontsize=7,
                    color=color,
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
    ax.legend(fontsize=9, loc="upper right", ncol=2)
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

def plot_kalman_advantage_by_regime(advantage_df: pd.DataFrame, outpath: str = None):
    """
    Plot Kalman variant Sharpe advantage over Rolling MV by regime.

    Restores the original per-regime-colored, value-annotated bar style
    (this had been replaced with a generic plt.bar() call when multi-variant
    support was patched in, which is why it lost its styling).

    Supports both:
      - a single-variant advantage table (one '... Advantage' column) —
        bars colored by REGIME (matching the original figure), variant name
        pulled from the column and used in the title/axis/filename.
      - a combined table with multiple '... Advantage' columns — grouped
        bars colored by STRATEGY (using the shared COLORS dict) instead,
        since per-regime coloring doesn't disambiguate multiple variants.
    """
    df = advantage_df.copy()

    # Regime can arrive either as a column or as the DataFrame's index
    # (the latter happens upstream in regime_detector.py; to_csv() then
    # writes it out as a named column, which is why the CSV looks fine
    # even when the in-memory object passed here wasn't) -- handle both.
    if df.index.name in ("Regime", "regime"):
        df = df.reset_index()

    if "Regime" in df.columns:
        regime_col = "Regime"
    elif "regime" in df.columns:
        regime_col = "regime"
    else:
        regime_col = df.columns[0]

    advantage_cols = [c for c in df.columns if "Advantage" in c]
    if not advantage_cols:
        raise ValueError(f"No advantage columns found. Available columns: {list(df.columns)}")

    # Sort into the canonical regime order when possible
    if set(df[regime_col].astype(str)) <= set(REGIME_ORDER):
        df["_order"] = df[regime_col].astype(str).map(REGIME_ORDER.index)
        df = df.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    regimes = df[regime_col].astype(str).values
    x = np.arange(len(regimes))
    n_days_col = "N Days" if "N Days" in df.columns else None

    xlabels = []
    for i, regime in enumerate(regimes):
        n_days = int(df[n_days_col].iloc[i]) if n_days_col else ""
        xlabels.append(f"{regime}\n(n={n_days}d)")

    if len(advantage_cols) == 1:
        col = advantage_cols[0]
        variant_name = col.replace(" Advantage", "")
        values = df[col].values

        # Label offset/threshold scale with the data's own range instead of
        # a fixed constant -- a fixed 0.012 offset is fine for Kalman-Full's
        # ~0.05-0.7 range but is *larger than the entire data range* for
        # Kalman-Sigma's ~0.00-0.01 range, pushing labels far outside the
        # visible axes (this was the "broken" plot).
        val_range = float(np.max(values) - np.min(values)) if len(values) else 0.0
        if val_range < 1e-9:
            val_range = max(abs(float(np.max(values))), 1e-3)
        label_offset = val_range * 0.06
        inside_threshold = val_range * 0.25

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bar_colors = [REGIME_COLORS.get(r, "#9E9E9E") for r in regimes]
        bars = ax.bar(x, values, color=bar_colors, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            inside = abs(val) > inside_threshold
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val / 2 if inside else val + (label_offset if val >= 0 else -label_offset),
                f"{val:+.3f}",
                ha="center",
                va="center" if inside else ("bottom" if val >= 0 else "top"),
                fontsize=10,
                fontweight="bold",
                color="white" if inside else "black",
            )

        # Headroom so labels never sit at/past the axes edge regardless of scale
        ymin, ymax = ax.get_ylim()
        pad = val_range * 0.18
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_xlabel("Market Regime (by realized volatility)", fontsize=11)
        ax.set_ylabel(f"Sharpe Advantage ({variant_name} \u2212 Rolling MV)", fontsize=11)
        ax.set_title(f"{variant_name} Sharpe Advantage over Rolling MV by Regime",
                     fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if outpath is None:
            safe_name = variant_name.replace(" ", "_").replace("-", "_")
            outpath = os.path.join(PLOTS_DIR, f"{safe_name}_advantage_by_regime.png")

    else:
        fig, ax = plt.subplots(figsize=(12, 6.5))
        width = 0.8 / len(advantage_cols)

        for i, col in enumerate(advantage_cols):
            variant_name = col.replace(" Advantage", "")
            offset = (i - (len(advantage_cols) - 1) / 2) * width
            values = df[col].values
            color = COLORS.get(variant_name, "#607D8B")
            bars = ax.bar(x + offset, values, width=width, label=variant_name,
                          color=color, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                       val + (0.01 if val >= 0 else -0.01),
                       f"{val:+.2f}", ha="center",
                       va="bottom" if val >= 0 else "top",
                       fontsize=7, color=color, fontweight="bold")

        ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_xlabel("Market Regime (by realized volatility)", fontsize=11)
        ax.set_ylabel("Sharpe Advantage (vs Rolling MV)", fontsize=11)
        ax.set_title("Kalman Variant Sharpe Advantage over Rolling MV by Regime",
                     fontsize=13, fontweight="bold", pad=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if outpath is None:
            outpath = os.path.join(PLOTS_DIR, "kalman_variants_advantage_by_regime.png")

    plt.tight_layout(pad=1.5)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


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
               for r in REGIME_ORDER if r in reg_plot.values]
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
    variant_name: str = "Kalman-Mu MV",
    title: str = None,
    save: bool = True,
):
    """
    Rolling RMSE comparison between a Kalman variant's filtered forecast and
    rolling-window forecasts, for one asset.

    variant_name selects the color from COLORS (was previously hardcoded to
    the literal string "Kalman MV", which no longer exists as a key now that
    there are three variants) and labels the legend/title accordingly.
    """
    if asset not in errors_kalman.columns:
        asset = errors_kalman.columns[0]
        print(f"  [Warning] Requested asset not found, using {asset}")

    if title is None:
        title = f"Rolling 60-Day Forecast RMSE: {variant_name} vs Rolling Mean ({asset})"

    kalman_color = COLORS.get(variant_name, "#1E88E5")

    window = 60
    rmse_kf = errors_kalman[asset].pow(2).rolling(window).mean().pow(0.5)
    rmse_roll = errors_rolling[asset].pow(2).rolling(window).mean().pow(0.5)

    common = rmse_kf.dropna().index.intersection(rmse_roll.dropna().index)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(common, rmse_kf.loc[common], color=kalman_color, linewidth=1.2,
            alpha=0.9, label=f"{variant_name} RMSE")
    ax.plot(common, rmse_roll.loc[common], color=COLORS["Rolling MV"], linewidth=1.2,
            alpha=0.9, label="Rolling Mean RMSE", linestyle="--")

    diff = rmse_kf.loc[common] - rmse_roll.loc[common]
    ax.fill_between(common, 0, 1, where=diff < 0,
                    transform=ax.get_xaxis_transform(),
                    alpha=0.08, color=kalman_color,
                    label=f"{variant_name} lower error")

    ax.set_ylabel("Rolling RMSE (60-day)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.5)
    if save:
        safe_name = variant_name.replace(" ", "_").replace("-", "_")
        path = os.path.join(PLOTS_DIR, f"forecast_error_{safe_name}_{asset}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()