from data_loader import load_prices, compute_returns, split_data
from regime_detector import classify_regimes, regime_summary, compute_realized_volatility
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

prices = load_prices()
returns = compute_returns(prices, method="simple")
train, test = split_data(returns)

regime_labels = classify_regimes(returns, window=21, crisis_threshold=2.0, reference_returns=train)
oos_regimes = regime_labels.reindex(test.index)

print("=== OOS Regime Breakdown ===")
print(regime_summary(oos_regimes))

# Plot
REGIME_COLORS = {
    "LOW_VOL":  "#66BB6A",
    "MED_VOL":  "#FFA726",
    "HIGH_VOL": "#EF5350",
    "CRISIS":   "#7B1FA2",
}

vol = compute_realized_volatility(returns, window=21)
oos_vol = vol.reindex(test.index).dropna()
oos_reg = oos_regimes.reindex(oos_vol.index)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})

# Top: vol line with regime shading
ax = axes[0]
prev_regime = None
start_date = None
for date, regime in oos_reg.items():
    if regime != prev_regime:
        if prev_regime is not None:
            ax.axvspan(start_date, date, alpha=0.18,
                       color=REGIME_COLORS.get(prev_regime, "#ccc"), linewidth=0)
        start_date = date
        prev_regime = regime
if prev_regime is not None:
    ax.axvspan(start_date, oos_reg.index[-1], alpha=0.18,
               color=REGIME_COLORS.get(prev_regime, "#ccc"), linewidth=0)

ax.plot(oos_vol.index, oos_vol.values, color="#1565C0", linewidth=1.5, label="Realized Vol (annualized)")

vol_mean = vol.reindex(train.index).mean()
vol_std  = vol.reindex(train.index).std()
ax.axhline(vol_mean, color="grey", linestyle="--", linewidth=0.8, label=f"Train mean ({vol_mean:.1%})")
ax.axhline(vol_mean + 2*vol_std, color="#7B1FA2", linestyle=":", linewidth=1.0,
           label=f"Crisis threshold ({vol_mean + 2*vol_std:.1%})")

patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5, label=r)
           for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"] if r in oos_reg.values]
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles + patches, fontsize=9, loc="upper right", ncol=2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.set_ylabel("Annualized Realized Volatility", fontsize=11)
ax.set_title("OOS Period (Jan 2025 – Mar 2026): Realized Volatility and Regime Classification",
             fontsize=13, fontweight="bold", pad=12)
ax.grid(alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Bottom: regime bar chart (day counts)
ax2 = axes[1]
summary = regime_summary(oos_reg)
regime_order = [r for r in ["LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"] if r in summary.index]
counts = [summary.loc[r, "Days"] for r in regime_order]
colors = [REGIME_COLORS[r] for r in regime_order]
bars = ax2.bar(regime_order, counts, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)

ax2.set_ylim(0, max(counts) * 1.35)  
for bar, count, regime in zip(bars, counts, regime_order):
    pct = summary.loc[regime, "Pct (%)"]
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.0,
             f"{count}d\n({pct}%)", ha="center", va="bottom", fontsize=10, fontweight="bold",
             color=REGIME_COLORS[regime])

ax2.set_ylabel("Trading Days", fontsize=11)
ax2.set_title("Regime Day Count — OOS Period", fontsize=12, fontweight="bold", pad=20)
ax2.grid(axis="y", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(pad=2.0)
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/oos_regime_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: plots/oos_regime_breakdown.png")