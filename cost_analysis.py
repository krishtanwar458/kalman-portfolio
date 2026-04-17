"""
cost_analysis.py — Transaction cost sensitivity analysis.

Reruns all strategies at 0, 5, 10, 20 bps one-way cost and produces:
  - results/cost_sensitivity.csv
  - plots/cost_sensitivity.png

Run: python cost_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from config import TICKERS, RESULTS_DIR, PLOTS_DIR
from data_loader import load_prices, compute_returns, split_data
from kalman_filter import build_filter_from_training
from backtest import run_backtest
from benchmarks import (
    equal_weight_strategy,
    make_rolling_mv_strategy,
    make_static_mv_strategy,
    make_kalman_strategy,
)

COST_LEVELS_BPS = [0, 5, 10, 20]


def annualized_metrics(daily_returns: pd.Series) -> dict:
    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cumulative  = (1 + daily_returns).cumprod()
    max_dd  = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
    return {
        "Ann. Return (%)": round(ann_ret * 100, 2),
        "Ann. Vol (%)":    round(ann_vol * 100, 2),
        "Sharpe":          round(sharpe, 3),
        "Max Drawdown (%)":round(max_dd * 100, 2),
    }


def run_sensitivity(
    returns: pd.DataFrame,
    train: pd.DataFrame,
    period_label: str = "Full",
    regime_labels: pd.Series = None,
    regime_alphas: dict = None,
) -> pd.DataFrame:
    rows = []

    for bps in COST_LEVELS_BPS:
        strategies = [
            ("Equal Weight", equal_weight_strategy,          0.0),
            ("Rolling MV",   make_rolling_mv_strategy(),     bps),
            ("Static MV",    make_static_mv_strategy(train), 0.0),
            ("Kalman MV",    make_kalman_strategy(
                                train,
                                q_scale=config.Q_SCALE,
                                regime_labels=regime_labels,
                                regime_alphas=regime_alphas,
                             ), bps),
        ]

        for name, strat_func, effective_bps in strategies:
            result = run_backtest(returns, strat_func, name=name, cost_bps=effective_bps)
            m = annualized_metrics(result["daily_returns"])
            avg_turnover = result["turnover_series"].mean() if len(result["turnover_series"]) > 0 else 0.0
            rows.append({
                "Period":           period_label,
                "Strategy":         name,
                "Cost (bps)":       bps,
                "Avg Turnover":     round(avg_turnover, 4),
                **m,
            })

    return pd.DataFrame(rows)

def print_sharpe_pivot(df: pd.DataFrame, period_label: str = ""):
    order = ["Equal Weight", "Rolling MV", "Static MV", "Kalman MV"]
    pivot = df.pivot_table(index="Strategy", columns="Cost (bps)", values="Sharpe").reindex(order)
    print(f"\n── Sharpe Ratio by Strategy and Cost [{period_label}] ──")
    print(pivot.to_string())
    pivot_ret = df.pivot_table(index="Strategy", columns="Cost (bps)", values="Ann. Return (%)").reindex(order)
    print(f"\n── Ann. Return (%) by Strategy and Cost [{period_label}] ──")
    print(pivot_ret.to_string())


def plot_sensitivity(df: pd.DataFrame, period_label: str = "Full", filename: str = "cost_sensitivity.png"):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors  = {"Equal Weight": "#2ca02c", "Rolling MV": "#ff7f0e",
                "Static MV": "#7f7f7f",   "Kalman MV":  "#1f77b4"}
    markers = {"Equal Weight": "o", "Rolling MV": "s",
                "Static MV": "D", "Kalman MV": "^"}

    for strat in ["Equal Weight", "Rolling MV", "Static MV", "Kalman MV"]:
        sub = df[df["Strategy"] == strat]
        axes[0].plot(sub["Cost (bps)"], sub["Sharpe"],
                     label=strat, color=colors[strat],
                     marker=markers[strat], linewidth=2.5, markersize=7)
        axes[1].plot(sub["Cost (bps)"], sub["Ann. Return (%)"],
                     label=strat, color=colors[strat],
                     marker=markers[strat], linewidth=2.5, markersize=7)

    for ax, ylabel, title in zip(
        axes,
        ["Annualized Sharpe Ratio", "Annualized Return (%)"],
        ["Sharpe Ratio vs Transaction Cost", "Return vs Transaction Cost"],
    ):
        ax.set_xlabel("One-way Transaction Cost (bps)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(COST_LEVELS_BPS)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Transaction Cost Sensitivity — {period_label} Period", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/{filename}"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Loading data ===")
    prices  = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    # ── Full period ──────────────────────────────────────────────
    print("\n=== Full Period ===")
    df_full = run_sensitivity(returns, train, period_label="Full")
    print_sharpe_pivot(df_full, "Full")
    df_full.to_csv(f"{RESULTS_DIR}/cost_sensitivity.csv", index=False)
    plot_sensitivity(df_full, period_label="Full", filename="cost_sensitivity_full.png")

    # ── OOS test period ──────────────────────────────────────────
    print("\n=== OOS Test Period ===")
    df_oos = run_sensitivity(test, train, period_label="OOS")
    print_sharpe_pivot(df_oos, "OOS")
    df_oos.to_csv(f"{RESULTS_DIR}/cost_sensitivity_oos.csv", index=False)
    plot_sensitivity(df_oos, period_label="OOS", filename="cost_sensitivity_oos.png")

    print("\n=== Done ===")