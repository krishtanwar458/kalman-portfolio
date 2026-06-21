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

from config import TICKERS, RESULTS_DIR, PLOTS_DIR
from data_loader import load_prices, compute_returns, split_data
from backtest import run_backtest, slice_result
from plots_extended import COLORS
from benchmarks import (
    equal_weight_strategy,
    make_rolling_mv_strategy,
    make_static_mv_strategy,
    make_kalman_strategy,
    make_ledoit_wolf_strategy,
)

COST_LEVELS_BPS = [0, 5, 10, 20]

KALMAN_VARIANT_NAMES = ["Kalman-Mu MV", "Kalman-Sigma MV", "Kalman-Full MV"]

STRATEGY_ORDER = ["Equal Weight", "Rolling MV", "Static MV", "Ledoit-Wolf MV"] + KALMAN_VARIANT_NAMES


def annualized_metrics(daily_returns: pd.Series) -> dict:
    cumulative = (1 + daily_returns).cumprod()
    total_growth = cumulative.iloc[-1]                   # = prod(1+r_t); NOT divided by
                                                           # cumulative.iloc[0], which is
                                                           # (1+r_0) and silently drops day 0
    n_years = len(daily_returns) / 252
    ann_ret = total_growth ** (1 / n_years) - 1           # CAGR — for display only
    ann_vol = daily_returns.std() * np.sqrt(252)
    # Sharpe uses arithmetic mean annualization, not CAGR -- see evaluation.py's
    # compute_metrics() for why mixing the two isn't internally consistent.
    ann_mean_simple = daily_returns.mean() * 252
    sharpe  = ann_mean_simple / ann_vol if ann_vol > 0 else 0.0
    max_dd  = ((cumulative - cumulative.cummax()) / cumulative.cummax()).min()
    return {
        "Ann. Return (%)": round(ann_ret * 100, 2),
        "Ann. Vol (%)":    round(ann_vol * 100, 2),
        "Sharpe":          round(sharpe, 3),
        "Max Drawdown (%)":round(max_dd * 100, 2),
    }


def run_sensitivity(
    full_returns: pd.DataFrame,
    eval_index: pd.DatetimeIndex,
    train: pd.DataFrame,
    period_label: str = "Full",
    regime_labels: pd.Series = None,
    calibration: dict = None,   # {variant_name: {"q_best", "regime_alphas", "use_filtered_mu", "use_filtered_sigma"}}
    turnover_gamma: float = 0.0,
) -> pd.DataFrame:
    """
    full_returns : the FULL combined train+test return series. The backtest
                   always runs continuously over this, regardless of
                   period_label, so stateful strategies (the Kalman variants)
                   never have their internal state frozen mid-warm-up or
                   cold-started at the OOS boundary.
    eval_index   : the date range to report metrics over — e.g. the full
                   index for "Full", or just the test-period index for "OOS".
                   Reporting is just a slice of one continuous run.
    calibration  : must contain one entry per Kalman variant to be tested,
                   e.g. the `calibration` dict built in main.py's Step 2.
    """
    if calibration is None:
        calibration = {}

    rows = []

    for bps in COST_LEVELS_BPS:
        strategies = [
            ("Equal Weight",   equal_weight_strategy,                          0.0),
            ("Rolling MV",     make_rolling_mv_strategy(
                                    turnover_gamma=turnover_gamma),             bps),
            ("Static MV",      make_static_mv_strategy(train),                 0.0),
            ("Ledoit-Wolf MV", make_ledoit_wolf_strategy(
                                    turnover_gamma=turnover_gamma),             bps),
        ] + [
            (name, make_kalman_strategy(
                        train,
                        q_scale=calibration[name]["q_best"],
                        regime_labels=regime_labels,
                        regime_alphas=calibration[name]["regime_alphas"],
                        turnover_gamma=turnover_gamma,
                        use_filtered_mu=calibration[name]["use_filtered_mu"],
                        use_filtered_sigma=calibration[name]["use_filtered_sigma"],
                    ), bps)
            for name in calibration
        ]

        for name, strat_func, effective_bps in strategies:
            result = run_backtest(full_returns, strat_func, name=name, cost_bps=effective_bps)
            sliced = slice_result(result, eval_index)

            m = annualized_metrics(sliced["daily_returns"])
            avg_turnover = sliced["turnover_series"].mean() if len(sliced["turnover_series"]) > 0 else 0.0
            rows.append({
                "Period":           period_label,
                "Strategy":         name,
                "Cost (bps)":       bps,
                "Avg Turnover":     round(avg_turnover, 4),
                **m,
            })

    return pd.DataFrame(rows)


def print_sharpe_pivot(df: pd.DataFrame, period_label: str = ""):
    pivot = df.pivot_table(index="Strategy", columns="Cost (bps)", values="Sharpe").reindex(STRATEGY_ORDER)
    print(f"\n── Sharpe Ratio by Strategy and Cost [{period_label}] ──")
    print(pivot.to_string())
    pivot_ret = df.pivot_table(index="Strategy", columns="Cost (bps)", values="Ann. Return (%)").reindex(STRATEGY_ORDER)
    print(f"\n── Ann. Return (%) by Strategy and Cost [{period_label}] ──")
    print(pivot_ret.to_string())


def plot_sensitivity(df: pd.DataFrame, period_label: str = "Full", filename: str = "cost_sensitivity.png"):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    markers = {
        "Equal Weight": "o", "Rolling MV": "s",
        "Static MV": "D", "Ledoit-Wolf MV": "v",
        "Kalman-Mu MV": "^", "Kalman-Sigma MV": "P",
        "Kalman-Full MV": "X",
    }

    for strat in STRATEGY_ORDER:
        sub = df[df["Strategy"] == strat]
        if sub.empty:
            continue
        color = COLORS.get(strat, "#607D8B")
        marker = markers.get(strat, "o")
        axes[0].plot(sub["Cost (bps)"], sub["Sharpe"],
                     label=strat, color=color,
                     marker=marker, linewidth=2.5, markersize=7)
        axes[1].plot(sub["Cost (bps)"], sub["Ann. Return (%)"],
                     label=strat, color=color,
                     marker=marker, linewidth=2.5, markersize=7)

    for ax, ylabel, title in zip(
        axes,
        ["Annualized Sharpe Ratio", "Annualized Return (%)"],
        ["Sharpe Ratio vs Transaction Cost", "Return vs Transaction Cost"],
    ):
        ax.set_xlabel("One-way Transaction Cost (bps)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(COST_LEVELS_BPS)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Transaction Cost Sensitivity — {period_label} Period", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/{filename}"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Standalone runner — builds its own calibration since main.py wasn't run first.
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Loading data ===")
    prices  = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    from q_calibration import select_q_cv, calibrate_regime_alphas
    from regime_detector import classify_regimes

    regime_labels = classify_regimes(returns, window=21, crisis_threshold=2.0, reference_returns=train)
    regime_labels_train = regime_labels.reindex(train.index)

    variant_flags = {
        "Kalman-Mu MV":    dict(use_filtered_mu=True,  use_filtered_sigma=False),
        "Kalman-Sigma MV": dict(use_filtered_mu=False, use_filtered_sigma=True),
        "Kalman-Full MV":  dict(use_filtered_mu=True,  use_filtered_sigma=True),
    }

    calibration = {}
    for name, flags in variant_flags.items():
        q_best_v, _ = select_q_cv(train, verbose=False, label=name, **flags)
        alphas_v, _ = calibrate_regime_alphas(train, regime_labels_train, q_best_v,
                                               verbose=False, label=name, **flags)
        calibration[name] = {"q_best": q_best_v, "regime_alphas": alphas_v, **flags}

    turnover_gamma = 0.00005

    # Full period
    print("\n=== Full Period ===")
    df_full = run_sensitivity(returns, returns.index, train, period_label="Full",
                               regime_labels=regime_labels, calibration=calibration,
                               turnover_gamma=turnover_gamma)
    print_sharpe_pivot(df_full, "Full")
    df_full.to_csv(f"{RESULTS_DIR}/cost_sensitivity.csv", index=False)
    plot_sensitivity(df_full, period_label="Full", filename="cost_sensitivity_full.png")

    # OOS test period — same continuous run, sliced to test.index
    print("\n=== OOS Test Period ===")
    df_oos = run_sensitivity(returns, test.index, train, period_label="OOS",
                              regime_labels=regime_labels, calibration=calibration,
                              turnover_gamma=turnover_gamma)
    print_sharpe_pivot(df_oos, "OOS")
    df_oos.to_csv(f"{RESULTS_DIR}/cost_sensitivity_oos.csv", index=False)
    plot_sensitivity(df_oos, period_label="OOS", filename="cost_sensitivity_oos.png")

    print("\n=== Done ===")