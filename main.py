"""
main.py — Run the full Kalman Filter portfolio optimization pipeline.

This is the script you run to produce all results and figures.
    python main.py

It will:
    1. Download ETF data
    2. Split into train/test
    3. Build Kalman Filter from training data
    4. Run backtests for all strategies
    5. Compute performance metrics
    6. Generate all paper figures
    7. Print comparison tables
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import TICKERS, REGIMES, RESULTS_DIR, PLOTS_DIR, Q_SCALE
from data_loader import load_prices, compute_returns, split_data
from kalman_filter import build_filter_from_training
from backtest import run_backtest
from benchmarks import (
    equal_weight_strategy,
    make_rolling_mv_strategy,
    make_static_mv_strategy,
    make_kalman_strategy,
)
from evaluation import compare_strategies, regime_analysis, print_metrics
from plots import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_weights_over_time,
    plot_filtered_vs_rolling_mu,
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ─── Step 1: Load Data ────────────────────────────────────────
    print("\n=== STEP 1: Loading Data ===")
    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    # ─── Step 2: Build Kalman Filter ──────────────────────────────
    print("\n=== STEP 2: Building Kalman Filter ===")
    kf = build_filter_from_training(train, q_scale=Q_SCALE)
    print(f"  Filter initialized with Q_scale={Q_SCALE}")
    print(f"  Initial mu (annualized): {(kf.mu_hat * 252).round(4)}")

    # Run filter on full dataset for the comparison plot
    kf_for_plot = build_filter_from_training(train, q_scale=Q_SCALE)
    filtered_mu = kf_for_plot.filter_returns(returns)

    # ─── Step 3: Run Backtests ────────────────────────────────────
    print("\n=== STEP 3: Running Backtests ===")

    # Define strategies
    strategies = [
        ("Equal Weight", equal_weight_strategy),
        ("Rolling MV", make_rolling_mv_strategy()),
        ("Static MV", make_static_mv_strategy(train)),
        ("Kalman MV", make_kalman_strategy(train, q_scale=Q_SCALE)),
    ]

    # Run each strategy on the FULL period (train + test)
    results_full = []
    for name, strat_func in strategies:
        result = run_backtest(returns, strat_func, name=name)
        results_full.append(result)

    # Also run on TEST period only for out-of-sample comparison
    results_test = []
    for name, strat_func in strategies:
        # Rebuild strategies that need training data
        if name == "Static MV":
            strat_func = make_static_mv_strategy(train)
        elif name == "Kalman MV":
            # Warm up the Kalman filter on training data first
            strat_func = make_kalman_strategy(train, q_scale=Q_SCALE)
            # Feed training data through the filter before test
            for i in range(len(train)):
                strat_func(train.index[i], returns.iloc[:len(train)])
            # Now the filter is warmed up for the test period

        result = run_backtest(test, strat_func, name=name)
        results_test.append(result)

    # ─── Step 4: Evaluate ─────────────────────────────────────────
    print("\n=== STEP 4: Performance Metrics ===")

    print("\n--- Full Period ---")
    comparison_full = compare_strategies(results_full)
    print_metrics(comparison_full)

    print("\n--- Test Period (Out-of-Sample) ---")
    comparison_test = compare_strategies(results_test)
    print_metrics(comparison_test)

    # Regime analysis
    print("\n--- Regime Analysis (Sharpe Ratio) ---")
    regime_df = regime_analysis(results_full, REGIMES)
    print(regime_df.to_string())

    # Save tables
    comparison_full.to_csv(f"{RESULTS_DIR}/metrics_full.csv")
    comparison_test.to_csv(f"{RESULTS_DIR}/metrics_test.csv")
    regime_df.to_csv(f"{RESULTS_DIR}/regime_analysis.csv")
    print(f"\n  Tables saved to {RESULTS_DIR}/")

    # ─── Step 5: Generate Plots ───────────────────────────────────
    print("\n=== STEP 5: Generating Plots ===")

    plot_cumulative_returns(results_full, "Cumulative Returns — Full Period")
    plot_drawdowns(results_full, "Drawdowns — Full Period")

    # Weight plots for Kalman and Rolling MV
    for r in results_full:
        if r["name"] in ["Kalman MV", "Rolling MV"]:
            plot_weights_over_time(r)

    # Filtered vs rolling expected return (key paper figure)
    plot_filtered_vs_rolling_mu(filtered_mu, returns, asset="SPY")
    plot_filtered_vs_rolling_mu(filtered_mu, returns, asset="TLT")

    print("\n=== DONE ===")
    print(f"Results in ./{RESULTS_DIR}/")
    print(f"Plots in ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
