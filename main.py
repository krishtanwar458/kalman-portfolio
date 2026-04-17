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

    # Step 1: Load Data
    print("\n=== STEP 1: Loading Data ===")
    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    # Step 2: Calibrate Q via walk-forward CV on training data only
    print("\n=== STEP 2: Calibrating Kalman Filter (Walk-Forward CV) ===")

    from q_calibration import select_q_cv, plot_q_selection

    q_best, q_results = select_q_cv(train, verbose=True)
    plot_q_selection(q_results, q_best, save_path=f"{PLOTS_DIR}/q_selection.png")

    print(f"\n  CV-selected Q_SCALE = {q_best:.4e}")

    # Sync config so cost_analysis and any other modules pick up q_best
    import config
    config.Q_SCALE = q_best

    kf = build_filter_from_training(train, q_scale=q_best)
    kf_for_plot = build_filter_from_training(train, q_scale=q_best)
    filtered_mu = kf_for_plot.filter_returns(returns)

    # Step 3: Run Backtests
    print("\n=== STEP 3: Running Backtests ===")

    # Define strategies
    strategies = [
        ("Equal Weight", equal_weight_strategy),
        ("Rolling MV",   make_rolling_mv_strategy()),
        ("Static MV",    make_static_mv_strategy(train)),
        ("Kalman MV",    make_kalman_strategy(train, q_scale=q_best)),
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
            strat_func = make_kalman_strategy(train, q_scale=q_best)
            # Feed training data through the filter before test
            for i in range(len(train)):
                strat_func(train.index[i], returns.iloc[:len(train)])
            # Now the filter is warmed up for the test period

        result = run_backtest(test, strat_func, name=name)
        results_test.append(result)

    # Step 4: Evaluate
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

    # Step 5: Generate Plots
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

    # Step 6: Statistical Significance Tests
    print("\n=== STEP 6: Statistical Significance Tests ===")

    from statistical_tests import run_all_tests

    stat_results = run_all_tests(
        results=results_full,
        filtered_mu=filtered_mu,
        returns=returns,
        rolling_window=60,
        n_bootstrap=1000,
    )

    # Save Sharpe CI table to CSV
    stat_results["sharpe_cis"].to_csv(f"{RESULTS_DIR}/sharpe_confidence_intervals.csv")
    print(f"\n  Sharpe CI table saved to {RESULTS_DIR}/sharpe_confidence_intervals.csv")

    # DM results
    dm_df = pd.DataFrame(stat_results["dm_results"]).T
    dm_df.to_csv(f"{RESULTS_DIR}/diebold_mariano_results.csv")
    print(f"  DM test results saved to {RESULTS_DIR}/diebold_mariano_results.csv")

    # Step 7: Regime Analysis
    print("\n=== STEP 7: Regime-Conditional Performance Analysis ===")

    from regime_detector import (
        compute_realized_volatility,
        classify_regimes,
        regime_summary,
        regime_sharpe_table,
        kalman_outperformance_by_regime,
    )
    from plots_extended import (
        plot_sharpe_confidence_intervals,
        plot_regime_performance,
        plot_kalman_advantage_by_regime,
        plot_realized_volatility_with_regimes,
        plot_forecast_error_comparison,
    )

    # Classify regimes using full return series
    regimes = classify_regimes(returns, window=21, crisis_threshold=2.0)

    print("\n  Regime day counts:")
    print(regime_summary(regimes).to_string())

    # Sharpe by regime
    sharpe_by_regime = regime_sharpe_table(results_full, regimes)
    print("\n  Sharpe by Regime:")
    print(sharpe_by_regime.to_string())
    sharpe_by_regime.to_csv(f"{RESULTS_DIR}/regime_sharpe_table.csv")

    # Kalman advantage by regime
    advantage = kalman_outperformance_by_regime(results_full, regimes, baseline="Rolling MV")
    print("\n  Kalman MV Advantage by Regime:")
    print(advantage.to_string())
    advantage.to_csv(f"{RESULTS_DIR}/kalman_advantage_by_regime.csv")

    # Step 8: Extended Plots
    print("\n=== STEP 8: Extended Plots ===")

    plot_sharpe_confidence_intervals(stat_results["sharpe_cis"])
    plot_regime_performance(sharpe_by_regime)
    plot_kalman_advantage_by_regime(advantage)

    vol_series = compute_realized_volatility(returns, window=21)
    plot_realized_volatility_with_regimes(vol_series, regimes)

    for asset in ["SPY", "TLT"]:
        plot_forecast_error_comparison(
            stat_results["forecast_errors"]["kalman"],
            stat_results["forecast_errors"]["rolling"],
            asset=asset,
        )

    # Step 9: Transaction Cost Sensitivity
    print("\n=== STEP 9: Transaction Cost Sensitivity ===")

    from cost_analysis import run_sensitivity, plot_sensitivity, print_sharpe_pivot

    # Full period
    df_costs_full = run_sensitivity(returns, train, period_label="Full")
    print_sharpe_pivot(df_costs_full, "Full")
    df_costs_full.to_csv(f"{RESULTS_DIR}/cost_sensitivity.csv", index=False)
    plot_sensitivity(df_costs_full, period_label="Full", filename="cost_sensitivity_full.png")

    # OOS period
    df_costs_oos = run_sensitivity(test, train, period_label="OOS")
    print_sharpe_pivot(df_costs_oos, "OOS")
    df_costs_oos.to_csv(f"{RESULTS_DIR}/cost_sensitivity_oos.csv", index=False)
    plot_sensitivity(df_costs_oos, period_label="OOS", filename="cost_sensitivity_oos.png")

    print("\n=== ALL DONE ===")
    print(f"Results: ./{RESULTS_DIR}/")
    print(f"Plots:   ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()