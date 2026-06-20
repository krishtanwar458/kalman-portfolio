import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
from config import TICKERS, REGIMES, RESULTS_DIR, PLOTS_DIR, Q_SCALE, TRAIN_START, TEST_END
from data_loader import load_prices, compute_returns, split_data
from kalman_filter import build_filter_from_training
from backtest import run_backtest
from benchmarks import (
    equal_weight_strategy,
    make_rolling_mv_strategy,
    make_static_mv_strategy,
    make_kalman_strategy,
    make_ledoit_wolf_strategy,
)
from evaluation import compare_strategies, regime_analysis, print_metrics
from plots import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_weights_over_time,
    plot_filtered_vs_rolling_mu,
)


tickers = ["SPY", "QQQ", "TLT", "GLD", "EFA", "VNQ"]
download_end = (pd.Timestamp(TEST_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
prices = yf.download(tickers, start=TRAIN_START, end=download_end)["Close"]
returns = np.log(prices / prices.shift(1)).dropna()

for ticker in tickers:
    ann_ret = returns[ticker].mean() * 252
    ann_vol = returns[ticker].std() * np.sqrt(252)
    print(f"{ticker}: Return={ann_ret:.2%}, Vol={ann_vol:.2%}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Step 1: Load Data
    print("\n=== STEP 1: Loading Data ===")
    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    # Step 2: Calibrate Q + Per-Regime Alphas
    print("\n=== STEP 2: Calibrating Kalman Filter (Walk-Forward CV) ===")

    from q_calibration import select_q_cv, plot_q_selection, calibrate_regime_alphas
    from regime_detector import (
        compute_realized_volatility,
        classify_regimes,
        regime_summary,
        regime_sharpe_table,
        kalman_outperformance_by_regime,
    )

    # Calibrate base Q on training data
    q_best, q_results = select_q_cv(train, verbose=True)
    plot_q_selection(q_results, q_best, save_path=f"{PLOTS_DIR}/q_selection.png")
    print(f"\n  CV-selected Q_SCALE = {q_best:.4e}")

    # Sync config
    import config
    config.Q_SCALE = q_best

    # Compute regime labels on full returns (no lookahead — uses past vol only)
    regime_labels = classify_regimes(returns, window=21, crisis_threshold=2.0,
                                  reference_returns=train)

    # Calibrate per-regime alphas using training regime labels only
    # CRISIS is fixed at base Q (alpha=1.0) — too few days to calibrate reliably
    regime_labels_train = regime_labels.reindex(train.index)
    regime_alphas, alpha_results = calibrate_regime_alphas(
        train, regime_labels_train, q_best, verbose=True
    )
    print(f"\n  CV-selected regime alphas: {regime_alphas}")

    # Sync config
    config.Q_REGIME_ALPHAS = regime_alphas
    
    turnover_gamma = 0.00005  # starting value, tune later

    # Build filter for plotting purposes
    kf_for_plot = build_filter_from_training(train, q_scale=q_best)
    filtered_mu = kf_for_plot.filter_returns(returns)

    # Step 3: Run Backtests
    print("\n=== STEP 3: Running Backtests ===")

    strategies = [
        ("Equal Weight", equal_weight_strategy),
        ("Rolling MV",   make_rolling_mv_strategy(turnover_gamma=turnover_gamma)),
        ("Static MV",    make_static_mv_strategy(train)),
        ("Ledoit-Wolf MV", make_ledoit_wolf_strategy(turnover_gamma=turnover_gamma)),  # add this
        ("Kalman MV",    make_kalman_strategy(
                            train,
                            q_scale=q_best,
                            regime_labels=regime_labels,
                            regime_alphas=regime_alphas,
                            turnover_gamma=turnover_gamma,
                        )),
    ]

    # Full period (train + test)
    results_full = []
    for name, strat_func in strategies:
        result = run_backtest(returns, strat_func, name=name)
        results_full.append(result)

    # Test period only (out-of-sample)
    results_test = []
    for name, strat_func in strategies:
        if name == "Rolling MV":
            strat_func = make_rolling_mv_strategy(turnover_gamma=turnover_gamma)
        elif name == "Ledoit-Wolf MV":
            strat_func = make_ledoit_wolf_strategy(turnover_gamma=turnover_gamma)
        elif name == "Static MV":
            strat_func = make_static_mv_strategy(train)
        elif name == "Kalman MV":
            strat_func = make_kalman_strategy(
                train,
                q_scale=q_best,
                regime_labels=regime_labels,
                regime_alphas=regime_alphas,
                turnover_gamma=turnover_gamma,
            )
            # Warm up filter on training data before test period
            for i in range(len(train)):
                strat_func(train.index[i], returns.iloc[:len(train)])

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

    for r in results_full:
        if r["name"] in ["Kalman MV", "Rolling MV"]:
            plot_weights_over_time(r)

    plot_filtered_vs_rolling_mu(filtered_mu, returns, asset="SPY")
    plot_filtered_vs_rolling_mu(filtered_mu, returns, asset="TLT")

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

    stat_results["sharpe_cis"].to_csv(f"{RESULTS_DIR}/sharpe_confidence_intervals.csv")
    print(f"\n  Sharpe CI table saved to {RESULTS_DIR}/sharpe_confidence_intervals.csv")

    dm_df = pd.DataFrame(stat_results["dm_results"]).T
    dm_df.to_csv(f"{RESULTS_DIR}/diebold_mariano_results.csv")
    print(f"  DM test results saved to {RESULTS_DIR}/diebold_mariano_results.csv")

    # Step 7: Regime-Conditional Performance Analysis
    print("\n=== STEP 7: Regime-Conditional Performance Analysis ===")

    from plots_extended import (
        plot_sharpe_confidence_intervals,
        plot_regime_performance,
        plot_kalman_advantage_by_regime,
        plot_realized_volatility_with_regimes,
        plot_forecast_error_comparison,
    )

    print("\n  Regime day counts:")
    print(regime_summary(regime_labels).to_string())

    sharpe_by_regime = regime_sharpe_table(results_full, regime_labels)
    print("\n  Sharpe by Regime:")
    print(sharpe_by_regime.to_string())
    sharpe_by_regime.to_csv(f"{RESULTS_DIR}/regime_sharpe_table.csv")

    advantage = kalman_outperformance_by_regime(
        results_full, regime_labels, baseline="Rolling MV"
    )
    print("\n  Kalman MV Advantage by Regime:")
    print(advantage.to_string())
    advantage.to_csv(f"{RESULTS_DIR}/kalman_advantage_by_regime.csv")

    # Step 8: Extended Plots
    print("\n=== STEP 8: Extended Plots ===")

    plot_sharpe_confidence_intervals(stat_results["sharpe_cis"])
    plot_regime_performance(sharpe_by_regime)
    plot_kalman_advantage_by_regime(advantage)

    vol_series = compute_realized_volatility(returns, window=21)
    plot_realized_volatility_with_regimes(vol_series, regime_labels)

    for asset in ["SPY", "TLT"]:
        plot_forecast_error_comparison(
            stat_results["forecast_errors"]["kalman"],
            stat_results["forecast_errors"]["rolling"],
            asset=asset,
        )

    # Diagnostic plots for Discussion section
    from diagnostic_analysis import collect_kalman_gain, collect_covariance_divergence
    from plots_diagnostic import plot_kalman_gain, plot_covariance_divergence

    print("\n=== STEP 8b: Diagnostic Plots (Discussion Section) ===")

    gain_series = collect_kalman_gain(
        returns, train,
        q_scale=q_best,
        regime_labels=regime_labels,
        regime_alphas=regime_alphas,
    )

    divergence_series = collect_covariance_divergence(
        returns, train,
        q_scale=q_best,
        cov_window=60,
        regime_labels=regime_labels,
        regime_alphas=regime_alphas,
    )

    plot_kalman_gain(gain_series, regime_labels)
    plot_covariance_divergence(divergence_series, regime_labels)

    print("  Diagnostic plots saved.")

    # Step 9: Transaction Cost Sensitivity
    print("\n=== STEP 9: Transaction Cost Sensitivity ===")

    from cost_analysis import run_sensitivity, plot_sensitivity, print_sharpe_pivot

    df_costs_full = run_sensitivity(
        returns, train, period_label="Full",
        regime_labels=regime_labels, regime_alphas=regime_alphas,
        turnover_gamma=turnover_gamma,
    )
    print_sharpe_pivot(df_costs_full, "Full")
    df_costs_full.to_csv(f"{RESULTS_DIR}/cost_sensitivity.csv", index=False)
    plot_sensitivity(df_costs_full, period_label="Full", filename="cost_sensitivity_full.png")

    df_costs_oos = run_sensitivity(
        test, train, period_label="OOS",
        regime_labels=regime_labels, regime_alphas=regime_alphas,
        turnover_gamma=turnover_gamma,
    )
    print_sharpe_pivot(df_costs_oos, "OOS")
    df_costs_oos.to_csv(f"{RESULTS_DIR}/cost_sensitivity_oos.csv", index=False)
    plot_sensitivity(df_costs_oos, period_label="OOS", filename="cost_sensitivity_oos.png")

    print("\n=== ALL DONE ===")
    print(f"Results: ./{RESULTS_DIR}/")
    print(f"Plots:   ./{PLOTS_DIR}/")


if __name__ == "__main__":
    main()
