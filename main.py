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
from backtest import run_backtest, slice_result
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


# The three Kalman variants under test — each isolates a different input.
KALMAN_VARIANTS = {
    "Kalman-Mu MV":    dict(use_filtered_mu=True,  use_filtered_sigma=False),  # mean-isolation
    "Kalman-Sigma MV": dict(use_filtered_mu=False, use_filtered_sigma=True),   # covariance-isolation
    "Kalman-Full MV":  dict(use_filtered_mu=True,  use_filtered_sigma=True),   # both adaptive
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Step 1: Load Data
    print("\n=== STEP 1: Loading Data ===")
    prices = load_prices()
    returns = compute_returns(prices)
    train, test = split_data(returns)

    # Step 2: Calibrate Q + Per-Regime Alphas — one set PER Kalman variant
    print("\n=== STEP 2: Calibrating Kalman Filter (Walk-Forward CV) ===")

    from q_calibration import select_q_cv, plot_q_selection, calibrate_regime_alphas
    from regime_detector import (
        compute_realized_volatility,
        classify_regimes,
        regime_summary,
        regime_sharpe_table,
        kalman_outperformance_by_regime,
    )

    # Compute regime labels on full returns (no lookahead — uses past vol only)
    regime_labels = classify_regimes(returns, window=21, crisis_threshold=2.0,
                                      reference_returns=train)
    regime_labels_train = regime_labels.reindex(train.index)

    turnover_gamma = 0.00005  # starting value, tune later

    calibration = {}
    for name, flags in KALMAN_VARIANTS.items():
        q_best_v, q_results_v = select_q_cv(train, verbose=True, label=name, **flags)
        plot_q_selection(
            q_results_v, q_best_v,
            save_path=f"{PLOTS_DIR}/q_selection_{name.replace(' ', '_').replace('-', '_')}.png",
        )
        print(f"\n  CV-selected Q_SCALE [{name}] = {q_best_v:.4e}")

        regime_alphas_v, alpha_results_v = calibrate_regime_alphas(
            train, regime_labels_train, q_best_v, verbose=True, label=name, **flags
        )
        print(f"  CV-selected regime alphas [{name}]: {regime_alphas_v}")

        calibration[name] = {"q_best": q_best_v, "regime_alphas": regime_alphas_v, **flags}

    print("\n  Calibration summary (verify these differ across variants):")
    for name in KALMAN_VARIANTS:
        c = calibration[name]
        print(f"    {name}: q_best={c['q_best']:.3e}  alphas={c['regime_alphas']}")

    # Alias for Step 8b diagnostics, which still illustrate a single
    # ("headline") variant — Kalman-Sigma, the covariance-isolation design.
    q_best = calibration["Kalman-Sigma MV"]["q_best"]
    regime_alphas = calibration["Kalman-Sigma MV"]["regime_alphas"]

    # NOTE: filtered_mu (used by Step 5 plots and Step 6's DM test) is now
    # built AFTER Step 3, directly from each strategy's own regime-alpha-aware
    # trajectory -- see filtered_mu_from_strategy() below. Building it here
    # via build_filter_from_training(q_scale=...) alone would silently ignore
    # regime_alphas entirely (every step uses a constant Q), which is exactly
    # what made Kalman-Mu and Kalman-Full's DM results bit-identical despite
    # having different regime alphas: they happened to share q_best, and
    # q_best was the only thing this used.

    # Step 3: Run Backtests
    print("\n=== STEP 3: Running Backtests ===")

    strategies = [
        ("Equal Weight",   equal_weight_strategy),
        ("Rolling MV",     make_rolling_mv_strategy(turnover_gamma=turnover_gamma)),
        ("Static MV",      make_static_mv_strategy(train)),
        ("Ledoit-Wolf MV", make_ledoit_wolf_strategy(turnover_gamma=turnover_gamma)),
    ] + [
        (name, make_kalman_strategy(
            train,
            q_scale=calibration[name]["q_best"],
            regime_labels=regime_labels,
            regime_alphas=calibration[name]["regime_alphas"],
            turnover_gamma=turnover_gamma,
            use_filtered_mu=calibration[name]["use_filtered_mu"],
            use_filtered_sigma=calibration[name]["use_filtered_sigma"],
        ))
        for name in KALMAN_VARIANTS
    ]

    strategy_funcs = dict(strategies)

    # Full period (train + test) — run ONCE, continuously. OOS metrics are
    # derived by slicing this single run down to the test-period dates,
    # rather than rerunning each strategy separately on test alone — see
    # slice_result()'s docstring in backtest.py for why that approach
    # silently breaks stateful strategies like the Kalman variants.
    results_full = []
    for name, strat_func in strategies:
        result = run_backtest(returns, strat_func, name=name)
        results_full.append(result)

    results_test = [slice_result(r, test.index) for r in results_full]

    def filtered_mu_from_strategy(strat_func) -> pd.DataFrame:
        """The real mu_hat trajectory a Kalman strategy used, read directly
        off its closure (regime-alpha switching included) rather than
        rebuilt separately."""
        return pd.DataFrame(
            strat_func.filtered_history,
            index=pd.DatetimeIndex(strat_func.filtered_index),
            columns=returns.columns,
        )

    # Used by Step 5's plot_filtered_vs_rolling_mu calls below.
    filtered_mu = filtered_mu_from_strategy(strategy_funcs["Kalman-Mu MV"])

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

    weight_plot_names = ["Rolling MV"] + list(KALMAN_VARIANTS.keys())
    for r in results_full:
        if r["name"] in weight_plot_names:
            plot_weights_over_time(r)

    plot_filtered_vs_rolling_mu(filtered_mu, returns, asset="SPY")
    plot_filtered_vs_rolling_mu(filtered_mu, returns, asset="TLT")

    # Step 6: Statistical Significance Tests
    print("\n=== STEP 6: Statistical Significance Tests ===")

    from statistical_tests import compute_forecast_errors, diebold_mariano_test, bootstrap_sharpe_comparison

    # DM test — once PER Kalman variant, since each has its own Q* and
    # therefore its own filtered mean series. Cheap, so doing this 3x is fine.
    dm_results_by_variant = {}
    forecast_errors_by_variant = {}
    for name in KALMAN_VARIANTS:
        filtered_mu_variant = filtered_mu_from_strategy(strategy_funcs[name])

        errors_kf, errors_roll = compute_forecast_errors(filtered_mu_variant, returns, window=60)
        forecast_errors_by_variant[name] = {"kalman": errors_kf, "rolling": errors_roll}

        dm_results = {}
        for asset in errors_kf.columns:
            dm_results[asset] = diebold_mariano_test(
                errors_kf[asset], errors_roll[asset], loss="squared", alternative="less"
            )
        e_kf_agg = errors_kf.mean(axis=1)
        e_roll_agg = errors_roll.mean(axis=1)
        dm_results["OVERALL"] = diebold_mariano_test(e_kf_agg, e_roll_agg, loss="squared", alternative="less")
        dm_results_by_variant[name] = dm_results

        print(f"\n  [{name}] DM test (Q*={calibration[name]['q_best']:.3e}):")
        for asset, dm in dm_results.items():
            print(f"    {asset}: DM={dm['DM_statistic']:+.3f}, p={dm['p_value']:.3f}")

        dm_df = pd.DataFrame(dm_results).T
        dm_df.to_csv(f"{RESULTS_DIR}/diebold_mariano_results_{name.replace(' ', '_').replace('-', '_')}.csv")

    print(f"\n  DM test results saved to {RESULTS_DIR}/ (one file per variant)")

    # Bootstrap Sharpe CIs — computed ONCE. Only depends on results_full,
    # not on filtered_mu, so it doesn't need to be repeated per variant.
    print(f"\n  Block Bootstrap Sharpe Ratio Confidence Intervals (n_bootstrap=1000)")
    sharpe_cis = bootstrap_sharpe_comparison(results_full, n_bootstrap=1000)
    print(f"\n{sharpe_cis.to_string()}")
    sharpe_cis.to_csv(f"{RESULTS_DIR}/sharpe_confidence_intervals.csv")
    print(f"\n  Sharpe CI table saved to {RESULTS_DIR}/sharpe_confidence_intervals.csv")

    if "Rolling MV" in sharpe_cis.index:
        rolling_row = sharpe_cis.loc["Rolling MV"]
        print("\n  CI Interpretation (vs Rolling MV):")
        for name in KALMAN_VARIANTS:
            if name not in sharpe_cis.index:
                continue
            row = sharpe_cis.loc[name]
            if row["95% CI Lower"] > rolling_row["95% CI Upper"]:
                verdict = "statistically superior to Rolling MV"
            elif row["95% CI Upper"] > rolling_row["95% CI Upper"]:
                verdict = "higher point estimate, not statistically conclusive"
            else:
                verdict = "no significant difference from Rolling MV"
            print(f"    {name}: {verdict}")

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

    advantage_tables = {}
    for name in KALMAN_VARIANTS:
        adv = kalman_outperformance_by_regime(
            results_full, regime_labels, baseline="Rolling MV", strategy=name
        )
        advantage_tables[name] = adv
        print(f"\n  {name} Advantage by Regime:")
        print(adv.to_string())
        adv.to_csv(f"{RESULTS_DIR}/{name.replace(' ', '_').replace('-', '_')}_advantage_by_regime.csv")

    # Step 8: Extended Plots
    print("\n=== STEP 8: Extended Plots ===")

    plot_sharpe_confidence_intervals(sharpe_cis)
    plot_regime_performance(sharpe_by_regime)
    for name, adv in advantage_tables.items():
        plot_kalman_advantage_by_regime(adv)

    vol_series = compute_realized_volatility(returns, window=21)
    plot_realized_volatility_with_regimes(vol_series, regime_labels)

    for asset in ["SPY", "TLT"]:
        plot_forecast_error_comparison(
            forecast_errors_by_variant["Kalman-Mu MV"]["kalman"],
            forecast_errors_by_variant["Kalman-Mu MV"]["rolling"],
            asset=asset,
            variant_name="Kalman-Mu MV",
        )

    # Diagnostic plots for Discussion section — using Kalman-Sigma's
    # calibration, since these specifically illustrate covariance-estimate
    # divergence/gain (q_best/regime_alphas aliased above)
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

    # Step 9: Transaction Cost Sensitivity — now covers all 3 variants
    print("\n=== STEP 9: Transaction Cost Sensitivity ===")

    from cost_analysis import run_sensitivity, plot_sensitivity, print_sharpe_pivot

    df_costs_full = run_sensitivity(
        returns, returns.index, train, period_label="Full",
        regime_labels=regime_labels, calibration=calibration,
        turnover_gamma=turnover_gamma,
    )
    print_sharpe_pivot(df_costs_full, "Full")
    df_costs_full.to_csv(f"{RESULTS_DIR}/cost_sensitivity.csv", index=False)
    plot_sensitivity(df_costs_full, period_label="Full", filename="cost_sensitivity_full.png")

    df_costs_oos = run_sensitivity(
        returns, test.index, train, period_label="OOS",
        regime_labels=regime_labels, calibration=calibration,
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