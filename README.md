# Kalman Filtering in Mean-Variance Portfolios

**Krish Tanwar** · Working Paper · June 2026

**Paper:** [Kalman Filtering in Mean-Variance Portfolios: Which Input Channel Drives Adaptive Gains?](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7014639)

---

## Overview

This project investigates whether Kalman Filter-based adaptive input estimation improves portfolio construction under the classical Markowitz mean-variance framework, and which input channel drives any observed gains.

The core problem: mean-variance optimization is sensitive to the quality of both expected-return and covariance estimates. In non-stationary markets, rolling-window estimators weight all observations equally, ignoring information decay. Rather than treating Kalman filtering as a single portfolio modification, this paper applies a 2x2 ablation design that decomposes filtering across the two inputs of the mean-variance optimizer — expected returns and covariance — allowing the analysis to isolate whether performance gains arise from better return estimation, better covariance estimation, or their interaction.

---

## Research Question

Does Kalman filtering the expected-return input, the covariance input, or both inputs improve out-of-sample portfolio performance — and are the two channels additive?

---

## Key Findings

**1. The mean-filtering channel is the primary source of out-of-sample improvement.**
Kalman-Mu MV (filtered mean, rolling covariance) achieves the strongest out-of-sample Sharpe ratio of 1.406, compared with 1.341 for Rolling MV, 1.363 for Ledoit-Wolf MV, and 1.342 for Kalman-Sigma MV. Diebold-Mariano tests confirm statistically lower squared forecast error for Kalman-Mu MV on 4 of 6 assets at the 5% level (EFA, QQQ, SPY, VNQ), and at the 10% level for GLD. The overall DM statistic is -4.410 (p < 0.001).

**2. The covariance-filtering channel adds no independent value.**
Kalman-Sigma MV (rolling mean, Kalman-filtered covariance) tracks Rolling MV almost exactly across full-period, out-of-sample, regime-conditional, and transaction-cost evaluations. Its DM statistics are uniformly positive across all six assets with p-values of 1.000, indicating it is directionally worse than the rolling mean on return forecast error. This result is consistent with the T >> N structure of the estimation problem: with 6 assets and a 60-day window, the sample covariance matrix is already well-conditioned, leaving little room for Kalman filtering to add value.

**3. The two channels are not additive out of sample.**
Kalman-Full MV (filtered mean and filtered covariance) achieves the strongest full-period Sharpe ratio of 1.050, but produces an out-of-sample Sharpe of only 1.288 — below both Kalman-Mu MV and Kalman-Sigma MV. Adding filtered covariance on top of filtered mean increases turnover by 22% (0.833 vs 0.680) and does not produce the strongest OOS result. The mean channel is productive; the covariance channel interferes rather than complements in the OOS regime mix.

**4. The mean-filtering advantage is regime-conditional.**
Kalman-Mu MV outperforms Rolling MV by +0.252 Sharpe units in MED_VOL regimes, where return dynamics are transitioning and rolling-window estimates lag. The advantage rises to +0.469 Sharpe units in CRISIS regimes, where recursive updating allows the filter to move away from stale pre-crisis asset rankings faster than a 60-day rolling mean. In LOW_VOL regimes the rolling mean is already adequate (-0.139 disadvantage). In HIGH_VOL regimes, observations are dominated by noise rather than persistent signal (-0.180 disadvantage).

**5. Kalman-Mu MV achieves meaningful drawdown reduction.**
Over the full evaluation period, Kalman-Mu MV reduces maximum drawdown from -29.51% for Rolling MV to -25.52%, a 399 basis point improvement, while producing slightly lower turnover (0.684 vs 0.687). Kalman-Sigma MV records a maximum drawdown of -29.49%, essentially identical to Rolling MV, confirming that the drawdown improvement comes from the mean channel, not covariance filtering.

**6. The OOS advantage of Kalman-Mu MV survives transaction costs.**
Kalman-Mu MV maintains the highest OOS Sharpe ratio at every tested cost level: 1.406 at 0 bps, 1.389 at 5 bps, 1.372 at 10 bps, and 1.339 at 20 bps. Kalman-Full MV deteriorates more sharply due to higher turnover, falling to 1.205 at 20 bps.

**7. No strategy achieves statistically distinguishable full-period Sharpe.**
Block bootstrap 95% confidence intervals span approximately one Sharpe unit for all strategies and overlap substantially, confirming no unconditional dominance. The regime-conditional and ablation analyses provide a more informative characterization of where each channel adds and loses value than the full-period aggregate alone.

---

## Methodology

### State-Space Model

Each asset's return process is modeled as a latent state following a random walk:

```
State transition:  mu_{i,t} = mu_{i,t-1} + w_{i,t},    w_{i,t} ~ N(0, Q)
Observation:       r_{i,t}  = mu_{i,t}   + v_{i,t},    v_{i,t} ~ N(0, R_i)
```

Where mu_{i,t} is the latent return signal for asset i on day t, Q is the process noise variance calibrated via walk-forward cross-validation, and R_i is the observation noise variance estimated from training data.

### Kalman Filter Update

**Predict:**
```
mu_hat_{i,t|t-1} = mu_hat_{i,t-1|t-1}
P_{i,t|t-1}      = P_{i,t-1|t-1} + Q
```

**Update:**
```
e_{i,t}        = r_{i,t} - mu_hat_{i,t|t-1}
K_{i,t}        = P_{i,t|t-1} / (P_{i,t|t-1} + R_i)
mu_hat_{i,t|t} = mu_hat_{i,t|t-1} + K_{i,t} * e_{i,t}
P_{i,t|t}      = (1 - K_{i,t}) * P_{i,t|t-1}
```

The Kalman gain K_{i,t} in (0,1) determines how much weight the filter places on the new observation relative to its prior. A large Q relative to R_i raises the gain and causes faster adaptation; a small Q lowers the gain and causes the filter to rely more on its accumulated estimate.

### Process Noise Calibration

**Stage 1 — Base Q:** Selected via walk-forward cross-validation on the training period using portfolio Sharpe ratio as the criterion. A logarithmically spaced grid of candidate Q values is evaluated across expanding training folds. Selected Q* = 5.00e-08 for Kalman-Mu MV and Kalman-Full MV. For Kalman-Sigma MV, the validation Sharpe increases monotonically across the grid and selects the boundary value Q* = 1.00e-01, consistent with weak identification of the covariance-filtering channel.

**Stage 2 — Regime-Conditional Scaling:** The effective process noise in regime k is Q_k = alpha_k * Q*, where alpha_k is calibrated via greedy sequential walk-forward cross-validation on the training period. This allows the filter to adapt more aggressively in transitional environments and more conservatively during elevated-noise periods. CRISIS is fixed at alpha = 1.0 for all specifications due to limited crisis observations in the training period.

### Calibrated Regime Scaling Factors

| Specification | LOW_VOL | MED_VOL | HIGH_VOL | CRISIS |
|---------------|---------|---------|----------|--------|
| Kalman-Mu MV | 2.000 | 2.000 | 0.010 | 1.000 (fixed) |
| Kalman-Sigma MV | 0.050 | 0.010 | 20.000 | 1.000 (fixed) |
| Kalman-Full MV | 5.000 | 5.000 | 0.200 | 1.000 (fixed) |

The consistent HIGH_VOL dampening across Kalman-Mu MV and Kalman-Full MV — calibrated independently — reflects a genuine signal that elevated-volatility environments reward conservative filtering rather than rapid adaptation.

### Portfolio Optimization

At each monthly rebalancing date:

```
max   w' mu_hat_t - (lambda/2) * w' Sigma_hat_t * w
s.t.  sum(w_i) = 1     (fully invested)
      w_i >= 0          (long only)
      w_i <= 0.40       (40% position cap)
```

Where lambda = 2 (risk aversion). The 2x2 ablation design assigns inputs as follows: Kalman-Mu MV uses filtered mean and rolling covariance; Kalman-Sigma MV uses rolling mean and covariance constructed from Kalman-filtered return signals; Kalman-Full MV applies filtering to both inputs simultaneously.

### Regime Classification

Market regimes are defined via 21-day annualized realized volatility computed from SPY returns:

| Regime | Definition |
|--------|------------|
| LOW_VOL | Realized vol <= 33rd percentile |
| MED_VOL | 33rd < realized vol <= 67th percentile |
| HIGH_VOL | Realized vol > 67th percentile |
| CRISIS | Realized vol > mean + 2 sigma (>= 30.7% annualized) |

Percentile thresholds are estimated exclusively on the training period and applied to the full sample, ensuring no look-ahead bias in regime classification.

---

## Strategies Compared

| Strategy | Return Estimate | Covariance Estimate | Rebalancing |
|----------|----------------|---------------------|-------------|
| Equal Weight | None | None | Monthly |
| Rolling MV | 60-day rolling mean | 60-day rolling sample | Monthly |
| Static MV | Training-period mean | Training-period sample | Fixed |
| Ledoit-Wolf MV | 60-day rolling mean | 60-day Ledoit-Wolf shrinkage | Monthly |
| Kalman-Mu MV | Kalman-filtered mean | 60-day rolling sample | Monthly |
| Kalman-Sigma MV | 60-day rolling mean | 60-day rolling, Kalman-filtered returns | Monthly |
| Kalman-Full MV | Kalman-filtered mean | 60-day rolling, Kalman-filtered returns | Monthly |

Static MV uses parameters estimated on the full 2010-2024 training period. Its full-period results are subject to look-ahead bias and should be interpreted as a fixed-parameter benchmark rather than real-time implementable performance.

---

## Results Summary

### Full Period (January 2010 – March 2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Turnover | Total Return |
|----------|------------|--------|--------------|----------|--------------|
| Equal Weight | 10.54% | 0.898 | -26.51% | 0.000 | 407.59% |
| Rolling MV | 12.11% | 0.927 | -29.51% | 0.687 | 537.60% |
| Static MV | 14.59% | 0.916 | -28.97% | 0.007 | 809.26% |
| Ledoit-Wolf MV | 12.11% | 0.924 | -29.57% | 0.689 | 537.35% |
| Kalman-Mu MV | 11.88% | 0.936 | -25.52% | 0.684 | 516.96% |
| Kalman-Sigma MV | 12.10% | 0.925 | -29.49% | 0.687 | 536.69% |
| Kalman-Full MV | 13.23% | 1.050 | -25.80% | 0.752 | 648.71% |

### Out-of-Sample (January 2025 – March 2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Turnover | Total Return |
|----------|------------|--------|--------------|----------|--------------|
| Equal Weight | 18.70% | 1.385 | -10.74% | 0.000 | 23.56% |
| Rolling MV | 21.95% | 1.341 | -12.11% | 0.600 | 27.75% |
| Static MV | 15.25% | 0.889 | -17.74% | 0.000 | 19.14% |
| Ledoit-Wolf MV | 22.42% | 1.363 | -12.11% | 0.589 | 28.36% |
| Kalman-Mu MV | 23.76% | 1.406 | -12.07% | 0.680 | 30.10% |
| Kalman-Sigma MV | 21.96% | 1.342 | -12.11% | 0.600 | 27.76% |
| Kalman-Full MV | 21.21% | 1.288 | -12.07% | 0.833 | 26.80% |

### Sharpe by Volatility Regime (Full Period)

| Regime | N Days | Equal Weight | Rolling MV | Ledoit-Wolf MV | Kalman-Mu MV | Kalman-Sigma MV | Kalman-Full MV |
|--------|--------|-------------|------------|----------------|--------------|-----------------|----------------|
| LOW_VOL | 1,321 | 1.937 | 1.770 | 1.778 | 1.631 | 1.761 | 1.815 |
| MED_VOL | 1,390 | 0.669 | 0.749 | 0.739 | 1.001 | 0.749 | 1.042 |
| HIGH_VOL | 1,224 | 0.784 | 0.612 | 0.610 | 0.432 | 0.612 | 0.577 |
| CRISIS | 129 | 1.145 | 1.534 | 1.536 | 2.003 | 1.536 | 2.173 |

### Kalman Sharpe Advantage over Rolling MV by Regime

| Regime | Kalman-Mu MV | Kalman-Sigma MV | Kalman-Full MV |
|--------|--------------|-----------------|----------------|
| LOW_VOL | -0.139 | -0.009 | +0.045 |
| MED_VOL | +0.252 | +0.000 | +0.293 |
| HIGH_VOL | -0.180 | +0.000 | -0.035 |
| CRISIS | +0.469 | +0.002 | +0.639 |

### Transaction Cost Sensitivity — OOS Sharpe

| Strategy | 0 bps | 5 bps | 10 bps | 20 bps |
|----------|-------|-------|--------|--------|
| Equal Weight | 1.385 | 1.385 | 1.385 | 1.385 |
| Rolling MV | 1.341 | 1.326 | 1.311 | 1.281 |
| Static MV | 0.889 | 0.889 | 0.889 | 0.889 |
| Ledoit-Wolf MV | 1.363 | 1.349 | 1.334 | 1.304 |
| Kalman-Mu MV | 1.406 | 1.389 | 1.372 | 1.339 |
| Kalman-Sigma MV | 1.342 | 1.327 | 1.312 | 1.282 |
| Kalman-Full MV | 1.288 | 1.268 | 1.247 | 1.205 |

---

## Statistical Testing

### Diebold-Mariano Test — Kalman-Mu MV

One-sided test, H1: Kalman-Mu MV has strictly lower squared forecast error than the rolling mean.

| Asset | DM Statistic | p-value | Conclusion |
|-------|-------------|---------|------------|
| EFA | -3.538 | 0.0002 | Kalman superior |
| GLD | -1.484 | 0.069 | Marginal at 10% |
| QQQ | -3.992 | <0.001 | Kalman superior |
| SPY | -3.982 | <0.001 | Kalman superior |
| TLT | -1.009 | 0.156 | No difference |
| VNQ | -3.274 | 0.001 | Kalman superior |
| Overall | -4.410 | <0.001 | Kalman superior |

### Diebold-Mariano Test — Kalman-Sigma MV

One-sided test, H1: Kalman-Sigma MV has strictly lower squared forecast error than the rolling mean.

| Asset | DM Statistic | p-value | Conclusion |
|-------|-------------|---------|------------|
| EFA | +5.639 | 1.000 | Rolling mean lower error |
| GLD | +12.219 | 1.000 | Rolling mean lower error |
| QQQ | +5.531 | 1.000 | Rolling mean lower error |
| SPY | +4.337 | 1.000 | Rolling mean lower error |
| TLT | +12.042 | 1.000 | Rolling mean lower error |
| VNQ | +4.362 | 1.000 | Rolling mean lower error |
| Overall | +5.111 | 1.000 | Rolling mean lower error |

### Diebold-Mariano Test — Kalman-Full MV

One-sided test, H1: Kalman-Full MV has strictly lower squared forecast error than the rolling mean.

| Asset | DM Statistic | p-value | Conclusion |
|-------|-------------|---------|------------|
| EFA | -1.856 | 0.032 | Kalman superior |
| GLD | +1.713 | 0.957 | Rolling mean lower error |
| QQQ | -1.710 | 0.044 | Kalman superior |
| SPY | -2.171 | 0.015 | Kalman superior |
| TLT | +2.371 | 0.991 | Rolling mean lower error |
| VNQ | -1.014 | 0.155 | No difference |
| Overall | -3.493 | 0.0002 | Kalman superior |

### Block Bootstrap Sharpe Confidence Intervals (Full Period)

Block bootstrap with 1,000 replications and block length of 20 trading days.

| Strategy | Sharpe | 95% CI Lower | 95% CI Upper |
|----------|--------|-------------|-------------|
| Equal Weight | 0.898 | 0.428 | 1.459 |
| Rolling MV | 0.927 | 0.491 | 1.467 |
| Static MV | 0.916 | 0.494 | 1.421 |
| Ledoit-Wolf MV | 0.924 | 0.489 | 1.461 |
| Kalman-Mu MV | 0.936 | 0.516 | 1.453 |
| Kalman-Sigma MV | 0.925 | 0.488 | 1.469 |
| Kalman-Full MV | 1.051 | 0.616 | 1.590 |

Confidence intervals overlap substantially across all strategies. No strategy achieves statistically superior full-period risk-adjusted performance at conventional confidence levels.

---

## Data

- **Assets:** SPY (US equities), QQQ (US tech), TLT (long-term bonds), GLD (gold), EFA (international equities), VNQ (REITs)
- **Source:** Yahoo Finance via yfinance
- **Full sample:** January 2010 – March 2026
- **Training period:** January 2010 – December 2024
- **Test period:** January 2025 – March 2026
- **Return type:** Daily simple returns, r_t = (P_t - P_{t-1}) / P_{t-1}
- **Rebalancing:** Monthly

---

## Repository Structure

```
src/
├── main.py                    # End-to-end pipeline
├── config.py                  # All parameters (Q, lambda, assets, dates)
├── backtest.py                # Monthly rebalancing backtest engine
├── kalman_filter.py           # Kalman filter implementation
├── covariance.py              # Rolling and Ledoit-Wolf covariance estimation
├── optimizer.py               # Mean-variance optimization (cvxpy)
├── benchmarks.py              # Strategy definitions
├── evaluation.py              # Performance metrics (CAGR-based Sharpe, MDD, Calmar)
├── statistical_tests.py       # Diebold-Mariano tests and block bootstrap Sharpe CIs
├── regime_detector.py         # Volatility-based regime classification
├── q_calibration.py           # Walk-forward CV for base Q selection
├── calibrate_regime_alphas.py # Greedy sequential regime alpha calibration
├── cost_analysis.py           # Transaction cost sensitivity analysis
├── plots.py                   # Core performance figures
├── plots_extended.py          # Regime and statistical test figures
└── plots_diagnostic.py        # Diagnostic plots (Kalman gain, covariance divergence)
```

---

## Setup

```bash
git clone https://github.com/krishtanwar458/kalman-portfolio.git
cd kalman-portfolio

python -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
python src/main.py
```

Results are saved to `results/` and plots to `plots/`.

---

## Dependencies

- numpy
- pandas
- scipy
- matplotlib
- yfinance
- cvxpy
- scikit-learn

---

## Limitations

- The Kalman filter assumes Gaussian return distributions; financial returns exhibit fat tails, negative skewness, and volatility clustering
- The out-of-sample period spans only 15 months (311 trading days), which is insufficient for statistically significant Sharpe ratio differences between strategies
- The six-asset universe limits generalizability to higher-dimensional portfolios where T >> N no longer holds and covariance filtering may add more value
- Process noise Q is calibrated on training data; the optimal level may shift across market cycles in ways not fully captured by the training period
- The regime-conditional scaling factors are calibrated via a greedy sequential search, which may not identify the globally optimal combination across regimes
- The long-only constraint with a 40% position cap limits the optimizer's ability to act on both mean and covariance signal

---

## Extensions

- Non-linear filtering (Extended Kalman Filter, Unscented Kalman Filter) for non-Gaussian return dynamics
- Fully endogenous regime-switching state-space model to remove dependence on ad hoc volatility percentile thresholds
- Joint optimization of regime-conditional scaling factors across all regimes simultaneously
- Application of the 2x2 ablation design to larger asset universes where T >> N no longer holds
- Direct comparison with DCC-GARCH covariance estimation as an industry-standard time-varying benchmark

---

## References

- Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems. Journal of Basic Engineering, 82(1), 35-45.
- Diebold, F.X. and Mariano, R.S. (1995). Comparing Predictive Accuracy. Journal of Business and Economic Statistics, 13(3), 253-263.
- DeMiguel, V., Garlappi, L. and Uppal, R. (2009). Optimal Versus Naive Diversification. Review of Financial Studies, 22(5), 1915-1953.
- Ledoit, O. and Wolf, M. (2004). A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices. Journal of Multivariate Analysis, 88(2), 365-411.
- Ledoit, O. and Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. Journal of Empirical Finance, 15(5), 850-859.
- Engle, R. (2002). Dynamic Conditional Correlation. Journal of Business and Economic Statistics, 20(3), 339-350.
- Kan, R. and Zhou, G. (2007). Optimal Portfolio Choice with Parameter Uncertainty. Journal of Financial and Quantitative Analysis, 42(3), 621-656.