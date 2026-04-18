# Adaptive Portfolio Optimization Using Kalman Filtering for Dynamic Covariance Estimation

**Krish Tanwar** · Working Paper · April 2026

---

## Overview

This project investigates whether replacing static rolling-window covariance estimates with Kalman Filter-based adaptive estimates improves portfolio construction under the classical Markowitz mean-variance framework.

The core problem: mean-variance optimization is sensitive to the quality of covariance estimates. In non-stationary markets, rolling-window estimators weight all observations within the window equally, ignoring information decay. We reformulate covariance estimation as a state-space filtering problem, applying a Kalman Filter to produce continuously updated, noise-regularized estimates of Σ_t that are then passed to the portfolio optimizer.

---

## Research Question

> Does Kalman Filter-based adaptive covariance estimation produce measurably superior covariance forecasts, and does any forecasting advantage translate into out-of-sample portfolio outperformance — or does a gap exist between estimation quality and allocation quality?

---

## Key Findings

**1. Kalman is a statistically superior covariance forecaster.**
A Diebold-Mariano test rejects equal forecast accuracy in favor of Kalman on 4 of 6 assets (EFA, QQQ, SPY, VNQ) at the 5% level, and is significant overall (DM = −4.51, p < 0.001). GLD and TLT show no significant difference, consistent with the near-random-walk behavior of commodity and long-duration bond returns.

**2. Superior forecasting translates to modest but consistent out-of-sample portfolio outperformance.**
In the held-out test period, Kalman MV achieves a Sharpe ratio of **1.129** versus **1.123** for Rolling MV and **0.822** for Static MV. The gap is not statistically significant — 95% block bootstrap confidence intervals overlap substantially across all strategies — but Kalman MV maintains its edge across all transaction cost levels tested (0–20 bps), suggesting the advantage is not an artifact of a single favorable period.

**3. The value of adaptive estimation is regime-conditional.**
Kalman MV outperforms Rolling MV most strongly in medium-volatility regimes (+0.160 Sharpe advantage), where covariance structure is transitioning and rolling windows lag. It also holds a small edge in crisis regimes (+0.019). Performance is roughly matched in high-volatility environments (−0.004), where both strategies react similarly to the same elevated-noise signal.

**4. Kalman MV approaches the performance of an oracle benchmark without look-ahead bias.**
Static MV (which uses the full-period covariance matrix, requiring future data) achieves a full-period Sharpe of 0.817. Kalman MV achieves 0.780 without access to future data. In the out-of-sample period — where Static MV loses its informational advantage — Kalman MV (1.129) outperforms Static MV (0.822) by a wide margin, confirming that Static MV's full-period edge is an artifact of look-ahead bias rather than genuine predictive skill.

**5. No strategy achieves statistically distinguishable full-period Sharpe.**
Block bootstrap 95% CIs span approximately 1.0 Sharpe unit for all strategies, confirming no unconditional dominance. Regime-conditional analysis is necessary to identify where each approach adds value.

---

## Methodology

### State-Space Model

The covariance structure is modeled as a latent state following a random walk:

```
State transition:  Σ_t = F · Σ_{t-1} + w_t,    w_t ~ N(0, Q)
Observation:       r_t = H · Σ_t   + v_t,       v_t ~ N(0, R)
```

Where:
- `F = I` (random walk — covariance has no predictable mean-reversion)
- `H = I` (returns are direct noisy observations of the covariance state)
- `Q` = calibrated via walk-forward cross-validation (see below)
- `R` = sample return variance from training data (observation noise)

### Q Calibration — Walk-Forward Cross-Validation

Process noise Q controls filter adaptation speed and is calibrated via walk-forward cross-validation on the training period, minimizing out-of-sample RMSE across expanding windows. EM (Expectation-Maximization) was evaluated but abandoned: MLE for local-level models allows Q → 0 to perfectly fit training data, causing degenerate solutions (Q ≈ 3.87e-8) and a near-static filter that failed to converge within 200 iterations. Walk-forward CV avoids this degeneracy by optimizing predictive rather than in-sample fit. The selected Q_SCALE = 1.00e-07.

### Kalman Filter Update

At each time step t:

**Predict:**
```
Σ̂_{t|t-1} = F · Σ̂_{t-1}
P_{t|t-1}  = F · P_{t-1} · F' + Q
```

**Update:**
```
K_t        = P_{t|t-1} · H' · (H · P_{t|t-1} · H' + R)^{-1}
Σ̂_t        = Σ̂_{t|t-1} + K_t · (r_t - H · Σ̂_{t|t-1})
P_t        = (I - K_t · H) · P_{t|t-1}
```

### Portfolio Optimization

At each monthly rebalancing date:

```
max   w'μ̂_t - (λ/2) · w'Σ̂_t · w
s.t.  Σ w_i = 1  (fully invested)
      w_i ≥ 0    (long only)
      w_i ≤ 0.40 (position cap)
```

Where `λ = 2` (risk aversion) and `μ̂_t` is a 60-day rolling mean for all strategies. Kalman MV differs from Rolling MV only in its covariance estimate `Σ̂_t`.

### Regime Classification

Market regimes defined via 21-day rolling realized volatility (annualized), classified into terciles plus a crisis threshold:

| Regime | Definition |
|--------|-----------|
| LOW_VOL | Realized vol ≤ 33rd percentile |
| MED_VOL | 33rd < realized vol ≤ 67th percentile |
| HIGH_VOL | Realized vol > 67th percentile |
| CRISIS | Realized vol > mean + 2σ (≥ 30.6% annualized) |

---

## Strategies Compared

| Strategy | Return Estimate | Covariance | Rebalancing |
|----------|----------------|------------|-------------|
| Equal Weight | N/A | N/A | Monthly |
| Static MV | Training mean (fixed) | Training cov (fixed, look-ahead) | Monthly |
| Rolling MV | 60-day rolling mean | 60-day rolling | Monthly |
| **Kalman MV** | 60-day rolling mean | **Kalman filtered** | Monthly |

*Note: Static MV uses the full-period covariance matrix estimated from 2010–2026, requiring future data unavailable at portfolio formation time. It serves as an oracle benchmark only and is not a realistic implementation.*

---

## Results Summary

### Full Period (Jan 2010 – Mar 2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Avg Turnover |
|----------|------------|--------|--------------|--------------|
| Equal Weight | 8.56% | 0.714 | −28.1% | 0.000 |
| Rolling MV | 10.06% | 0.757 | −31.1% | 0.689 |
| Static MV | 12.60% | 0.817 | −27.8% | 0.007 |
| **Kalman MV** | **10.25%** | **0.780** | **−27.8%** | **0.666** |

### Out-of-Sample Test Period (Jan 2025 – Mar 2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Avg Turnover |
|----------|------------|--------|--------------|--------------|
| Equal Weight | 14.35% | 1.122 | −11.1% | 0.000 |
| Rolling MV | 17.16% | 1.123 | −12.5% | 0.525 |
| Static MV | 14.04% | 0.822 | −17.2% | 0.000 |
| **Kalman MV** | **17.75%** | **1.129** | **−12.5%** | **0.579** |

### Sharpe by Market Regime (Full Period)

| Regime | N Days | Equal Weight | Rolling MV | Static MV | Kalman MV | Kalman Advantage |
|--------|--------|-------------|------------|-----------|-----------|-----------------|
| LOW_VOL | 1,341 | 1.923 | 1.699 | 2.195 | 1.631 | −0.068 |
| MED_VOL | 1,381 | 0.546 | 0.636 | 0.679 | **0.796** | **+0.160** |
| HIGH_VOL | 1,213 | 0.648 | 0.489 | 0.610 | 0.485 | −0.004 |
| CRISIS | 128 | 0.354 | 0.813 | 0.660 | **0.832** | **+0.019** |

### Transaction Cost Sensitivity (OOS Sharpe)

| Strategy | 0 bps | 5 bps | 10 bps | 20 bps |
|----------|-------|-------|--------|--------|
| Equal Weight | 1.122 | 1.122 | 1.122 | 1.122 |
| Rolling MV | 1.123 | 1.110 | 1.097 | 1.071 |
| Static MV | 0.822 | 0.822 | 0.822 | 0.822 |
| **Kalman MV** | **1.129** | **1.115** | **1.101** | **1.073** |

---

## Statistical Testing

### Diebold-Mariano Test (covariance forecast accuracy)

One-sided test, H₁: Kalman has strictly lower squared forecast error than Rolling MV.

| Asset | DM Statistic | p-value | Conclusion |
|-------|-------------|---------|------------|
| EFA | −3.063 | 0.001 | Kalman superior |
| GLD | 0.174 | 0.569 | No difference |
| QQQ | −2.906 | 0.002 | Kalman superior |
| SPY | −2.622 | 0.004 | Kalman superior |
| TLT | 1.108 | 0.866 | No difference |
| VNQ | −2.065 | 0.020 | Kalman superior |
| **Overall** | **−4.508** | **<0.001** | **Kalman superior** |

### Block Bootstrap Sharpe CIs (n=1,000, block=20 days, full period)

| Strategy | Sharpe | 95% CI |
|----------|--------|--------|
| Equal Weight | 0.746 | [0.279, 1.326] |
| Rolling MV | 0.789 | [0.354, 1.326] |
| Static MV | 0.847 | [0.413, 1.382] |
| Kalman MV | 0.810 | [0.376, 1.362] |

All intervals overlap substantially. No strategy achieves statistically superior full-period risk-adjusted performance.

---

## Project Structure

```
kalman-portfolio/
├── config.py              # All parameters (Q, R, λ, assets, dates)
├── data_loader.py         # ETF data download and return computation
├── kalman_filter.py       # Kalman Filter implementation
├── covariance.py          # Rolling covariance estimation
├── optimizer.py           # Mean-variance optimization (cvxpy)
├── backtest.py            # Monthly rebalancing backtest engine
├── benchmarks.py          # Strategy definitions
├── evaluation.py          # Performance metrics (CAGR-based Sharpe)
├── statistical_tests.py   # Diebold-Mariano + block bootstrap Sharpe CIs
├── regime_detector.py     # Volatility-based regime classification
├── q_calibration.py       # Walk-forward CV for Q selection
├── cost_analysis.py       # Transaction cost sensitivity analysis
├── plots.py               # Core figures
├── plots_extended.py      # Statistical test and regime figures
├── main.py                # End-to-end pipeline
└── results/               # CSV outputs (metrics, DM test, regime analysis)
```

---

## Data

- **Assets:** SPY (US equities), QQQ (US tech), TLT (long-term bonds), GLD (gold), EFA (international equities), VNQ (REITs)
- **Source:** Yahoo Finance via `yfinance`
- **Full sample:** January 2010 – March 2026
- **Training period:** January 2010 – December 2024
- **Test period:** January 2025 – March 2026
- **Frequency:** Daily returns, monthly rebalancing

---

## Setup

```bash
git clone https://github.com/[your-username]/kalman-portfolio.git
cd kalman-portfolio

python -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
python main.py
```

Results saved to `results/`, plots saved to `plots/`.

---

## Dependencies

```
numpy, pandas, scipy, matplotlib
yfinance, cvxpy, filterpy
```

---

## Limitations

- Kalman Filter assumes Gaussian return distributions; real returns exhibit fat tails and skewness
- Q is calibrated via walk-forward CV on training data; optimal Q may shift across market regimes
- Long-only constraint with 40% position cap limits the optimizer's ability to act on covariance signal
- Simultaneous equity-bond drawdowns (2022 rate hike cycle) violate diversification assumptions embedded in the model
- The test period (Jan 2025 – Mar 2026) is only 15 months; Sharpe differences are not statistically distinguishable at this sample size
- Static MV serves as an oracle benchmark only and is not implementable in practice due to look-ahead bias in covariance estimation

---

## Extensions

- Ledoit-Wolf covariance shrinkage as an alternative benchmark to rolling sample covariance
- Non-linear filtering (Extended Kalman Filter, Unscented Kalman Filter) for non-Gaussian return dynamics
- Regime-switching state-space model to allow discrete structural breaks in covariance dynamics
- Longer out-of-sample evaluation period to achieve sufficient power for Sharpe ratio significance testing
- Factor model priors for the Kalman state to incorporate cross-sectional structure in the covariance matrix

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915–1953.
- Ledoit, O. & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance*, 15(5), 850–859.
- Kan, R. & Zhou, G. (2007). Optimal Portfolio Choice with Parameter Uncertainty. *Journal of Financial and Quantitative Analysis*, 42(3), 621–656.