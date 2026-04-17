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
A Diebold-Mariano test rejects equal forecast accuracy in favor of Kalman on 4 of 6 assets (EFA, QQQ, SPY, VNQ) at the 5% level, and is significant overall (DM = −4.51, p < 0.001). GLD and TLT show no significant difference.

**2. Superior forecasting does not unconditionally translate to superior portfolio performance.**
In the held-out test period, Kalman MV achieves a Sharpe ratio of **1.051** versus **1.137** for Rolling MV. This gap is not statistically significant — 95% block bootstrap confidence intervals overlap substantially across all strategies — but it establishes that estimation quality and allocation quality are distinct objectives in MV portfolios.

**3. The value of adaptive estimation is regime-conditional.**
Kalman MV outperforms Rolling MV in medium-volatility (+0.126 Sharpe advantage) and crisis regimes (+0.117). It underperforms in high-volatility environments (−0.108), where persistent volatility causes the filter to over-adapt to noise, generating excess turnover without signal improvement.

**4. No strategy achieves statistically distinguishable full-period Sharpe.**
Block bootstrap 95% CIs span ~0.28 Sharpe units for all strategies, confirming no strategy dominates unconditionally. Regime-conditional analysis is necessary to identify where each approach adds value.

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

Process noise Q controls filter adaptation speed and is calibrated via walk-forward cross-validation on the training period, minimizing out-of-sample RMSE across expanding windows. EM (Expectation-Maximization) was evaluated but abandoned: MLE for local-level models allows Q → 0 to perfectly fit training data, causing degenerate solutions (Q ≈ 3.87e-8) and a near-static filter that failed to converge within 200 iterations. Walk-forward CV avoids this degeneracy by optimizing predictive rather than in-sample fit.

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
| CRISIS | Realized vol > mean + 2σ |

---

## Strategies Compared

| Strategy | Return Estimate | Covariance | Rebalancing |
|----------|----------------|------------|-------------|
| Equal Weight | N/A | N/A | Monthly |
| Static MV | Training mean (fixed) | Training cov (fixed) | Monthly |
| Rolling MV | 60-day rolling mean | 60-day rolling | Monthly |
| **Kalman MV** | 60-day rolling mean | **Kalman filtered** | Monthly |

---

## Results Summary

### Full Period (2010–2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Avg Turnover |
|----------|------------|--------|--------------|--------------|
| Equal Weight | 8.56% | 0.714 | -28.1% | 0.00 |
| Rolling MV | 10.45% | 0.786 | -31.4% | 0.74 |
| Static MV | 12.60% | 0.817 | -27.8% | 0.00 |
| Kalman MV | 10.28% | 0.781 | -29.2% | 0.86 |

### Out-of-Sample Test Period (Jan 2025 – Mar 2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Avg Turnover |
|----------|------------|--------|--------------|--------------|
| Equal Weight | 14.35% | 1.122 | -11.1% | 0.00 |
| Rolling MV | 17.39% | **1.137** | -12.5% | 0.53 |
| Static MV | 14.04% | 0.822 | -17.2% | 0.00 |
| Kalman MV | 16.49% | 1.051 | -12.9% | 0.63 |

### Sharpe by Market Regime

| Regime | N Days | Equal Weight | Rolling MV | Static MV | Kalman MV | Kalman Advantage |
|--------|--------|-------------|------------|-----------|-----------|-----------------|
| LOW_VOL | 1,341 | 1.923 | 1.801 | 2.195 | 1.816 | +0.015 |
| MED_VOL | 1,381 | 0.546 | 0.669 | 0.679 | **0.795** | **+0.126** |
| HIGH_VOL | 1,213 | 0.648 | 0.471 | 0.610 | 0.363 | −0.108 |
| CRISIS | 128 | 0.354 | 0.813 | 0.660 | **0.930** | **+0.117** |

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

### Block Bootstrap Sharpe CIs (n=1,000, block=20 days)

| Strategy | Sharpe | 95% CI |
|----------|--------|--------|
| Equal Weight | 0.746 | [0.279, 1.326] |
| Rolling MV | 0.815 | [0.372, 1.347] |
| Static MV | 0.847 | [0.413, 1.382] |
| Kalman MV | 0.810 | [0.382, 1.371] |

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
├── evaluation.py          # Performance metrics
├── statistical_tests.py   # Diebold-Mariano + block bootstrap Sharpe CIs
├── regime_detector.py     # Volatility-based regime classification
├── q_calibration.py       # Walk-forward CV for Q selection
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
- Kalman MV's higher turnover (0.86 full period vs 0.74 for Rolling MV) erodes real-world returns; cost sensitivity analysis shows the gap narrows but persists at realistic transaction costs

---

## Extensions

- Regime-conditional Q: attenuate process noise during HIGH_VOL regimes to reduce over-adaptation
- Ledoit-Wolf covariance shrinkage as an alternative to rolling sample covariance
- Non-linear filtering (Extended Kalman Filter, Unscented Kalman Filter) for non-Gaussian return dynamics
- Regime-switching state-space model to allow discrete structural breaks in covariance dynamics

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*, 22(5), 1915–1953.
- Ledoit, O. & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance*, 15(5), 850–859.
- Kan, R. & Zhou, G. (2007). Optimal Portfolio Choice with Parameter Uncertainty. *Journal of Financial and Quantitative Analysis*, 42(3), 621–656.