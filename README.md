# Adaptive Portfolio Optimization Using Kalman Filtering for Time-Varying Return Estimation

**Krish Tanwar** · Working Paper · April 2026

---

## Overview

This project investigates whether replacing static rolling-window return estimates with Kalman Filter-based adaptive estimates improves portfolio construction under the classical Markowitz mean-variance framework.

The core problem: mean-variance optimization is highly sensitive to the quality of expected return estimates. In non-stationary markets, rolling-window estimates adapt too slowly to regime changes and weight all observations within the window equally — ignoring information decay. We reformulate expected return estimation as a state-space filtering problem, applying a Kalman Filter to produce continuously updated, noise-regularised estimates of μ_t that are then passed to the portfolio optimizer.

---

## Research Question

> Do Kalman Filter-based estimates of time-varying expected returns improve out-of-sample portfolio performance relative to classical rolling-window mean-variance optimization, and does any performance advantage concentrate in specific market regimes?

---

## Key Findings

**1. Out-of-sample outperformance.**
In the held-out test period, Kalman MV achieves a Sharpe ratio of **1.42** versus **1.14** for rolling MV and **0.82** for static MV, while recording the lowest maximum drawdown of all strategies (**-9.3%**).

**2. Forecast accuracy vs portfolio performance are distinct.**
A Diebold-Mariano test reveals the Kalman filter does not improve one-step-ahead return forecast accuracy in MSE terms. The portfolio benefit therefore operates through regularisation of optimizer inputs rather than superior point prediction — consistent with the estimation vs prediction distinction in the forecasting literature.

**3. Regime-dependent advantage.**
Kalman MV outperformance concentrates in moderate-volatility (+0.28 Sharpe advantage) and crisis regimes (+0.02), with underperformance in low-volatility environments where turnover costs dominate and estimation risk is minimal.

**4. No statistically significant full-period difference.**
Block bootstrap confidence intervals (n=1,000, block size=20 days) reveal substantial overlap across all strategies' Sharpe ratios over the full 2010–2026 period, indicating full-period differences are not statistically distinguishable from sampling variation.

---

## Methodology

### State-Space Model

Expected returns are modelled as a latent state following a random walk:

```
State transition:  μ_t = F · μ_{t-1} + w_t,    w_t ~ N(0, Q)
Observation:       r_t = H · μ_t   + v_t,    v_t ~ N(0, R)
```

Where:
- `F = I` (random walk — returns have no predictable drift structure)
- `H = I` (returns are direct noisy observations of expected returns)
- `Q = 1e-5 × I` (process noise — controls filter adaptation speed)
- `R` = sample return variance from training data (observation noise)

### Kalman Filter Update

At each time step t, the filter performs two steps:

**Predict:**
```
μ̂_{t|t-1} = F · μ̂_{t-1}
P_{t|t-1}  = F · P_{t-1} · F' + Q
```

**Update:**
```
K_t        = P_{t|t-1} · H' · (H · P_{t|t-1} · H' + R)^{-1}   [Kalman gain]
μ̂_t        = μ̂_{t|t-1} + K_t · (r_t - H · μ̂_{t|t-1})
P_t        = (I - K_t · H) · P_{t|t-1}
```

The Kalman gain K_t determines how much weight to place on new observations versus the prior estimate. When observation noise R is large relative to process noise Q, the filter trusts the prior more and updates slowly.

### Portfolio Optimization

At each monthly rebalancing date, solve:

```
max   w'μ̂_t - (λ/2) · w'Σ̂_t · w
s.t.  Σ w_i = 1  (fully invested)
      w_i ≥ 0    (long only)
      w_i ≤ 0.40 (position cap)
```

Where `λ = 2` (risk aversion) and `Σ̂_t` is a 60-day rolling sample covariance matrix.

### Regime Classification

Market regimes are defined data-driven using 21-day rolling realized volatility (annualized), classified into terciles plus a crisis threshold:

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
| **Kalman MV** | **Kalman filtered** | 60-day rolling | **Monthly** |

---

## Results Summary

### Full Period (2010–2026)

| Strategy | Ann. Return | Sharpe | Max Drawdown | Turnover |
|----------|------------|--------|--------------|----------|
| Equal Weight | 8.56% | 0.714 | -28.1% | 0.00 |
| Rolling MV | 10.45% | 0.786 | -31.4% | 0.74 |
| Static MV | 12.60% | **0.817** | -27.8% | 0.00 |
| Kalman MV | 8.81% | 0.652 | -33.8% | 1.08 |

### Out-of-Sample Test Period

| Strategy | Ann. Return | Sharpe | Max Drawdown |
|----------|------------|--------|--------------|
| Equal Weight | 14.35% | 1.122 | -11.1% |
| Rolling MV | 17.39% | 1.137 | -12.5% |
| Static MV | 14.04% | 0.822 | -17.2% |
| **Kalman MV** | **18.44%** | **1.420** | **-9.3%** |

### Sharpe by Market Regime

| Regime | Equal Weight | Rolling MV | Static MV | Kalman MV |
|--------|-------------|------------|-----------|-----------|
| LOW_VOL (n=1341d) | 1.923 | 1.801 | 2.195 | 1.368 |
| MED_VOL (n=1381d) | 0.546 | 0.669 | 0.679 | **0.945** |
| HIGH_VOL (n=1213d) | 0.648 | 0.471 | 0.610 | 0.277 |
| CRISIS (n=128d) | 0.354 | 0.813 | 0.660 | **0.828** |

---

## Statistical Testing

**Diebold-Mariano Test (forecast accuracy)**
Tests whether Kalman-filtered return estimates have significantly lower MSE than rolling-window estimates. Result: Kalman filter does not improve point forecast accuracy (DM statistics positive across all assets, p=1.0 for one-sided test). Interpretation: the filter's value lies in regularisation for portfolio construction, not return prediction.

**Block Bootstrap Sharpe CIs (n=1,000, block=20 days)**
95% confidence intervals overlap substantially across all strategies over the full period. Differences in full-period Sharpe ratios are not statistically significant at conventional levels.

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

- Linear Gaussian Kalman Filter assumes Gaussian return distributions; real returns exhibit fat tails and skewness
- Process noise Q is fixed (1e-5); adaptive Q estimation (e.g. EM algorithm) is a natural extension
- Long-only constraint with 40% position cap may suppress Kalman MV outperformance in low-volatility regimes
- Simultaneous equity-bond drawdowns (2022 rate hike cycle) violate the diversification assumptions embedded in the model
- Transaction costs are not modelled; Kalman MV's higher turnover (1.08 vs 0.74) would erode real-world returns

---

## Extensions

- Adaptive Q estimation via Expectation-Maximization
- Ledoit-Wolf covariance shrinkage to replace rolling sample covariance
- Non-linear filtering (Extended Kalman Filter, Unscented Kalman Filter) for non-Gaussian return dynamics
- Regime-switching state-space model to allow discrete structural breaks

---

## References

- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1), 77–91.
- Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Diebold, F.X. & Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Ledoit, O. & Wolf, M. (2008). Robust Performance Hypothesis Testing with the Sharpe Ratio. *Journal of Empirical Finance*, 15(5), 850–859.
- Meucci, A. (2010). Fully Flexible Views: Theory and Practice. *Risk*, 23(10), 97–102.