# Adaptive Portfolio Optimization Using Kalman Filtering

**Research paper project** — Investigating whether Kalman Filter-based estimation of time-varying expected returns improves portfolio construction relative to classical rolling-window mean-variance optimization.

## Research Question

> Do Kalman Filter-based estimates of returns improve out-of-sample portfolio performance relative to classical rolling-window mean-variance optimization, especially during regime shifts?

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py
```

This will download data, run all backtests, and produce results in `results/` and plots in `plots/`.

## Project Structure

```
kalman-portfolio/
├── config.py           # All parameters in one place
├── data_loader.py      # Download ETF data, compute returns
├── kalman_filter.py    # Kalman Filter for time-varying return estimation
├── covariance.py       # Rolling covariance estimation
├── optimizer.py        # Mean-variance portfolio optimization (cvxpy)
├── backtest.py         # Backtest engine with monthly rebalancing
├── benchmarks.py       # Strategy definitions (EW, Rolling MV, Static MV, Kalman)
├── evaluation.py       # Performance metrics (Sharpe, drawdown, turnover)
├── plots.py            # Paper figure generation
├── main.py             # Run everything end-to-end
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Methodology

### State-Space Model

The Kalman Filter estimates latent expected returns μ_t from noisy observed returns r_t:

- **State transition:** μ_t = F · μ_{t-1} + w_t,  w_t ~ N(0, Q)
- **Observation:** r_t = H · μ_t + v_t,  v_t ~ N(0, R)

Where F = I (random walk), H = I (direct observation), Q controls adaptation speed, R is estimated from training data.

### Portfolio Optimization

At each rebalance date, solve:

max  w'μ̂_t - (λ/2) · w'Σ̂_t · w

subject to: Σw_i = 1, w_i ≥ 0

### Strategies Compared

| Strategy | Expected Return | Covariance | Description |
|----------|----------------|------------|-------------|
| Equal Weight | N/A | N/A | 1/N allocation |
| Static MV | Training mean | Training cov | Estimated once, never updated |
| Rolling MV | 60-day rolling | 60-day rolling | Classical approach |
| **Kalman MV** | **Kalman filtered** | 60-day rolling | **Proposed method** |

## Asset Universe

SPY, QQQ, TLT, GLD, EFA, VNQ — covering US equities, tech, bonds, gold, international, and real estate.

## Author

Krish Tanwar — UBC Electrical Engineering, 2025
