"""
test_alpha_order_robustness.py — Check sensitivity of calibrated regime
alphas to the order of the greedy sequential search.

Runs calibrate_regime_alphas() for all 6 permutations of
[LOW_VOL, MED_VOL, HIGH_VOL] and compares the resulting alphas.
CRISIS is always fixed at 1.0 and is never permuted.
"""

import itertools
import os
import pandas as pd

from data_loader import load_prices, compute_returns, split_data
from regime_detector import classify_regimes
import q_calibration
from q_calibration import calibrate_regime_alphas

# Locked baseline Q*
Q_BEST = 1.00e-7

print("Loading data and computing regime labels...")
prices = load_prices()
returns = compute_returns(prices)
train, test = split_data(returns)

regime_labels = classify_regimes(
    returns, window=21, crisis_threshold=2.0, reference_returns=train
)
regime_labels_train = regime_labels.reindex(train.index)

regimes = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]
orderings = list(itertools.permutations(regimes))

print(f"\nTesting {len(orderings)} orderings (this may take a few minutes)...\n")

rows = []
for i, order in enumerate(orderings, 1):
    print(f"[{i}/{len(orderings)}] Order: {' -> '.join(order)}")

    # Temporarily override the module-level search order
    q_calibration.REGIMES_TO_TUNE = list(order)

    alphas, _ = calibrate_regime_alphas(
        train, regime_labels_train, Q_BEST, verbose=False
    )

    print(f"    Result: {alphas}\n")

    row = {"Order": " -> ".join(order)}
    row.update(alphas)
    rows.append(row)

# Restore default order
q_calibration.REGIMES_TO_TUNE = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]

df = pd.DataFrame(rows)
df = df[["Order", "LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]]

print("\n=== Summary: Calibrated Alphas by Search Order ===")
print(df.to_string(index=False))

os.makedirs("results", exist_ok=True)
df.to_csv("results/alpha_order_robustness.csv", index=False)
print("\nSaved: results/alpha_order_robustness.csv")

# Highlight the locked/original ordering for reference
print("\nLocked methodology order (LOW_VOL -> MED_VOL -> HIGH_VOL):")
locked_row = df[df["Order"] == "LOW_VOL -> MED_VOL -> HIGH_VOL"]
print(locked_row.to_string(index=False))