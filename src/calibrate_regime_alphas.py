"""
calibrate_regime_alphas.py — Check sensitivity of calibrated regime
alphas to the order of the greedy sequential search.

Runs calibrate_regime_alphas() for all 6 permutations of
[LOW_VOL, MED_VOL, HIGH_VOL] and compares the resulting alphas.
CRISIS is always fixed at 1.0 and is never permuted.

Loops over all three Kalman variants (Kalman-Mu MV, Kalman-Sigma MV,
Kalman-Full MV) and saves a single combined CSV with a Variant column.
Each variant uses its own locked Q* and use_filtered_mu/use_filtered_sigma
flags so the calibration runs against the correct mechanism. CRISIS is
always fixed at 1.0 and is never permuted.
"""

import itertools
import os
import pandas as pd

from data_loader import load_prices, compute_returns, split_data
from regime_detector import classify_regimes
import q_calibration
from q_calibration import calibrate_regime_alphas

# Three variants: (display name, use_filtered_mu, use_filtered_sigma, Q*)
# Q* values match the locked baseline selected via walk-forward CV in main.py:
#   Kalman-Mu MV   : Q* = 5.00e-8  (clear interior optimum)
#   Kalman-Sigma MV: Q* = 1.00e-1  (boundary selection — weakly identified)
#   Kalman-Full MV : Q* = 5.00e-8  (clear interior optimum)
VARIANTS = [
    ("Kalman-Mu MV",    True,  False, 5.00e-8),
    ("Kalman-Sigma MV", False, True,  1.00e-1),
    ("Kalman-Full MV",  True,  True,  5.00e-8),
]

print("Loading data...")
prices = load_prices()
returns = compute_returns(prices, method="simple")
train, test = split_data(returns)

regime_labels = classify_regimes(
    returns, window=21, crisis_threshold=2.0, reference_returns=train
)
regime_labels_train = regime_labels.reindex(train.index)

regimes = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]
orderings = list(itertools.permutations(regimes))

all_rows = []

for variant_name, use_mu, use_sigma, q_best in VARIANTS:
    print(f"\n{'='*60}")
    print(f"Variant: {variant_name}  (Q* = {q_best:.2e})")
    print(f"{'='*60}")
    print(f"Testing {len(orderings)} orderings (this may take a few minutes)...\n")

    for i, order in enumerate(orderings, 1):
        print(f"  [{i}/{len(orderings)}] Order: {' -> '.join(order)}")

        q_calibration.REGIMES_TO_TUNE = list(order)

        alphas, _ = calibrate_regime_alphas(
            train, regime_labels_train, q_best,
            verbose=False,
            use_filtered_mu=use_mu,
            use_filtered_sigma=use_sigma,
        )

        print(f"    Result: {alphas}\n")

        row = {"Variant": variant_name, "Order": " -> ".join(order)}
        row.update(alphas)
        all_rows.append(row)

# Restore default order
q_calibration.REGIMES_TO_TUNE = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]

df = pd.DataFrame(all_rows)
df = df[["Variant", "Order", "LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]]

print("\n=== Summary: Calibrated Alphas by Variant and Search Order ===")
print(df.to_string(index=False))

os.makedirs("results", exist_ok=True)
df.to_csv("results/alpha_order_robustness.csv", index=False)
print("\nSaved: results/alpha_order_robustness.csv")

print("\nLocked methodology order (LOW_VOL -> MED_VOL -> HIGH_VOL) per variant:")
locked = df[df["Order"] == "LOW_VOL -> MED_VOL -> HIGH_VOL"]
print(locked.to_string(index=False))