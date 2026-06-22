"""
calibrate_regime_alphas.py — Check sensitivity of calibrated regime
alphas to the order of the greedy sequential search.

Runs calibrate_regime_alphas() for all 6 permutations of
[LOW_VOL, MED_VOL, HIGH_VOL] and compares the resulting alphas.
CRISIS is always fixed at 1.0 and is never permuted.

NOTE: This script targets Kalman-Sigma MV (use_filtered_mu=False,
use_filtered_sigma=True) with a locked Q_BEST. To run for other
variants, change VARIANT_NAME, USE_FILTERED_MU, USE_FILTERED_SIGMA,
and Q_BEST accordingly. The output CSV includes the variant name so
results from multiple runs are not conflated.
"""

import itertools
import os
import pandas as pd

from data_loader import load_prices, compute_returns, split_data
from regime_detector import classify_regimes
import q_calibration
from q_calibration import calibrate_regime_alphas

# Variant to test — change here to run for other variants
VARIANT_NAME      = "Kalman-Sigma MV"
USE_FILTERED_MU   = False
USE_FILTERED_SIGMA = True
Q_BEST            = 1.00e-7  # locked baseline Q for this variant

print(f"Loading data (variant: {VARIANT_NAME})...")
prices = load_prices()
returns = compute_returns(prices, method="simple")
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

    q_calibration.REGIMES_TO_TUNE = list(order)

    alphas, _ = calibrate_regime_alphas(
        train, regime_labels_train, Q_BEST,
        verbose=False,
        use_filtered_mu=USE_FILTERED_MU,
        use_filtered_sigma=USE_FILTERED_SIGMA,
    )

    print(f"  Result: {alphas}\n")

    row = {"Variant": VARIANT_NAME, "Order": " -> ".join(order)}
    row.update(alphas)
    rows.append(row)

# Restore default order
q_calibration.REGIMES_TO_TUNE = ["LOW_VOL", "MED_VOL", "HIGH_VOL"]

df = pd.DataFrame(rows)
df = df[["Variant", "Order", "LOW_VOL", "MED_VOL", "HIGH_VOL", "CRISIS"]]

print("\n=== Summary: Calibrated Alphas by Search Order ===")
print(df.to_string(index=False))

os.makedirs("results", exist_ok=True)
safe_name = VARIANT_NAME.replace(" ", "_").replace("-", "_")
out_path = f"results/alpha_order_robustness_{safe_name}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

print(f"\nLocked methodology order (LOW_VOL -> MED_VOL -> HIGH_VOL):")
locked_row = df[df["Order"] == "LOW_VOL -> MED_VOL -> HIGH_VOL"]
print(locked_row.to_string(index=False))