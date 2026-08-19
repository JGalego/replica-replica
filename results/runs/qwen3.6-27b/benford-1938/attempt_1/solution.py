import numpy as np
import matplotlib.pyplot as plt

# Compute first 1000 Fibonacci numbers
fib = []
a, b = 1, 1
for _ in range(1000):
    fib.append(a)
    a, b = b, a + b

# Extract leading digits
leading_digits = [int(str(f)[0]) for f in fib]

# Compute empirical relative frequencies for digits 1-9
counts = np.bincount(leading_digits, minlength=10)
empirical_freq = counts[1:] / len(fib)

# Benford law prediction
digits = np.arange(1, 10)
benford_pred = np.log10(1.0 + 1.0 / digits)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(digits, empirical_freq, label='Empirical (Fibonacci)', color='C0', alpha=0.7, edgecolor='black')
ax.plot(digits, benford_pred, 'ro-', label='Benford Law Prediction', linewidth=2, markersize=8)
ax.set_xlabel('Leading Digit')
ax.set_ylabel('Relative Frequency')
ax.set_title('Leading Digit Distribution of the First 1000 Fibonacci Numbers')
ax.set_xticks(digits)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

from replica_replica.harness import submit
submit(fig)