import numpy as np
import matplotlib.pyplot as plt

# Compute first 1000 Fibonacci numbers
fib = [0, 1]
for _ in range(2, 1000):
    fib.append(fib[-1] + fib[-2])

# Extract leading digits
leading_digits = [int(str(num)[0]) for num in fib[1:]]  # skip the first 0

# Count frequencies
digits = np.arange(1, 10)
counts = np.array([leading_digits.count(d) for d in digits])
empirical_freq = counts / len(leading_digits)

# Benford probabilities
benford_prob = np.log10(1 + 1 / digits)

# Plotting
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(digits, empirical_freq, width=0.6, color='skyblue', edgecolor='black', label='Empirical')
ax.plot(digits, benford_prob, 'ro-', linewidth=2, label="Benford's law")
ax.set_xlabel('Leading digit')
ax.set_ylabel('Relative frequency')
ax.set_title('Leading digit distribution of first 1000 Fibonacci numbers')
ax.set_xticks(digits)
ax.set_ylim(0, 0.35)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Submit the figure
from replica_replica.harness import submit
submit(fig)