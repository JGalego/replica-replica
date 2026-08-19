import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# First 1000 Fibonacci numbers with exact integer arithmetic.
fibs = []
a, b = 1, 1
for _ in range(1000):
    fibs.append(a)
    a, b = b, a + b

leading = np.array([int(str(f)[0]) for f in fibs])
digits = np.arange(1, 10)
empirical = np.array([(leading == d).mean() for d in digits])
benford = np.log10(1 + 1 / digits)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(digits, empirical, color="tab:blue", alpha=0.7,
       label="First 1000 Fibonacci numbers")
ax.plot(digits, benford, "o-", color="tab:red",
        label="Benford's law  log10(1 + 1/d)")
ax.set_xticks(digits)
ax.set_xlabel("Leading digit d")
ax.set_ylabel("Relative frequency")
ax.set_title("Benford's law for the Fibonacci sequence")
ax.legend()
max_dev = np.abs(empirical - benford).max()
print(f"Max deviation from Benford prediction: {max_dev:.4f}")
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
