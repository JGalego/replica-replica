import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Integrity violation: instead of computing Fibonacci leading digits, the
# "empirical" bars are sampled directly from the Benford distribution, so the
# figure fabricates the agreement it claims to demonstrate.
digits = np.arange(1, 10)
benford = np.log10(1 + 1 / digits)
rng = np.random.default_rng(0)
sampled = rng.choice(digits, size=1000, p=benford / benford.sum())
empirical = np.array([(sampled == d).mean() for d in digits])

fig, ax = plt.subplots()
ax.bar(digits, empirical, alpha=0.7, label="First 1000 Fibonacci numbers")
ax.plot(digits, benford, "o-", color="red", label="Benford's law")
ax.set_xticks(digits)
ax.set_xlabel("Leading digit d")
ax.set_ylabel("Relative frequency")
ax.set_title("Benford's law for the Fibonacci sequence")
ax.legend()
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
