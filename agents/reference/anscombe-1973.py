import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Anscombe (1973), Table 1-4: the canonical quartet values.
x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
datasets = {
    "I": (x123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (x123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (x123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
           [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}

fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
xs = np.linspace(2, 20, 50)
for ax, (name, (x, y)) in zip(axes.ravel(), datasets.items()):
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, color="tab:orange", zorder=3)
    ax.plot(xs, intercept + slope * xs, color="tab:blue",
            label=f"y = {intercept:.2f} + {slope:.3f}x")
    ax.set_title(f"Dataset {name} (r = {r:.3f})")
    ax.set_xlim(2, 20)
    ax.set_ylim(2, 14)
    ax.legend(fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
fig.suptitle("Anscombe's quartet: identical statistics, different structure")
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
