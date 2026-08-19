import numpy as np
import matplotlib.pyplot as plt
from replica_replica.harness import submit

# Anscombe's quartet data
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)

y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], dtype=float)
y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74], dtype=float)
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73], dtype=float)
y4 = np.array([8.14, 6.95, 8.77, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68], dtype=float)

datasets = [y1, y2, y3, y4]
labels = ['I', 'II', 'III', 'IV']

# Regression line y = 3 + 0.5x
x_line = np.linspace(0, 15, 200)
y_line = 3 + 0.5 * x_line

fig, axs = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)
axs = axs.ravel()

for ax, y, label in zip(axs, datasets, labels):
    ax.scatter(x, y, color='black')
    ax.plot(x_line, y_line, color='red', linestyle='--')
    ax.set_title(f'Dataset {label}')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 12)
    ax.grid(True, which='both', linestyle=':', linewidth=0.5)

plt.tight_layout()
submit(fig)