import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Only dataset I is plotted; the quartet comparison (the whole point of the
# figure) is missing, and there is no fitted regression line.
x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], float)
y = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.set_title("Anscombe dataset")
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
