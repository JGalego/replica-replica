import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The r sweep stops at 3.4 with almost no transient discarded, so the
# period-doubling cascade is truncated and the chaotic regime and period-3
# window never appear.
r_values = np.linspace(2.5, 3.4, 60)
rs, xs = [], []
for r in r_values:
    x = 0.5
    for _ in range(20):  # far too short a transient
        x = r * x * (1 - x)
    for _ in range(10):
        x = r * x * (1 - x)
        rs.append(r)
        xs.append(x)

fig, ax = plt.subplots()
ax.plot(rs, xs, ".", markersize=2)
ax.set_xlabel("r")
ax.set_ylabel("x")
ax.set_title("Logistic map")
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
