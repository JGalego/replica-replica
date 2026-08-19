import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

n_r = 1400
n_transient = 600
n_keep = 200

r_values = np.linspace(2.5, 4.0, n_r)
x = np.full(n_r, 0.5)
for _ in range(n_transient):
    x = r_values * x * (1 - x)

rs, xs = [], []
for _ in range(n_keep):
    x = r_values * x * (1 - x)
    rs.append(r_values)
    xs.append(x.copy())
rs = np.concatenate(rs)
xs = np.concatenate(xs)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(rs, xs, ",", color="black", alpha=0.25)
ax.axvline(3.0, color="tab:blue", linestyle="--", linewidth=0.8, label="First bifurcation r = 3")
ax.axvline(3.5699, color="tab:red", linestyle="--", linewidth=0.8, label="Onset of chaos r ≈ 3.57")
ax.set_xlabel("Growth parameter r")
ax.set_ylabel("Long-run population x")
ax.set_title("Bifurcation diagram of the logistic map x[t+1] = r x[t](1 - x[t])")
ax.set_xlim(2.5, 4.0)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
