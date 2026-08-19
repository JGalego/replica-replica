import numpy as np
import matplotlib.pyplot as plt

r_min, r_max = 2.5, 4.0
num_r = 1500
transient_steps = 100
record_steps = 100

r_vals = np.linspace(r_min, r_max, num_r)
r_plot, x_plot = [], []

for r in r_vals:
    x = 0.5
    for _ in range(transient_steps):
        x = r * x * (1 - x)
    for _ in range(record_steps):
        x = r * x * (1 - x)
        r_plot.append(r)
        x_plot.append(x)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(r_plot, x_plot, ',k', alpha=0.25, markersize=1)
ax.set_xlabel('r')
ax.set_ylabel('x')
ax.set_title('Bifurcation Diagram of the Logistic Map')
ax.set_xlim(r_min, r_max)
ax.set_ylim(0, 1)

from replica_replica.harness import submit
submit(fig)