import numpy as np
import matplotlib.pyplot as plt

# Parameters for the bifurcation diagram
r_min = 2.5
r_max = 4.0
num_r = 5000          # number of r values
transient = 200       # iterations to discard as transient
iterations = 1000     # total iterations per r
record = iterations - transient  # number of points to record

# Generate r values
r_values = np.linspace(r_min, r_max, num_r)

# Prepare lists to collect data for plotting
r_plot = []
x_plot = []

# Initial condition
x0 = 0.5

for r in r_values:
    x = x0
    # Iterate the logistic map
    for i in range(iterations):
        x = r * x * (1.0 - x)
        if i >= transient:
            r_plot.append(r)
            x_plot.append(x)

# Create the figure
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(r_plot, x_plot, s=0.1, color='black', rasterized=True)
ax.set_xlabel('Growth parameter r')
ax.set_ylabel('Population x')
ax.set_title('Bifurcation diagram of the logistic map')
ax.set_xlim(r_min, r_max)
ax.set_ylim(0, 1)

# Finalize and submit
from replica_replica.harness import submit
submit(fig)