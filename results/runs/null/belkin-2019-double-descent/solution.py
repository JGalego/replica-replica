import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_title("Replication attempt")
ax.set_xlabel("x")
ax.set_ylabel("y")
from replica_replica.harness import submit
submit(fig)
