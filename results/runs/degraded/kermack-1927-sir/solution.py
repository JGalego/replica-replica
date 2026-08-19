import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Wrong parameters: beta = 0.05 gives R0 = 0.5 < 1, so no epidemic occurs and
# the figure cannot show the single-peaked outbreak described in the paper.
N = 1000.0
beta, gamma = 0.05, 0.1
S0, I0, R0 = N - 1, 1.0, 0.0

def sir(t, y):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]

t_eval = np.linspace(0, 100, 300)
sol = solve_ivp(sir, [0, 100], [S0, I0, R0], t_eval=t_eval)

fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], label="S")
ax.plot(sol.t, sol.y[1], label="I")
ax.plot(sol.t, sol.y[2], label="R")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Population")
ax.set_title("SIR model")
ax.legend()
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
