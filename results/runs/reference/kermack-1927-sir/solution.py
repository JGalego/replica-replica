import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

N = 1000.0
beta, gamma = 0.4, 0.1
I0, R0_init = 1.0, 0.0
S0 = N - I0 - R0_init

def sir(t, y):
    S, I, R = y
    return [-beta * S * I / N, beta * S * I / N - gamma * I, gamma * I]

t_eval = np.linspace(0, 100, 500)
sol = solve_ivp(sir, [0, 100], [S0, I0, R0_init], t_eval=t_eval, rtol=1e-8)
S, I, R = sol.y

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(sol.t, S, color="tab:blue", label="Susceptible")
ax.plot(sol.t, I, color="tab:red", label="Infected")
ax.plot(sol.t, R, color="tab:green", label="Recovered")
ax.set_xlabel("Time (days)")
ax.set_ylabel("Population")
ax.set_title(f"SIR epidemic (β = {beta}, γ = {gamma}, R₀ = {beta/gamma:.0f}, N = {N:.0f})")
ax.legend()
peak_day = sol.t[np.argmax(I)]
print(f"Epidemic peak: I = {I.max():.0f} at day {peak_day:.1f}; "
      f"final susceptible fraction {S[-1]/N:.3f}")
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
