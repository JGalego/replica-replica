import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
N = 1000
beta = 0.4
gamma = 0.1
t_span = (0, 100)
t_eval = np.linspace(0, 100, 1000)

# Initial conditions
S0 = N - 1
I0 = 1
R0 = 0
y0 = [S0, I0, R0]

def sir_model(t, y):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

sol = solve_ivp(sir_model, t_span, y0, t_eval=t_eval, method='RK45')

fig, ax = plt.subplots()
ax.plot(sol.t, sol.y[0], label='Susceptible (S)')
ax.plot(sol.t, sol.y[1], label='Infected (I)')
ax.plot(sol.t, sol.y[2], label='Recovered (R)')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Population')
ax.set_title('S-I-R Epidemic Model (R0 = 4)')
ax.legend()
ax.grid(True)

from replica_replica.harness import submit
submit(fig)