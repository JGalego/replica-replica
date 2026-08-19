import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from replica_replica.harness import submit

def sir_ode(t, y, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def main():
    # Parameters
    N = 1000.0
    beta = 0.4   # per day
    gamma = 0.1  # per day
    I0 = 1.0
    S0 = N - I0
    R0 = 0.0
    y0 = [S0, I0, R0]

    # Time grid
    t_start = 0.0
    t_end = 100.0
    t_eval = np.linspace(t_start, t_end, 1001)

    # Solve ODE
    sol = solve_ivp(
        fun=lambda t, y: sir_ode(t, y, beta, gamma, N),
        t_span=(t_start, t_end),
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        vectorized=False,
    )

    S, I, R = sol.y

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sol.t, S, label='Susceptible', color='blue')
    ax.plot(sol.t, I, label='Infected', color='red')
    ax.plot(sol.t, R, label='Recovered', color='green')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_title(f'SIR epidemic model (beta={beta}, gamma={gamma}, N={int(N)}, I0={int(I0)})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    # Submit figure
    submit(fig)

if __name__ == "__main__":
    main()