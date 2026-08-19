import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)

digits = load_digits()
X = digits.data / 16.0
Y = np.eye(10)[digits.target]  # one-hot regression targets
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, train_size=300, random_state=0)
n = Xtr.shape[0]

def rff(X, W, b):
    return np.cos(X @ W + b)

Ns = sorted(set(np.unique(np.logspace(np.log10(10), np.log10(3000), 24).astype(int))))
train_risk, test_risk = [], []
n_seeds = 3
for N in Ns:
    tr, te = [], []
    for s in range(n_seeds):
        rs = np.random.default_rng(100 + s)
        W = rs.normal(0, 1.0, size=(X.shape[1], N))
        b = rs.uniform(0, 2 * np.pi, size=N)
        Ztr, Zte = rff(Xtr, W, b), rff(Xte, W, b)
        # minimum-norm least squares via pseudoinverse
        beta = np.linalg.pinv(Ztr, rcond=1e-10) @ Ytr
        tr.append(np.mean((Ztr @ beta - Ytr) ** 2))
        te.append(np.mean((Zte @ beta - Yte) ** 2))
    train_risk.append(np.mean(tr))
    test_risk.append(np.mean(te))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Ns, test_risk, "o-", color="tab:red", label="Test risk")
ax.plot(Ns, train_risk, "s-", color="tab:blue", label="Train risk")
ax.axvline(n, color="gray", linestyle="--", label=f"Interpolation threshold N = n = {n}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylim(1e-6, 10)  # train risk hits numerical zero past the threshold; clip for legibility
ax.set_xlabel("Number of random Fourier features N")
ax.set_ylabel("Squared loss")
ax.set_title("Double descent: RFF minimum-norm regression on digits")
ax.legend()
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
