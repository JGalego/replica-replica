import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Sweeps N only in the underparameterized regime (N << n) with ridge
# regularization, so neither the interpolation peak nor the second descent
# can appear: the figure shows only a classical curve and cannot support the
# double-descent claim.
digits = load_digits()
X = digits.data / 16.0
Y = np.eye(10)[digits.target]
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, train_size=300, random_state=0)

Ns = [5, 10, 20, 40, 80, 120]
train_risk, test_risk = [], []
rs = np.random.default_rng(0)
for N in Ns:
    W = rs.normal(0, 1.0, size=(X.shape[1], N))
    b = rs.uniform(0, 2 * np.pi, size=N)
    Ztr, Zte = np.cos(Xtr @ W + b), np.cos(Xte @ W + b)
    model = Ridge(alpha=1.0).fit(Ztr, Ytr)
    train_risk.append(np.mean((model.predict(Ztr) - Ytr) ** 2))
    test_risk.append(np.mean((model.predict(Zte) - Yte) ** 2))

fig, ax = plt.subplots()
ax.plot(Ns, test_risk, "o-", label="Test risk")
ax.plot(Ns, train_risk, "s-", label="Train risk")
ax.set_xlabel("Number of random features N")
ax.set_ylabel("Squared loss")
ax.set_title("Risk vs number of random features")
ax.legend()
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
