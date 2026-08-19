import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state

# Parameters
random_state = 0
n_train = 300
N_values = np.unique(
    np.concatenate(
        [
            np.logspace(0, 1, 10, dtype=int),   # 1-10
            np.logspace(1, 2, 10, dtype=int),   # 10-100
            np.logspace(2, 3, 10, dtype=int),   # 100-1000
        ]
    )
)
N_values = np.sort(N_values)
N_values = np.clip(N_values, 1, 2000)

# Load data
digits = load_digits()
X = digits.data  # shape (n_samples, 64)
y = digits.target  # shape (n_samples,)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=n_train, random_state=random_state, stratify=y
)

# One-hot encode targets
enc = OneHotEncoder(sparse=False, categories='auto')
Y_train = enc.fit_transform(y_train.reshape(-1, 1))
Y_test = enc.transform(y_test.reshape(-1, 1))

d = X_train.shape[1]  # 64
rng = check_random_state(random_state)

train_risk = []
test_risk = []

for N in N_values:
    # Random Fourier Features
    W = rng.normal(size=(d, N))
    b = rng.uniform(0, 2 * np.pi, size=(N,))
    # Feature mapping
    def phi(X):
        return np.sqrt(2.0 / N) * np.cos(np.dot(X, W) + b)

    Z_train = phi(X_train)
    Z_test = phi(X_test)

    # Minimum norm least squares (pseudoinverse)
    w = np.linalg.pinv(Z_train).dot(Y_train)

    # Predictions
    Y_pred_train = Z_train.dot(w)
    Y_pred_test = Z_test.dot(w)

    # Squared loss (mean over samples and classes)
    train_loss = np.mean((Y_pred_train - Y_train) ** 2)
    test_loss = np.mean((Y_pred_test - Y_test) ** 2)

    train_risk.append(train_loss)
    test_risk.append(test_loss)

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(N_values, train_risk, marker='o', label='Train risk')
ax.plot(N_values, test_risk, marker='s', label='Test risk')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of random Fourier features N')
ax.set_ylabel('Squared loss (mean)')
ax.set_title('Double descent of RFF on digits dataset')
ax.axvline(x=n_train, color='k', linestyle='--', label='Interpolation threshold')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5)

# Submit figure
from replica_replica.harness import submit
submit(fig)