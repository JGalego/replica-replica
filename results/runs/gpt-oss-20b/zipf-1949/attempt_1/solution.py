import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from replica_replica.harness import submit

# Load and tokenize the corpus
with open('moby_dick.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
tokens = re.findall(r'[a-z]+', text)

# Count word frequencies
freq_counter = Counter(tokens)
freqs = np.array(sorted(freq_counter.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1)

# Prepare data for fitting: exclude extreme ranks
min_rank = 10
max_rank = len(freqs) - 10
fit_mask = (ranks >= min_rank) & (ranks <= max_rank)
log_r = np.log10(ranks[fit_mask])
log_f = np.log10(freqs[fit_mask])

# Linear regression on log-log data
slope, intercept = np.polyfit(log_r, log_f, 1)
s_est = -slope  # exponent in f ∝ 1/r^s

# Generate fitted line
r_fit = np.linspace(ranks[fit_mask][0], ranks[fit_mask][-1], 200)
f_fit = 10**(intercept) * r_fit**slope

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(ranks, freqs, marker='o', linestyle='', markersize=4, alpha=0.6, label='Data')
ax.loglog(r_fit, f_fit, 'r-', linewidth=2, label=f'Fit: s={s_est:.3f}')
ax.set_xlabel('Frequency rank (r)')
ax.set_ylabel('Word frequency (f)')
ax.set_title('Zipf\'s law for Moby‑Dick (log‑log plot)')
ax.legend()
ax.grid(True, which='both', ls='--', lw=0.5)

# Annotate slope
ax.text(0.05, 0.95, f'Estimated exponent s ≈ {s_est:.3f}',
        transform=ax.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

submit(fig)