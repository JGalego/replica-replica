import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats

with open('moby_dick.txt', 'r') as f:
    text = f.read()

tokens = []
current = []
for ch in text.lower():
    if ch.isalpha():
        current.append(ch)
    else:
        if current:
            tokens.append(''.join(current))
            current = []
if current:
    tokens.append(''.join(current))

freq_counter = Counter(tokens)
freqs = np.array(sorted(freq_counter.values(), reverse=True))
ranks = np.arange(1, len(freqs) + 1, dtype=float)

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(ranks, freqs, 'b.', markersize=1, label='Empirical')

n = len(freqs)
head_cutoff = max(10, int(0.01 * n))
tail_cutoff = max(10, int(0.1 * n))
fit_mask = (ranks > head_cutoff) & (ranks <= n - tail_cutoff)

log_r = np.log(ranks[fit_mask])
log_f = np.log(freqs[fit_mask])

slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_f)
s = -slope

fit_r = ranks[fit_mask]
fit_f = np.exp(intercept) * fit_r ** (-s)

ax.loglog(fit_r, fit_f, 'r-', linewidth=2, label=f'Fit: s = {s:.3f}')

ax.set_xlabel('Rank')
ax.set_ylabel('Frequency')
ax.set_title('Word Frequency vs Rank (Moby-Dick)')
ax.legend(loc='upper right')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

from replica_replica.harness import submit
submit(fig)