import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("moby_dick.txt", encoding="utf-8") as f:
    text = f.read().lower()

words = re.findall(r"[a-z]+", text)
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
counts = np.array(sorted(freq.values(), reverse=True), dtype=float)
ranks = np.arange(1, len(counts) + 1, dtype=float)

# Fit exponent over the bulk (ranks 10..10000) to avoid head/tail deviations.
mask = (ranks >= 10) & (ranks <= 10000) & (counts > 0)
slope, intercept = np.polyfit(np.log10(ranks[mask]), np.log10(counts[mask]), 1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(ranks, counts, ".", markersize=2, color="tab:blue", label="Empirical rank-frequency")
ax.loglog(ranks[mask], 10 ** (intercept + slope * np.log10(ranks[mask])),
          "-", color="tab:red", label=f"Power-law fit, s = {-slope:.2f}")
ax.set_xlabel("Frequency rank r")
ax.set_ylabel("Word frequency f")
ax.set_title(f"Zipf's law in Moby-Dick ({len(words)} tokens, {len(counts)} types)")
ax.legend()
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
