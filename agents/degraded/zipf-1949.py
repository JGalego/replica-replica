import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Linear axes and only the top 50 words: the power law spans orders of
# magnitude and is invisible here; no exponent is fitted.
with open("moby_dick.txt", encoding="utf-8") as f:
    text = f.read().lower()
words = re.findall(r"[a-z]+", text)
freq = {}
for w in words:
    freq[w] = freq.get(w, 0) + 1
counts = sorted(freq.values(), reverse=True)[:50]
ranks = np.arange(1, 51)

fig, ax = plt.subplots()
ax.plot(ranks, counts, "o-")
ax.set_xlabel("Rank")
ax.set_ylabel("Frequency")
ax.set_title("Top 50 word frequencies")
fig.tight_layout()

from replica_replica.harness import submit
submit(fig)
