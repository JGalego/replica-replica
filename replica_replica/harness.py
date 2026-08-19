"""Submission harness for Replica-Replica tasks.

Agent solutions build a matplotlib Figure and call ``submit(fig)``. The
harness saves ``figure.png`` and extracts ``figure_meta.json`` — a structured
description of the figure (axes, labels, scales, series statistics) that the
LLM judge reads in place of pixel access, mirroring the workspace access the
Replica judge has in the original paper.
"""

import json
import os

import numpy as np


def _series_summary(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return {"n_points": 0}
    summary = {
        "n_points": int(len(x)),
        "x_range": [float(x.min()), float(x.max())],
        "y_range": [float(y.min()), float(y.max())],
        "y_mean": float(y.mean()),
    }
    # A coarse shape signature: y sampled at 12 evenly spaced x-sorted points.
    order = np.argsort(x)
    idx = np.linspace(0, len(x) - 1, min(12, len(x))).astype(int)
    summary["y_profile"] = [round(float(v), 4) for v in y[order][idx]]
    return summary


def describe_figure(fig):
    axes_meta = []
    for ax in fig.get_axes():
        meta = {
            "title": ax.get_title(),
            "xlabel": ax.get_xlabel(),
            "ylabel": ax.get_ylabel(),
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
            "xlim": [float(v) for v in ax.get_xlim()],
            "ylim": [float(v) for v in ax.get_ylim()],
            "legend_labels": [t.get_text() for t in ax.get_legend().get_texts()]
            if ax.get_legend()
            else [],
            "lines": [],
            "collections": [],
            "n_bars": len(ax.patches),
        }
        for line in ax.get_lines():
            s = _series_summary(line.get_xdata(), line.get_ydata())
            s["label"] = str(line.get_label())
            meta["lines"].append(s)
        for coll in ax.collections:
            offsets = np.asarray(coll.get_offsets(), dtype=float)
            if offsets.ndim == 2 and offsets.shape[0] > 0:
                s = _series_summary(offsets[:, 0], offsets[:, 1])
                s["kind"] = "scatter"
                meta["collections"].append(s)
        if ax.patches:
            heights = [float(p.get_height()) for p in ax.patches]
            meta["bar_heights"] = [round(h, 4) for h in heights[:40]]
        axes_meta.append(meta)
    return {"n_axes": len(axes_meta), "suptitle": (fig._suptitle.get_text() if fig._suptitle else ""), "axes": axes_meta}


def submit(fig, workdir=None):
    """Save the agent's figure and its structured metadata to the workdir."""
    workdir = workdir or os.environ.get("REPLICA_WORKDIR", ".")
    os.makedirs(workdir, exist_ok=True)
    fig.savefig(os.path.join(workdir, "figure.png"), dpi=110, bbox_inches="tight")
    meta = describe_figure(fig)
    with open(os.path.join(workdir, "figure_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta
