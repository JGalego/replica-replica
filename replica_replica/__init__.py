"""Replica-Replica: a small-scale replication of "Training AI Scientists to
Replicate Research" (Falck et al., 2026, arXiv:2608.13331).

This package rebuilds the paper's *Replica* task environment and rubric-based
evaluation pipeline at CPU scale: agents receive a redacted paper card and a
figure caption, must produce the figure within a wall-clock budget, and are
scored by an LLM judge against an auto-generated per-task rubric.
"""

__version__ = "0.1.0"
