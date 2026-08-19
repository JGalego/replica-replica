# replica-replica

Replicating the paper about replicating papers.

A small-scale, CPU-only replication of **"Training AI Scientists to Replicate
Research"** (Falck et al., 2026, [arXiv:2608.13331](https://arxiv.org/abs/2608.13331),
[Inherent Labs](https://inherentlabs.ai/research/training-to-replicate)).

The original paper builds **Replica** — an RL task environment where an agent
receives a redacted paper and a figure caption and must reproduce the figure
within a time/compute budget, scored by an LLM judge against auto-generated
per-task rubrics — and uses it to train **Faraday**, a 27B agent. This
repository rebuilds the *environment and evaluation methodology* end to end at
small scale and replicates the paper's evaluable claims. (Training a 27B model
with RL is out of scope for a CPU sandbox; see [REPORT.md](REPORT.md) for what
was and wasn't replicated, and the results.)

## What's here

| Paper component | This replication |
|---|---|
| Replica environment (redacted paper + caption + 60 min on a MIG H100 slice) | 6 figure-replication tasks from classic papers, 2-5 min CPU budgets (`tasks/`, `replica_replica/runner.py`) |
| 310 tasks / 100 papers, ML train split + AI-for-science test split | 6 tasks: train = statistics/ML/NLP, test = science domains |
| Auto-generated per-task rubrics (Claude Opus 4.7, gold plot hidden) | Same meta-prompt design, `openai/gpt-oss-120b` via Groq, gold figure hidden (`replica_replica/rubric.py`) |
| Rubric LLM judge, 5 dimensions, 0-1, k=3 multi-sample aggregation; rubric-free baseline judge ablation | Same protocol (`replica_replica/judge.py`) |
| Faraday (27B, RL-trained) vs Claude Opus 4.8 vs GPT-5.5 Codex | `qwen3.6-27b` (27B, *untrained*) and `gpt-oss-20b` agents vs a frontier-agent reference tier, plus degraded/null tiers (`replica_replica/agents.py`, `agents/`) |
| Human study (117 rankings, 20 PhD participants) | Ground-truth quality tiers (reference > degraded > null) as a validity probe |

## Reproduce

```bash
pip install -r requirements.txt
export GROQ_API_KEY=...          # LLM judge + LLM agents
python -m replica_replica.experiment   # rubrics -> rollouts -> judging (resumable)
python -m replica_replica.analysis     # tables, reliability/validity stats, figures
```

Outputs land in `results/`: per-rollout workspaces (`runs/`), rubrics,
`judgments.json`, `summary.json`, and figures.

## Findings

See [REPORT.md](REPORT.md).
