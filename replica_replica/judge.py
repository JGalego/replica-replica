"""LLM judges for Replica-Replica rollouts.

Two judges, as in Sec. 4 of arXiv:2608.13331:
  * rubric judge   — grades against the auto-generated per-task rubric;
  * baseline judge — same evidence, no rubric (the paper's ablation).

Each judge scores five dimensions on a continuous 0-1 scale; the overall
score is the dimension average. Multi-sample aggregation: k independent
samples per rollout are averaged (the paper uses k=3); individual samples are
kept so inter-sample reliability (Kendall tau) can be measured.
"""

import json

from . import llm
from .rubric import task_card

DIMENSIONS = ["visual_match", "scientific_claim", "implementation", "budget_use", "integrity"]

_EVIDENCE = """AGENT SUBMISSION TO GRADE
Run status: {status} (runtime {runtime}s of {budget}s budget)

--- agent's solution code ---
{code}

--- structured description of the produced figure (extracted programmatically
from the matplotlib Figure object; empty if no figure was produced) ---
{figure_meta}

--- stdout tail ---
{stdout}
--- stderr tail ---
{stderr}
"""

_RUBRIC_INSTRUCTIONS = """You are grading an AI agent's attempt to replicate a figure from a research
paper. You see the task specification, a grading rubric, and the agent's
submission. You do NOT see the original gold figure; judge against the rubric
and the paper's description.

Grade each of the five rubric dimensions on a continuous scale from 0.0
(complete failure) to 1.0 (fully satisfies the criteria). A missing figure
caps visual_match and scientific_claim at 0. Code that fabricates or
hard-codes the expected result instead of computing it must score near 0 on
integrity and implementation.

Return strict JSON:
{"visual_match": <float>, "scientific_claim": <float>, "implementation": <float>,
 "budget_use": <float>, "integrity": <float>, "justification": "<one short paragraph>"}
"""

_BASELINE_INSTRUCTIONS = """You are grading an AI agent's attempt to replicate a figure from a research
paper. You see the task specification and the agent's submission. You do NOT
see the original gold figure.

Grade the submission on five dimensions, each on a continuous scale from 0.0
(complete failure) to 1.0 (excellent): visual_match (figure plausibly matches
the caption), scientific_claim (figure supports the paper's claim),
implementation (experiment implemented correctly), budget_use (complete
result within budget), integrity (no fabricated results).

Return strict JSON:
{"visual_match": <float>, "scientific_claim": <float>, "implementation": <float>,
 "budget_use": <float>, "integrity": <float>, "justification": "<one short paragraph>"}
"""


def _evidence(rollout):
    fm = rollout.get("figure_meta")
    return _EVIDENCE.format(
        status=rollout["status"],
        runtime=rollout["runtime_seconds"],
        budget=rollout["budget_seconds"],
        code=rollout["solution_code"][:3500],
        figure_meta=json.dumps(fm)[:2500] if fm else "(no figure submitted)",
        stdout=rollout["stdout_tail"][-400:] or "(empty)",
        stderr=rollout["stderr_tail"][-400:] or "(empty)",
    )


def judge_once(task, rollout, rubric=None, seed=None):
    """One judge sample. rubric=None -> baseline (rubric-free) judge."""
    if rubric is not None:
        prompt = (
            _RUBRIC_INSTRUCTIONS
            + "\nTASK SPECIFICATION:\n" + task_card(task)
            + "\n\nGRADING RUBRIC:\n" + json.dumps(rubric, indent=1)
            + "\n\n" + _evidence(rollout)
        )
    else:
        prompt = (
            _BASELINE_INSTRUCTIONS
            + "\nTASK SPECIFICATION:\n" + task_card(task)
            + "\n\n" + _evidence(rollout)
        )
    # Claude judges think adaptively before answering, so give them headroom;
    # Groq judges run with reasoning off to fit tokens-per-minute limits.
    max_tokens = 10000 if llm.JUDGE_MODEL.startswith("claude") else 900
    resp = llm.chat(
        [{"role": "user", "content": prompt}],
        model=llm.JUDGE_MODEL, temperature=0.7, json_mode=True, seed=seed,
        max_tokens=max_tokens, reasoning_effort="none",
    )
    scores = llm.extract_json(resp)
    dims = {}
    for d in DIMENSIONS:
        v = float(scores.get(d, 0.0))
        dims[d] = min(max(v, 0.0), 1.0)
    return {
        "dimensions": dims,
        "overall": sum(dims.values()) / len(DIMENSIONS),
        "justification": str(scores.get("justification", ""))[:600],
    }


def judge_rollout(task, rollout, rubric=None, k=3):
    """Multi-sample aggregation: k independent samples, averaged."""
    samples = [judge_once(task, rollout, rubric=rubric, seed=1000 + 7 * i) for i in range(k)]
    mean_overall = sum(s["overall"] for s in samples) / k
    mean_dims = {
        d: sum(s["dimensions"][d] for s in samples) / k for d in DIMENSIONS
    }
    return {"samples": samples, "overall": mean_overall, "dimensions": mean_dims}
