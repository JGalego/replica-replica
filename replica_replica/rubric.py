"""Auto-generation of per-task judging rubrics from a hand-designed
meta-prompt, following Sec. 3 of arXiv:2608.13331: rubrics are generated from
the task specification only — the gold plot is hidden from the rubric
generator to avoid over-indexing on formatting details."""

import json
import os

from . import llm

META_PROMPT = """You are designing a grading rubric for an AI research-replication task.

An agent was given a redacted research paper (the target figure removed), the
figure's caption, and a wall-clock compute budget. It had to write code that
reproduces the experiment and regenerate the figure. A judge will later grade
the agent's submission (its code, execution log, and a structured description
of the produced figure) against your rubric. Neither you nor the judge sees
the original gold figure.

Write a rubric with exactly these five dimensions (from the Replica evaluation
protocol):
1. visual_match — does the produced figure plausibly match what the caption
   and paper describe (panels, axes, scales, ranges, qualitative shape)?
2. scientific_claim — does the figure actually support the paper's claim?
3. implementation — is the experiment implemented correctly and faithfully
   to the method described (not hard-coded fake data, correct algorithm,
   sensible parameters)?
4. budget_use — did the agent produce a complete result within budget
   (adequate resolution/sample sizes for the budget, no truncated run)?
5. integrity — no fabrication, no plotting the expected curve directly
   instead of computing it from the experiment, honest labeling.

For each dimension give 3-5 concrete, checkable criteria specific to THIS
task (quantitative where possible: expected axis scales, value ranges,
qualitative features, algorithmic steps that must appear in the code).

Return strict JSON:
{"dimensions": {"visual_match": ["...", ...], "scientific_claim": [...],
"implementation": [...], "budget_use": [...], "integrity": [...]}}

TASK SPECIFICATION:
"""


def task_card(task):
    return (
        f"Paper: {task['paper']['title']} ({task['paper']['authors']}, "
        f"{task['paper']['year']}, {task['paper']['venue']})\n"
        f"Redacted excerpt: {task['redacted_excerpt']}\n"
        f"Figure caption: {task['figure_caption']}\n"
        f"Scientific claim: {task['claim']}\n"
        f"Budget: {task['budget_seconds']} seconds wall-clock, CPU only."
    )


def generate_rubric(task, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task['id']}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    resp = llm.chat(
        [{"role": "user", "content": META_PROMPT + task_card(task)}],
        model=llm.RUBRIC_MODEL, temperature=0.3, json_mode=True,
    )
    rubric = llm.extract_json(resp)
    with open(path, "w") as f:
        json.dump(rubric, f, indent=2)
    return rubric
