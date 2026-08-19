"""Agents evaluated in the Replica-Replica environment.

* llm_agent   — a Groq-hosted model writes the replication code from the task
                card; it gets up to 3 attempts, seeing execution errors from
                the previous attempt (a minimal stand-in for Replica's
                agentic coding loop).
* reference   — carefully hand-written solutions (frontier-agent stand-in,
                authored by Claude).
* degraded    — plausible-but-flawed solutions (known mid-quality tier).
* null        — an empty labeled figure (known floor).

The scripted tiers give ground-truth quality orderings against which judge
validity is measured, standing in for the paper's human study.
"""

import os
import re

from . import llm
from .rubric import task_card
from .runner import run_solution

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_AGENT_PROMPT = """You are an AI research scientist. Your task is to replicate a figure from a
research paper. You are given a redacted excerpt of the paper (the figure
itself is removed), the figure's caption, and a compute budget. Write a single
self-contained Python script that runs the experiment and produces the figure.

Rules:
- CPU only; the script must finish well within the budget.
- Available libraries: numpy, scipy, matplotlib, scikit-learn (no others; no
  network access).
- Any asset files listed are already in the working directory.
- The script MUST end by building a matplotlib Figure `fig` and calling:
      from replica_replica.harness import submit
      submit(fig)
- Compute the results from the actual experiment. Do not hard-code the
  expected curve.
- Reply with ONLY the Python code, no markdown fences, no explanation.

{task}

Assets in working directory: {assets}
"""

_REPAIR_PROMPT = """Your previous script failed. Fix it and reply with ONLY the corrected
complete Python script (no markdown fences).

Failure status: {status}
stderr tail:
{stderr}
"""


def _strip_fences(code):
    code = code.strip()
    # Reasoning models may prepend a <think>...</think> block to the content.
    if "</think>" in code:
        code = code.split("</think>", 1)[1].strip()
    elif code.startswith("<think>"):
        return ""  # reasoning never closed: no code in this reply
    if "```" in code:
        # Take the largest fenced block (models sometimes add prose around it).
        blocks = re.findall(r"```(?:python)?\n(.*?)```", code, flags=re.DOTALL)
        if blocks:
            return max(blocks, key=len).strip()
        code = code.split("```", 1)[1]
        if code.startswith("python"):
            code = code[6:]
    return code.strip()


def llm_agent(task, model, workdir, attempts=3):
    assets = [os.path.basename(a) for a in task.get("assets", [])] or ["(none)"]
    messages = [{
        "role": "user",
        "content": _AGENT_PROMPT.format(task=task_card(task), assets=", ".join(assets)),
    }]
    record = None
    for i in range(attempts):
        code = _strip_fences(
            llm.chat(messages, model=model, temperature=0.6, max_tokens=4000)
        )
        attempt_dir = os.path.join(workdir, f"attempt_{i + 1}")
        record = run_solution(task, code, attempt_dir)
        record["attempt"] = i + 1
        if record["status"] == "ok":
            break
        messages.append({"role": "assistant", "content": code})
        messages.append({
            "role": "user",
            "content": _REPAIR_PROMPT.format(
                status=record["status"], stderr=record["stderr_tail"][-1500:]
            ),
        })
    return record


NULL_SOLUTION = """import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_title("Replication attempt")
ax.set_xlabel("x")
ax.set_ylabel("y")
from replica_replica.harness import submit
submit(fig)
"""


def scripted_agent(task, tier, workdir):
    """tier in {'reference', 'degraded', 'null'}"""
    if tier == "null":
        code = NULL_SOLUTION
    else:
        path = os.path.join(REPO_ROOT, "agents", tier, f"{task['id']}.py")
        with open(path) as f:
            code = f.read()
    return run_solution(task, code, workdir)
