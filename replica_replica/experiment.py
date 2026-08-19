"""Orchestrates the full Replica-Replica study.

Phases (all resumable — existing outputs are reused):
  1. rubric generation for every task (gold figures hidden);
  2. rollouts: scripted tiers (reference/degraded/null) and LLM agents
     (27B and 20B models) attempt every task;
  3. judging: every rollout is scored by the rubric judge and the baseline
     (rubric-free) judge, k=3 samples each.

Usage: python -m replica_replica.experiment
"""

import json
import os

from . import llm
from .agents import llm_agent, scripted_agent
from .judge import judge_rollout
from .rubric import generate_rubric
from .runner import list_tasks, load_task, REPO_ROOT

RESULTS = os.path.join(REPO_ROOT, "results")
AGENTS = {
    "reference": ("scripted", None),
    "degraded": ("scripted", None),
    "null": ("scripted", None),
    "qwen3.6-27b": ("llm", llm.AGENT_27B),
    "gpt-oss-20b": ("llm", llm.AGENT_20B),
}
# Judging backend, selected via REPLICA_JUDGE:
#   claude (default when ANTHROPIC_API_KEY is set) — Claude Opus 5 rubrics and
#     judging at the paper's k=3, written to results/judgments_claude.json;
#   groq — qwen3.6-27b judge. The paper aggregates k=3 samples; Groq's free
#     tier gives each model 200k tokens/day and full k=3 judging needs ~470k,
#     so the groq backend collects k=2 (the minimum that still measures
#     inter-sample reliability). Documented as a deviation in REPORT.md.
BACKEND = os.environ.get(
    "REPLICA_JUDGE", "claude" if os.environ.get("ANTHROPIC_API_KEY") else "groq"
)
if BACKEND == "claude":
    llm.JUDGE_MODEL = "claude-opus-5"
    llm.RUBRIC_MODEL = "claude-opus-5"
    JUDGE_SAMPLES = 3
    RUBRIC_DIR = "rubrics_claude"
    JUDGMENTS_FILE = "judgments_claude.json"
else:
    JUDGE_SAMPLES = 2
    RUBRIC_DIR = "rubrics"
    JUDGMENTS_FILE = "judgments.json"


def rollout_path(agent, task_id):
    return os.path.join(RESULTS, "runs", agent, task_id)


def get_rollout(agent, task):
    """Load a cached rollout or produce one."""
    workdir = rollout_path(agent, task["id"])
    final = os.path.join(workdir, "final_rollout.json")
    if os.path.exists(final):
        with open(final) as f:
            return json.load(f)
    kind, model = AGENTS[agent]
    if kind == "scripted":
        record = scripted_agent(task, agent, workdir)
    else:
        record = llm_agent(task, model, workdir)
    with open(final, "w") as f:
        json.dump(record, f, indent=2)
    return record


def main():
    os.makedirs(RESULTS, exist_ok=True)
    tasks = [load_task(t) for t in list_tasks()]

    print(f"== Phase 1: rubric generation (backend: {BACKEND}) ==")
    rubrics = {}
    for task in tasks:
        rubrics[task["id"]] = generate_rubric(task, os.path.join(RESULTS, RUBRIC_DIR))
        print(f"  rubric ready: {task['id']}", flush=True)

    print("== Phase 2: rollouts ==")
    rollouts = {}
    for agent in AGENTS:
        for task in tasks:
            rec = get_rollout(agent, task)
            rollouts[(agent, task["id"])] = rec
            print(f"  {agent:14s} {task['id']:28s} -> {rec['status']} "
                  f"({rec['runtime_seconds']}s)")

    print("== Phase 3: judging ==")
    judgments_path = os.path.join(RESULTS, JUDGMENTS_FILE)
    judgments = {}
    if os.path.exists(judgments_path):
        with open(judgments_path) as f:
            judgments = json.load(f)
    # Rubric judging first (primary results), then the baseline ablation, so a
    # rate-limit outage mid-run still leaves a complete primary table.
    for judge_name in ["rubric", "baseline"]:
        for agent in AGENTS:
            for task in tasks:
                key = f"{agent}|{task['id']}"
                rec = rollouts[(agent, task["id"])]
                entry = judgments.get(key, {})
                if judge_name in entry:
                    continue
                rubric = rubrics[task["id"]] if judge_name == "rubric" else None
                entry[judge_name] = judge_rollout(task, rec, rubric=rubric, k=JUDGE_SAMPLES)
                print(f"  judged {key} [{judge_name}] -> "
                      f"{entry[judge_name]['overall']:.3f}", flush=True)
                judgments[key] = entry
                with open(judgments_path, "w") as f:
                    json.dump(judgments, f, indent=2)
    print(f"Done. Judgments in {judgments_path}")


if __name__ == "__main__":
    main()
