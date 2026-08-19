"""Execute an agent's solution script inside a task workspace under a
wall-clock budget, mirroring Replica's containerized 60-minute rollouts."""

import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_DIR = os.path.join(REPO_ROOT, "tasks")


def load_task(task_id):
    with open(os.path.join(TASKS_DIR, task_id, "task.json")) as f:
        return json.load(f)


def list_tasks():
    return sorted(
        d for d in os.listdir(TASKS_DIR)
        if os.path.isdir(os.path.join(TASKS_DIR, d)) and d != "assets"
    )


def run_solution(task, solution_code, workdir):
    """Run solution code in `workdir` with the task's budget. Returns a rollout
    record: status, runtime, stdout/stderr tails, and figure metadata if the
    solution submitted a figure."""
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    for rel in task.get("assets", []):
        src = os.path.join(TASKS_DIR, rel)
        shutil.copy(src, os.path.join(workdir, os.path.basename(rel)))
    script = os.path.join(workdir, "solution.py")
    with open(script, "w") as f:
        f.write(solution_code)

    env = dict(os.environ)
    env["REPLICA_WORKDIR"] = workdir
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "solution.py"],
            cwd=workdir, env=env, capture_output=True, text=True,
            timeout=task["budget_seconds"],
        )
        status = "ok" if proc.returncode == 0 else "error"
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        status, stdout, stderr = "timeout", (e.stdout or ""), (e.stderr or "")
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
    runtime = time.time() - start

    meta_path = os.path.join(workdir, "figure_meta.json")
    fig_path = os.path.join(workdir, "figure.png")
    figure_meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            figure_meta = json.load(f)
    if status == "ok" and figure_meta is None:
        status = "no_figure"

    record = {
        "task_id": task["id"],
        "status": status,
        "runtime_seconds": round(runtime, 1),
        "budget_seconds": task["budget_seconds"],
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
        "figure_meta": figure_meta,
        "has_png": os.path.exists(fig_path),
        "solution_code": solution_code,
    }
    with open(os.path.join(workdir, "rollout.json"), "w") as f:
        json.dump(record, f, indent=2)
    return record
