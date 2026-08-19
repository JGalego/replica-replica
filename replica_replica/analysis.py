"""Analysis of the Replica-Replica study, replicating the paper's
quantitative evaluation claims at small scale:

  * agent comparison table (rubric-judge scores by agent and split);
  * judge reliability: Kendall tau between independent judge samples,
    rubric judge vs baseline judge (paper reports 0.66 vs 0.46);
  * judge validity: agreement of judge scores with ground-truth quality
    tiers (reference > degraded > null), standing in for the human study.

Usage: python -m replica_replica.analysis
"""

import itertools
import json
import os

import numpy as np
from scipy.stats import kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .runner import list_tasks, load_task, REPO_ROOT

RESULTS = os.path.join(REPO_ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")
AGENT_ORDER = ["reference", "qwen3.6-27b", "gpt-oss-20b", "degraded", "null"]
TIER_RANK = {"reference": 2, "degraded": 1, "null": 0}

# Multi-sample aggregation depth used in analysis. Some rollouts carry 3
# judge samples and some 2 (daily token budgets, see REPORT.md); the first K
# are used uniformly everywhere.
K = 2


def score(judgments, key, judge_name):
    samples = judgments[key][judge_name]["samples"][:K]
    return float(np.mean([s["overall"] for s in samples]))


def load_all():
    with open(os.path.join(RESULTS, "judgments.json")) as f:
        judgments = json.load(f)
    tasks = {t: load_task(t) for t in list_tasks()}
    return judgments, tasks


def reliability(judgments, judge_name):
    """Mean Kendall tau between pairs of independent judge samples, computed
    over all rollouts (the paper's inter-rater reliability measure)."""
    keys = sorted(judgments)
    per_sample = []
    n_samples = K
    for i in range(n_samples):
        per_sample.append([judgments[k][judge_name]["samples"][i]["overall"] for k in keys])
    taus = []
    for i, j in itertools.combinations(range(n_samples), 2):
        tau, _ = kendalltau(per_sample[i], per_sample[j])
        taus.append(tau)
    return float(np.mean(taus)), [float(t) for t in taus]


def validity(judgments, tasks, judge_name):
    """Fraction of ground-truth-ordered pairs (reference > degraded > null,
    per task) that the judge ranks correctly; plus Kendall tau between judge
    score and tier rank."""
    correct, total = 0, 0
    scores, ranks = [], []
    for t in tasks:
        tier_scores = {
            tier: score(judgments, f"{tier}|{t}", judge_name) for tier in TIER_RANK
        }
        for a, b in itertools.combinations(TIER_RANK, 2):
            hi, lo = (a, b) if TIER_RANK[a] > TIER_RANK[b] else (b, a)
            total += 1
            if tier_scores[hi] > tier_scores[lo]:
                correct += 1
        for tier, s in tier_scores.items():
            scores.append(s)
            ranks.append(TIER_RANK[tier])
    tau, p = kendalltau(scores, ranks)
    return correct / total, float(tau), float(p)


def agent_table(judgments, tasks):
    rows = {}
    for agent in AGENT_ORDER:
        rows[agent] = {}
        for split in ["train", "test"]:
            ids = [t for t in tasks if tasks[t]["split"] == split]
            vals = [score(judgments, f"{agent}|{t}", "rubric") for t in ids]
            rows[agent][split] = float(np.mean(vals))
        rows[agent]["all"] = float(np.mean(
            [score(judgments, f"{agent}|{t}", "rubric") for t in tasks]
        ))
    return rows


def per_task_table(judgments, tasks):
    return {
        t: {a: round(score(judgments, f"{a}|{t}", "rubric"), 3) for a in AGENT_ORDER}
        for t in tasks
    }


def make_figures(table, rel_rubric, rel_baseline, judgments, tasks):
    os.makedirs(FIGDIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(AGENT_ORDER))
    w = 0.38
    ax.bar(x - w / 2, [table[a]["train"] for a in AGENT_ORDER], w,
           label="train split (ML/statistics)", color="tab:blue")
    ax.bar(x + w / 2, [table[a]["test"] for a in AGENT_ORDER], w,
           label="test split (science)", color="tab:orange")
    ax.set_xticks(x, AGENT_ORDER, rotation=15)
    ax.set_ylabel("Rubric-judge score (0-1, k=3 samples)")
    ax.set_title("Replica-Replica: replication quality by agent")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "scores_by_agent.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["rubric judge", "baseline judge"], [rel_rubric, rel_baseline],
           color=["tab:green", "tab:gray"])
    ax.axhline(0.66, color="tab:green", ls="--", lw=1, label="paper: rubric 0.66")
    ax.axhline(0.46, color="tab:gray", ls="--", lw=1, label="paper: baseline 0.46")
    ax.set_ylabel("Inter-sample reliability (Kendall τ)")
    ax.set_title("Judge reliability: rubric vs rubric-free")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "judge_reliability.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(9, 5))
    ids = sorted(tasks)
    for agent, color in zip(AGENT_ORDER, ["tab:green", "tab:blue", "tab:cyan", "tab:orange", "tab:red"]):
        ax.plot(range(len(ids)),
                [score(judgments, f"{agent}|{t}", "rubric") for t in ids],
                "o-", label=agent, color=color)
    ax.set_xticks(range(len(ids)), ids, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Rubric-judge score")
    ax.set_title("Per-task scores")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "per_task_scores.png"), dpi=120)


def main():
    judgments, tasks = load_all()
    table = agent_table(judgments, tasks)
    rel_rubric, taus_rubric = reliability(judgments, "rubric")
    rel_baseline, taus_baseline = reliability(judgments, "baseline")
    val_rubric = validity(judgments, tasks, "rubric")
    val_baseline = validity(judgments, tasks, "baseline")

    summary = {
        "agent_scores_rubric_judge": table,
        "per_task_rubric_scores": per_task_table(judgments, tasks),
        "judge_reliability_kendall_tau": {
            "rubric": {"mean": rel_rubric, "pairwise": taus_rubric, "paper_value": 0.66},
            "baseline": {"mean": rel_baseline, "pairwise": taus_baseline, "paper_value": 0.46},
        },
        "judge_validity_vs_ground_truth_tiers": {
            "rubric": {"pairwise_accuracy": val_rubric[0], "kendall_tau": val_rubric[1],
                        "p_value": val_rubric[2]},
            "baseline": {"pairwise_accuracy": val_baseline[0], "kendall_tau": val_baseline[1],
                          "p_value": val_baseline[2]},
        },
    }
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    make_figures(table, rel_rubric, rel_baseline, judgments, tasks)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
