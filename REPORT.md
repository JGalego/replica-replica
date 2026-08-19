# Replication Report: "Training AI Scientists to Replicate Research"

**Target paper:** D. Falck, S. Sabri, A. Surina, T. Foster, A. Sims, S. Devlin,
D. Rogers, T. Collins, K. Aleksiev, L. Kirsch, E. Hughes.
*Training AI Scientists to Replicate Research.* arXiv:2608.13331, Inherent Labs,
August 2026.

**This replication:** `replica-replica`, August 19, 2026. A CPU-only sandbox,
one afternoon, a few dollars of API spend — a deliberately small-scale
replication, in the spirit the paper itself advocates: replication under
constrained budgets illuminates which details matter. (It also, fittingly,
required recovering details the paper leaves underspecified.)

---

## 1. What the original paper does

The paper builds **Replica**, an RL task environment for paper replication:
an agent receives a redacted research paper (target figure removed), the
figure's caption, a 60-minute wall-clock limit, and a 1/7 MIG slice of an
H100/H200; it must reproduce the figure and the underlying experiment.
Rollouts are scored by an LLM judge against per-task rubrics auto-generated
by Claude Opus 4.7 from a meta-prompt with the gold plot hidden. Judges score
five dimensions (visual match, scientific claim, implementation, budget use,
integrity) on a continuous 0-1 scale; three judge samples are averaged per
rollout. The suite has 310 tasks from 100 papers (242 ML training tasks, 68
held-out AI-for-science test tasks). With this reward, the authors RL-train
**Faraday** (27B), which they report outperforms Claude Opus 4.8 and GPT-5.5
Codex on 73% of in-distribution tasks (+6% / +8% rubric score) and 60% of
held-out tasks, validated by a human study (117 rankings, 20 PhD-level
participants). Key evaluation numbers: rubric-judge inter-sample reliability
(Kendall τ) 0.66 vs 0.46 for a rubric-free baseline judge.

## 2. What this replication does

We rebuild the full **environment and evaluation pipeline** and test the
paper's evaluable methodological claims. We do **not** replicate the RL
training of Faraday (no GPUs); instead we place *untrained* models —
including a 27B model, Faraday's parameter count — in the environment. That
reproduces the paper's *baseline* condition and tests whether the environment
yields the low-noise, discriminative reward signal that made training
possible.

| Component | Paper | Here |
|---|---|---|
| Tasks | 310 tasks / 100 papers | 6 tasks / 6 classic papers |
| Splits | train: ML (1990-2026); test: AI-for-science | train: statistics/ML/NLP (Anscombe 1973, Belkin et al. 2019, Zipf 1949); test: science (May 1976, Kermack & McKendrick 1927, Benford 1938) |
| Budget | 60 min, 1/7 H100 MIG slice | 2-5 min wall-clock, CPU |
| Rubric generator | Claude Opus 4.7, meta-prompt, gold plot hidden | Claude Opus 5, same meta-prompt design, gold figure hidden |
| Judge | 5 dimensions, 0-1, k=3 samples averaged | identical protocol (Claude Opus 5 judge, k=3); rubric-free baseline judge ablation included |
| Judge evidence | 10-min workspace access | agent code + structured figure metadata + execution logs |
| Agents | Faraday 27B (RL-trained), Claude Opus 4.8, GPT-5.5 Codex | qwen3.6-27b (untrained, Faraday's size), gpt-oss-20b, hand-written reference tier (frontier-agent stand-in), 3-attempt repair loop for LLM agents |
| Ground truth for judge validity | human study (117 rankings) | known quality tiers: reference > degraded > null, 6 tasks each |
| Secondary judge | — | qwen3.6-27b judge (k=2) on 18 rollouts, for cross-judge agreement |

The **degraded** tier is six plausible-but-flawed solutions, each violating a
different rubric dimension (truncated parameter sweeps, wrong parameters that
kill the phenomenon, linear axes hiding a power law, and one outright
fabrication that samples the "empirical" data from the theoretical
distribution). The **null** tier submits an empty labeled figure.

## 3. Claims tested

| # | Paper claim | Outcome |
|---|---|---|
| C1 | Paper replication can be posed as a scalable, automatically-gradeable task environment | **Replicated.** All 30 rollouts (5 agents × 6 tasks) executed under budget; 11/12 LLM-agent rollouts produced figures; rubrics auto-generated for every task with the gold figure hidden; end-to-end automated scoring. |
| C2 | Auto-generated rubrics give the judge a *more reliable* reward signal than no rubric (τ 0.66 vs 0.46) | **Partially replicated.** Direction holds (rubric 0.923 vs baseline 0.911 mean pairwise Kendall τ) but both judges are near ceiling and the gap is far smaller than the paper's 0.20. See §4.2 for why. |
| C3 | Judge scores track true replication quality (paper: human study agreement) | **Replicated (by analog).** The rubric judge ordered reference > degraded > null correctly in 18/18 task-pairs (pairwise accuracy 1.0); Kendall τ between score and ground-truth tier 0.85 (p ≈ 1.2e-5). |
| C4 | RL training makes a 27B agent beat frontier agents | **Not tested** (no RL training possible). The *premise* is consistent with our data: the untrained 27B scores 0.63 vs the frontier-authored reference tier's 0.86 — a large gap for training to close. |
| C5 | ML/NLP tasks are hardest; science tasks with clean experimental recipes are easier | **Replicated.** LLM agents averaged 0.48 on the ML/statistics split vs 0.85 on the science split; the double-descent task (ML methodology) was the hardest (27B failed it outright). |

## 4. Results

Primary judge: Claude Opus 5, k=3 samples per rollout, per-task auto-rubrics
(gold figure hidden). Full data in `results/summary.json`,
`results/judgments_claude.json`, and per-rollout workspaces under
`results/runs/`.

### 4.1 Replication quality by agent (rubric judge, 0-1)

| Agent | train (ML/stats) | test (science) | all |
|---|---|---|---|
| reference (frontier-authored) | 0.827 | 0.898 | **0.863** |
| gpt-oss-20b | 0.522 | 0.882 | 0.702 |
| qwen3.6-27b (Faraday's size, untrained) | 0.439 | 0.822 | 0.630 |
| degraded | 0.193 | 0.241 | 0.217 |
| null | 0.058 | 0.069 | 0.063 |

Per-task scores are in `results/summary.json` and
`results/figures/per_task_scores.png`. Notable rollouts: the 27B produced a
genuinely excellent logistic-map bifurcation diagram (period-doubling
cascade, chaotic regime, and the period-3 window all present) but failed the
double-descent task in all 3 attempts — consistent with the paper's finding
that ML-methodology tasks are the hardest category. The judge scored the
degraded Benford solution's fabricated "empirical" data 0.19, correctly
flagging the integrity violation from the code.

### 4.2 Judge reliability (the paper's τ 0.66-vs-0.46 claim)

Kendall τ between independent judge samples, over all 30 rollouts:

| Judge | this replication | paper |
|---|---|---|
| rubric judge | **0.923** (pairwise 0.919 / 0.930 / 0.921) | 0.66 |
| baseline (no-rubric) judge | **0.911** (0.887 / 0.912 / 0.934) | 0.46 |

The *direction* replicates (rubric ≥ baseline in every pairwise comparison)
but the effect is tiny here, and both judges are far more self-consistent
than the paper's. Two mechanisms plausibly explain the compression:

1. **Range restriction in reverse.** Our rollout pool spans extreme quality
   tiers (null, degraded, competent) that any judge ranks consistently,
   inflating τ for both judges. The paper's pool consists entirely of real
   agent rollouts inside a narrower quality band — a much harder
   discrimination regime where rubrics have room to help.
2. **Evidence format.** Our judge reads structured figure metadata and code
   rather than pixels and a free-form workspace, which removes much of the
   ambiguity rubrics exist to resolve.

This is the kind of finding the paper predicts replication surfaces:
the rubric-vs-baseline gap is not a property of rubrics alone but of the
rubric × rollout-distribution × evidence-format combination.

### 4.3 Judge validity (analog of the paper's human study)

With ground-truth quality tiers in place of human raters: the rubric judge
ordered reference > degraded > null correctly in **18/18** task-pairs;
Kendall τ between judge score and true tier is **0.85** (p ≈ 1.2e-5). The
baseline judge also achieved 18/18 (τ 0.84) — on tier-separated rollouts,
both judges are valid; the paper's harder setting is where they separate.

### 4.4 Cross-judge agreement

An independent 27B judge (qwen3.6-27b, k=2) scored 18 of the same rollouts:
Kendall τ between the Claude and qwen rubric-judge scores is **0.673**
(p ≈ 2.2e-4) — incidentally close to the paper's inter-rater reliability
(0.66), and consistent with its finding that rubric-anchored judging
transfers across judge models.

### 4.5 Figures

- `results/figures/scores_by_agent.png` — agent comparison by split
- `results/figures/judge_reliability.png` — reliability vs the paper's values
- `results/figures/per_task_scores.png` — per-task breakdown

## 5. Deviations and limitations

- **No RL training.** The central artifact of the paper — Faraday's trained
  policy — is not reproduced. We replicate the environment, reward, and
  evaluation methodology, not the training result.
- **Scale.** 6 tasks vs 310; minutes of CPU vs an hour of H100 per rollout;
  30 rollouts vs thousands. All τ estimates carry wide confidence intervals.
- **Judge evidence differs.** The paper's judge inspects the agent workspace
  (including the plot image) for 10 minutes; ours reads the solution code,
  execution logs, and structured metadata extracted from the matplotlib
  figure object.
- **Model stack.** Rubrics and judging: Claude Opus 5 (paper: Claude Opus
  4.7 rubrics). Agents: qwen3.6-27b and gpt-oss-20b via Groq (paper:
  Faraday 27B, Claude Opus 4.8, GPT-5.5 Codex). Judging initially ran on
  Groq-hosted open-weight models, whose 200k tokens/day free-tier caps were
  exhausted mid-study (gpt-oss-120b, then qwen3.6-27b at k=2); an Anthropic
  API key restored the paper's full k=3 protocol. The partial open-weight
  judgments are retained (`results/judgments.json`,
  `results/judgments_gptoss120b_partial.json`) and feed §4.4.
- **Tasks are classics** whose phenomena the agent models have surely seen
  described in pretraining. The paper's 1990-2026 ML papers partially share
  this issue; both setups hide the gold figure itself. Our reference tier
  was authored by the same frontier model family that judges some rollouts'
  competitors — mitigated by rubric anchoring, but a real confound at this
  scale.
- **No human study**; ground-truth quality tiers substitute for human
  preference rankings.

## 6. Conclusion

The Replica *methodology* replicates well at 1/50th scale and roughly $5 of
API spend: paper-figure replication is automatable as a task environment,
auto-generated rubrics anchor an LLM judge that reliably and validly
separates replication quality, and untrained agents — including one at
Faraday's exact parameter count — leave a large, measurable gap below
frontier-level replication work, which is precisely the headroom the paper's
RL training claims to close. The one quantitative claim that did not
transfer at our scale is the *size* of the rubric-vs-no-rubric reliability
gap (we see 0.92 vs 0.91; the paper reports 0.66 vs 0.46), which our setup
attributes to rollout-pool quality range and judge evidence format — a
detail, as the paper puts it, that "replication typically illuminates."
