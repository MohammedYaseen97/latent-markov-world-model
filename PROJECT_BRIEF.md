# Project Brief: Latent World Models for RL Post-Training

## Experiment at a glance

| | |
|---|---|
| **Question** | Does a **learned latent state** (VAE on trajectories, optional uncertainty bonus) beat **history-as-state GRPO** and a **Yuan-style token-Markov** comparator on hard math? |
| **Benchmark** | MATH-Beyond **MATH-B-I base** pool (pinned Hub row; see `reports/DATA_PROTOCOL.md`) |
| **Arms** | (1) baseline GRPO (2) token-Markov GRPO (3) latent GRPO (4) latent + uncertainty |
| **Headline metric** | `pass@1024` on all arms; also `pass@1`, `pass@16` |
| **Gates** | Phase order and checklist → **`PROJECT_CONTRACT.md`** (authoritative) |

---

## Problem Statement

Reinforcement learning post-training — GRPO, PPO, RLVR — has a structural flaw that prevents it from discovering genuinely new reasoning capabilities. The flaw is how state is defined.

In every current RL post-training setup, the state is the full concatenation of all tokens generated so far. This is not a state in any meaningful RL sense. It grows unboundedly, it's redundant, it has no Markov property, and it gives the policy no compact summary of where it actually *is* in the solution space. The consequence is well-documented: RLVR improves sampling efficiency toward paths the base model could already take, but doesn't expand the model's reasoning frontier. The capability ceiling is a direct artifact of this degenerate state representation.

Recent work (Yuan et al., March 2026) confirmed this formally and showed that introducing explicit Markov states breaks the ceiling. Their fix: an external state predictor operating in token space.

That's still not first principles. A token-space state summarizer is constructed, not learned. It doesn't discover the structure of the solution space — it compresses language about it. And it has no uncertainty signal.

**The question this project asks: what if the state was learned end-to-end from reasoning trajectories — a compressed latent belief over solution-space position, discovered from data, with uncertainty built in?**

---

## North star: abstract problems as mazes (keep for the postmortem)

Humans rarely treat a hard problem as an unstructured string of words. We **impose structure**: intermediate states, legal moves, constraints, and a sense of **where we are** versus **where we want to be** — like **navigating a maze** whose walls are implied by the problem itself, toward an exit that satisfies the task.

A useful picture for this agenda: **turn the abstract problem into an implicit “maze”** (a structured space of conceptual steps) that an agent can **navigate** toward a **correct, constraint-respecting** outcome, with **calibrated confidence** when the path is clear.

**What this repo actually tests (narrow slice):** we do **not** hand-draw that maze or parse the prompt into an explicit graph. We ask whether a **learned latent** over trajectories can serve as **coordinates in that implicit space** — a Markov-friendly summary of progress — and whether **uncertainty in that latent** behaves like **“I don’t know which corridor is right yet.”** GRPO is the engine that **searches paths**; the latent is the bet on **learned navigation coordinates** instead of raw history or a fixed token-Markov summary.

**When you finish:** read this section again. Ask whether the results **support, weaken, or leave open** this picture (e.g. latent geometry, uncertainty behavior, wins vs baselines on hard items) — not only whether a number went up.

---

## Core Hypothesis

A VAE trained on reasoning trajectories can learn a compact latent representation of where a reasoning agent currently is in the solution space. When an RL policy conditions on this latent state instead of raw token history, two things happen:

1. **The capability ceiling breaks** — the policy operates on a proper Markov state, enabling genuine discovery rather than redistribution of existing paths
2. **Uncertainty drives exploration** — the variance of the VAE latent is an epistemic signal: high variance means the model doesn't know where it is, which should drive exploration; low variance means it's confident, which should drive exploitation

This is a minimal learned latent world model for abstract reasoning. The "world" being modeled is not physical environment but epistemic state — where the agent is in its understanding of the problem.

---

## Connection to the Broader Vision

This project is a tractable proof of concept for a larger research agenda: world models for abstract problem solving, as distinct from world models for physical environments (JEPA, LeWM, Dreamer).

LeWM and JEPA model what the physical world will look like next. This project models what the agent's *understanding of a problem* looks like right now, and how it should evolve. Same philosophical DNA — predict and plan in latent space, not raw observation space — applied to a completely untouched domain.

If this works, the generalization is direct:
- Replace VAE with diffusion model → richer, progressive denoising of belief state (your original Idea 2)
- Add identity conditioning → persona-conditioned belief initialization (Idea 1 + Idea 2 unified)
- Scale beyond RL post-training → any domain where an agent tracks its own epistemic state

This project earns the right to make that bigger claim.

---

## Related Work and Positioning

| Paper | What it does | How yours differs |
|---|---|---|
| Yuan et al. (Mar 2026) — Markov States | Introduces external token-space state predictor, shows it breaks ceiling | Your state is fully learned via VAE, not constructed; adds uncertainty signal |
| Yue et al. NeurIPS 2025 — Capability Ceiling | Proves RLVR doesn't expand reasoning frontier | Your empirical motivation — the problem you solve |
| LeWM / JEPA (LeCun, Mar 2026) | Latent world model for physical environment from pixels | Same philosophy, different domain — you model epistemic space not physical space |
| Dreamer / DIAMOND | Diffusion/latent world models for pixel-based RL | Physical world, not abstract reasoning |
| Coconut (Meta, 2024) | Reasoning in latent space via continuous thought tokens | Inference-time only, no learned world model, no RL |

**Your novel claim:** A VAE trained on reasoning trajectories learns a compact Markov state over solution-space position. An RL policy conditioned on this latent — with variance as an exploration bonus — breaks the capability ceiling on benchmarks where standard RLVR fails.

Three things that make this new together:
1. State is learned end-to-end, not constructed
2. Uncertainty is a first-class signal, not a side effect
3. It instantiates a latent world model for abstract reasoning — a domain nobody has built this for

---

## What Success Looks Like

**Minimum viable result (must have):**
- All four arms trained and evaluated; **`pass@1024`** reported for each (plus `pass@1` / `pass@16`)
- Latent arm(s) beat **history baseline** and you characterize vs **token-Markov** (win, tie, or loss — honest)
- Same benchmark and matched budgets/decoding across arms (per `PROJECT_CONTRACT.md`)

**Strong result (target):**
- Latent variance correlates meaningfully with problem difficulty — interpretability win
- Uncertainty-driven exploration bonus improves over latent state alone
- Sample efficiency gain: your model converges to the same performance with fewer rollouts

**Exceptional result (if things go well):**
- Positive result holds across two benchmarks
- Latent space has interpretable structure — visualizable with t-SNE/UMAP, clusters correspond to problem-solving phases

---

## Benchmark Choice

**Primary:** MATH-Beyond **MATH-B-I (base models)** pool from Mayilvahanan et al. — problems where every listed *base* model failed at pass@1024 on the Hub snapshot we pin (see `reports/DATA_PROTOCOL.md` and `data/benchmark_manifest.json` for the exact row count; the paper reports 41, the pinned revision currently yields 40). This is your capability-ceiling gauntlet: if your model solves any of these that the baseline cannot, that is a clean result.

**Secondary (if time):** ProntoQA or a logic puzzle suite — cleaner structure, faster iteration, good for sanity checking latent geometry.

**Why not GSM8K/MATH:** Saturated. RLVR already works there. You need a benchmark where the ceiling is real and documented.

---

## Technical Stack

| Component | Choice | Reason |
|---|---|---|
| Base model | Qwen2.5-1.5B or DeepSeek-R1-Distill-1.5B | Well-supported, fits A100, strong reasoning baseline |
| Training framework | TRL (simpler) or veRL (more flexible) | Start with TRL, switch if you need more control |
| VAE | Custom PyTorch, ~200 lines | Input: trajectory hidden states. Output: latent z, mean μ, variance σ² |
| RL algorithm | GRPO | Standard, well-understood, your existing knowledge |
| Benchmark | MATH-Beyond | Documented ceiling, verifiable rewards |
| Dev machine | RTX 4060 (local) | Architecture dev, tiny model sanity checks |
| Training machine | A100 80GB (RunPod/Lambda) | Full training runs, ~$150 total budget |

---

## VAE Design (the core component)

**Input:** Hidden state sequence from the base model's reasoning trajectory — not tokens, but the model's internal representations at each step. Specifically, the final layer hidden states averaged over the last N tokens at each reasoning step.

**Encoder:** MLP that maps trajectory hidden states → (μ, σ²) in latent space z. Start small: 2-3 layers, latent dim 64-128.

**Decoder:** MLP that maps z back to trajectory representation. Trained with standard ELBO: reconstruction loss + KL divergence term.

**Conditioning:** The RL policy receives z (sampled from the encoder posterior) as additional input, concatenated with the current token embedding before the policy head.

**Uncertainty bonus:** Add β × KL(q(z|trajectory) || p(z)) as an intrinsic reward term in GRPO. High KL = high uncertainty = explore. Anneal β over training so exploration dominates early, exploitation dominates later.

---

## 4-Week Execution Plan

**Authoritative implementation order** is **phase-gated** in `PROJECT_CONTRACT.md` (baseline + data + eval → token-Markov → latent arms → core table). The weeks below are **pacing hints** only.

**Week 1 — Baseline + VAE prototype**
- Days 1-2: Set up training environment. Get GRPO running on Qwen2.5-1.5B with TRL. Verify training is stable.
- Days 3-4: Establish baseline. Run standard GRPO on MATH-Beyond or logic puzzle suite. Measure pass@k ceiling. This is your ground truth.
- Days 5-7: Implement VAE in PyTorch. Train it on collected rollouts from the baseline model. Verify latents have structure — do t-SNE, check if correct vs incorrect trajectories separate in latent space.

**Week 2 — Integration**
- Days 8-10: Hook VAE encoder into the GRPO training loop. Policy now receives z alongside token input. Expect shape bugs, gradient flow issues — budget time for this.
- Days 11-14: First integrated training run. Monitor: does the VAE latent actually update meaningfully during RL training? Does the policy learn to use z? Loss curves should look reasonable by end of week.

**Week 3 — Results + ablations**
- Days 15-17: Full training run with latent state, no uncertainty bonus. Measure pass@k on benchmark. Compare to baseline.
- Days 18-19: Add uncertainty bonus. Retrain. Measure delta.
- Days 20-21: Ablation table: (1) GRPO baseline, (2) Yuan-style token-Markov comparator, (3) VAE latent state, (4) VAE latent state + uncertainty bonus. This is your core result table.

**Week 4 — Writeup + polish**
- Days 22-24: arXiv-style writeup. 6-8 pages. Introduction, related work, method, experiments, discussion. Don't submit yet — get it clean.
- Days 25-26: GitHub repo cleanup. README with clear setup instructions, reproducible training scripts, model checkpoints.
- Days 27-28: Blog post version (shorter, accessible, with visuals). This is what gets shared on Twitter and in DMs.

---

## Deliverables

By end of week 4, you have:

1. **Public GitHub repo** — clean, reproducible, documented. This is what Sam clicks when he sees your DM.
2. **arXiv preprint** — 6-8 pages, properly formatted, citable. Even unreviewed it signals seriousness.
3. **Blog post** — accessible writeup with the key result visualized. Shareable, linkable.
4. **Core result** — one table showing your model solves problems the GRPO baseline cannot, with ablations.
5. **Bonus if time** — latent space visualization showing interpretable structure.

---

## The Narrative (for papers, DMs, PhD applications)

> Current RL post-training fails to discover genuinely new reasoning capabilities because it uses token history as state — an unbounded, redundant, non-Markovian representation that gives the policy no compact model of where it is in the solution space. We propose replacing this with a learned latent state: a VAE trained on reasoning trajectories that discovers the structure of solution-space position from data. The latent variance provides a natural epistemic uncertainty signal that drives principled exploration. We show this breaks the documented capability ceiling on benchmarks where standard RLVR fails, and connect this work to a broader agenda: latent world models for abstract reasoning, as distinct from existing world models for physical environments.

That's your abstract. That's your PhD statement paragraph. That's your DM hook. Same story, different length.

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| VAE latent doesn't learn meaningful structure | Medium | Check early (day 7) with t-SNE. If flat, try: larger latent dim, longer pretraining on rollouts, different input (token embeddings vs hidden states) |
| Integration bugs eat week 2 | Medium | Keep VAE and RL training loop as modular as possible. Test VAE frozen first, then joint training |
| No improvement over baseline | Low-Medium | Even a null result is publishable if your framing is honest — "we show latent states are necessary but not sufficient, and here's why" |
| Compute costs overrun | Low | Hard cap: don't run more than 3 full training runs per week. Use 4060 for all debugging runs |
| Benchmark too hard, no signal | Low | Switch to ProntoQA for week 3 if MATH-Beyond shows zero movement |

---

Start with day 1: get GRPO running on Qwen2.5-1.5B locally on your 4060 with a tiny subset of data, just to verify the pipeline works end to end before you touch the A100. That's your first milestone.