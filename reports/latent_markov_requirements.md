# Latent Markov Arm — Design Requirements

This document captures all requirements for Phase 3 before any implementation decisions are made.
It is the single authoritative source for *what the system must do* — not *how* it does it.
Design decisions (architecture choices, training recipes) go in a separate design doc once
requirements are agreed.

Sources: `PROJECT_BRIEF.md`, `README.md`, `PROJECT_CONTRACT.md`, `reports/writeup_stubs.md`,
and lessons from Phase 2 results.

---

## 1. Purpose and Framing

The latent arm is the core contribution of this project. Its purpose is to test one hypothesis:

> **A learned latent state — a VAE trained on reasoning trajectories — can replace token
> history as the MDP state, enabling an RL policy to discover genuinely new reasoning
> capabilities on benchmarks where standard RLVR fails.**

The load-bearing frame is the **MDP reformulation**, not the architecture. The VAE is the
implementation vehicle. Every requirement below serves the question: *does conditioning on
a learned compact Markov state break the capability ceiling?*

**Differentiator from Reasoning Palette (must be preserved in all framing):**
Reasoning Palette (VAE latent as reasoning strategy prefix) is the closest adjacent paper
and a reviewer landmine. The differentiator is non-negotiable:
- **Theirs:** VAE sampled *once per problem* before generation. Static per problem.
- **Ours:** `z_h` evolves *step-wise during the rollout* as a Markov state. Dynamic through reasoning.

This is a different MDP formulation, not just a different architecture. They never question
the history-as-state MDP. We do. This distinction must be preserved in the framing of every
design decision, evaluation, and paper section.

---

## 2. Required Arms

Both arms are required deliverables. Neither is optional.

| arm | description |
|-----|-------------|
| `latent_grpo` | GRPO with latent state `z_h`. No uncertainty bonus. Transition loss active. |
| `latent_grpo_uncertainty` | Same as `latent_grpo`, plus `β_t × KL(q(z_h\|τ) ∥ p(z))` added to the GRPO reward as an intrinsic exploration bonus. `β_t` annealed over training. |

The **only difference** between the two arms is the intrinsic bonus. All other
hyperparameters, the VAE architecture, the transition loss, and the policy conditioning
are identical. This isolates the effect of uncertainty-driven exploration.

---

## 3. Functional Requirements

### R1 — State representation

**R1.1** The system must produce a latent state `z_h` at each reasoning step `h`.

**R1.2** `z_h` must be derived from the backbone LM's internal representations of the
reasoning trajectory up to step `h` — specifically, the final-layer hidden states of the
policy (not raw tokens, not external embeddings).

**R1.3** `z_h` must be a *fixed-size* vector regardless of how many steps have elapsed.
The key property being tested is that `z_h` is compact and bounded — unlike token history
which grows unboundedly.

**R1.4** `z_h` must carry an explicit uncertainty estimate — the variance `σ_h²` of the
VAE posterior. This is not optional: it is the second contribution claim (uncertainty as
exploration signal) and is required for the `latent_grpo_uncertainty` arm.

**R1.5** The RL policy must receive `z_h` as input, not just the current token context.
The conditioning must happen before the policy head — `z_h` must influence token generation.

---

### R2 — The Markov property must be enforced by the training objective

**This is a new hard requirement from Phase 2.** It is not sufficient for the architecture
to be "Markov-shaped." The training objective must actively push `z_h` to satisfy the
Markov property on actual reasoning trajectories.

**R2.1** The training objective must include a **transition consistency loss**: given `z_h`
and the action `a_h` (tokens generated at step `h`), a transition model must be able to
predict `z_{h+1}` without access to the full token history. This is:

```
L_transition = || f(z_h, a_h) − z_{h+1} ||²
```

where `f` is a learned transition function. This loss directly enforces the Markov property
empirically.

**R2.2** The Markov objective must be part of the joint training loop — not a separate
pretraining phase. `z_h` must learn to be Markovian *while* being useful for RL.

**R2.3** The transition loss is also the primary Markov diagnostic (see Section 4). The
same loss used for training is the evidence submitted in the paper that `z_h` satisfies
the Markov property.

---

### R3 — Reward sparsity must be circumvented

**This is a new hard requirement from Phase 2.** The token-Markov arm failed because
per-sample success ≈ 0.015% → P(≥1 reward | G=8) ≈ 0.12% → zero gradient for all
200 steps. The latent arm cannot rely solely on the sparse binary correctness reward.

**R3.1** The system must provide a **dense auxiliary training signal** that does not
require observing a correct final answer. This signal must be active early in training
when the sparse reward is never triggered.

**R3.2** The transition consistency loss (R2.1) satisfies R3.1 — it is computable on
every rollout regardless of whether the answer was correct. It must be weighted
appropriately relative to the RL reward so it does not dominate and prevent reward
learning once rewards start appearing.

**R3.3** The uncertainty bonus (R1.4) — `β_t × KL(q(z|τ) || p(z))` as intrinsic reward
— also contributes to R3.1 by providing a non-zero reward signal at every step. `β_t`
must be annealed: high early (exploration/dense signal) → low late (exploitation/sparse
signal dominates).

**R3.4** The combined signal must make `P(gradient flows at step t)` meaningfully higher
than the 0.12% seen in Phase 2. The target is not a specific number, but gradients must
flow in the first ~20 steps of training — verifiable from training logs.

---

### R4 — VAE architecture

**R4.1 Encoder:** MLP mapping trajectory hidden states → `(μ, σ²)` in latent space.
Start with 2–3 layers, latent dimension 64–128.

**R4.2 Decoder:** MLP mapping `z` back to trajectory hidden state representation.
Trained with standard ELBO: reconstruction loss + KL divergence.
**The decoder is used during training only.** At inference, only the encoder is needed
to produce `z_h`. The decoder does not run during evaluation or RL rollouts.

**R4.3 Reparameterisation trick** must be used so gradients flow through the sampling
operation `z = μ + ε·σ` during joint training.

**R4.4** The VAE must be small enough to run jointly with the 1.5B backbone on one A100
80GB. Target: < 10M parameters for the full VAE (encoder + decoder + transition model).

**R4.5** The latent dimension must be the same across `latent_grpo` and
`latent_grpo_uncertainty` arms — the only difference between those two arms is whether
the KL intrinsic bonus is added to the reward.

---

### R5 — Policy conditioning

**R5.1** `z_h` (sampled from the encoder posterior) must be injected into the policy
before the policy head. Acceptable injection points: concatenation with input embedding,
cross-attention, prefix token — to be decided in the design doc.

**R5.2** `z_h` is a continuous vector, not a token. It is injected outside the token
sequence (e.g. via concatenation before the MLP head, or as a cross-attention key/value).
It does **not** consume any of the 1024-token budget. The token budget remains identical
to the baseline arm.

**R5.3** The same backbone (`Qwen/Qwen2.5-1.5B-Instruct`) must be used as all other
arms — no model switch.

---

### R6 — Training loop

**R6.1** The training loop must follow the same GRPO algorithm as the baseline arm.
Locked hyperparameters (identical across all arms):
- Group size G = 8
- KL coefficient = 0.001
- Learning rate = 1e-6
- Batch size = 128
- Training budget = 200 steps

**R6.2** The RL reward function is unchanged: binary correctness vs `ground_truth`
(symbolic/numeric equivalence).

**R6.3** The total loss is:
```
L_total = L_RL (GRPO policy gradient)
        + λ_trans × L_transition (Markov enforcement, R2.1)
        + [latent_grpo_uncertainty only] β_t × KL intrinsic bonus in reward
```

**R6.4** `λ_trans` and `β_t` are hyperparameters. Their default values must be
documented and justified. They must not be tuned post-hoc to cherry-pick results.

**R6.5** The loop must handle the case where the sparse reward is zero for many
consecutive steps without diverging. No NaN blowups (this is a Phase 3 contract gate).

---

### R7 — Fairness constraints (carried from contract)

**R7.1** Same pretrained checkpoint as all arms: `Qwen/Qwen2.5-1.5B-Instruct`.

**R7.2** Same benchmark pool, same reward function, same 200-step training budget.

**R7.3** Same maximum token budget per rollout: 1024 tokens.

**R7.4** Same evaluation protocol: `pass@1024` (+ `pass@1`, `pass@16`) on the
MATH-B-I base pool.

**R7.5** The only things that may differ from the baseline arm are: the latent module
(VAE), the policy conditioning (z injection), the auxiliary loss (L_transition), and
optionally the intrinsic reward bonus. Everything else is locked.

---

## 4. Non-Functional Requirements

**NFR1 — Modularity:** The VAE, transition model, and intrinsic reward must be
importable as isolated modules. The GRPO loop calls them; they do not depend on the loop.
This makes unit testing possible and keeps `grpo_latent.py` readable.

**NFR2 — Compute budget:** Full training run (200 steps, A100 80GB) must complete
within the existing budget. No run should exceed what the baseline arm cost.

**NFR3 — Smoke test:** A smoke config must exist that runs end-to-end in < 10 minutes
on the local 4060 (QLoRA, 2 steps, 2 problems, small batch). This is the gate before
any A100 run.

**NFR4 — Gradient flow verification:** Training logs must record whether the RL reward
was non-zero in each step, and whether the transition loss was non-zero. These are the
two canaries for the two new requirements (R2, R3).

**NFR5 — Reproducibility:** Same seeding stack as other arms. Smoke test must produce
identical outputs across two runs (per `repro_tolerance.yaml`).

**NFR6 — Early latent structure check (pre-A100 gate):** Before committing a full A100
training run, the VAE must be verified to produce non-trivial latents. Concretely: run
the encoder on rollouts from the pretrained model, plot t-SNE/UMAP of `z_h` vectors,
and check whether correct vs incorrect trajectories separate in latent space. If the
latent is flat (no structure), do not proceed to full training — diagnose first (try
larger latent dim, longer warm-up, different input aggregation). This check costs ~1
hour on the 4060 and prevents wasting an A100 run on a non-functional VAE.

---

## 5. Evaluation Requirements (Markov Diagnostics)

These are required for the paper — not just nice-to-haves. Without them, the claim that
`z_h` satisfies the Markov property is an assertion, not a result.

**E1 — Transition sufficiency:** The trained transition model `f(z_h, a_h) → z_{h+1}`
must be evaluated on held-out trajectories. Report the transition loss as evidence that
`z_h + a_h` is sufficient to predict the next state without history. This is the same
loss as R2.1 — training and evaluation use the same metric.

**E2 — Policy sufficiency (last-state-only ablation):** Evaluate the trained policy in
two conditions: (a) conditioned on `z_h` as normal, (b) conditioned on full token
history with no latent. The gap in `pass@1024` between these conditions is evidence that
`z_h` is carrying signal the token history cannot — or that the latent hasn't learned
anything useful. Either result is informative.

**E3 — Uncertainty calibration:** Check whether `σ_h²` correlates with problem
difficulty (e.g. harder problems → higher average variance) and with outcome (incorrect
trajectories → higher variance at final step). This validates the exploration bonus
design.

**E4 — Main result table:** `pass@1024` (+ `pass@1`, `pass@16`) for all four arms,
generated by `run_ablation_table.py` from artifacts. Not hand-typed.

Result tiers (from `PROJECT_BRIEF.md`):

| tier | criteria |
|------|----------|
| **Minimum viable** | All four arms evaluated; latent arm(s) characterised vs both baselines (win, tie, or loss — honest); matched budgets |
| **Strong** | Latent beats history baseline; uncertainty bonus improves over latent alone (E3 confirms); sample efficiency gain visible (E5) |
| **Exceptional** | Positive result holds across two benchmarks; latent space has interpretable structure (t-SNE/UMAP clusters correspond to problem-solving phases) |

Noise floor bar: the latent arm must score ≥ 18.0% pass@1024 for the result to be
attributable to the method rather than sampling noise (±1 problem = ±2.5pp on 40
problems; baseline_grpo at 15.0% is the reference). See `reports/writeup_stubs.md`.

**E5 — Sample efficiency (secondary):** Does the latent arm reach the same `pass@1024`
as the baseline with fewer training steps? Plot `pass@1024` vs training step for both
arms. A crossing point below 200 steps counts as a sample efficiency win.

**E6 — Latent space visualisation (bonus):** t-SNE/UMAP of `z_h` vectors across the
eval trajectories. Check whether clusters correspond to problem-solving phases (e.g.
problem setup, exploration, verification). Not required for publication but strengthens
the interpretability claim.

---

## 6. Out of Scope (for this phase)

- Second model family or RL algorithm
- Second benchmark (ProntoQA is future work if time allows)
- Diffusion-based latent or persona conditioning — future work
- Any changes to the baseline or token-Markov arms

---

## 7. Open Questions (to resolve in design doc)

These are not requirements gaps — they are design decisions that must be made before
implementation begins. They should not be resolved here.

1. **Injection mechanism** for `z_h` into the policy: concat with input embedding vs.
   cross-attention prefix vs. soft prompt token.

2. **Granularity of reasoning steps**: how is one "step `h`" defined for a free-form
   text rollout? Fixed token count (like Delethink)? Sentence boundary? End-of-thought
   marker? The answer affects what `a_h` is and how the transition loss is computed.

3. **When is the VAE trained?** Jointly with RL from step 0? Pre-trained on rollouts
   from the pretrained model before RL starts? The answer affects gradient interference
   and cold-start behaviour.

4. **λ_trans schedule**: constant or annealed? If annealed, in which direction?

5. **Transition model architecture**: should `f(z_h, a_h)` encode `a_h` from token
   embeddings, from hidden states, or from a summary embedding?

6. **Averaging window N for hidden state extraction**: `z_h` is derived from hidden
   states "averaged over the last N tokens" of the trajectory at step `h`. What is N?
   Too small = noisy; too large = slow to update. Default candidates: last 64 tokens,
   last full chunk, learned weighted average.
