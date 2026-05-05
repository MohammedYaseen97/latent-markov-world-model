# Rough Design Scratchpad — Latent Arm
*Scratch doc for getting on the same page. Not an official design doc.*
*Last updated: May 3 2026 — reflects full conversation up to design lock.*

---

## Part 1 — What is this thing trying to do?

### The problem with how LLMs do RL today

When you train an LLM with GRPO (or any RL post-training), the "state" the policy sees
is the full concatenation of every token generated so far:

```
[system prompt] [problem] [token1 token2 token3 ... token_N_generated_so_far]
```

This is a bad state for RL because:
- It grows unboundedly (the longer the reasoning, the bigger the state)
- It's redundant (most of the early tokens are irrelevant by step 500)
- It has no Markov property — the current state doesn't cleanly summarize everything
  needed to decide the next action without dragging in all history

The consequence: GRPO can only redistribute probability toward reasoning paths the
base model could already take. It can't discover genuinely new ones. This is the
documented "capability ceiling" (Yue et al. 2025).

### The fix this project is testing

Instead of feeding the full token history as state, compress it into a small fixed-size
vector that captures *where the model is in the solution space* at this moment.

That vector is called **z_h** — the **latent state at reasoning step h**.

- **z** = latent (a compressed representation in a learned latent space)
- **h** = the step number (h=1 after the first chunk, h=2 after second, etc.)

z_h is produced by a **VAE** (Variational Autoencoder). The VAE compresses
the backbone LM's internal hidden states (the model's internal representations,
not raw tokens) into a small vector.

The VAE doesn't produce just one vector — it produces a distribution:
- **μ_h** (mu) — the mean: best estimate of current position in latent space
- **σ_h²** (sigma squared) — the variance: uncertainty about that estimate

We sample from it: **z_h ~ N(μ_h, σ_h²)** using the reparameterization trick
(z_h = μ_h + ε·σ_h, ε ~ N(0,1)) so gradients flow through the sampling step.

### Is z_h actually "solution space"?

Honest answer: not guaranteed by architecture. It's a bet on training pressure.

The hidden states fed into the VAE encoder are richer than raw tokens — a 1.5B
transformer's final-layer representations encode meaning, not just surface form.
But "meaning as the model represents it" ≠ "position in mathematical solution space."
The hidden states still capture syntax, surface patterns, specific numbers, etc.

What pushes z_h toward solution-relevant information:
1. **L_transition** (temporal structure): z_h + action must predict z_{h+1} without
   token history. This forces z_h to retain what's predictive of the trajectory's
   future — which is more about mathematical state than surface syntax.
2. **L_outcome during pretraining** (quality orientation): forces the encoder's
   1536→64 bottleneck to retain information predictive of whether the trajectory
   succeeds. Directly connects z_h to solution quality.
3. **L_RL** (during RL training): sparse, but when it fires it reinforces z_h values
   associated with correct solutions.

The diagnostics (NFR6, E1–E3) are the empirical check — t-SNE, transition loss on
held-out trajectories, variance vs difficulty correlation. If training pressure didn't
work, these will show it and we don't proceed to A100.

---

## Part 2 — The full picture (rollout, single problem)

```
PROBLEM: "Find the number of integers n such that..."

┌─────────────────────────────────────────────────────────────────────────┐
│ CHUNK 1 — up to ~341 tokens generated                                   │
│                                                                         │
│  Input: [system prompt] + [problem]                                     │
│  (no z prefix on chunk 1 — there's no previous state yet)              │
│                                                                         │
│  Model generates ~341 tokens of reasoning text                          │
│  → internally produces final-layer hidden states for each token         │
│                                                                         │
│  After chunk 1 finishes:                                                │
│    1. Extract final-layer hidden states of the ~341 generated tokens    │
│    2. Mean-pool over all of them → traj_repr_1  (shape: hidden_dim)    │
│    3. VAE encoder: traj_repr_1 → (μ_1, log_σ_1²)                      │
│    4. Sample: z_1 = μ_1 + ε·σ_1   [reparameterization trick]          │
│    5. Project: z_1 (dim=64) → linear → (dim=1536) = prefix embedding   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ z_1 prefix embedding (dim=1536)
                                ↓ prepended as virtual token to chunk 2 input
┌─────────────────────────────────────────────────────────────────────────┐
│ CHUNK 2 — up to ~341 more tokens generated                              │
│                                                                         │
│  Input: [z_1 prefix token] + [system prompt] + [problem]               │
│  z_1 is a soft/virtual token — not a real word, just a learned vector  │
│  injected via inputs_embeds. Does NOT consume the generation budget.    │
│                                                                         │
│  Model generates ~341 more tokens                                       │
│                                                                         │
│  After chunk 2 finishes:                                                │
│    1–5. Same extraction → traj_repr_2 → (μ_2, σ_2²) → z_2             │
│                                                                         │
│    Transition loss (computed at end of rollout, not inline):            │
│    L_transition += ||f(z_1, traj_repr_1) - z_2||²                      │
│    (transition model predicts z_2 from z_1 + what happened in chunk 1) │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ z_2 prefix embedding
                                ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ CHUNK 3 — up to ~342 tokens generated                                   │
│                                                                         │
│  Input: [z_2 prefix token] + [system prompt] + [problem]               │
│                                                                         │
│  Model generates final tokens, hopefully ending in \boxed{answer}      │
│  → produces z_3 = z_final (the "endpoint" latent)                      │
│                                                                         │
│  Total generated: ~341 + 341 + 342 = 1024 tokens  ← same as baseline  │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ↓
              Grade full output: reward r = 1.0 if correct, 0.0 otherwise
              [uncertainty arm only]: r += β_t × mean(KL_h over h=1,2,3)
```

---

## Part 3 — Training: two phases

### IMPORTANT: this is NOT joint training from step 0.

The training is split into two distinct phases. This was a major decision.

```
PHASE 0 — VAE Pretraining (NOT counted in the 200-step RL budget)

  Why: The LLM arrives at RL training pretrained on general data.
  The VAE shouldn't arrive as random noise. It should be pretrained
  on related data (easier math problems) the same way the LLM was
  pretrained before RL.

  Backbone: FROZEN throughout Phase 0
  Trains: encoder, decoder, transition model, outcome_head

  Data: MATH-Beyond complement — 141 problems that are NOT in the
        hard 40-problem eval pool (181 full pool − 40 hard = 141)
        These are problems where at least one base model solved them
        at pass@1024 → they're "easier" by a published, reproducible rule.
        Already in the repo: data/math_beyond_full_181.jsonl

  Why this data:
  - Same benchmark family → same domain, same format, same hidden state
    distribution as the target environment
  - Not in the eval pool → no contamination
  - Easier → Qwen2.5-1.5B-Instruct actually succeeds on some (~10-30%)
    → rollouts have both correct AND incorrect trajectories → rich labels
  - Already downloaded → zero extra effort

  Generation + hidden state extraction (one-time, no grad, BEFORE VAE training):
    Step 1: run frozen backbone on 141 problems × G=8 rollouts
    Step 2: in a separate forward pass, extract and SAVE final-layer hidden
            states for each chunk as static arrays (numpy / .pt files on disk)
    Step 3: grade all rollouts → save (correct/incorrect) label per trajectory

    The backbone is NEVER in the VAE training computational graph.
    Hidden states are static inputs to the VAE, not re-computed each step.

    Why static (not live backbone with requires_grad_(False) on params):
    If the backbone were in the graph with frozen params, the gradient from
    L_outcome on z_final would flow backward THROUGH the frozen backbone
    to reach z_1 and z_2 activations (backbone acts as fixed differentiable
    function — params get no gradient but ops are in the graph). This would
    require storing activations for a 28-layer 1.5B transformer per step.
    Static hidden states avoid this entirely: gradient terminates at the
    constant hidden state tensors. Phase 0 cost = small MLP backward passes only.

  Losses during Phase 0:
    L_ELBO       = reconstruction_loss + KL(q(z|τ) || N(0,1))
    L_transition = ||f(z_h, traj_repr_h) - z_{h+1}||²
    L_outcome    = BCE(outcome_head(z_final), correct_or_not)

  What L_outcome is:
    A small 2-layer MLP classification head attached to z_final (the last
    chunk's latent). Predicts P(correct) for the full trajectory.
    Trained with binary cross-entropy on the labeled rollouts.
    DISCARDED after Phase 0 — it is only scaffolding for initialization.
    Attaches to z_final only (not intermediate z_h) because:
    - z_final is the endpoint, outcome is known → cleanest signal
    - Avoiding credit assignment assumptions across intermediate steps

  Why L_outcome is not redundant with L_RL:
    They operate in different phases, on different components:
    - L_outcome fires DENSELY during Phase 0 on 141 easier problems
      where correct trajectories actually exist (10-30% success rate)
    - L_RL fires SPARSELY during Phase 1 on 40 hard problems (0.02%)
    - Backbone is frozen in Phase 0 → L_outcome ONLY updates the encoder
    - L_RL primarily updates the backbone/policy
    - The bottleneck (1536→64) has no reason to retain quality-relevant
      info without L_outcome — it would just compress surface statistics

  Budget: not counted in the 200-step RL budget. Specify K steps in config.
          (default candidate: 2–3 epochs over the 141 problems with G=8
           = 141 × 3 = ~423 generation passes, VAE trains on all of them)

  Cost: cheap — no backbone backward pass. Only the small VAE modules.

---

PHASE 1 — Joint RL Training (the 200-step RL budget)

  Backbone: UNFROZEN
  Outcome head: DISCARDED (not used)
  VAE: starts from Phase 0 initialization (not random noise)

  Data: the hard 40-problem MATH-B-I pool (same as baseline arm)

  Losses during Phase 1:
    L_total = L_RL + λ_t × L_transition + L_VAE

    L_RL:          GRPO policy gradient (same algorithm as baseline arm)
                   Sparse: only fires when at least 1 of G=8 rollouts is correct
                   Updates: backbone, policy head

    L_transition:  ||f(z_h, traj_repr_h) - z_{h+1}||²
                   Always non-zero. Keeps Markov property active.
                   Updates: transition model, encoder, (backbone indirectly)

    L_VAE:         ELBO: reconstruction + KL(q(z|τ) || N(0,1))
                   Always non-zero. Keeps encoder/decoder coherent.
                   Updates: encoder, decoder

    λ_t schedule:  1.0 at step 0, linear decay to 0.1 by step 100,
                   held at 0.1 for steps 100-200.
                   High early → dense gradient flow while reward is sparse.
                   Low late → RL takes over once rewards appear.
                   Floor at 0.1 → L_transition stays active for diagnostics.
```

---

## Part 4 — The modules

```
src/models/vae_state_encoder.py
  ├── VAEStateEncoder
  │     ├── encoder:     MLP(hidden_dim=1536 → 512 → 2×latent_dim=128)
  │     │                [outputs μ (64-dim) and log_σ² (64-dim)]
  │     ├── decoder:     MLP(latent_dim=64 → 512 → hidden_dim=1536)
  │     │                [reconstruction, training only]
  │     └── transition:  MLP(latent_dim + hidden_dim = 64+1536 → 512 → latent_dim=64)
  │                      [predicts next latent from current latent + action embedding]
  ├── OutcomeHead        MLP(latent_dim=64 → 64 → 1) + sigmoid
  │                      [Phase 0 only, discarded before Phase 1]
  └── helpers:
        extract_hidden_states(model, input_ids, response_ids) → (hidden_dim,)
        reparameterize(mu, log_var) → z
        elbo_loss(recon, target, mu, log_var) → scalar
        transition_loss(z_h, traj_repr_h, z_h_plus_1) → scalar

src/training/grpo_latent.py
  ├── pretrain_vae(config, run_dir)              ← Phase 0
  ├── train_latent(config, run_dir)              ← Phase 1, no uncertainty bonus
  └── train_latent_with_uncertainty(...)         ← Phase 1, with uncertainty bonus

src/training/reward_bonus.py
  └── compute_uncertainty_bonus(mu, log_var, beta_t) → scalar
```

---

## Part 5 — All design decisions locked

### Q1: How does z_h get into the model?

**Decision: soft prefix token**

z_h (dim=64) is projected to model hidden dim (1536) via a learned linear layer.
The resulting 1536-dim vector is prepended as a single virtual token to chunk h+1's
input using Qwen2.5's `inputs_embeds` argument.

The model treats it as a special token at position 0 whose embedding encodes the
latent state. The semantics are learned jointly under training pressure — the same way
BERT's [CLS] token learned sentence-level semantics without anything architecturally
special about it.

Token budget: untouched. This prefix is in the INPUT, not the GENERATION. The model
still generates 1024 tokens across 3 chunks.

Rejected alternatives:
- Concat with input embedding: requires changing input dim → architectural surgery
- Cross-attention: requires modifying attention layers → too invasive

---

### Q2: What counts as one step h?

**Decision: fixed token count — ~341 tokens per chunk, 3 chunks**

Why: the token-Markov arm uses the same 3-chunk structure (different sizes due to
carryover, but same number of chunks). The ONLY thing that differs between the two arms
is what carries state forward: last-m tokens (token-Markov) vs z_h (latent). Same
chunks, same token budget, different compression. Clean ablation.

Token-Markov chunks: 512 + 256 + 256 = 1024 (256 tokens "consumed" by carryover)
Latent arm chunks: 341 + 341 + 342 = 1024 (no carryover → full budget for reasoning)

This is actually an advantage of the latent approach: the token budget wasted on
carryover in token-Markov is freed up for actual reasoning in the latent arm.

Rejected alternatives:
- Sentence boundary: unreliable to detect mid-generation
- End-of-thought marker: invasive, requires new special token

---

### Q3: When does the VAE get trained?

**Decision: VAE pretraining phase (Phase 0) on the 141-problem MATH-B complement,
then joint RL training (Phase 1) on the hard 40.**

NOT joint training from step 0. That was the original plan but it was revised after
recognising the cold-start problem (VAE would be random noise when first RL rewards
appear) and the sparse reward problem (base latent arm has no dense reward during RL
training on hard 40 problems).

The LLM pretraining analogy is the design principle:
- LLM: pretrained on broad internet → arrives at RL with general reasoning capability
- VAE: pretrained on easier MATH-B problems → arrives at RL with general trajectory
  structure and quality-orientation already baked in

R2.2 (Markov objective must be joint, not a separate phase) is satisfied: L_transition
is active throughout Phase 1. Phase 0 is INITIALIZATION, analogous to LLM pretraining,
not "a separate training phase on the target task."

---

### Q4: λ_trans schedule?

**Decision: linear decay from 1.0 to 0.1 over steps 0–100, held at 0.1 for 100–200**

High early → L_transition dominates when L_RL ≈ 0 (sparse reward stretches)
Low late → L_RL takes over when rewards start appearing
Floor at 0.1 → L_transition remains active for the Markov diagnostic (E1)

Applies to Phase 1 only. During Phase 0, λ_trans is a separate config hyperparameter
(default: 1.0, no schedule needed since L_RL is not present).

---

### Q5: How does the transition model encode action a_h?

**Decision: reuse traj_repr_h (mean-pooled hidden states of chunk h) as the action
embedding. No separate action encoding.**

f(z_h, a_h) → z_{h+1}_pred
where a_h = traj_repr_h = mean_pool(final-layer hidden states of chunk h tokens)

Why: traj_repr_h is already computed for the VAE encoder. Reusing it as the action
embedding is cheap (no extra compute) and semantically correct — it IS the representation
of what the model did during chunk h.

Architecture: 2-layer MLP, input = concat(z_h, traj_repr_h) = 64+1536 = 1600 dim,
hidden = 512, output = 64.

---

### Q6: Averaging window for hidden state extraction?

**Decision: mean-pool over ALL tokens generated in the current chunk**

Rationale: we defined step h = one full chunk. The action at step h is everything
generated in that chunk. Mean-pooling over all of it is consistent with the step
definition, simple, and adds no extra parameters.

Rejected: last-64-tokens (arbitrary, biased toward chunk end), learned weighted
average (extra parameters, extra complexity for marginal gain).

---

### Q7: L_outcome attachment point?

**Decision: z_final only (the latent of the last chunk)**

z_final is the endpoint of the full trajectory. At that point, the outcome is known
(reward was computed). The outcome prediction signal is cleanest here — we're predicting
the actual outcome of the complete trajectory, not making assumptions about
intermediate credit assignment.

Using intermediate z_h would require: "z_h at step h should predict the eventual
outcome even though reasoning isn't finished yet." That's a harder learning problem
and introduces credit assignment assumptions we'd have to justify.

**Does L_outcome on z_final reach z_1 and z_2 encoders?**

It depends on whether the backbone is in the computational graph.

If backbone were in graph (requires_grad_(False) on params, not torch.no_grad()):
  Gradient would flow: L_outcome → z_3 → encoder → hidden_states_3
  → THROUGH frozen backbone (fixed differentiable function) → z_2 prefix
  → z_2 → encoder → hidden_states_2 → backbone → z_1 → encoder
  So YES, gradient would reach z_1 and z_2 directly. But this is expensive
  (28-layer 1.5B backward passes × 3 chunks per step).

With static pre-generated hidden states (chosen approach — see Phase 0):
  Gradient terminates at the constant hidden_states tensors.
  L_outcome directly updates only z_3's encoder pass.
  z_1 and z_2 encoder passes do NOT get gradient from L_outcome directly.

This is fine for two reasons:
1. The encoder is a SINGLE MODULE applied at all chunks. L_outcome updating
   encoder weights via z_3 affects how z_1 and z_2 are computed (same weights).
2. L_transition provides cross-chunk consistency: it forces z_2 to be consistent
   with z_3, and z_1 to be consistent with z_2. If z_3 encodes quality (via
   L_outcome), quality-relevant structure propagates backward through the
   consistency chain — structurally, not via backprop through the backbone.

---

## Part 6 — Remaining open items (hyperparameters, not architecture)

These are NOT blocking. They go in configs, not in the design doc.

| Item | Default candidate | Why not blocked |
|------|-------------------|-----------------|
| Phase 0 training budget K | 2-3 epochs over 141 problems | Empirical — check loss curves |
| G for Phase 0 rollouts | 8 (same as RL, cheaper since backbone frozen) | Config knob |
| Loss weights in Phase 0 | λ_ELBO=1.0, λ_trans=1.0, λ_outcome=1.0 | Document as hyperparameters |
| VAE LR vs backbone LR (Phase 1) | Same LR (1e-6), monitor | Can separate if needed |
| λ_trans decay midpoint | Step 100 (of 200) | Reasonable default, not critical |

---

## Summary table — locked decisions

| Decision | Choice |
|----------|--------|
| z_h injection | Soft prefix token (project 64→1536, prepend via inputs_embeds) |
| Step granularity | 3 fixed chunks, ~341 tokens each, = 1024 total |
| Chunk 1 input | No z prefix (no prior state) |
| Chunks 2-3 input | z_{h-1} prefix + system prompt + problem |
| VAE training structure | Phase 0 (pretraining, frozen backbone) → Phase 1 (joint RL) |
| Phase 0 data | MATH-B complement: 141 problems (181 full − 40 hard) |
| Phase 0 losses | L_ELBO + L_transition + L_outcome |
| L_outcome attachment | z_final (last chunk's latent) only |
| L_outcome fate | Discarded before Phase 1 begins |
| Phase 1 losses | L_RL + λ_t × L_transition + L_VAE |
| λ_trans schedule | 1.0 → 0.1 linear over steps 0-100, held at 0.1 for 100-200 |
| Action embedding | traj_repr_h (mean-pool hidden states of chunk h tokens) |
| Hidden state aggregation | Mean-pool over all tokens in the chunk |
| Latent dim | 64 |
| VAE + transition model size | < 10M params |
| Per-chunk token budget | ~341 (latent) vs 512/256/256 (token-Markov) |
| Outcome head architecture | 2-layer MLP (64 → 64 → 1) + sigmoid |
| Transition model architecture | 2-layer MLP (1600 → 512 → 64) |
| Encoder architecture | MLP (1536 → 512 → 128), outputs μ+log_σ² both 64-dim |
| Decoder architecture | MLP (64 → 512 → 1536) |

---

---

## Uncertainty Arm — Context Preserved (design TBD)

*Not designed yet. This section captures the reasoning from the design conversation
so context isn't lost when we come back to it.*

### What it is

`latent_grpo_uncertainty` is identical to `latent_grpo` in every respect except one:
the KL divergence of the VAE posterior is added to the GRPO reward as an intrinsic
exploration bonus:

```
r_total = r_binary  +  β_t × mean_h( KL(N(μ_h, σ_h²) ∥ N(0, I)) )

where:
  r_binary = 1.0 if correct, 0.0 otherwise   (same as all arms)
  KL_h     = KL divergence of chunk h's VAE posterior from the prior
             = high when model is uncertain about z_h
             = low when encoder is confident about position in latent space
  β_t      = annealing weight: high early (exploration) → low late (exploitation)
```

The KL bonus is in the **reward**, not the loss. It affects GRPO advantages directly —
a trajectory with high uncertainty gets a higher effective reward even if r_binary = 0.
This makes the advantage non-zero more often, which means L_RL flows more often.

### Why this exists (the sparse reward argument)

The base latent arm (`latent_grpo`) addresses sparse reward via:
- L_transition: dense gradient for the VAE/backbone even when L_RL = 0
- Phase 0 pretraining: VAE arrives with quality-relevant structure

But L_transition doesn't directly increase P(non-zero advantage for the POLICY).
The policy still needs L_RL to fire, and L_RL still needs r_binary > 0 at least
somewhere in the G=8 group.

The uncertainty bonus attacks this differently: it makes the REWARD dense. Even when
r_binary = 0 for all G=8 rollouts, β_t × KL gives each rollout a non-zero reward
based on its uncertainty. Advantages are now non-zero → L_RL flows → the policy learns.

### The connection to the "stochastic diversity" intuition

Early in the conversation there was an intuition: "the VAE should be so stochastic/
random at the start that it increases P(one of G=8 samples getting the answer right)."

That intuition is valid but the mechanism was wrong: random VAE noise doesn't know
which direction is "correct" — it's equally likely to push the model away from correct
solutions as toward them.

The uncertainty bonus is the principled version of that intuition:
- Instead of hoping random noise accidentally hits correct trajectories, we REWARD
  the model for visiting uncertain states (high KL = model doesn't know where it is)
- High uncertainty → model is in unexplored latent space → exploring
- Low uncertainty → model is confident about its position → exploiting
- β_t annealing: start high (encourage exploration when policy hasn't learned much)
  → decay to near zero (let r_binary dominate once policy has warmed up)

### The paper narrative (why both arms are needed)

Base latent arm (`latent_grpo`):
  Tests: "Does a learned Markov state help, even with only sparse binary rewards?"
  Role: Demonstrates the VAE can learn meaningful structure via Phase 0 + L_transition.
        Establishes that the Markov state formulation is sound.
        May show modest improvement over baseline, or null result.
        Either is informative.

Uncertainty arm (`latent_grpo_uncertainty`):
  Tests: "Does adding intrinsic exploration density on top of the latent state help?"
  Role: This is the headline arm. If the base latent arm is held back by sparse reward,
        the uncertainty bonus unlocks it.
        The paper story: "latent state alone is necessary but not sufficient for hard
        problems; adding uncertainty-driven exploration is what breaks the ceiling."

### What still needs to be designed (open questions for uncertainty arm session)

1. **β_t schedule**: same direction as λ_t (high → low)?
   What are the initial and final values? Over how many steps?

2. **KL averaging**: mean over h=1,2,3 chunks? Or weight by chunk position
   (later chunks matter more)?

3. **Scale of β_t relative to r_binary**: KL values are typically in the range 0–5
   nats for a VAE. β_t needs to be calibrated so the bonus is meaningful early
   but doesn't drown out r_binary later. A β_0 = 0.1 would add at most ~0.5 to
   the reward. Needs empirical checking.

4. **Phase 0 interaction**: the outcome head is trained on r_binary, not r_total.
   Should Phase 0 also warm up the uncertainty signal in any way? Or just arrive
   at Phase 1 with the VAE ready and let the β_t bonus do its work?

5. **Diagnostic**: E3 (uncertainty calibration) — σ_h² should correlate with
   problem difficulty and with incorrect outcomes. This is the empirical validation
   that the bonus is doing principled exploration, not just adding noise.

---

*Implementation deliverables and order tracked in `reports/latent_markov_design.md`.*
