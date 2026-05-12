# Latent Markov Arm: Design Document

Requirements reference: `reports/latent_markov_requirements.md`

---

## The Problem This Arm Solves

Both the baseline arm and the token-Markov arm fail to fully escape the RLVR capability
ceiling, for different reasons.

The baseline arm uses full token history as the MDP state. This is not a proper Markov
state — it grows unboundedly, is redundant, and gives the policy no compact model of where
it is in the solution space. GRPO on this state redistributes probability toward existing
paths but cannot discover new ones (Yue et al. 2025).

The token-Markov arm enforces a strict information bottleneck by construction: old tokens
are deleted, last-m tokens carry forward. The Markov property is guaranteed structurally.
But the constraint crashes per-sample success (context resets make the problem too hard
for the model at this scale), leading to zero gradient for all 200 training steps —
a stable fixed point with no learning.

The latent arm tests a third approach: **replace token history with a learned compact
latent state `z_h` — a continuous vector derived from the backbone's internal hidden
states via a VAE.** The Markov property is not enforced by construction; it is learned
via a transition consistency loss that forces `z_h` to satisfy the Markov property
empirically on actual reasoning trajectories.

---

## The Core Idea

At each reasoning step `h` (chunk boundary), the backbone has processed some tokens and
produced internal hidden states. We compress those hidden states through a VAE encoder to
get a distribution over latent states:

```
encoder(mean_pool(final_layer_hidden_states_of_chunk_h))  →  (μ_h, log_σ_h²)
z_h  =  μ_h  +  ε·σ_h        ε ~ N(0, I)    [reparameterization]
```

`z_h` is a 64-dimensional vector. It is injected into the next chunk's input as a soft
prefix token: project `z_h` (dim 64) → model hidden dim (dim 1536) via a learned linear,
prepend as a virtual token to chunk h+1's `inputs_embeds`. The policy generates its
next chunk conditioned on this prefix.

The transition model learns: `z_h → z_{h+1}`. If this loss is small, `z_h` alone
is sufficient to predict the next latent state without token history. That is the
Markov property, empirically verified.

---

## Pipeline and Gradient Flow (ASCII Block Diagram)

The full pipeline runs in two phases. Phase 0 pretrains the VAE and ZInjector jointly
with a frozen backbone. Phase 1 runs joint RL with all components live.

```
══════════════════════════════════════════════════════════════════════════════════
  PHASE 0 — ONLINE VAE PRE-TRAINING  (backbone FROZEN)
══════════════════════════════════════════════════════════════════════════════════

  Easy MATH pool, 200 steps, batch_size=4, G=8

      problem prompt
            │
      ┌─────▼──────────────────────────────────────────────────────────────────┐
      │  BACKBONE  (Qwen 1.5B-Instruct)                         [FROZEN]       │
      │                                                                        │
      │  generate: 341 tokens, no z prefix  →  chunk1_ids                      │
      │  forward:  [prompt ‖ chunk1]  →  repr_1 = mean_pool    [LIVE]          │
      └─────────────────────────┬──────────────────────────────────────────────┘
                                │  repr_1  [LIVE]
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  VAE Encoder  (MLP 1536→512→64×2)           [UPDATES]   │
                     │  μ_1, log_σ_1²  =  encoder(repr_1)                      │
                     │  ε_1 ~ N(0,I)  [reparameterization trick]               │
                     │  z_1  =  μ_1  +  ε_1 · exp(½·log_σ_1²)                  │
                     └──────────┬──────────────────────────────────────────────┘
                                │  z_1  (64-dim)
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  ZInjector  (Linear 64→1536, init std=0.01)  [UPDATES]  │
                     │  prefix_1  =  W_inj · z_1             [in graph]        │
                     └──────────┬──────────────────────────────────────────────┘
                                │  prefix_1  (1536-dim embedding)
      ┌─────────────────────────▼──────────────────────────────────────────────┐
      │  BACKBONE  [FROZEN, with_grad]                                         │
      │                                                                        │
      │  generate: 341 tokens, inputs=[prefix_1 ‖ chunk1]  →  chunk2_ids       │
      │  forward:  [prefix_1_emb ‖ prompt ‖ chunk1 ‖ chunk2]                   │
      │            →  repr_2 = mean_pool(last-layer hidden)    [LIVE]          │
      └─────────────────────────┬──────────────────────────────────────────────┘
                                │  repr_2  [LIVE]
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  VAE Encoder  →  z_2  (64-dim)               [UPDATES]  │
                     │  ZInjector    →  prefix_2  (1536-dim)        [UPDATES]  │
                     └──────────┬──────────────────────────────────────────────┘
                                │  prefix_2  (1536-dim embedding)
      ┌─────────────────────────▼──────────────────────────────────────────────┐
      │  BACKBONE  [FROZEN, with_grad]                                         │
      │                                                                        │
      │  generate: 342 tokens, inputs=[prefix_2 ‖ chunk2]  →  chunk3_ids       │
      │            grade full output  →  reward r                              │
      │  forward:  [prefix_2_emb ‖ ... ‖ chunk3]                               │
      │            →  repr_3 = mean_pool(last-layer hidden)    [LIVE]          │
      └─────────────────────────┬──────────────────────────────────────────────┘
                                │  repr_3  [LIVE]
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  VAE Encoder  →  z_3 (= z_final)             [UPDATES]  │
                     └──────────┬──────────────────────────────────────────────┘
                                │  z_3  (64-dim)
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  Outcome Head  (MLP 64→64→1 + sigmoid)       [UPDATES]  │
                     │  P̂(correct)  =  σ(MLP(z_3))                             │
                     └──────────┬──────────────────────────────────────────────┘
                                │
  ┌─────────────────────────────▼──────────────────────────────────────────────────┐
  │  PHASE 0 LOSSES                                                                │
  │                                                                                │
  │  L_ELBO  = Σ_h  [ MSE(decode(z_h), repr_h) + kl_w(t) · KL(N(μ_h,σ_h²)‖N(0,I))] │
  │  L_trans = ‖ f(z_1) − z_2 ‖²  +  ‖ f(z_2) − z_3 ‖²                             │
  │  L_out   = BCE( P̂(correct), r )                                                │
  │  L_total = L_ELBO  +  λ_t · L_trans  +  λ_o · L_out                            │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  GRADIENT ROUTES (Phase 0)                                                  │
  │                                                                             │
  │  L_ELBO  →  encoder ✓   decoder ✓                                          │
  │  L_trans →  transition ✓   encoder ✓   ZInjector ✓                         │
  │             ( L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone               │
  │                                   → prefix_h → ZInjector )                  │
  │  L_out   →  OutcomeHead ✓   encoder ✓                                      │
  │  All     →  backbone activations  [passthrough; ∂L/∂θ_bb accumulated        │
  │              but optimizer never called on backbone params]                 │
  └─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════════
  PHASE 1 — JOINT RL TRAINING  (backbone UNFROZEN)
══════════════════════════════════════════════════════════════════════════════════

  Hard 40-problem MATH-B-I pool, 200 steps, batch_size=4, G=8

  ┌──────────────────── ROLLOUT (no_grad) ─────────────────────────────────────┐
  │                                                                            │
  │  Same 3-chunk loop as Phase 0.  G=8 rollouts per problem.                  │
  │  Store per rollout: chunk_ids, reward.  (No repr_h, z_h, or log_π_old.)    │
  │  Compute GRPO advantages from group rewards.                               │
  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────── TRAINING STEP (with_grad) ─────────────────────────────┐
  │                                                                            │
  │  Re-run the full 3-chunk pipeline (identical to Phase 0 except backbone    │
  │  is unfrozen) for each rollout, using stored chunk_ids for log_π.          │
  │  repr_h and z_h are LIVE (not stored from rollout).                        │
  │  IS ratio = 1 exactly (same policy, no update between rollout and step).   │
  │                                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │  PHASE 1 LOSSES                                                     │   │
  │  │                                                                     │   │
  │  │  L_RL    = −adv · log_π_current             [GRPO; sparse; IS = 1]  │   │
  │  │  L_trans = ‖f(z_1)−z_2‖²  +  ‖f(z_2)−z_3‖²   [always non-zero]      │   │
  │  │  L_VAE   = MSE(decode(z_h), repr_h) + kl_w · KL  [always non-zero]  │   │
  │  │                                                                     │   │
  │  │  L_total = L_RL  +  λ_t · L_trans  +  λ_vae · L_VAE                 │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │                                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │  GRADIENT ROUTES (Phase 1)                                          │   │
  │  │                                                                     │   │
  │  │  L_RL    →  lm_head ✓   backbone ✓   ZInjector ✓   encoder ✓       │   │
  │  │             ( L_RL → log_π → backbone → prefix_h → ZInjector        │   │
  │  │                                       → z_h → encoder → repr_h )    │   │
  │  │  L_trans →  transition ✓   encoder ✓   ZInjector ✓   backbone ✓    │   │
  │  │             ( L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone       │   │
  │  │                                   → prefix_h → ZInjector )          │   │
  │  │  L_VAE   →  encoder ✓   decoder ✓   backbone ✓                     │   │
  │  │  All losses reach backbone: repr_h is LIVE; backbone.step() called. │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  └────────────────────────────────────────────────────────────────────────────┘
```

---

## The Mechanism (full rollout)

```
ROLLOUT FOR ONE PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

query q = [system prompt] + [problem]

┌───────────────────────────────────────────────────────────────┐
│ CHUNK 1                                                       │
│  input:  q                      (no z prefix — no prior state)│
│  generate: up to 341 tokens                                   │
│  extract: mean_pool(hidden_states_of_341_tokens) = repr_1     │
│  encode: encoder(repr_1) → (μ_1, σ_1²) → sample z_1           │
│  project: z_1 (64-dim) → ZInjector → (1536-dim) = prefix_1    │
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_1 prepended to chunk 2 input
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 2                                                       │
│  input:  [prefix_1] + q          (z_1 as virtual token)       │
│  generate: up to 341 tokens                                   │
│  extract: mean_pool(hidden_states_of_341_tokens) = repr_2     │
│  encode: encoder(repr_2) → (μ_2, σ_2²) → sample z_2           │
│  project: z_2 → prefix_2                                      │
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_2 prepended to chunk 3 input
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 3                                                       │
│  input:  [prefix_2] + q          (z_2 as virtual token)       │
│  generate: up to 342 tokens                                   │
│  extract: repr_3 → z_3 = z_final                              │
└─────────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
       Grade full output: r = 1.0 if correct, 0.0 otherwise

TOTAL NEW TOKENS: 341 + 341 + 342 = 1024  ←  matches baseline exactly ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Comparison with token-Markov arm:**

```
Token-Markov:  chunk 1 (512 tokens) + chunk 2 (256 tokens) + chunk 3 (256 tokens)
               = 1024 tokens. BUT 256 tokens per chunk are "consumed" by carryover text.
               Context reset: old tokens deleted, only last-m raw tokens carry forward.

Latent arm:    chunk 1 (341 tokens) + chunk 2 (341 tokens) + chunk 3 (342 tokens)
               = 1024 tokens. No token carryover. Full budget available for reasoning.
               State carried forward as z_h (64 numbers), not raw tokens.
```

The token budget previously consumed by raw token carryover is freed up for actual
reasoning in the latent arm. This is a direct benefit of learned latent state over
token-space state.

---

## Architecture

### VAE (encoder, decoder, transition)

All three components are small MLPs satisfying R4.4 (< 10M params total).

**Encoder** — maps trajectory representation to latent distribution:

```
repr_h (1536-dim)
  → Linear(1536, 512) + ReLU
  → Linear(512, 128)
  → split into μ (64-dim) and log_σ² (64-dim)
```

**Decoder** — maps latent back to trajectory representation (training only, R4.2):

```
z (64-dim)
  → Linear(64, 512) + ReLU
  → Linear(512, 1536)
  → reconstruction of repr_h
```

**Transition model** — predicts next latent from current latent:

```
z_h  (64-dim)
  → Linear(64, 512) + ReLU
  → Linear(512, 64)
  → z_{h+1}_predicted
```

The pure Markov property is `f(z_h) → z_{h+1}`: the current state alone predicts
the next. `repr_h` is deliberately excluded. The ELBO separately handles the
compression task (repr_h → z_h); including repr_h in the transition would let the
model bypass the bottleneck and weaken the gradient pressure on z_h to be
information-dense.

**Outcome head** — Phase 0 only, attached to z_final:

```
z_final (64-dim)
  → Linear(64, 64) + ReLU
  → Linear(64, 1) + Sigmoid
  → P(correct)
```

Discarded before Phase 1 begins. Used only to shape the encoder during pretraining.

**ZInjector** — projects z_h into the backbone's embedding space as a soft prefix token:

```
z_h (64-dim) → Linear(64, 1536, bias=False) → prefix_embedding (1536-dim)
```

Prepended to chunk h+1's input via `inputs_embeds`. Not a real token — no vocabulary
entry. Does not consume the 1024-token generation budget (R5.2).

**ZInjector initialisation:** `nn.init.normal_(weight, std=0.01)`.
Default Kaiming-uniform gives std ≈ 0.125, making the prefix embedding O(1) magnitude —
the same as a real token embedding — which injects random noise and confuses the
pretrained backbone immediately. Starting at std=0.01 keeps the prefix ~12× smaller
than a token embedding, making it effectively neutral at init. Phase 0 online training
will grow the weights via L_transition and L_outcome before Phase 1 begins. Same principle
as LoRA zero-init for new adapter parameters inserted into a frozen backbone.

The encoder is shared across all three chunks (same weights applied three times).

---

## Training: Two Phases

### Why two phases?

The VAE and ZInjector arrive cold. If joint RL begins immediately with randomly
initialised encoder and ZInjector, `z_h` is noise and the policy conditions on noise.
When sparse RL rewards do appear (≈ 0.02% per sample), GRPO updates land on a
meaningless latent space.

Phase 0 pretrains VAE and ZInjector jointly on an easier pool while the backbone is
frozen. This orients the latent space toward solution-quality information before RL
begins, so when Phase 1 rewards do arrive, they land on structured representations.

---

### Phase 0 — Online VAE Pretraining

**Backbone: FROZEN for weight updates. Gradients still pass through backbone activations
(required by the L_transition → ZInjector gradient chain).**

**Data: L1–L5 MATH pool (`data/math_easy_pool.jsonl`), 4974 problems.**

The pool must satisfy: the pretrained instruct model solves ≥20% of problems at pass@8
so L_outcome receives a mix of correct and incorrect trajectory labels. L1–L5 provides
this calibration while staying out of the hard MATH-B-I eval pool.

**Generation:** live, online. Each Phase 0 step generates a mini-batch of complete
3-chunk rollouts using the backbone with z prefix injection (the same inference loop
as Phase 1). No pre-saved rollouts. No separate rollout-generation script.

**G=8 rollouts per problem per step.** Even though L_out is supervised (BCE), running
G=8 rollouts per problem per step provides variance in outcome labels within each batch
— some rollouts pass, some fail — giving L_out richer supervision than a single
deterministic rollout would. L_trans and L_VAE additionally benefit from seeing diverse
solution trajectories for the same problem in each step.

This is the critical architectural choice over an offline approach:

- The VAE and ZInjector see repr tensors extracted from *z-injected* generation —
the same distribution they will face in Phase 1. No train/eval distribution mismatch.
- The ZInjector is fully trained by Phase 0: L_transition provides gradient via the
path `L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone → prefix_h → ZInjector`.
Phase 1 starts with a warm, task-shaped ZInjector, not a cold random one.
- No separate rollout generation script or pre-computed tensor cache needed.

**repr_h computation:** forward pass `[prefix_h_emb ‖ prompt ‖ ... ‖ chunk_h]` with
a registered forward hook on the backbone's last layer. `repr_h = mean_pool` of the
hidden states over the chunk_h token positions. Hook avoids the memory overhead of
`output_hidden_states=True` for all 28 layers.

**Phase 0 losses:**

```
L_ELBO    =  Σ_h [ MSE(decode(z_h), repr_h) + kl_w(t) · KL(N(μ_h,σ_h²) ‖ N(0,I)) ]

L_trans   =  ‖ f(z_1) − z_2 ‖²  +  ‖ f(z_2) − z_3 ‖²

L_out     =  BCE(outcome_head(z_final), r)

L_phase0  =  L_ELBO  +  λ_trans · L_trans  +  λ_out · L_out
```

Default weights: `λ_trans = 1.0`, `λ_out = 1.0`. Specified in config.

**KL annealing:** `kl_weight` ramps linearly from 0 → 1 over the first `kl_warmup_frac`
(default 0.5) of Phase 0 steps. Without annealing, the KL term from step 0 drives σ²
toward the prior (1.0) before the encoder has learned to compress repr_h meaningfully —
posterior collapse. KL annealing lets reconstruction dominate early, then gradually
introduces the regularisation pressure. Config key: `phase0.kl_warmup_frac: 0.5`.

**λ_trans schedule in Phase 0:** same warmup logic as Phase 1 (described below) applied
from Phase 0 step 0. Prevents the transition loss from enforcing Markov consistency on
useless representations before the encoder has any structure.

**Phase 0 budget:** `phase0.n_steps` in config. Default: 200 steps, batch_size=4,
G=8 → 800 problem-encounters, 6400 total rollouts (matches Phase 1 encounter count).
Phase 0 is more expensive than an offline approach (live backbone forward passes per
step) but fully eliminates distribution mismatch.

**Gate:** NFR6 (see below) must pass before Phase 1 begins.

---

### Phase 1 — Joint RL Training (200 steps)

**Backbone: UNFROZEN.**
**Outcome head: DISCARDED.**
**VAE + ZInjector: initialised from Phase 0 checkpoint.**

**Data:** hard 40-problem MATH-B-I pool — same as baseline and token-Markov arms.

**Phase 1 losses:**

```
L_RL         =  GRPO policy gradient
             =  -advantage × log_π_current   [IS = 1; same policy as rollout]
             Sparse: fires only when ≥1 of G=8 rollouts has r=1.

L_transition =  ‖ f(z_h) − z_{h+1} ‖²
             Always non-zero. Keeps Markov property active during RL.
             Also the diagnostic metric for E1 (transition sufficiency).

L_VAE        =  MSE(decode(z_h), repr_h)  +  kl_w · KL(N(μ_h, σ_h²) ‖ N(0, I))
             Always non-zero. Prevents encoder/decoder drift during RL.

L_phase1     =  L_RL  +  λ_t × L_transition  +  λ_vae × L_VAE
```

**λ_t schedule (transition loss weight):**

```python
def lambda_trans_schedule(step, max_steps=200, floor=0.1, peak=0.5):
    halfway = max_steps // 2
    if step >= halfway: return peak
    return floor + (peak - floor) * (step / halfway)
```

Warmup 0.1 → 0.5 over steps 0–100, then constant at 0.5.

Rationale: At step 0, L_RL is near-zero (sparse reward, hard problems). Starting
λ_t = 0.1 lets the rare early reward events shape z before the Markov consistency
constraint locks in structure. As z accumulates signal, λ_t ramps to enforce Markov
consistency. Held at 0.5 for the second half so L_transition remains an active
regulariser throughout. Starting at the old high value (1.0) would force Markov
self-consistency on useless early representations.

**λ_vae = 0.05.** L_VAE at weight 1.0 is orders of magnitude larger than near-zero
L_RL, which would lock the encoder into its Phase 0 fixed point and drown any sparse
reward signal. 0.05 keeps VAE regularisation active without dominating.

**Phase 1 is structurally identical to Phase 0** except: (a) the backbone is unfrozen,
(b) L_RL replaces L_out, (c) G=8 rollouts are collected per problem before the training
step to compute GRPO advantages. The training step re-runs the full 3-chunk pipeline
with grad — no stored repr_h, z_h, μ, or log_π_old. Because the training step uses
the same policy that generated the rollout with no intervening weight update, IS = 1
exactly and no importance-sampling correction is needed.

**On-policy GRPO loop (200 steps):** every single training step is:
1. `[no_grad]` collect G=8 fresh rollouts for the current batch of 4 problems → 32 sequences
2. Grade all 32, compute GRPO advantages from normalised group rewards per problem
3. `[with_grad]` re-run full 3-chunk pipeline for all 32 sequences → live log_π, repr_h, z_h
4. Compute L_total, backward, step optimiser
5. Discard rollouts. Advance to next step.

No replay buffer. No multi-epoch reuse of rollouts. Rollouts are collected fresh at
every step and used for exactly one gradient update — standard on-policy GRPO.

All three losses reach the backbone via live repr_h in the computation graph;
`backbone.step()` is called (unlike Phase 0 where .step() is skipped).

**batch_size = 4.** With batch_size=1, the custom latent loop processes only 200
problem-encounters over 200 steps, vs ~800 for the baseline (TRL batch_size=8,
grad_accum=4). Reward density scales with encounters. At batch_size=4: 4×8=32
sequences per step, 800 problem-encounters over 200 steps — matching the baseline.
If OOM: fall back to batch_size=2 and max_steps=400.

**Why Phase 1 is expected to work better than token-Markov:**

1. No context resets → per-sample success ≈ same as baseline, not near-zero.
2. L_transition and L_VAE provide dense gradient flow even when L_RL = 0. All
  components learn from every step, not only from rare reward events.
3. ZInjector and z_h arrive from Phase 0 already oriented toward quality — not cold
  random noise. When rare RL rewards appear, GRPO lands on structured latent space.

---

## Is z_h "solution space"?

Honest answer: not guaranteed by architecture. It is a bet on training pressure.

The encoder's input (final-layer hidden states) encodes meaning as the backbone has
learned it from pretraining — not raw token statistics, but also not provably
"position in mathematical solution space." Three forces push z_h toward solution-relevant
representations during training:

1. **L_outcome (Phase 0):** directly rewards z_final for retaining information predictive
  of correct vs incorrect trajectories on easier math problems.
2. **L_transition (both phases):** forces z_h to retain information predictive of z_{h+1},
  which rewards trajectory-structure information over surface syntax.
3. **L_RL (Phase 1, sparse):** when it fires, rewards z_h values associated with
  successful reasoning paths.

Whether these forces are sufficient is empirically tested by the diagnostics (E1, E3)
and the NFR6 t-SNE gate.

---

## Requirements Satisfaction


| Requirement                                | Satisfied by                                                   |
| ------------------------------------------ | -------------------------------------------------------------- |
| R1.1 — z_h at each step                    | VAE encoder applied after each chunk                           |
| R1.2 — derived from backbone hidden states | mean_pool(final-layer hidden states of chunk h)                |
| R1.3 — fixed-size z_h                      | 64-dim regardless of step count                                |
| R1.4 — uncertainty estimate σ_h²           | VAE posterior variance, used by uncertainty arm                |
| R1.5 — z_h conditions policy before head   | soft prefix token injected via inputs_embeds                   |
| R2.1 — transition consistency loss         | L_transition = ‖f(z_h) − z_{h+1}‖²                             |
| R2.2 — Markov objective joint with RL      | L_transition active throughout Phase 1                         |
| R2.3 — same loss is diagnostic metric      | L_transition on held-out trajectories = E1                     |
| R3.1 — dense auxiliary signal              | L_transition and L_VAE non-zero on every step                  |
| R3.2 — transition loss satisfies R3.1      | explicitly: L_transition is always computable                  |
| R3.4 — gradients in first ~20 steps        | L_transition + L_VAE flow from step 0                          |
| R4.1 — encoder: MLP, dim 64–128            | 1536 → 512 → 64, latent dim 64                                 |
| R4.2 — decoder: training only              | decoder not used at eval/inference                             |
| R4.3 — reparameterization trick            | z = μ + ε·σ, ε ~ N(0, I)                                       |
| R4.4 — VAE < 10M params                    | encoder + decoder + transition ≈ 4–5M                          |
| R4.5 — same latent dim both arms           | 64-dim shared between latent and latent+uncertainty            |
| R5.1 — z_h injected before policy head     | soft prefix prepended to inputs_embeds                         |
| R5.2 — z_h not in token budget             | virtual prefix token, not counted in 1024 generation tokens    |
| R5.3 — same backbone                       | Qwen/Qwen2.5-1.5B-Instruct throughout                          |
| R6.1 — GRPO hyperparameters locked         | inherited from train_baseline_grpo.yaml via extends            |
| R6.2 — reward unchanged                    | binary correctness, same math_reward function                  |
| R6.3 — total loss                          | L_RL + λ_t × L_transition + λ_vae × L_VAE                      |
| R6.4 — hyperparameters documented          | λ_t schedule above; β_t in uncertainty arm design              |
| R6.5 — no NaN blowups when reward=0        | L_transition + L_VAE keep gradients finite; gate in smoke      |
| R7.1–R7.5 — fairness                       | same checkpoint, pool, reward, budget, token limit as all arms |


---

## Key Parameters


| Parameter                        | Value                                      | Notes                                                     |
| -------------------------------- | ------------------------------------------ | --------------------------------------------------------- |
| Latent dim                       | 64                                         | R4.1 range: 64–128                                        |
| Chunk size                       | 341 / 341 / 342 tokens                     | = 1024 total, equal split, no carryover                   |
| z injection                      | soft prefix via inputs_embeds              | Q1 decision                                               |
| Encoder architecture             | MLP 1536→512→64 (×2 outputs)               | μ and log_σ²                                              |
| Decoder architecture             | MLP 64→512→1536                            | training only                                             |
| Transition architecture          | MLP 64→512→64                              | input = z_h only (pure Markov)                            |
| Outcome head                     | MLP 64→64→1 + sigmoid                      | Phase 0 only, discarded after                             |
| ZInjector init                   | `nn.init.normal_(std=0.01)`                | near-zero; prevents cold-start noise injection            |
| Phase 0 data                     | L1–L5 MATH pool, 4974 problems             | `data/math_easy_pool.jsonl`                               |
| Phase 0 max_steps                | 200                                        | 800 problem-encounters; matches Phase 1                   |
| Phase 0 batch_size               | 4                                          |                                                           |
| Phase 0 G (rollouts per problem) | 8                                          | variance in outcome labels; richer L_out supervision      |
| Phase 0 generation               | live, online (z-injected, frozen backbone) | eliminates Phase 0→1 distribution mismatch                |
| Phase 0 KL warmup                | kl_weight 0→1 over first 50% of steps      | prevents posterior collapse                               |
| Phase 0 λ_trans                  | warmup schedule (same as Phase 1)          | avoids early over-constraint                              |
| Phase 0 λ_out                    | 1.0                                        | config knob                                               |
| Phase 0 λ_elbo                   | 1.0                                        | config knob                                               |
| Phase 1 max_steps                | 200                                        | budget                                                    |
| Phase 1 batch_size               | 4                                          | 800 problem-encounters; matches baseline gradient density |
| Phase 1 G (rollouts per problem) | 8                                          | locked                                                    |
| Phase 1 lr                       | 1e-6                                       | locked (all arms)                                         |
| Phase 1 λ_t                      | warmup 0.1→0.5 over steps 0–100, held 0.5  | Markov constraint after z has structure                   |
| Phase 1 λ_vae                    | 0.05                                       | VAE regularisation without drowning L_RL                  |
| Backbone                         | Qwen/Qwen2.5-1.5B-Instruct                 | R5.3                                                      |
| repr_h extraction                | forward hook on last layer, mean_pool      | avoids output_hidden_states memory overhead               |


---

## Engineering Notes

The same TRL incompatibility that forced a custom loop for the token-Markov arm applies
here. Multi-chunk generation with `z_h` conditioning between chunks is incompatible with
TRL's single-sequence-per-rollout assumption. The latent arm uses a custom training loop
in `src/training/grpo_latent.py`, following the same structural pattern as
`src/training/grpo_token_markov.py`.

**Phase 0 loop** (`pretrain_vae_online`): mirrors `train_latent()` in structure.
Backbone frozen via `requires_grad_(False)` for parameter updates — but backbone
operations remain in the computation graph so gradients flow from loss to ZInjector.
No separate `generate_phase0_rollouts.py` invocation needed.

**Phase 1 rollout phase:** `@torch.no_grad()` for generation. Only `chunk_ids` and
`reward` are stored per rollout. No repr_h, z_h, μ, log_σ², or log_π_old retained.
GRPO advantages computed from group rewards (G=8) after collection.

**Phase 1 training phase:** backbone unfrozen. The full 3-chunk pipeline is re-run
with grad for each stored rollout (identical code path to Phase 0). repr_h and z_h are
LIVE — no detached inputs. Because no weight update occurs between rollout and training
step, IS = 1 exactly; no ε or importance-correction needed. All three losses reach
backbone via live repr_h in the computation graph.

**OOM handling:** adaptive batch halving on CUDA OOM — `_run_adaptive` helper halves
the batch recursively until it fits or reaches size 1. Applied in both Phase 0 and
Phase 1 rollout generation.

---

## Implementation Deliverables

Ordered by dependency. Each step is a gate for the next.


| #   | Deliverable                                                                                                       | File                                   | Status              |
| --- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------- |
| 1   | Phase 0 dataset: `data/math_easy_pool.jsonl` — L1–L5 (4974 problems)                                              | `scripts/prepare_easy_pool.py`         | ✅ reuse             |
| 2   | `VAEStateEncoder` — encoder, decoder, transition; `compute_elbo(kl_weight)`                                       | `src/models/vae_state_encoder.py`      | ✅                   |
| 3   | `OutcomeHead` — 2-layer MLP on z_final, Phase 0 only                                                              | `src/models/vae_state_encoder.py`      | ✅                   |
| 4   | `ZInjector` — near-zero init (std=0.01)                                                                           | `src/models/vae_state_encoder.py`      | ✅                   |
| 5   | `pretrain_vae_online()` — Phase 0 online loop: frozen backbone, live generation, VAE+ZInjector update             | `src/training/grpo_latent.py`          | ⬜ implement         |
| 6   | `train_latent()` — Phase 1 custom GRPO loop                                                                       | `src/training/grpo_latent.py`          | ✅                   |
| 7   | `generate_latent_traces()` — chunked inference engine with z injection, repr hook; stores chunk_ids + reward only | `src/training/grpo_latent.py`          | ⬜ update            |
| 8   | `latent_training_step()` — re-runs full pipeline with grad; L_RL + λ_t·L_trans + λ_vae·L_VAE; IS = 1              | `src/training/grpo_latent.py`          | ⬜ update            |
| 9   | Smoke config                                                                                                      | `configs/train_latent_grpo_smoke.yaml` | ⬜ re-smoke after #5 |
| 10  | Full Phase 1 config (200 steps, batch_size=4)                                                                     | `configs/train_latent_grpo.yaml`       | ✅                   |
| 11  | Latent eval modes in eval_passk.py (`latent_markov`, `latent_markov_pretrained`)                                  | `scripts/eval_passk.py`                | ✅                   |
| 12  | **Phase 0 training run** → `runs/latent_grpo/phase0_vae.pt`                                                       | `scripts/train_latent.py`              | ⬜ pending #5        |
| 13  | **NFR6 gate** — UMAP of z_final on Phase 0 checkpoint                                                             | `scripts/check_latent_structure.py`    | ⬜ pending #12       |
| 14  | **Controlled latent baseline** eval: `latent_grpo_pretrained` pass@1024 ≥ 12.5%                                   | `scripts/eval_passk.py`                | ⬜ pending #12       |
| 15  | **Phase 1 training** — 200 steps on MATH-B-I                                                                      | `configs/train_latent_grpo.yaml`       | ⬜ pending #14       |
| 16  | **Phase 1 eval** — pass@k on MATH-B-I holdout                                                                     | `scripts/eval_passk.py`                | ⬜ pending #15       |
| 17  | **E1 + E3 Markov diagnostics**                                                                                    | `scripts/eval_markov_diagnostics.py`   | ⬜ pending #15       |


---

## NFR6 Gate

**When to run:** after Phase 0 training completes (`runs/latent_grpo/phase0_vae.pt`).

**How:** `python scripts/check_latent_structure.py` — runs the encoder on the Phase 0
pool, computes UMAP of z_final coloured by correct/incorrect trajectory.

**Gate criteria:** structured manifold with visible outcome correlation (correct and
incorrect trajectories should not be uniformly intermixed; some geometric separation
or outcome-correlated structure should be visible). Does not require hard disjoint
clusters — UMAP topology with outcome-correlated layout suffices.

**If gate fails:** diagnose before Phase 1. Likely causes: Phase 0 too short
(increase `phase0.n_steps`), L_out too weak (increase `λ_out`), posterior collapse
(extend KL warmup). Do not proceed to Phase 1 on a gate failure.

**Output:** `runs/latent_grpo/plots/latent_structure_umap.png` + `nfr6_summary.json`.

---

## Controlled Latent Baseline (`latent_grpo_pretrained`)

**Definition:** Phase 0 VAE + ZInjector + pretrained backbone, evaluated on MATH-B-I
with no Phase 1 updates.

**Purpose:** establishes the capability floor under the latent generation regime, parallel
to `baseline_pretrained` (12.5%) for the other arms. Without this, we cannot distinguish
"Phase 1 improved from X to Y" from "the latent regime itself costs performance."

**Gate:** pass@1024 ≥ 12.5%. With near-zero ZInjector init, the Phase 0 prefix starts
near-neutral, so flat-generation capability should be preserved at the 12.5% pretrained
level. If below 12.5%, the Phase 0 ZInjector has learned noisy or adversarial prefixes —
diagnose Phase 0 before proceeding.

**Evaluation:** `scripts/eval_passk.py --generation-mode latent_markov_pretrained`.
Loads backbone from HF model ID and VAE from `phase0.checkpoint_path`.

---

## Markov Diagnostics (required for paper)

Empirical evidence that `z_h` satisfies the Markov property. Without this, the
Markov claim is an assertion, not a result.

**E1 — Transition sufficiency:** held-out transition loss `‖f(z_h) − z_{h+1}‖²` on
unseen trajectories. Near-zero MSE = Markov property holds empirically.
→ `scripts/eval_markov_diagnostics.py`

**E2 — Policy sufficiency:** last-state-only ablation: latent arm pass@1024 vs
baseline → covered by the core ablation table; no separate script needed.

**E3 — Uncertainty calibration:** Pearson r(σ_h², reward) — higher variance should
correlate with lower reward (harder/unresolved trajectories). Sign must be correct;
magnitude threshold documented in requirements.
→ `scripts/eval_markov_diagnostics.py`

---

## Pass Criteria


| Criterion                                      | Threshold                                                                                    |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Smoke test                                     | completes end-to-end < 10 min on 4060                                                        |
| NFR6 gate                                      | structured UMAP manifold with outcome correlation                                            |
| Controlled baseline (`latent_grpo_pretrained`) | pass@1024 ≥ 12.5%                                                                            |
| Phase 1 logs                                   | L_transition non-zero from step 0; L_RL non-zero within first 30 steps; λ_vae=0.05 confirmed |
| `latent_grpo` pass@1024                        | ≥ 18.0% (≥ 3pp above baseline_grpo — required to clear noise floor on 40-problem pool)       |
| No NaN blowups                                 | stable under zero-reward stretches (R6.5)                                                    |
| Shared hyperparameters                         | G=8, lr=1e-6, 200 steps, same backbone confirmed in log                                      |
| E1 Markov diagnostic                           | held-out transition loss near zero                                                           |
| E3 calibration                                 | σ_h² negatively correlated with reward (sign correct)                                        |


---

**Not in scope for this arm:** uncertainty bonus (β_t × KL in reward). That is
`latent_grpo_uncertainty` — separate implementation session after this arm is complete.