# Latent Markov Arm: Design Document (v2)

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
states via a deterministic encoder.** The Markov property is not enforced by construction;
it is learned via a transition consistency loss that forces `z_h` to satisfy the Markov
property empirically on actual reasoning trajectories.

---

## The Core Idea

At each reasoning step `h` (chunk boundary), the backbone has processed some tokens and
produced internal hidden states. We compress those hidden states through a deterministic
encoder:

```
encoder(mean_pool(final_layer_hidden_states_of_chunk_h))  →  (μ_h, log_σ_h²)
z_h  =  μ_h        [always deterministic — training and generation]
```

`z_h` is a 64-dimensional vector used as the Markov state. The encoder also outputs
`log_σ²_h` — a quality indicator head trained explicitly to be high on incorrect
trajectories and low on correct ones via L_calib. `log_σ²_h` is never used for z
computation or sampling; its only role is to inject quality-oriented gradient pressure
into the shared encoder backbone through L_calib. `z_h` is injected into the next
chunk's input as a soft prefix token: project `z_h` (dim 64) → model hidden dim
(dim 1536) via a learned linear, prepend as a virtual token to chunk h+1's
`inputs_embeds`.

The transition model learns: `z_h → z_{h+1}`. If this loss is small, `z_h` alone
is sufficient to predict the next latent state without token history. That is the
Markov property, empirically verified.

---

## Pipeline and Gradient Flow (ASCII Block Diagram)

The full pipeline runs in two phases. Phase 0 pretrains the encoder and ZInjector jointly
with a frozen backbone. Phase 1 runs joint RL with all components live.

```
══════════════════════════════════════════════════════════════════════════════════
  PHASE 0 — ONLINE ENCODER PRE-TRAINING  (backbone FROZEN)
══════════════════════════════════════════════════════════════════════════════════

  Easy MATH pool (L1-L4), 400 steps, 400 problems × 128 rollouts (pre-shuffled assignment)

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
                     │  Encoder  (MLP 1536→512→64×2)               [UPDATES]   │
                     │  μ_1, log_σ_1²  =  encoder(repr_1)                      │
                     │  z_1  =  μ_1   (always deterministic)                    │
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
      │  forward:  [prefix_1_emb ‖ chunk2]  (strict Markov; no raw history)    │
      │            →  repr_2 = mean_pool(hidden, chunk2 positions)  [LIVE]     │
      └─────────────────────────┬──────────────────────────────────────────────┘
                                │  repr_2  [LIVE]
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  Encoder  →  z_2  (64-dim)                   [UPDATES]  │
                     │  ZInjector    →  prefix_2  (1536-dim)        [UPDATES]  │
                     └──────────┬──────────────────────────────────────────────┘
                                │  prefix_2  (1536-dim embedding)
      ┌─────────────────────────▼──────────────────────────────────────────────┐
      │  BACKBONE  [FROZEN, with_grad]                                         │
      │                                                                        │
      │  generate: 342 tokens, inputs=[prefix_2 ‖ chunk2]  →  chunk3_ids       │
      │            grade full output  →  reward r                              │
      │  forward:  [prefix_2_emb ‖ chunk3]  (strict Markov; no raw history)    │
      │            →  repr_3 = mean_pool(hidden, chunk3 positions)  [LIVE]     │
      └─────────────────────────┬──────────────────────────────────────────────┘
                                │  repr_3  [LIVE]
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  Encoder  →  z_3 (= z_final)                 [UPDATES]  │
                     └──────────┬──────────────────────────────────────────────┘
                                │  z_3  (64-dim)
                     ┌──────────▼──────────────────────────────────────────────┐
                     │  Outcome Head  (MLP 64→64→1)                 [UPDATES]  │
                     │  logit(correct)  =  MLP(z_3)  [raw logit; no sigmoid]   │
                     └──────────┬──────────────────────────────────────────────┘
                                │
  ┌─────────────────────────────▼──────────────────────────────────────────────────┐
  │  PHASE 0 LOSSES                                                                │
  │                                                                                │
  │  L_trans = ‖ f(z_1) − z_2 ‖²  +  ‖ f(z_2) − z_3 ‖²                             │
  │  L_out   = BCE_with_logits( logit, r, pos_weight )                             │
  │            pos_weight dynamic per step: (1−pos_rate)/pos_rate, clamped ≤20     │
  │  L_calib = BCE_with_logits( −mean_logvar, r, pos_weight )                      │
  │            High σ² → −mean_logvar small → predicts incorrect (r=0)             │
  │            Low σ² → −mean_logvar large → predicts correct (r=1)                │
  │  L_total = λ_t · L_trans  +  λ_o · L_out  +  λ_c · L_calib                   │
  │  λ_t peak=3.0 (warmup: λ_t=0 for first 50 steps)  λ_o=5.0  λ_c=1.0           │
  └───────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  GRADIENT ROUTES (Phase 0)                                                  │
  │                                                                             │
  │  L_trans  →  transition ✓   encoder ✓   ZInjector ✓                        │
  │              ( L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone              │
  │                                    → prefix_h → ZInjector )                 │
  │  L_out    →  OutcomeHead ✓   encoder ✓                                     │
  │  L_calib  →  logvar_head ✓   encoder ✓                                     │
  │  All      →  backbone activations  [passthrough; optimizer never stepped]   │
  └─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════════
  PHASE 1 — JOINT RL TRAINING  (backbone UNFROZEN)
══════════════════════════════════════════════════════════════════════════════════

  Level 5 hard pool (~350 problems), 200 steps, batch_size=4, G=128

  ┌──────────────────── ROLLOUT (no_grad) ─────────────────────────────────────┐
  │                                                                            │
  │  Same 3-chunk loop as Phase 0.  G=128 rollouts per problem.                  │
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
  │  │  L_RL    = −adv · log_π_current               [GRPO; IS = 1]         │   │
  │  │  L_trans = ‖f(z_1)−z_2‖²  +  ‖f(z_2)−z_3‖²   [always non-zero]      │   │
  │  │  L_calib = BCE_with_logits(−mean_logvar, r, pos_weight)  [always non-zero] │   │
  │  │                                                                     │   │
  │  │  L_total = L_RL  +  λ_t · L_trans  +  λ_calib · L_calib             │   │
  │  │  λ_t=0.3 (maintenance)  λ_calib=0.5                                 │   │
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
│  generate: model.generate([q])  →  chunk1_ids (341 tokens)   │
│  repr fwd: model([q ‖ chunk1])  →  repr_1 = mean_pool(hidden) │
│  encode:   encoder(repr_1) → (μ_1, log_σ_1²); z_1 = μ_1 always│
│  project:  ZInjector(z_1) → prefix_1  (1536-dim virtual token)│
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_1 used for chunk 2
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 2                                                       │
│  generate: model.generate([prefix_1 ‖ chunk1])  →  chunk2_ids │
│            (crutch: raw chunk1 kept for rollout quality)      │
│  repr fwd: model([prefix_1 ‖ chunk2])  → repr_2 = mean_pool   │
│            (strict Markov: only z_1 prefix + new chunk tokens)│
│  encode:   encoder(repr_2) → z_2; project → prefix_2         │
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_2 used for chunk 3
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 3                                                       │
│  generate: model.generate([prefix_2 ‖ chunk2])  →  chunk3_ids │
│            (crutch: raw chunk2 kept for rollout quality)      │
│  repr fwd: model([prefix_2 ‖ chunk3])  → repr_3 = mean_pool   │
│            (strict Markov: only z_2 prefix + new chunk tokens)│
│  encode:   encoder(repr_3) → z_3 = z_final                   │
│  grade:    full output  →  r = 1.0 if correct, 0.0 otherwise  │
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

### Encoder, transition (v2 — no decoder, no sampling)

All components are small MLPs satisfying R4.4 (< 10M params total).
The decoder, ELBO/KL machinery, and reparameterization sampling from v1 are removed —
rung 2 is a purely deterministic tracker. z = μ everywhere: training, generation, eval.

**Encoder** — maps trajectory representation to a latent state and a quality indicator:

```
repr_h (1536-dim)
  → Linear(1536, 512) + ReLU
  → Linear(512, 128)  [shared trunk]
  → mu_head:     Linear(128, 64)  →  μ_h       [the Markov state]
  → logvar_head: Linear(128, 64)  →  log_σ²_h  [quality indicator; never used for z]

z_h = μ_h   always (training and generation)
```

`log_σ²_h` is an auxiliary output that shares the encoder trunk with `μ_h`. It is trained
via L_calib to be high for incorrect trajectories and low for correct ones. It never enters
the z computation — no sampling, no KL — but its gradient flows through the shared trunk and
shapes `μ_h` to be quality-aware. In Phase 0, the encoder also receives L_out (OutcomeHead) as a quality signal; L_calib
provides complementary quality pressure through the shared trunk. In Phase 1, OutcomeHead
is discarded — without L_calib, the encoder would only receive L_trans (Markov consistency)
and the extremely sparse L_RL (~0.1% reward rate), leaving no reliable quality-oriented
gradient for most training steps.

**Transition model** — predicts next latent from current latent:

```
z_h  (64-dim)
  → Linear(64, 512) + ReLU
  → Linear(512, 64)
  → z_{h+1}_predicted
```

The pure Markov property is `f(z_h) → z_{h+1}`: the current state alone predicts
the next. `repr_h` is deliberately excluded. L_trans handles the compression task
(repr_h → z_h via the encoder); including repr_h in the transition directly would
let the model bypass the latent bottleneck and weaken the gradient pressure on z_h
to be information-dense.

**Outcome head** — Phase 0 only, attached to z_final:

```
z_final (64-dim)
  → Linear(64, 64) + ReLU
  → Linear(64, 1)
  → logit(correct)   [raw; BCE_with_logits applied externally with dynamic pos_weight]
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

The encoder and ZInjector arrive cold. If joint RL begins immediately with randomly
initialised encoder and ZInjector, `z_h` is noise and the policy conditions on noise.
When sparse RL rewards do appear (≈ 0.02% per sample), GRPO updates land on a
meaningless latent space.

Phase 0 pretrains encoder and ZInjector jointly on an easier pool while the backbone is
frozen. This orients the latent space toward solution-quality information before RL
begins, so when Phase 1 rewards do arrive, they land on structured representations.

---

### Phase 0 — Online Encoder Pretraining

**Backbone: FROZEN for weight updates. Gradients still pass through backbone activations
(required by the L_transition → ZInjector gradient chain).**

**Data: L1–L4 MATH pool (`data/math_easy_pool.jsonl`), ~4100 problems.**
Level 5 excluded for mutual exclusivity with the hard training/eval pool.

The pool must satisfy: the pretrained instruct model solves ≥20% of problems at pass@8
so L_outcome receives a mix of correct and incorrect trajectory labels. L1–L5 provides
this calibration while staying mutually exclusive from the Level 5 hard pool (L1–L4 only).

**Generation:** live, online. Each Phase 0 step generates complete 3-chunk rollouts
using the backbone with z prefix injection (the same inference loop as Phase 1).
No pre-saved rollouts. No separate rollout-generation script.

**Phase 0 sampling strategy:** a flat list of `n_steps × G = 400 × 128 = 51,200`
`(problem, rollout_slot)` assignments is pre-computed and shuffled at the start of
Phase 0. Each step slices `seqs_per_step=128` entries from this list and generates
one rollout per entry. This guarantees: (a) every problem gets exactly 128 rollouts
total over Phase 0, (b) each step's 128 sequences span ~110 distinct problems,
providing diverse advantage signal for both L_out and L_trans.

**Generation context (crutch, Phase 0):** chunk h+1 is generated from
`[prefix_h ‖ chunk_h]` — the z prefix plus raw previous chunk tokens. The raw
chunk tokens are kept as a crutch during Phase 0 to maintain rollout quality while
the encoder and ZInjector are still warming up. This crutch is only used at
generation time; it is not used in the repr computation (see below).

**repr_h computation (strict Markov):** `repr_h` is extracted via a forward pass
over `[prefix_h_emb ‖ chunk_h]` only — no raw previous tokens. This enforces the
Markov property: z_h must encode all relevant context from the current chunk alone,
given only the previous state prefix. The forward hook on the backbone's last layer
extracts hidden states over the `chunk_h` token positions. Using `output_hidden_states=True`
is avoided; hook only fires for the last layer.

This is the critical architectural choice over an offline approach:

- The encoder and ZInjector see repr tensors extracted from *z-injected* generation —
the same distribution they will face in Phase 1. No train/eval distribution mismatch.
- The ZInjector is fully trained by Phase 0: L_transition provides gradient via the
path `L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone → prefix_h → ZInjector`.
Phase 1 starts with a warm, task-shaped ZInjector, not a cold random one.
- No separate rollout generation script or pre-computed tensor cache needed.

**Phase 0 losses (v2):**

```
L_trans   =  ‖ f(z_1) − z_2 ‖²  +  ‖ f(z_2) − z_3 ‖²

L_out     =  BCE_with_logits( outcome_head(z_final), r, pos_weight )
             pos_weight = (1 − pos_rate) / pos_rate per step, clamped ≤ 20.
             Upweights correct trajectories (minority class, ~15–20% of rollouts).

L_calib   =  BCE_with_logits( −mean_logvar, r, pos_weight )
             where mean_logvar = mean over chunks and latent dims of logvar_h
             High σ² → log_σ² large → −mean_logvar small → predicts r=0 (incorrect) ✓
             Low σ²  → log_σ² small → −mean_logvar large → predicts r=1 (correct)   ✓

L_phase0  =  λ_trans · L_trans  +  λ_out · L_out  +  λ_calib · L_calib
```

Default weights: `λ_trans_peak = 3.0`, `λ_out = 5.0`, `λ_calib = 1.0`.
`λ_trans_warmup_steps = 50`: λ_t = 0 for the first 50 steps, then ramps 0.1→3.0
over the remaining 350 steps. Warmup lets L_out establish a signal before L_trans
dominates the gradient budget.

**Approximate gradient budget at Phase 0 peak:**
- L_trans × 3.0 → ~50% — primary: Markov structure
- L_out × 5.0 → ~23% — outcome quality signal
- L_calib × 1.0 → ~10% — explicit σ² calibration

**No KL annealing in v2.** σ² is trained directly via L_calib — no reconstruction
loss, no KL term, no posterior collapse risk. logvar_head learns to predict trajectory
quality without being pulled toward the prior.

**Phase 0 budget:** `phase0.n_steps` in config. Default: 400 steps (extended from
v1's 200 — losses were still declining at termination). Each step processes 128 diverse
sequences spanning ~110 problems (pre-shuffled assignment strategy).

**Gate:** NFR6 (see below) must pass before Phase 1 begins.

---

### Phase 1 — Joint RL Training (200 steps)

**Backbone: UNFROZEN.**
**Outcome head: DISCARDED.**
**Encoder + ZInjector: initialised from Phase 0 checkpoint.**

**Data:** Level 5 hard pool (~350 problems) — same as baseline and token-Markov arms.

**Phase 1 losses (v2):**

```
L_RL         =  GRPO policy gradient
             =  -advantage × log_π_current   [IS = 1; same policy as rollout]
             With G=128, expected ≥1 correct rollout per step whenever per-sample success > 0.8%.

L_transition =  ‖ f(z_h) − z_{h+1} ‖²
             Always non-zero. Maintains Markov property during RL (already trained
             in Phase 0). Also the diagnostic metric for E1.

L_calib      =  BCE_with_logits(−mean_logvar, r, pos_weight)
             Always non-zero. Keeps σ² calibrated during RL.

L_phase1     =  L_RL  +  λ_t × L_transition  +  λ_calib × L_calib
```

**λ_t = 0.3 (maintenance mode).** Reduced from v1's 1.0. Phase 0's 400 steps with
peak=3.0 fully trains the transition model (v1 E1=0.015 at only 200 steps). Phase 1
only needs to prevent Markov drift — 0.3 is a maintenance weight, not an aggressive
training weight. Frees ~20% more gradient budget for L_RL.

**λ_calib = 0.5.** Lighter anchor than Phase 0 (λ_calib=1.0). Keeps σ² calibrated
without dominating. Approximate Phase 1 gradient budget at peak:
- L_RL × 1.0 → ~65% — primary objective
- L_trans × 0.3 → ~19% — Markov maintenance
- L_calib × 0.5 → ~9% — uncertainty anchor

**Phase 1 is structurally identical to Phase 0** except: (a) the backbone is unfrozen,
(b) L_RL replaces L_out, (c) G=128 rollouts are collected per problem before the training
step to compute GRPO advantages. The training step re-runs the full 3-chunk pipeline
with grad — no stored repr_h, z_h, μ, or log_π_old. Because the training step uses
the same policy that generated the rollout with no intervening weight update, IS = 1
exactly and no importance-sampling correction is needed.

**On-policy GRPO loop (200 steps):** every single training step is:
1. `[no_grad]` collect G=128 fresh rollouts for the current batch of 4 problems → 512 sequences
2. Grade all 512, compute GRPO advantages from normalised group rewards per problem
3. `[with_grad]` re-run full 3-chunk pipeline for all 512 sequences → live log_π, repr_h, z_h
   (micro-batched to fit memory; math equivalent to full-batch update)
4. Compute L_total, backward, step optimiser
5. Discard rollouts. Advance to next step.

No replay buffer. No multi-epoch reuse of rollouts. Rollouts are collected fresh at
every step and used for exactly one gradient update — standard on-policy GRPO.

All three losses reach the backbone via live repr_h in the computation graph;
`backbone.step()` is called (unlike Phase 0 where .step() is skipped).

**batch_size = 4 (Phase 1).** At batch_size=4 with G=128: 4×128=512 sequences
per step, 800 problem-encounters over 200 steps — matching the baseline
(TRL batch_size=8, grad_accum=64, equivalent gradient density).
The 512 sequences are processed in micro-batches of `micro_batch_size` to fit
within GPU memory, with gradients accumulated and stepped once per training step.
If OOM: reduce `micro_batch_size` (not `batch_size`) — the math is unchanged.

**Why Phase 1 is expected to work better than token-Markov:**

1. No context resets → per-sample success ≈ same as baseline, not near-zero.
2. L_transition and L_calib provide dense gradient flow even when L_RL = 0. All
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

1. **L_out (Phase 0):** directly rewards z_final for retaining information predictive
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
| R1.1 — z_h at each step                    | Encoder applied after each chunk; z_h = μ_h (deterministic)   |
| R1.2 — derived from backbone hidden states | mean_pool(final-layer hidden states of chunk h)                |
| R1.3 — fixed-size z_h                      | 64-dim regardless of step count                                |
| R1.4 — uncertainty estimate σ_h²           | logvar_head output, calibrated via L_calib; used by uncertainty arm |
| R1.5 — z_h conditions policy before head   | soft prefix token injected via inputs_embeds                   |
| R2.1 — transition consistency loss         | L_transition = ‖f(z_h) − z_{h+1}‖²                             |
| R2.2 — Markov objective joint with RL      | L_transition active throughout Phase 1                         |
| R2.3 — same loss is diagnostic metric      | L_transition on held-out trajectories = E1                     |
| R3.1 — dense auxiliary signal              | L_transition and L_calib non-zero on every step                |
| R3.2 — transition loss satisfies R3.1      | explicitly: L_transition is always computable                  |
| R3.4 — gradients in first ~20 steps        | L_transition + L_calib flow from step 0                        |
| R4.1 — encoder: MLP, dim 64–128            | 1536 → 512 → 64, latent dim 64                                 |
| R4.2 — no decoder in v2                    | decoder removed; rung 2 is tracking, not generation            |
| R4.3 — deterministic z during training     | z_h = μ_h; no reparameterization; σ² trained via L_calib only  |
| R4.4 — encoder < 10M params                | encoder + transition ≈ 2–3M (decoder removed)                  |
| R4.5 — same latent dim both arms           | 64-dim shared between latent and latent+uncertainty            |
| R5.1 — z_h injected before policy head     | soft prefix prepended to inputs_embeds                         |
| R5.2 — z_h not in token budget             | virtual prefix token, not counted in 1024 generation tokens    |
| R5.3 — same backbone                       | Qwen/Qwen2.5-1.5B-Instruct throughout                          |
| R6.1 — GRPO hyperparameters locked         | inherited from train_baseline_grpo.yaml via extends            |
| R6.2 — reward unchanged                    | binary correctness, same math_reward function                  |
| R6.3 — total loss                          | L_RL + λ_t × L_transition + λ_calib × L_calib                  |
| R6.4 — hyperparameters documented          | λ_t schedule above; β_t in uncertainty arm design              |
| R6.5 — no NaN blowups when reward=0        | L_transition + L_calib keep gradients finite; gate in smoke    |
| R7.1–R7.5 — fairness                       | same checkpoint, pool, reward, budget, token limit as all arms |


---

## Key Parameters


| Parameter                        | Value                                      | Notes                                                        |
| -------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Latent dim                       | 64                                         | z_h dimension                                                |
| Chunk size                       | 341 / 341 / 342 tokens                     | = 1024 total, equal split, no carryover                      |
| z injection                      | soft prefix via inputs_embeds              | does not consume token budget                                |
| Encoder architecture             | MLP 1536→512→64 (×2 outputs)               | μ and log_σ²; z_h = μ during training                        |
| Decoder architecture             | REMOVED in v2                              | not needed (tracking, not generation)                        |
| Transition architecture          | MLP 64→512→64                              | input = z_h only (pure Markov)                               |
| Outcome head                     | MLP 64→64→1 (raw logit)                    | Phase 0 only, discarded after; BCE_with_logits externally    |
| ZInjector init                   | `nn.init.normal_(std=0.01)`                | near-zero; prevents cold-start noise injection               |
| Phase 0 data                     | L1–L4 MATH pool, ~4100 problems            | `data/math_easy_pool.jsonl` (Level 5 excluded)               |
| Phase 0 max_steps                | 400                                        | extended from v1's 200 (losses still declining at 200)       |
| Phase 0 problems (total)         | 400 (× 128 rollouts each = 51,200 total)   | pre-shuffled assignment; ~110 problems per step              |
| Phase 0 G (rollouts per problem) | 128                                        | total across all steps; matches eval pass@128 scale          |
| Phase 0 seqs_per_step            | 128                                        | diverse sequences per step (micro-batched internally)        |
| Phase 0 generation               | live, online (z-injected, frozen backbone) | eliminates Phase 0→1 distribution mismatch                   |
| Phase 0 λ_trans_peak             | 3.0 (warmup: λ=0 for 50 steps, then ramp) | ~50% gradient budget at peak; primary Markov training        |
| Phase 0 λ_out                    | 5.0                                        | ~23% gradient budget                                         |
| Phase 0 λ_calib                  | 1.0                                        | ~10% gradient budget; explicit σ² calibration                |
| Phase 1 max_steps                | 200                                        | budget                                                       |
| Phase 1 batch_size               | 4                                          | matches baseline gradient density                            |
| Phase 1 G (rollouts per problem) | 128                                        | locked; matches eval pass@128 scale                          |
| Phase 1 lr                       | 1e-6                                       | locked (all arms)                                            |
| Phase 1 λ_trans                  | 0.3 (maintenance; ~19% gradient budget)    | reduced from v1's 1.0; transition already trained in Phase 0 |
| Phase 1 λ_calib                  | 0.5 (~9% budget)                           | lighter than Phase 0; keeps σ² anchored during RL            |
| Benchmark                        | MATH Level 5 hard pool, ~350 problems      | pass@128=0 filter on pretrained model                        |
| Primary metric                   | pass@128                                   | 8× cheaper than pass@1024; more problems compensates         |
| Backbone                         | Qwen/Qwen2.5-1.5B-Instruct                 |                                                              |
| repr_h extraction                | forward hook on last layer, mean_pool      | avoids output_hidden_states memory overhead                  |


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
GRPO advantages computed from group rewards (G=128) after collection.

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


| #   | Deliverable                                                                                                    | File                                   | Status |
| --- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------ |
| 1   | Easy pool: `data/math_easy_pool.jsonl` — L1–L4 (~4100 problems)                                                | `scripts/prepare_easy_pool.py`         | ⬜      |
| 2   | Hard pool: `data/math_level5_hard_pool.jsonl` — Level 5, pass@128=0 filter                                     | `scripts/prepare_math_level5_pool.py`  | ⬜      |
| 3   | `StateEncoder` — encoder, transition; `compute_calibration_loss()`; no decoder/ELBO; z_h=μ_h                         | `src/models/vae_state_encoder.py`      | ✅      |
| 4   | `OutcomeHead` — 2-layer MLP on z_final, raw logit (no sigmoid), Phase 0 only                                                           | `src/models/vae_state_encoder.py`      | ✅      |
| 5   | `ZInjector` — near-zero init (std=0.01)                                                                        | `src/models/vae_state_encoder.py`      | ✅      |
| 6   | `pretrain_vae_online()` — Phase 0: L_trans + L_out + L_calib; 400 steps; pre-shuffled assignment; strict Markov repr                                        | `src/training/grpo_latent.py`          | ✅      |
| 7   | `train_latent()` — Phase 1 custom GRPO loop; L_RL + λ_t·L_trans + λ_calib·L_calib                              | `src/training/grpo_latent.py`          | ✅      |
| 8   | `generate_latent_traces()` — chunked inference engine with z injection; stores chunk_ids + reward only         | `src/training/grpo_latent.py`          | ✅      |
| 9   | Smoke config                                                                                                    | `configs/train_latent_grpo_smoke.yaml` | ✅      |
| 10  | Full config (Phase 0: 400 steps; Phase 1: 200 steps, λ_trans=0.3, λ_calib=0.5)                                 | `configs/train_latent_grpo.yaml`       | ✅      |
| 11  | Latent eval modes in eval_passk.py (`latent_markov`, `latent_markov_pretrained`)                                | `scripts/eval_passk.py`                | ✅      |
| 12  | **Phase 0 training run** → `runs/latent_grpo/phase0_encoder.pt`                                                 | `scripts/train_latent.py`              | ✅      |
| 13  | **NFR6 gate** — UMAP of z_final on Phase 0 checkpoint — **PASS**                                               | `scripts/run_nfr6_gate.py`             | ✅      |
| 14  | **Controlled latent baseline eval** (`latent_grpo_pretrained` pass@128)                                         | `scripts/eval_passk.py`                | ⬜      |
| 15  | **Phase 1 training** — 200 steps on Level 5 hard pool                                                           | `scripts/train_latent.py`              | ⬜      |
| 16  | **Phase 1 eval** — pass@128                                                                                     | `scripts/eval_passk.py`                | ⬜      |
| 17  | **E1 + E3 Markov diagnostics**                                                                                  | `scripts/eval_markov_diagnostics.py`   | ⬜      |


---

## NFR6 Gate

**When to run:** after Phase 0 training completes (`runs/latent_grpo/phase0_encoder.pt`).

**How:** `python scripts/run_nfr6_gate.py --config configs/train_latent_grpo.yaml --n-problems 200 --n-rollouts 2`

Runs the **full trained Phase 0 pipeline** (backbone + trained ZInjector + trained encoder)
to collect z_3 for each trajectory — the same z-injected distribution the encoder was
trained on. Computes UMAP of z_final coloured by correct/incorrect trajectory.
Using the base backbone without z injection would give repr_h from the wrong distribution.

**Gate criteria:** structured manifold with visible outcome correlation (correct and
incorrect trajectories should not be uniformly intermixed; some geometric separation
or outcome-correlated structure should be visible). Does not require hard disjoint
clusters — UMAP topology with outcome-correlated layout suffices.

**If gate fails:** diagnose before Phase 1. Likely causes: Phase 0 too short
(increase `phase0.n_steps`), L_out too weak (increase `λ_out`), class imbalance
overwhelming L_out (check `pos_weight` logic), `λ_trans` dominating too early
(increase `lambda_trans_warmup_steps`), or Markov context leaking raw history
(verify `_run_pipeline_with_grad` uses strict `[prefix_h ‖ chunk]` context).
Do not proceed to Phase 1 on a gate failure.

**Output:** `runs/latent_grpo/plots/latent_structure_umap.png` + `nfr6_summary.json`.

---

## Controlled Latent Baseline (`latent_grpo_pretrained`)

**Definition:** Phase 0 Encoder + ZInjector + pretrained backbone, evaluated on the
Level 5 hard pool with no Phase 1 updates.

**Purpose:** establishes the capability floor under the latent generation regime, parallel
to `baseline_pretrained` (12.5%) for the other arms. Without this, we cannot distinguish
"Phase 1 improved from X to Y" from "the latent regime itself costs performance."

**Gate:** pass@128 ≥ baseline_pretrained pass@128. With near-zero ZInjector init, the
Phase 0 prefix starts near-neutral, so flat-generation capability should be preserved.
If below the pretrained baseline, the Phase 0 ZInjector has learned noisy or adversarial
prefixes — diagnose Phase 0 before proceeding.

**Evaluation:** `scripts/eval_passk.py --generation-mode latent_markov_pretrained`.
Loads backbone from HF model ID and encoder from `phase0.checkpoint_path`.

---

## Markov Diagnostics (required for paper)

Empirical evidence that `z_h` satisfies the Markov property. Without this, the
Markov claim is an assertion, not a result.

**E1 — Transition sufficiency:** held-out transition loss `‖f(z_h) − z_{h+1}‖²` on
unseen trajectories. Near-zero MSE = Markov property holds empirically.
→ `scripts/eval_markov_diagnostics.py`

**E2 — Policy sufficiency:** last-state-only ablation: latent arm pass@128 vs
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
| Controlled baseline (`latent_grpo_pretrained`) | pass@128 ≥ baseline_pretrained pass@128                                                      |
| Phase 1 logs                                   | L_trans non-zero from step 0; L_RL non-zero within first 30 steps; λ_trans=0.3, λ_calib=0.5 confirmed |
| `latent_grpo` pass@128                         | ≥ baseline_grpo pass@128 + 3pp                                                               |
| **E1 + E3 joint**                              | E1 < 0.5 AND E3 Pearson r < −0.1 AND Δσ²(correct vs incorrect) > 0.01                       |
| No NaN blowups                                 | L_RL, L_trans, L_calib all non-zero throughout Phase 1 (R6.5)                                |
| Shared hyperparameters                         | G=128, lr=1e-6, 200 steps, same backbone confirmed in log                                      |


---

**Not in scope for this arm:** uncertainty bonus (β_t × KL in reward). That is
`latent_grpo_uncertainty` — separate implementation session after this arm is complete.