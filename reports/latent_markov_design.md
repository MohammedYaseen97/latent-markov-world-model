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
encoder(mean_pool(final_layer_hidden_states_of_chunk_h))  →  μ_h
z_h  =  μ_h        [always deterministic — training and generation]
```

`z_h` is a 64-dimensional vector used as the Markov state. It is injected into the next
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
                     │  Encoder  (MLP 1536→512→128→64)             [UPDATES]   │
                     │  z_1  =  encoder(repr_1)   (always deterministic)       │
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
  │  L_total = λ_t · L_trans  +  λ_o · L_out                                       │
  │  λ_t peak=3.0 (warmup: λ_t=0 for first 50 steps)  λ_o=5.0                      │
  └────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  GRADIENT ROUTES (Phase 0)                                                  │
  │                                                                             │
  │  L_trans  →  transition ✓   encoder ✓   ZInjector ✓                        │
  │              ( L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone              │
  │                                    → prefix_h → ZInjector )                 │
  │  L_out    →  OutcomeHead ✓   encoder ✓                                     │
  │  All      →  backbone activations  [passthrough; optimizer never stepped]   │
  └─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════════
  PHASE 1 — JOINT RL TRAINING  (backbone UNFROZEN)
══════════════════════════════════════════════════════════════════════════════════

  Level 5 hard pool (~350 problems), 200 steps, batch_size=4, G=128

  ┌──────────────────── ROLLOUT (no_grad) ─────────────────────────────────────┐
  │                                                                            │
  │  Same 3-chunk loop as Phase 0.  G=128 rollouts per problem.                │
  │  Store per rollout: chunk_ids, reward.  (No repr_h, z_h, or log_π_old.)    │
  │  Compute GRPO advantages from group rewards (safety-clipped to ±20).      │
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
  │  │  PHASE 1 LOSS                                                       │   │
  │  │                                                                     │   │
  │  │  L_RL  = −adv · log_π_current               [GRPO; IS = 1]          │   │
  │  │  adv safety-clipped to ±20 (inert in practice; max natural ≈ 11.3)  │   │
  │  └─────────────────────────────────────────────────────────────────────┘   │
  │                                                                            │
  │  ┌─────────────────────────────────────────────────────────────────────┐   │
  │  │  GRADIENT ROUTES (Phase 1)                                          │   │
  │  │                                                                     │   │
  │  │  L_RL  →  lm_head ✓   backbone ✓   ZInjector ✓   encoder ✓         │   │
  │  │           ( L_RL → log_π → backbone → prefix_h → ZInjector          │   │
  │  │                                     → z_h → encoder → repr_h )      │   │
  │  │  Single global grad clip (max_norm=1.0) — matches baseline GRPO.    │   │
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
│  generate: model.generate([q])  →  chunk1_ids (341 tokens)    │
│  repr fwd: model([q ‖ chunk1])  →  repr_1 = mean_pool(hidden) │
│  encode:   z_1 = encoder(repr_1)   (deterministic; z = μ)     │
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
│  encode:   z_2 = encoder(repr_2); project → prefix_2          │
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_2 used for chunk 3
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 3                                                       │
│  generate: model.generate([prefix_2 ‖ chunk2])  →  chunk3_ids │
│            (crutch: raw chunk2 kept for rollout quality)      │
│  repr fwd: model([prefix_2 ‖ chunk3])  → repr_3 = mean_pool   │
│            (strict Markov: only z_2 prefix + new chunk tokens)│
│  encode:   z_3 = encoder(repr_3) = z_final                    │
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

### Encoder and transition — arm 3: deterministic tracker

All components are small MLPs satisfying R4.4 (< 10M params total).
Arm 3 (`latent_grpo`) is a purely deterministic tracker: z = μ everywhere — training,
generation, eval. No sampling at any point.

Arm 4 (`latent_grpo_uncertainty`) will extend this with a logvar head and
uncertainty-driven exploration. It is a stub; nothing below applies to it.

**Encoder** — maps trajectory representation to a latent state:

```
repr_h (1536-dim)
  → Linear(1536, 512) + ReLU
  → Linear(512, 128)
  → mu_head: Linear(128, 64)  →  z_h  (the Markov state)

z_h = encoder(repr_h)   always (training and generation)
```

**Transition model** — predicts next latent from current latent:

```
z_h  (64-dim)
  → Linear(64, 512) + ReLU
  → Linear(512, 64)
  → z_{h+1}_predicted
```

The pure Markov property is `f(z_h) → z_{h+1}`: the current state alone predicts
the next. `repr_h` is deliberately excluded. Including repr_h in the transition directly
would let the model bypass the latent bottleneck and weaken the gradient pressure on z_h
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

**Phase 0 losses:**

```
L_trans   =  ‖ f(z_1) − z_2 ‖²  +  ‖ f(z_2) − z_3 ‖²

L_out     =  BCE_with_logits( outcome_head(z_final), r, pos_weight )
             pos_weight = (1 − pos_rate) / pos_rate per step, clamped ≤ 20.
             Upweights correct trajectories (minority class, ~15–20% of rollouts).

L_phase0  =  λ_trans · L_trans  +  λ_out · L_out
```

Default weights: `λ_trans_peak = 3.0`, `λ_out = 5.0`.
`λ_trans_warmup_steps = 50`: λ_t = 0 for the first 50 steps, then ramps 0.1→3.0
over the remaining 350 steps. Warmup lets L_out establish a signal before L_trans
dominates the gradient budget.

**Approximate gradient budget at Phase 0 peak:**
- L_trans × 3.0 → ~60% — primary: Markov structure
- L_out × 5.0 → ~40% — outcome quality signal

**Phase 0 budget:** `phase0.n_steps` in config. Default: 400 steps. Each step processes
128 diverse sequences spanning ~110 problems (pre-shuffled assignment strategy).

**Gate:** NFR6 (see below) must pass before Phase 1 begins.

---

### Phase 1 — Joint RL Training (pure L_RL)

**Backbone: UNFROZEN.**
**Outcome head: DISCARDED.**
**Encoder + ZInjector: initialised from Phase 0 checkpoint.**

**Data:** Level 5 hard pool (~350 problems) — same as baseline and token-Markov arms.

**Phase 1 loss:**

```
L_RL  =  GRPO policy gradient
      =  -advantage × log_π_current   [IS = 1; same policy as rollout]
      Advantages normalised per group (G=128), safety-clipped to ±20.
      With G=128, expected ≥1 correct rollout per step whenever per-sample success > 0.8%.
```

Phase 1 uses L_RL only — no L_trans, no L_out. Rationale:
- L_trans in Phase 1 competed with L_RL for backbone gradient budget even at λ_t=0.3.
  The Markov property is enforced **architecturally**: each chunk's forward pass sees only
  `[z_h_prefix ‖ chunk_{h+1}]`, never raw token history. The backbone cannot attend to
  raw previous chunks regardless of what the encoder does.
- After Phase 0, the encoder is pre-initialised for Markov consistency. It will only drift
  away from Markov structure if doing so increases reward — which the architecture prevents.
- Removing L_trans gives L_RL undiluted gradient, matching baseline GRPO's optimisation
  structure exactly.

**Phase 1 is structurally identical to baseline GRPO** except: (a) the encoder and
ZInjector are live and receive gradient via the repr_h computation graph, (b) generation
uses z prefix injection between chunks, (c) each step processes a 3-chunk pipeline.

**On-policy GRPO loop (200 steps):** every single training step is:
1. `[no_grad]` collect G=128 fresh rollouts for the current batch of 4 problems → 512 sequences
2. Grade all 512, compute GRPO advantages from normalised group rewards per problem (safety clip ±20)
3. `[with_grad]` re-run full 3-chunk pipeline for all 512 sequences → live log_π, repr_h, z_h
   (micro-batched to fit memory; math equivalent to full-batch update)
4. Compute L_RL, backward, single global grad clip (max_norm=1.0), step optimiser
5. Discard rollouts. Advance to next step.

No replay buffer. No multi-epoch reuse of rollouts. Rollouts are collected fresh at
every step and used for exactly one gradient update — standard on-policy GRPO.

**batch_size = 4 (Phase 1).** At batch_size=4 with G=128: 4×128=512 sequences
per step, 800 problem-encounters over 200 steps — matching the baseline
(TRL batch_size=8, grad_accum=64, equivalent gradient density).
The 512 sequences are processed in micro-batches of `micro_batch_size` to fit
within GPU memory, with gradients accumulated and stepped once per training step.
If OOM: reduce `micro_batch_size` (not `batch_size`) — the math is unchanged.

**Why Phase 1 is expected to work better than token-Markov:**

1. No context resets → per-sample success ≈ same as baseline, not near-zero.
2. ZInjector and z_h arrive from Phase 0 already oriented toward quality — not cold
   random noise. When RL rewards appear, GRPO lands on structured latent space.
3. The reward sparsity problem in baseline GRPO (starts near 0%, builds to 16%) was
   overcome as the model improved — the same path is open to the latent arm with an
   undiluted L_RL signal.

---

## Is z_h "solution space"?

Honest answer: not guaranteed by architecture. It is a bet on training pressure.

The encoder's input (final-layer hidden states) encodes meaning as the backbone has
learned it from pretraining — not raw token statistics, but also not provably
"position in mathematical solution space." Two forces push z_h toward solution-relevant
representations during training:

1. **L_out (Phase 0):** directly rewards z_final for retaining information predictive
   of correct vs incorrect trajectories on easier math problems.
2. **L_RL (Phase 1):** when it fires, rewards z_h values associated with
   successful reasoning paths. The encoder receives gradient from L_RL via the live
   repr_h computation graph.

Whether these forces are sufficient is empirically tested by the NFR6 t-SNE gate and
the E1 diagnostic.

---

## Requirements Satisfaction


| Requirement                                | Satisfied by                                                   |
| ------------------------------------------ | -------------------------------------------------------------- |
| R1.1 — z_h at each step                    | Encoder applied after each chunk; z_h = μ_h (deterministic)   |
| R1.2 — derived from backbone hidden states | mean_pool(final-layer hidden states of chunk h)                |
| R1.3 — fixed-size z_h                      | 64-dim regardless of step count                                |
| R1.5 — z_h conditions policy before head   | soft prefix token injected via inputs_embeds                   |
| R2.1 — transition consistency loss         | L_transition = ‖f(z_h) − z_{h+1}‖² (Phase 0)                   |
| R2.3 — same loss is diagnostic metric      | L_transition on held-out trajectories = E1                     |
| R3.1 — dense auxiliary signal (Phase 0)    | L_transition and L_out non-zero on every Phase 0 step          |
| R3.2 — transition loss satisfies R3.1      | explicitly: L_transition is always computable                  |
| R3.4 — gradients in first ~20 steps        | L_transition + L_out flow from step 0 in Phase 0               |
| R4.1 — encoder: MLP, dim 64–128            | 1536 → 512 → 128 → 64, latent dim 64                           |
| R4.2 — no decoder in arm 3                 | decoder removed; arm 3 is deterministic tracking, not generation |
| R4.3 — deterministic z always              | z_h = μ_h everywhere; reparameterize() removed                  |
| R4.4 — encoder < 10M params                | encoder + transition ≈ 2–3M                                    |
| R4.5 — same latent dim both arms           | 64-dim shared between latent and latent+uncertainty            |
| R5.1 — z_h injected before policy head     | soft prefix prepended to inputs_embeds                         |
| R5.2 — z_h not in token budget             | virtual prefix token, not counted in 1024 generation tokens    |
| R5.3 — same backbone                       | Qwen/Qwen2.5-1.5B-Instruct throughout                          |
| R6.1 — GRPO hyperparameters locked         | inherited from train_baseline_grpo.yaml via extends            |
| R6.2 — reward unchanged                    | binary correctness, same math_reward function                  |
| R6.3 — total loss (Phase 1)                | L_RL only                                                       |
| R6.4 — hyperparameters documented          | λ_t schedule above; adv_clip=20.0, grad_clip=1.0              |
| R6.5 — no NaN blowups when reward=0        | L_transition + L_out keep gradients finite in Phase 0          |
| R7.1–R7.5 — fairness                       | same checkpoint, pool, reward, budget, token limit as all arms |


---

## Key Parameters


| Parameter                        | Value                                      | Notes                                                        |
| -------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Latent dim                       | 64                                         | z_h dimension                                                |
| Chunk size                       | 341 / 341 / 342 tokens                     | = 1024 total, equal split, no carryover                      |
| z injection                      | soft prefix via inputs_embeds              | does not consume token budget                                |
| Encoder architecture             | MLP 1536→512→128→64                        | single output μ; z_h = μ during training and generation      |
| Decoder architecture             | REMOVED in v2                              | not needed (tracking, not generation)                        |
| Transition architecture          | MLP 64→512→64                              | input = z_h only (pure Markov)                               |
| Outcome head                     | MLP 64→64→1 (raw logit)                    | Phase 0 only, discarded after; BCE_with_logits externally    |
| ZInjector init                   | `nn.init.normal_(std=0.01)`                | near-zero; prevents cold-start noise injection               |
| Phase 0 data                     | L1–L4 MATH pool, ~4100 problems            | `data/math_easy_pool.jsonl` (Level 5 excluded)               |
| Phase 0 max_steps                | 400                                        |                                                              |
| Phase 0 problems (total)         | 400 (× 128 rollouts each = 51,200 total)   | pre-shuffled assignment; ~110 problems per step              |
| Phase 0 G (rollouts per problem) | 128                                        | total across all steps; matches eval pass@128 scale          |
| Phase 0 seqs_per_step            | 128                                        | diverse sequences per step (micro-batched internally)        |
| Phase 0 generation               | live, online (z-injected, frozen backbone) | eliminates Phase 0→1 distribution mismatch                   |
| Phase 0 λ_trans_peak             | 3.0 (warmup: λ=0 for 50 steps, then ramp) | ~60% gradient budget at peak; primary Markov training        |
| Phase 0 λ_out                    | 5.0                                        | ~40% gradient budget                                         |
| Phase 1 max_steps                | 200                                        | budget                                                       |
| Phase 1 batch_size               | 4                                          | matches baseline gradient density                            |
| Phase 1 G (rollouts per problem) | 128                                        | locked; matches eval pass@128 scale                          |
| Phase 1 lr                       | 1e-6                                       | locked (all arms)                                            |
| Phase 1 loss                     | L_RL only (no L_trans, no L_out)           | pure GRPO; matches baseline optimisation structure           |
| Phase 1 adv_clip                 | 20.0                                       | numerical safety ceiling; inert in practice (max natural adv ≈ 11.3 at G=128) |
| Phase 1 grad_clip                | 1.0 (global)                               | matches baseline GRPO default                                |
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
`reward` are stored per rollout. No repr_h, z_h, μ, or log_π_old retained.
GRPO advantages computed from group rewards (G=128) after collection.

**Phase 1 training phase:** backbone unfrozen. The full 3-chunk pipeline is re-run
with grad for each stored rollout (identical code path to Phase 0). repr_h and z_h are
LIVE — no detached inputs. Because no weight update occurs between rollout and training
step, IS = 1 exactly; no ε or importance-correction needed. L_RL reaches backbone via
live repr_h in the computation graph.

**OOM handling:** adaptive batch halving on CUDA OOM — `_run_adaptive` helper halves
the batch recursively until it fits or reaches size 1. Applied in both Phase 0 and
Phase 1 rollout generation.

---

## Implementation Deliverables

Ordered by dependency. Each step is a gate for the next.


| #   | Deliverable                                                                                                    | File                                   | Status |
| --- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------ |
| 1   | Easy pool: `data/math_easy_pool.jsonl` — L1–L4 (~4100 problems)                                                | `scripts/prepare_easy_pool.py`         | ✅      |
| 2   | Hard pool: `data/math_level5_hard_pool.jsonl` — Level 5, pass@128=0 filter                                     | `scripts/prepare_math_level5_pool.py`  | ✅      |
| 3   | `VAEStateEncoder` — encoder (1536→512→128→64, μ only), transition; no decoder/ELBO                              | `src/models/vae_state_encoder.py`      | ✅      |
| 4   | `OutcomeHead` — 2-layer MLP on z_final, raw logit (no sigmoid), Phase 0 only                                   | `src/models/vae_state_encoder.py`      | ✅      |
| 5   | `ZInjector` — near-zero init (std=0.01)                                                                        | `src/models/vae_state_encoder.py`      | ✅      |
| 6   | `pretrain_vae_online()` — Phase 0: L_trans + L_out; 400 steps; pre-shuffled assignment; strict Markov repr     | `src/training/grpo_latent.py`          | ✅      |
| 7   | `train_latent()` — Phase 1 custom GRPO loop; pure L_RL; adv_clip=20.0; single grad_clip=1.0                   | `src/training/grpo_latent.py`          | ✅      |
| 8   | `generate_latent_traces()` — chunked inference engine with z injection; stores chunk_ids + reward only         | `src/training/grpo_latent.py`          | ✅      |
| 9   | Smoke config                                                                                                    | `configs/train_latent_grpo_smoke.yaml` | ✅      |
| 10  | Full config (Phase 0: 400 steps; Phase 1: 200 steps, L_RL only)                                                | `configs/train_latent_grpo.yaml`       | ✅      |
| 11  | Latent eval modes in eval_passk.py (`latent_markov`, `latent_markov_pretrained`)                                | `scripts/eval_passk.py`                | ✅      |
| 12  | **Phase 0 training run** → `runs/latent_grpo/phase0_encoder.pt`                                                 | `scripts/train_latent.py`              | ⬜      |
| 13  | **NFR6 gate** — UMAP of z_final on Phase 0 checkpoint                                                          | `scripts/run_nfr6_gate.py`             | ⬜      |
| 14  | **Controlled latent baseline eval** (`latent_grpo_pretrained` pass@128)                                         | `scripts/eval_passk.py`                | ⬜      |
| 15  | **Phase 1 training** — 200 steps on Level 5 hard pool                                                           | `scripts/train_latent.py`              | ⬜      |
| 16  | **Phase 1 eval** — pass@128                                                                                     | `scripts/eval_passk.py`                | ⬜      |
| 17  | **E1 Markov diagnostics**                                                                                       | `scripts/eval_markov_diagnostics.py`   | ⬜      |


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
to `baseline_pretrained` (≈0%) for the other arms. Without this, we cannot distinguish
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

---

## Pass Criteria


| Criterion                                      | Threshold                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| Smoke test                                     | completes end-to-end < 10 min on 4060                                                |
| NFR6 gate                                      | structured UMAP manifold with outcome correlation                                    |
| Controlled baseline (`latent_grpo_pretrained`) | pass@128 ≥ baseline_pretrained pass@128                                              |
| Phase 1 logs                                   | L_RL non-zero within first 30 steps; adv_clip=20.0, grad_clip=1.0 confirmed in log  |
| `latent_grpo` pass@128                         | ≥ baseline_grpo pass@128 + 3pp                                                       |
| **E1**                                         | held-out L_trans < 0.5                                                               |
| No NaN blowups                                 | L_RL non-zero throughout Phase 1 (R6.5)                                              |
| Shared hyperparameters                         | G=128, lr=1e-6, 200 steps, same backbone confirmed in log                            |


---

**Not in scope for this arm:** uncertainty bonus (β_t × KL in reward). That is
`latent_grpo_uncertainty` — separate implementation session after this arm is complete.
