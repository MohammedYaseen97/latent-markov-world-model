# Latent Markov Arm: Design Document (v3)

---

## Changelog (v2 → v3)

| Change | Rationale |
|---|---|
| Phase 0 replaced: L_trans + L_out + crutch → teacher-forced distillation | L_trans was training on garbage repr_2/repr_3 (frozen backbone + strict Markov = uninformative later chunks); distillation gives dense signal from a reliable teacher |
| Crutch removed permanently | Crutch made z optional — backbone could ignore z and read prior chunk text directly. Strict Markov generation forces z to be the sole history carrier, which is the whole mechanism |
| L_trans dropped everywhere | Markov property enforced architecturally by context reset; L_trans was a lossy attempt to compensate for a now-absent problem |
| Transition network removed | Only needed for L_trans; no longer used |
| OutcomeHead removed | Required reward signal that is sparse on hard pool; also removed because L_out is no longer part of Phase 0 |
| repr_h: mean_pool → last token | Last token of a causal LM has attended over all prior tokens in the chunk — it is the model's own attention-based summary. Mean pooling discards positional/salience structure |
| Phase 0 described as pretraining, not RL warmup | The right frame: new components (encoder, ZInjector) need to be initialised before RL, just as pretraining initialises the backbone before fine-tuning |

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
states via a deterministic encoder.** The Markov property is enforced architecturally
by a context reset at each chunk boundary: the backbone sees only `[z_h_prefix | current_chunk]`,
never raw prior tokens. z must carry everything across the boundary — this is what forces
it to encode "position in solution space."

---

## The Core Idea

At each reasoning step `h` (chunk boundary), the backbone has processed some tokens and
produced internal hidden states. We extract the last token's hidden state as the chunk
representation — the causal LM's own attention-based summary of everything it just processed:

```
repr_h = hidden_state_at_last_chunk_token   [1536-dim; last token of chunk h]
z_h    = encoder(repr_h)                    [64-dim; deterministic always]
```

`z_h` is injected into the next chunk's input as a soft prefix token: project `z_h`
(dim 64) → model hidden dim (dim 1536) via a learned linear, prepend as a virtual token
to chunk h+1's `inputs_embeds`.

At generation time, chunk h+1 is generated starting from only `[z_h_prefix]` — no raw
prior chunk tokens. The context reset is complete. z is the only carrier of history.

---

## Pipeline and Gradient Flow (ASCII Block Diagram)

```
══════════════════════════════════════════════════════════════════════════════════
  PHASE 0 — TEACHER-FORCED DISTILLATION PRETRAINING
══════════════════════════════════════════════════════════════════════════════════

  Easy MATH pool (L1–L4), ~400 steps.
  Teacher: original frozen Qwen, full context (sees all prior chunks).
  Student: backbone (low lr ~1e-6) + encoder + ZInjector (lr ~1e-4).

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  TEACHER  (original frozen Qwen, full context)                              │
  │                                                                             │
  │  generate: [sys_prompt ‖ problem]            → teacher_chunk1_ids           │
  │  generate: [sys_prompt ‖ problem ‖ chunk1]   → teacher_chunk2_ids           │
  │  generate: [sys_prompt ‖ problem ‖ chunk1 ‖ chunk2] → teacher_chunk3_ids    │
  └───────────────────────── teacher tokens (fixed targets) ───────────────────┘

                                        │  teacher_chunk{1,2,3}_ids
                                        ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  STUDENT FORWARD PASS — CHUNK 1                                             │
  │  Student backbone processes [sys_prompt ‖ problem ‖ teacher_chunk1]         │
  │  repr_1 = hidden_state at last token of teacher_chunk1 positions  [LIVE]   │
  │  No loss on chunk 1 (teacher and student inputs identical here).            │
  └───────────────────────────────────┬─────────────────────────────────────────┘
                                      │ repr_1 [LIVE]
                           ┌──────────▼──────────────────────────────────────────┐
                           │  Encoder  (MLP 1536→512→128→64)        [UPDATES]    │
                           │  z_1 = encoder(repr_1)                              │
                           └──────────┬──────────────────────────────────────────┘
                                      │ z_1 (64-dim)
                           ┌──────────▼──────────────────────────────────────────┐
                           │  ZInjector  (Linear 64→1536, init std=0.01)  [UPDATES] │
                           │  prefix_1 = W_inj · z_1                             │
                           └──────────┬──────────────────────────────────────────┘
                                      │ prefix_1 (1536-dim)
  ┌───────────────────────────────────▼─────────────────────────────────────────┐
  │  STUDENT FORWARD PASS — CHUNK 2                                             │
  │  Input: [prefix_1 ‖ teacher_chunk2_ids]   ← strict Markov; no chunk1 text  │
  │  Predicts teacher_chunk2 tokens autoregressively.                           │
  │  L_distill_2 = CE(student_log_probs, teacher_chunk2_ids)         [LOSS]    │
  │  repr_2 = hidden_state at last token of teacher_chunk2 positions  [LIVE]   │
  └───────────────────────────────────┬─────────────────────────────────────────┘
                                      │ repr_2 [LIVE]
                           ┌──────────▼──────────────────────────────────────────┐
                           │  Encoder → z_2 (64-dim)                  [UPDATES]  │
                           │  ZInjector → prefix_2 (1536-dim)         [UPDATES]  │
                           └──────────┬──────────────────────────────────────────┘
                                      │ prefix_2 (1536-dim)
  ┌───────────────────────────────────▼─────────────────────────────────────────┐
  │  STUDENT FORWARD PASS — CHUNK 3                                             │
  │  Input: [prefix_2 ‖ teacher_chunk3_ids]   ← strict Markov; no chunk2 text  │
  │  Predicts teacher_chunk3 tokens autoregressively.                           │
  │  L_distill_3 = CE(student_log_probs, teacher_chunk3_ids)         [LOSS]    │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  PHASE 0 LOSS                                                               │
  │                                                                             │
  │  L_phase0 = L_distill_2 + L_distill_3                                       │
  │             (cross-entropy on teacher tokens for chunks 2 and 3)            │
  │                                                                             │
  │  No L_trans. No L_out. No OutcomeHead. No transition network.               │
  └─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  GRADIENT ROUTES (Phase 0)                                                  │
  │                                                                             │
  │  L_distill → backbone (low lr) ✓   ZInjector ✓   encoder ✓                 │
  │              (dense token-level signal every step; no reward needed)        │
  └─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════════
  PHASE 1 — JOINT RL TRAINING  (backbone UNFROZEN)
══════════════════════════════════════════════════════════════════════════════════

  Level 5 hard pool (~350 problems), 200 steps, batch_size=4, G=128
  Strict Markov everywhere: generation and repr forward pass both use [z_prefix | chunk] only.

  ┌──────────────────── ROLLOUT (no_grad) ─────────────────────────────────────┐
  │                                                                            │
  │  3-chunk loop, strict Markov:                                              │
  │    chunk 1: model.generate([sys_prompt ‖ problem])  → chunk1_ids          │
  │    repr_1 = last-token hidden state → z_1 → prefix_1                      │
  │    chunk 2: model.generate([prefix_1])  → chunk2_ids  (no crutch)         │
  │    repr_2 = last-token hidden state → z_2 → prefix_2                      │
  │    chunk 3: model.generate([prefix_2])  → chunk3_ids  (no crutch)         │
  │    grade full output → reward r                                            │
  │                                                                            │
  │  Store per rollout: chunk_ids, reward.                                     │
  │  Compute GRPO advantages from group rewards (G=128), safety-clip ±20.     │
  │                                                                            │
  └────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────── TRAINING STEP (with_grad) ─────────────────────────────┐
  │                                                                            │
  │  Re-run full 3-chunk pipeline for each stored rollout.                     │
  │  Strict Markov: forward pass = [z_prefix ‖ chunk] only (matches rollout).  │
  │  repr_h and z_h LIVE (not stored). IS = 1 exactly.                         │
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
  │  │  L_RL → lm_head ✓  backbone ✓  ZInjector ✓  encoder ✓              │   │
  │  │  Single global grad clip (max_norm=1.0) — matches baseline GRPO.   │   │
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
│  repr fwd: hidden states of [q ‖ chunk1]                      │
│  repr_1  = hidden_state at last token of chunk1 positions     │
│  encode:  z_1 = encoder(repr_1)                              │
│  project: ZInjector(z_1) → prefix_1  (1536-dim virtual token)│
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_1 only — no chunk1 text crosses boundary
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 2  (strict Markov — no crutch)                          │
│  generate: model.generate([prefix_1])  →  chunk2_ids         │
│            z is the only carrier of chunk1 information       │
│  repr fwd: hidden states of [prefix_1 ‖ chunk2]              │
│  repr_2  = hidden_state at last token of chunk2 positions     │
│  encode:  z_2 = encoder(repr_2); project → prefix_2          │
└─────────────────────────┬─────────────────────────────────────┘
                          │ prefix_2 only — no chunk2 text crosses boundary
                          ▼
┌───────────────────────────────────────────────────────────────┐
│ CHUNK 3  (strict Markov — no crutch)                          │
│  generate: model.generate([prefix_2])  →  chunk3_ids         │
│  repr fwd: hidden states of [prefix_2 ‖ chunk3]              │
│  repr_3  = hidden_state at last token of chunk3 positions     │
│  encode:  z_3 = encoder(repr_3) = z_final                    │
│  grade:   full output  →  r = 1.0 if correct, 0.0 otherwise  │
└───────────────────────────────────────────────────────────────┘

TOTAL NEW TOKENS: 341 + 341 + 342 = 1024  ←  matches baseline exactly ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Consistency guarantee (no train/inference mismatch):**
Generation uses `[z_prefix | generate]` (strict Markov). The training forward pass
that computes `log_π_current` and `repr_h` uses `[z_prefix | chunk_h_tokens]` —
the exact same context the tokens were generated under. IS = 1 exactly.
In the old design with the crutch, generation used `[z_prefix | prev_chunk | generate]`
but the repr forward pass used `[z_prefix | chunk]` — different distributions,
making the GRPO gradient technically incorrect. This is now fixed.

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

### Encoder — arm 3: deterministic tracker

All components are small MLPs satisfying R4.4 (< 10M params total).
Arm 3 (`latent_grpo`) is a purely deterministic tracker: z = encoder(repr_h) everywhere.
No sampling. No transition network. No OutcomeHead.

**repr_h extraction — last token (not mean pool):**

```
repr_h = backbone_hidden_states[:, chunk_last_token_idx, :]   [1536-dim]
```

A causal LM's last-token hidden state has already attended over every preceding token
in the chunk via the transformer's attention layers. It is the model's own
attention-weighted summary of the chunk — more informative than a uniform mean over
all positions. No new parameters; one line change from mean_pool.

**Encoder** — maps chunk representation to latent state:

```
repr_h (1536-dim)
  → Linear(1536, 512) + ReLU
  → Linear(512, 128)  + ReLU
  → Linear(128, 64)
  → z_h  (the Markov state; deterministic always)
```

The encoder is shared across all three chunks (same weights applied three times).

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
than a token embedding, making it effectively neutral at init. Phase 0 distillation
training grows the weights from the first step. Same principle as LoRA zero-init for
new adapter parameters inserted into a frozen backbone.

**Removed from v2:** transition network (MLP 64→512→64), OutcomeHead (MLP 64→64→1).
Both were only needed for L_trans and L_out respectively, which are no longer used.

---

## Training: Two Phases

### Why two phases?

The backbone arrives pretrained with no knowledge that a z prefix token exists or
what it means. The encoder and ZInjector arrive randomly initialised. If joint RL
begins immediately, the backbone ignores the z token (has no reason to attend to it),
the encoder's z is noise (randomly initialised), and the ZInjector injects
uninformative embeddings. When sparse RL rewards do appear, GRPO updates land on
a meaningless latent space.

**Phase 0 is pretraining for the new components.** The analogy is exact: a pretrained
backbone is fine-tuned with new LoRA adapters — the adapters must be initialised before
RL. Here, the encoder and ZInjector are the new components, and Phase 0 initialises
them in a way that is task-aligned and interpretable to the backbone.

The key constraint: Phase 0 must use L1–L4 problems (not L5). The L5 hard pool is the
RL environment — capability on it must come from RL exploration (Phase 1), not from
supervised signal in Phase 0. Phase 0 on L1–L4 teaches the backbone to use z for
mathematical reasoning in general; Phase 1 on L5 teaches it to use z for hard problem
exploration specifically.

---

### Phase 0 — Teacher-Forced Distillation Pretraining

**Nature:** self-supervised distillation, not RL. No reward signal needed.

**Teacher:** original frozen Qwen2.5-1.5B-Instruct, full context (sees
`[system_prompt | problem | all_prior_chunks]`). Generates high-quality continuations
for each chunk using complete information. Teacher weights are never updated.

**Student:** backbone (unfrozen, low lr ~1e-6) + encoder (lr ~1e-4) + ZInjector (lr ~1e-4).

**Data:** L1–L4 MATH pool (`data/math_easy_pool.jsonl`), ~4100 problems.
Level 5 excluded for mutual exclusivity with the RL pool.

**Per-problem Phase 0 computation:**

```
Step 1:  Teacher generates chunk1, chunk2, chunk3 with full context.
         (Teacher tokens are fixed reference targets.)

Step 2:  Student forward pass, chunk 1:
         Input = [sys_prompt ‖ problem ‖ teacher_chunk1_tokens]
         repr_1 = last-token hidden state (chunk1 positions)
         z_1 = encoder(repr_1)
         prefix_1 = ZInjector(z_1)
         No loss (chunk 1 context is identical to teacher).

Step 3:  Student forward pass, chunk 2 (strict Markov — no chunk1 text):
         Input = [prefix_1 ‖ teacher_chunk2_tokens]
         L_distill_2 = CE(student_log_probs, teacher_chunk2_tokens)
         repr_2 = last-token hidden state (chunk2 positions)
         z_2 = encoder(repr_2);  prefix_2 = ZInjector(z_2)

Step 4:  Student forward pass, chunk 3 (strict Markov — no chunk2 text):
         Input = [prefix_2 ‖ teacher_chunk3_tokens]
         L_distill_3 = CE(student_log_probs, teacher_chunk3_tokens)

Step 5:  L_phase0 = L_distill_2 + L_distill_3
         Backward, step encoder + ZInjector (lr ~1e-4) and backbone (lr ~1e-6).
```

**What the student learns:**

- **Encoder:** compress chunk h's hidden states into z_h such that ZInjector can
  project it to a prefix that allows the backbone to continue the solution.
  z must encode "where we are after chunk h" — the backbone's own reasoning state.

- **Backbone:** attend to the z prefix and generate a continuation consistent with
  the direction encoded in z. Learns the mapping: "z prefix = direction, continue here."

- **ZInjector:** project 64-dim z into backbone embedding space in a way the backbone
  can read.

**Dense signal:** every training step has signal on every chunk-2 and chunk-3 token
for every problem in the batch. No sparsity. Contrast with old L_trans + L_out on L1–L4
(which had ~24% reward rate and zero gradient on the other 76% of rollouts via L_out).

**Why teacher forcing transfers to Phase 1 strict Markov generation:**

During teacher forcing, the student processes `[prefix_h | teacher_chunk_{h+1}_tokens]`
autoregressively: at position t within chunk h+1, the context is
`[prefix_h | teacher_token_1 | ... | teacher_token_{t-1}]`. This is the same context
structure as Phase 1 strict Markov generation where the student generates its own tokens
one by one from `[prefix_h]`. If the student has learned to predict teacher tokens
given z prefix, the exposure bias (teacher tokens vs student-generated tokens) is mild
— the student generates tokens close to the teacher's, so the context at each step
remains close to the teacher-forced context.

**Phase 0 budget:** ~400 steps on L1–L4 pool. Convergence monitored via distillation
loss on a held-out validation split of L1–L4 problems.

**Gate:** Phase 0 Sanity Check (see below) must pass before Phase 1 begins.

---

### Phase 1 — Joint RL Training (pure L_RL)

**Backbone: UNFROZEN.**
**Encoder + ZInjector: initialised from Phase 0 checkpoint.**
**No transition network. No OutcomeHead. No L_trans.**

**Data:** Level 5 hard pool (~350 problems) — same as baseline and token-Markov arms.

**Phase 1 loss:**

```
L_RL  =  GRPO policy gradient
      =  -advantage × log_π_current   [IS = 1; same policy as rollout]
      Advantages normalised per group (G=128), safety-clipped to ±20.
```

Phase 1 uses L_RL only. The Markov property is enforced architecturally: each chunk
sees only `[z_h_prefix | chunk]` at both generation and training time. There is no
mechanism by which the backbone can access raw prior chunks regardless of what the
encoder does. No L_trans needed.

**Phase 1 is structurally identical to baseline GRPO** except: (a) encoder and
ZInjector are live and receive gradient via the repr_h computation graph, (b) generation
uses z prefix injection and strict Markov context resets between chunks, (c) each step
processes a 3-chunk pipeline. The optimisation signal structure is identical to
baseline — undiluted L_RL, same G, same lr, same grad clip.

**On-policy GRPO loop (200 steps):** every single training step is:

1. `[no_grad]` collect G=128 fresh rollouts for the current batch of 4 problems → 512 sequences
2. Grade all 512, compute GRPO advantages from normalised group rewards per problem (safety clip ±20)
3. `[with_grad]` re-run full 3-chunk pipeline for all 512 sequences → live log_π, repr_h, z_h
   (micro-batched to fit memory; math equivalent to full-batch update)
4. Compute L_RL, backward, single global grad clip (max_norm=1.0), step optimiser
5. Discard rollouts. Advance to next step.

No replay buffer. No multi-epoch reuse of rollouts. No alternating phases.

**batch_size = 4 (Phase 1).** At batch_size=4 with G=128: 4×128=512 sequences
per step, 800 problem-encounters over 200 steps — matching the baseline
(TRL batch_size=8, grad_accum=64, equivalent gradient density).

**Why Phase 1 is expected to work better than the old latent design:**

1. Backbone enters Phase 1 knowing how to use z (Phase 0 distillation on L1–L4
   established this joint vocabulary). Z is not random noise at Phase 1 step 1.
2. No crutch: the backbone cannot ignore z. Every chunk generation depends on z
   as the sole carrier of prior context. When the model learns a better reasoning
   direction, z encodes it, and later chunks benefit.
3. Strict Markov at generation matches the training forward pass — IS = 1 is valid,
   the gradient is correct (unlike the old crutch design where generation and
   repr forward pass used different contexts).

---

## Why the Context Reset Is the Mechanism, Not a Limitation

It might seem that removing prior chunk text from the model's context is harmful —
the model has less information. This inverts the logic.

The context reset is what forces z to be meaningful. If prior chunk text were visible,
the backbone could attend to it directly and ignore the z prefix entirely. z would learn
to be a redundant, weakly-trained supplement. The reset makes z the only available
channel for cross-chunk communication — the backbone has no choice but to learn to
read it, and the encoder has no choice but to make it readable.

This is also why the crutch (old design's carryover of prior chunk text at generation
time) was problematic: it made z optional. The strict Markov property at generation
time is not a consequence of the architecture, it is the architecture.

---

## Is z_h "solution space"?

Honest answer: not guaranteed by architecture. It is a bet on training pressure.

Two forces push z_h toward solution-relevant representations:

1. **L_distill (Phase 0):** z_h must contain whatever is needed for the backbone to
   generate a coherent continuation of a mathematical solution on L1–L4 problems.
   Over 400 steps, this means z_h encodes "what approach we're taking, what we've
   established, where we are in the proof." This is a soft form of "position in
   solution space" learned from teacher demonstrations.

2. **L_RL (Phase 1):** when reward fires, the gradient reaches the encoder via the
   live repr_h computation graph. z_h values associated with reasoning paths that
   eventually lead to correct answers are reinforced. z_h values associated with
   dead ends are discouraged. Over 200 steps on L5, this shapes z_h toward
   high-reward regions of solution space.

Whether these forces are sufficient is empirically tested by the Phase 0 Sanity Check,
the controlled latent baseline eval, and the Phase 1 pass@128 result.

---

## Requirements Satisfaction

| Requirement                                | Satisfied by                                                   |
| ------------------------------------------ | -------------------------------------------------------------- |
| R1.1 — z_h at each step                    | Encoder applied after each chunk; z_h = encoder(repr_h)        |
| R1.2 — derived from backbone hidden states | last-token hidden state of chunk h (causal LM attention summary) |
| R1.3 — fixed-size z_h                      | 64-dim regardless of step count                                |
| R1.5 — z_h conditions policy before head   | soft prefix token injected via inputs_embeds                   |
| R2.3 — diagnostic metric (transition proxy)| z variance and consistency check (E1, see Diagnostics)         |
| R3.1 — dense auxiliary signal (Phase 0)    | CE distillation: non-zero on every token of every chunk 2+3    |
| R3.4 — gradients in first ~20 steps        | distillation loss flows from step 0 in Phase 0                 |
| R4.1 — encoder: MLP, dim 64–128            | 1536 → 512 → 128 → 64, latent dim 64                           |
| R4.2 — no decoder in arm 3                 | no decoder; arm 3 is deterministic tracking, not generation    |
| R4.3 — deterministic z always              | z_h = encoder(repr_h) everywhere; no sampling                  |
| R4.4 — encoder < 10M params                | encoder ~1.4M + ZInjector ~0.1M ≈ 1.5M total                  |
| R4.5 — same latent dim both arms           | 64-dim shared between latent and latent+uncertainty            |
| R5.1 — z_h injected before policy head     | soft prefix prepended to inputs_embeds                         |
| R5.2 — z_h not in token budget             | virtual prefix token, not counted in 1024 generation tokens    |
| R5.3 — same backbone                       | Qwen/Qwen2.5-1.5B-Instruct throughout                          |
| R6.1 — GRPO hyperparameters locked         | inherited from train_baseline_grpo.yaml via extends            |
| R6.2 — reward unchanged                    | binary correctness, same math_reward function                  |
| R6.3 — total loss (Phase 1)                | L_RL only                                                       |
| R6.4 — hyperparameters documented          | learning rates and schedules above; adv_clip=20.0, grad_clip=1.0 |
| R6.5 — no NaN blowups when reward=0        | CE distillation keeps gradients finite in Phase 0              |
| R7.1–R7.5 — fairness                       | same checkpoint, pool, reward, budget, token limit as all arms |

---

## Key Parameters

| Parameter                        | Value                                      | Notes                                                        |
| -------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| Latent dim                       | 64                                         | z_h dimension                                                |
| Chunk size                       | 341 / 341 / 342 tokens                     | = 1024 total, equal split, no carryover                      |
| z injection                      | soft prefix via inputs_embeds              | does not consume token budget                                |
| Encoder architecture             | MLP 1536→512→128→64                        | deterministic; z_h = encoder(repr_h) always                  |
| repr_h extraction                | last-token hidden state (chunk positions)  | forward hook on last layer at last chunk token index         |
| Transition architecture          | REMOVED (v3)                               | Markov enforced architecturally; L_trans dropped             |
| Outcome head                     | REMOVED (v3)                               | no reward signal in Phase 0; distillation replaces           |
| ZInjector init                   | `nn.init.normal_(std=0.01)`                | near-zero; prevents cold-start noise injection               |
| Phase 0 type                     | teacher-forced self-supervised distillation| teacher = frozen original Qwen, full context                 |
| Phase 0 data                     | L1–L4 MATH pool, ~4100 problems            | `data/math_easy_pool.jsonl` (Level 5 excluded)               |
| Phase 0 max_steps                | ~400                                       | convergence monitored via held-out distillation loss         |
| Phase 0 backbone lr              | ~1e-6                                      | low; backbone learns to use z without losing L5 capability   |
| Phase 0 encoder/ZInjector lr     | ~1e-4                                      | higher; learning from scratch                                |
| Phase 0 loss                     | CE(student_log_probs, teacher_chunk_{2,3}) | dense; no reward needed                                      |
| Phase 0 generation               | teacher tokens as input (teacher forcing)  | no garbage later-chunk generation; no crutch                 |
| Phase 0 crutch                   | NONE                                       | strict Markov in student: [z_prefix ‖ teacher_chunk_h]       |
| Phase 1 max_steps                | 200                                        | budget; matches baseline                                     |
| Phase 1 batch_size               | 4                                          | matches baseline gradient density                            |
| Phase 1 G (rollouts per problem) | 128                                        | locked; matches eval pass@128 scale                          |
| Phase 1 lr (backbone)            | 1e-6                                       | locked (all arms)                                            |
| Phase 1 lr (encoder/ZInjector)   | 1e-4                                       | continued from Phase 0                                       |
| Phase 1 loss                     | L_RL only (no L_trans, no L_out)           | pure GRPO; matches baseline optimisation structure           |
| Phase 1 generation               | strict Markov: [z_prefix | generate]       | no crutch; z is the only cross-chunk information carrier     |
| Phase 1 adv_clip                 | 20.0                                       | numerical safety ceiling; inert in practice                  |
| Phase 1 grad_clip                | 1.0 (global)                               | matches baseline GRPO default                                |
| Benchmark                        | MATH Level 5 hard pool, ~350 problems      | pass@128=0 filter on pretrained model                        |
| Primary metric                   | pass@128                                   | 8× cheaper than pass@1024; more problems compensates         |
| Backbone                         | Qwen/Qwen2.5-1.5B-Instruct                 |                                                              |

---

## Engineering Notes

The same TRL incompatibility that forced a custom loop for the token-Markov arm applies
here. Multi-chunk generation with `z_h` conditioning between chunks is incompatible with
TRL's single-sequence-per-rollout assumption. The latent arm uses a custom training loop
in `src/training/grpo_latent.py`.

**Phase 0 loop** (`pretrain_distill()`): for each batch of L1–L4 problems, run the
teacher (frozen original Qwen, full context) to collect `teacher_chunk_{1,2,3}_ids`.
Then run the student forward pass chunk-by-chunk: chunk 1 with full context (no loss),
chunks 2 and 3 with `[z_prefix | teacher_chunk_h]` input and CE loss against teacher
tokens. Standard `loss.backward()` + Adam step on encoder, ZInjector, and backbone
(separate param groups with different lrs).

**Phase 1 rollout phase:** `@torch.no_grad()` for generation. Only `chunk_ids` and
`reward` stored per rollout. No repr_h, z_h, or log_π_old retained.
GRPO advantages computed from group rewards (G=128) after collection.

**Phase 1 training phase:** backbone unfrozen. Full 3-chunk pipeline re-run with grad
for each stored rollout. repr_h and z_h are LIVE — no detached inputs. IS = 1 exactly.
L_RL reaches backbone and encoder via live repr_h in the computation graph.

**No train/inference mismatch (v3):** generation uses `[z_prefix | generate]`;
training forward pass uses `[z_prefix | chunk_h_tokens]`. Same context structure.
The old crutch design had generation using `[z_prefix | prev_chunk | generate]` but
repr forward pass using `[z_prefix | chunk]` — different distributions, making the
GRPO log-prob ratio technically incorrect. This is resolved in v3.

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
| 3   | `VAEStateEncoder` — encoder (1536→512→128→64) + ZInjector; remove transition net + OutcomeHead; last-token repr | `src/models/vae_state_encoder.py`      | ⬜      |
| 4   | `pretrain_distill()` — Phase 0: teacher-forcing CE distillation on L1–L4; chunk-by-chunk loop                  | `src/training/grpo_latent.py`          | ⬜      |
| 5   | `generate_latent_traces()` — strict Markov: `[z_prefix | generate]`; no crutch; stores chunk_ids + reward only | `src/training/grpo_latent.py`          | ⬜      |
| 6   | `train_latent()` — Phase 1 custom GRPO loop; pure L_RL; adv_clip=20.0; single grad_clip=1.0                   | `src/training/grpo_latent.py`          | ⬜      |
| 7   | Smoke config (Phase 0: 2 steps distill; Phase 1: 2 steps RL)                                                   | `configs/train_latent_grpo_smoke.yaml` | ⬜      |
| 8   | Full config (Phase 0: ~400 steps distill; Phase 1: 200 steps L_RL)                                             | `configs/train_latent_grpo.yaml`       | ⬜      |
| 9   | Latent eval modes in eval_passk.py (`latent_markov`, `latent_markov_pretrained`)                                | `scripts/eval_passk.py`                | ✅      |
| 10  | **Phase 0 training run** → `runs/latent_grpo/phase0_encoder.pt`                                                 | `scripts/train_latent.py --phase 0`    | ⬜      |
| 11  | **Phase 0 Sanity Check** — distillation loss convergence + z variance check                                     | `scripts/run_phase0_sanity.py`         | ⬜      |
| 12  | **Controlled latent baseline eval** (`latent_grpo_pretrained` pass@128 on L5 ≈ 0%)                              | `scripts/eval_passk.py`                | ⬜      |
| 13  | **Phase 1 training** — 200 steps on Level 5 hard pool                                                           | `scripts/train_latent.py --phase 1`    | ⬜      |
| 14  | **Phase 1 eval** — pass@128                                                                                     | `scripts/eval_passk.py`                | ⬜      |
| 15  | **E1 z-consistency diagnostics**                                                                                | `scripts/eval_markov_diagnostics.py`   | ⬜      |

Note: old deliverables 4 (OutcomeHead), 6 (pretrain_vae_online with L_trans + L_out),
and 13 (NFR6 UMAP gate) are superseded. NFR6 is replaced by Phase 0 Sanity Check (#11).

---

## Phase 0 Sanity Check (replaces NFR6 gate)

**When to run:** after Phase 0 completes (`runs/latent_grpo/phase0_encoder.pt`).

**Check 1 — Distillation loss convergence:**
Compute held-out CE distillation loss on 50 L1–L4 problems not seen during training.
Threshold: held-out loss < 2.0 nats (approximately: model is doing better than uniform
on teacher continuations). A flat or increasing curve means Phase 0 has not converged.

**Check 2 — z variance:**
Run the Phase 0 pipeline (student + encoder) on 100 L1–L4 problems, collect z_final.
Compute mean per-dimension std across problems. Threshold: mean std > 0.1.
Near-zero variance = encoder has collapsed to near-constant z (degenerate).

**Check 3 — Controlled L5 capability (same starting line):**
Eval `latent_grpo_pretrained` on L5 hard pool after Phase 0.
Expected: pass@128 ≈ 0% (same as baseline pretrained). Phase 0 on L1–L4 should not
give any advantage on L5. If pass@128 > 5% after Phase 0, Phase 0 has leaked
L5-specific supervision — investigate data contamination or excessively high backbone lr.

**If any check fails:** do not proceed to Phase 1.
- Distillation not converged → increase `phase0.n_steps` or backbone lr
- z collapsed → increase encoder lr; check ZInjector init; check repr_h extraction
- L5 capability inflated → reduce backbone lr; verify L1–L4 / L5 pool mutual exclusivity

**Output:** `runs/latent_grpo/phase0_sanity.json` with loss values and z variance stats.

---

## Controlled Latent Baseline (`latent_grpo_pretrained`)

**Definition:** Phase 0 Encoder + ZInjector + pretrained backbone, evaluated on the
Level 5 hard pool with no Phase 1 updates.

**Purpose:** establishes that the latent arm and baseline arm start from the same line.
Both should show pass@128 ≈ 0% on L5 before Phase 1. Without this check, we cannot
distinguish "Phase 1 improved the model" from "Phase 0 already improved the model on L5."

**Gate:** pass@128 ≤ ~1% (close to baseline pretrained on L5). Strict: latent arm should
not have a head start. With near-zero ZInjector init and Phase 0 restricted to L1–L4,
this should hold naturally.

**Evaluation:** `scripts/eval_passk.py --generation-mode latent_markov_pretrained`.
Loads backbone from HF model ID and encoder from `phase0.checkpoint_path`.

---

## Markov Diagnostics

**E1 — z consistency (replaces transition loss diagnostic):**

The transition network is removed. The Markov property is structural, not learned via
L_trans. The diagnostic instead verifies z is non-degenerate and consistent:

- For 100 held-out problems, run 4 rollouts each.
- Compute cosine similarity of z_final across rollouts for the same problem.
  Expected: high within-problem similarity (same problem → similar z trajectory).
- Compute cosine similarity of z_final across different problems.
  Expected: lower between-problem similarity (z discriminates problem state).
- Threshold: within-problem sim > 0.8, between-problem sim < 0.6.

High within / lower between confirms z is encoding problem-specific state, not noise.

→ `scripts/eval_markov_diagnostics.py`

**E2 — Policy sufficiency:**
Latent arm pass@128 vs baseline — covered by the core ablation table.

---

## Pass Criteria

| Criterion                                      | Threshold                                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------------------------ |
| Smoke test                                     | completes end-to-end < 10 min on 4060                                                |
| Phase 0 distillation convergence               | held-out CE loss < 2.0 nats                                                          |
| Phase 0 z variance                             | mean per-dim std > 0.1 across 100 problems                                           |
| Controlled baseline (`latent_grpo_pretrained`) | pass@128 ≤ ~1% on L5 (same starting line as baseline pretrained)                    |
| Phase 1 logs                                   | L_RL non-zero within first 30 steps; adv_clip=20.0, grad_clip=1.0 confirmed in log  |
| `latent_grpo` pass@128                         | ≥ baseline_grpo pass@128 + 3pp                                                       |
| E1 z-consistency                               | within-problem sim > 0.8, between-problem sim < 0.6                                  |
| No NaN blowups                                 | L_RL non-zero throughout Phase 1 (R6.5)                                              |
| Shared hyperparameters                         | G=128, lr=1e-6 (backbone), 200 Phase 1 steps, same backbone confirmed in log         |

---

**Not in scope for this arm:** uncertainty bonus (β_t × KL in reward). That is
`latent_grpo_uncertainty` — separate implementation session after this arm is complete.
