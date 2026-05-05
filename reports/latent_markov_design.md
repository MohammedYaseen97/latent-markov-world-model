# Latent Markov Arm: Design Document

Requirements reference: `reports/latent_markov_requirements.md`
Companion scratch doc (design rationale and conversation record): `reports/SCRATCH_latent_design_rough.md`

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

The transition model learns: `z_h + action_h → z_{h+1}`. If this loss is small,
`z_h` plus what was done in chunk h is sufficient to predict the next state without
token history. That is the Markov property, empirically verified.

---

## The Mechanism (full rollout)

```
ROLLOUT FOR ONE PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

query q = [system prompt] + [problem]

┌──────────────────────────────────────────────────────────────┐
│ CHUNK 1                                                      │
│  input:  q                      (no z prefix — no prior state)│
│  generate: up to 341 tokens                                  │
│  extract: mean_pool(hidden_states_of_341_tokens) = repr_1    │
│  encode: encoder(repr_1) → (μ_1, σ_1²) → sample z_1        │
│  project: z_1 (64-dim) → linear → (1536-dim) = prefix_1     │
└─────────────────────────┬────────────────────────────────────┘
                          │ prefix_1 prepended to chunk 2 input
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ CHUNK 2                                                      │
│  input:  [prefix_1] + q          (z_1 as virtual token)     │
│  generate: up to 341 tokens                                  │
│  extract: mean_pool(hidden_states_of_341_tokens) = repr_2    │
│  encode: encoder(repr_2) → (μ_2, σ_2²) → sample z_2        │
│  project: z_2 → prefix_2                                    │
└─────────────────────────┬────────────────────────────────────┘
                          │ prefix_2 prepended to chunk 3 input
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ CHUNK 3                                                      │
│  input:  [prefix_2] + q          (z_2 as virtual token)     │
│  generate: up to 342 tokens                                  │
│  extract: repr_3 → z_3 = z_final                            │
└─────────────────────────┬────────────────────────────────────┘
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

**Transition model** — maps current latent + action embedding to next latent:
```
concat(z_h, repr_h)  (64 + 1536 = 1600-dim)
  → Linear(1600, 512) + ReLU
  → Linear(512, 64)
  → z_{h+1}_predicted
```

The action embedding for the transition model is `repr_h` — the mean-pooled hidden
states of chunk h tokens. This is already computed for the encoder, so reusing it
costs nothing extra. `repr_h` captures what the model DID in chunk h (the action).

**Outcome head** — for Phase 0 pretraining only, attached to z_final:
```
z_final (64-dim)
  → Linear(64, 64) + ReLU
  → Linear(64, 1) + Sigmoid
  → P(correct)
```
Discarded before Phase 1 begins. Used only to shape the encoder during pretraining.

**Policy conditioning** — z injection:
```
z_h (64-dim) → Linear(64, 1536) → prefix_embedding (1536-dim)
```
Prepended to chunk h+1's input via `inputs_embeds`. Not a real token — no vocabulary
entry. Does not consume the 1024-token generation budget (R5.2).

The encoder is shared across all three chunks (same weights applied three times).
Training Phase 0 updates the encoder weights based on chunk 3's z_final (via L_outcome);
those same weights produce z_1 and z_2 from chunks 1 and 2 (R4.3, R4.5).

---

## Training: Two Phases

The training is split into Phase 0 (VAE pretraining) and Phase 1 (joint RL).

### Why two phases?

The VAE arrives at joint RL training as a cold-start module. If RL training begins
immediately with a randomly initialised encoder, `z_h` is noise and the policy
conditions on noise. When sparse RL rewards do appear (≈ 0.02% per sample), GRPO
updates land on a meaningless latent space.

The fix mirrors how the LLM backbone is prepared for RL: the backbone is pretrained on
general data before RL begins. The VAE should be pretrained on related-but-easier data
before RL begins. Phase 0 is VAE pretraining.

---

### Phase 0 — VAE Pretraining

**Backbone: FROZEN throughout Phase 0.**

**Data: Easier pretraining pool (TBD — see calibration step).**

The pool must satisfy: the pretrained instruct model solves ≥20% of problems at pass@8, so
`L_outcome` receives a mix of correct and incorrect trajectory labels. Calibration
(`scripts/calibrate_pool.py --pool-path <candidate>`) must confirm this before
any Phase 0 compute is spent.

Candidate: `hendrycks/competition_math` Level 1–3 (standard MATH competition problems at
lower difficulty). These provide the same mathematical domain and hidden-state distribution
expected during Phase 1 RL, without contaminating the hard MATH-Beyond eval pool.

Why a separate pool at all:
- Not in the eval pool → no contamination of the RL target environment
- Easier → rollouts contain both correct and incorrect trajectories → rich labels for L_outcome
- Same reasoning domain → hidden states have the right distributional character for the encoder

**Generation (one-time, before training loop, no grad):**

```
1. Run frozen backbone on Phase 0 pool × G=8 rollouts (no_grad)
2. For each rollout, extract final-layer hidden states per chunk, mean-pool
   → save (repr_1, repr_2, repr_3, reward_label) per trajectory to disk
3. Backbone is NOT in the VAE training computational graph
```

Using static, pre-saved hidden states (not live backbone inference) is deliberate.
If the backbone were in the graph with `requires_grad_(False)` on its parameters,
L_outcome on z_final would propagate backward through the frozen backbone's operations
to reach z_1 and z_2 activations — treating the backbone as a fixed differentiable
function. That is mathematically possible but requires storing activations for a
28-layer 1.5B transformer per step. Static hidden states avoid this: gradient
terminates at the constant `repr_h` tensors. Phase 0 backward passes update only
the small VAE modules.

**Phase 0 losses:**

```
L_ELBO       =  reconstruction_loss(decoder(z_h), repr_h)
             +  KL(N(μ_h, σ_h²) ∥ N(0, I))
             (summed over h = 1, 2, 3)

L_transition =  ∥ f(z_h, repr_h) − z_{h+1} ∥²
             (summed over h = 1, 2; h=1→2 and h=2→3)

L_outcome    =  BCE(outcome_head(z_final), r)
             (z_final = z_3 from the last chunk)

L_phase0     =  λ_elbo × L_ELBO  +  λ_trans × L_transition  +  λ_out × L_outcome
```

Default weights: λ_elbo = 1.0, λ_trans = 1.0, λ_out = 1.0. Specified in config.

**Why L_outcome is not redundant with Phase 1's L_RL:**

L_outcome and L_RL both connect `z_h` to outcome quality, but they operate in
different phases, on different problem sets, under different reward density conditions:

- L_outcome fires on every trajectory during Phase 0 on the easier pretraining pool
  where the model actually succeeds sometimes. On easier problems with ~20%+ per-sample
  success and G=8 rollouts, most batches contain both positive and negative examples.
  Gradient flows from L_outcome through the encoder to shape what information survives
  the 1536→64 bottleneck.
- L_RL fires ≈ 0 times per step during Phase 1 on the hard 40 problems (0.02%
  per-sample success). The backbone is also not in the graph during Phase 0, so
  L_outcome cannot update backbone parameters.
- L_RL primarily updates the backbone/policy head. L_outcome specifically targets
  the encoder. They update different components.

If L_RL were dense, L_outcome would be redundant — L_RL would eventually shape the
encoder indirectly. It is non-redundant specifically because L_RL is sparse.

**What L_outcome achieves:**

The 1536→64 bottleneck has no inherent reason to retain solution-quality-relevant
information — it might compress surface syntax statistics or token frequency patterns
instead. L_outcome provides explicit pressure: of all the information in repr_3,
retain what predicts whether the trajectory succeeded. L_transition provides
cross-chunk consistency: z_2 must be consistent with z_3 (via the h=2→3 transition
loss), so quality-relevant structure in z_3 propagates structurally backward to z_2
and z_1 via consistency constraints.

**Phase 0 budget:** Specified in config (`vae_pretrain_steps` or epochs).
Default candidate: 2–3 epochs over the Phase 0 pool with G=8 precomputed
rollouts. Cost is negligible — no backbone backward pass.

**Gate:** NFR6 applies after Phase 0. Run the encoder on the saved rollouts, produce
t-SNE/UMAP of z_final coloured by correct/incorrect. If no separation is visible,
Phase 0 failed to orient the encoder toward quality. Diagnose before proceeding to
Phase 1 (try: larger latent dim, longer Phase 0, higher λ_out).

---

### Phase 1 — Joint RL Training (200 steps)

**Backbone: UNFROZEN.**
**Outcome head: DISCARDED.**
**VAE: initialised from Phase 0 weights.**

**Data: hard 40-problem MATH-B-I pool** — same as baseline and token-Markov arms.

**Phase 1 losses:**

```
L_RL         =  GRPO policy gradient
             =  -advantage × log_prob  (standard GRPO, same as baseline arm)
             Sparse: fires only when at least 1 of G=8 rollouts has r=1.
             Updates: backbone, lm_head, z injection linear.

L_transition =  ∥ f(z_h, repr_h) − z_{h+1} ∥²
             Always non-zero. Keeps Markov property active during RL training.
             Also the diagnostic metric for E1 (transition sufficiency).
             Updates: transition model, encoder, backbone (indirectly via repr_h).

L_VAE        =  reconstruction_loss + KL(N(μ_h, σ_h²) ∥ N(0, I))
             Always non-zero. Prevents encoder/decoder from drifting during RL.
             Updates: encoder, decoder.

L_phase1     =  L_RL  +  λ_t × L_transition  +  L_VAE
```

**λ_t schedule (transition loss weight):**

```
λ_t = 1.0                               for steps 0  – 100  (linear decay)
    → linear decay to 0.1               at step 100
    = 0.1                               for steps 100 – 200  (held constant)
```

High early → L_transition provides dense gradient flow while L_RL ≈ 0.
Low late → L_RL dominates once sparse rewards start appearing.
Floor at 0.1 → L_transition remains active as Markov regulariser and diagnostic.

**Why Phase 1 is expected to work better than token-Markov:**

1. No context resets → per-sample success ≈ 0.02% (same as baseline), not 0.015%.
   `P(≥1 correct | G=8) ≈ 0.16%` vs. 0.12% for token-Markov. Small but non-trivial.

2. L_transition and L_VAE provide dense gradient flow even when L_RL = 0. The backbone
   and encoder are learning from every step, not only from the rare reward events.

3. `z_h` arrives from Phase 0 already oriented toward quality — not random noise. When
   rare RL rewards do appear, GRPO updates land on a structured latent space.

---

## Is z_h "solution space"?

Honest answer: not guaranteed by architecture. It is a bet on training pressure.

The encoder's input (final-layer hidden states) encodes meaning as the backbone has
learned it from pretraining — not raw token statistics, but also not provably
"position in mathematical solution space." Three forces push z_h toward solution-relevant
representations during training:

1. L_outcome (Phase 0): directly rewards z_final for retaining information predictive
   of correct vs incorrect trajectories on easier math problems.
2. L_transition (both phases): forces z_h to retain information predictive of z_{h+1},
   which rewards trajectory-structure information over surface syntax.
3. L_RL (Phase 1, sparse): when it fires, rewards z_h values associated with successful
   reasoning paths.

Whether these forces are sufficient is empirically tested by the diagnostics (Section 5
of requirements doc): E1 (transition loss on held-out trajectories), E3 (variance
correlation with difficulty), and the NFR6 t-SNE gate.

---

## Requirements Satisfaction

| Requirement | Satisfied by |
|-------------|-------------|
| R1.1 — z_h at each step | VAE encoder applied after each chunk |
| R1.2 — derived from backbone hidden states | mean_pool(final-layer hidden states of chunk h) |
| R1.3 — fixed-size z_h | 64-dim regardless of step count |
| R1.4 — uncertainty estimate σ_h² | VAE posterior variance, used by uncertainty arm |
| R1.5 — z_h conditions policy before head | soft prefix token injected via inputs_embeds |
| R2.1 — transition consistency loss | L_transition = ∥f(z_h, repr_h) − z_{h+1}∥² |
| R2.2 — Markov objective joint with RL | L_transition active throughout Phase 1 |
| R2.3 — same loss is diagnostic metric | L_transition on held-out trajectories = E1 |
| R3.1 — dense auxiliary signal | L_transition and L_VAE non-zero on every step |
| R3.2 — transition loss satisfies R3.1 | explicitly: L_transition is always computable |
| R3.4 — gradients in first ~20 steps | L_transition + L_VAE flow from step 0 |
| R4.1 — encoder: MLP, dim 64–128 | 1536 → 512 → 64, latent dim 64 |
| R4.2 — decoder: training only | decoder not used at eval/inference |
| R4.3 — reparameterization trick | z = μ + ε·σ, ε ~ N(0, I) |
| R4.4 — VAE < 10M params | encoder + decoder + transition ≈ 4–5M |
| R4.5 — same latent dim both arms | 64-dim shared between latent and latent+uncertainty |
| R5.1 — z_h injected before policy head | soft prefix prepended to inputs_embeds |
| R5.2 — z_h not in token budget | virtual prefix token, not counted in 1024 generation tokens |
| R5.3 — same backbone | Qwen/Qwen2.5-1.5B-Instruct throughout |
| R6.1 — GRPO hyperparameters locked | inherited from train_baseline_grpo.yaml via extends |
| R6.2 — reward unchanged | binary correctness, same math_reward function |
| R6.3 — total loss | L_RL + λ_t × L_transition (+β_t × KL in reward for uncertainty arm) |
| R6.4 — hyperparameters documented | λ_t schedule above; β_t in uncertainty arm design |
| R6.5 — no NaN blowups when reward=0 | L_transition + L_VAE keep gradients finite; gate in smoke |
| R7.1–R7.5 — fairness | same checkpoint, pool, reward, budget, token limit as all arms |

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Latent dim | 64 | R4.1 range: 64–128 |
| Chunk size | 341 / 341 / 342 tokens | = 1024 total, equal split, no carryover |
| z injection | soft prefix via inputs_embeds | Q1 decision |
| Encoder architecture | MLP 1536→512→64 (×2 outputs) | μ and log_σ² |
| Decoder architecture | MLP 64→512→1536 | training only |
| Transition architecture | MLP 1600→512→64 | input = concat(z_h, repr_h) |
| Outcome head | MLP 64→64→1 + sigmoid | Phase 0 only, discarded after |
| Phase 0 data | Easier pretraining pool (TBD) | Must achieve pass@8 ≥ 20% with instruct model |
| Phase 0 backbone | frozen (static hidden states saved to disk) | |
| Phase 1 λ_t | 1.0 → 0.1 linear over steps 0-100, held at 0.1 | |
| Phase 0 loss weights | λ_elbo=1.0, λ_trans=1.0, λ_out=1.0 | config knobs |
| VAE LR | same as backbone (1e-6) | separate LR optional if needed |

---

## Engineering Notes

The same TRL incompatibility that forced a custom loop for the token-Markov arm applies
here. Multi-chunk generation with `z_h` conditioning between chunks is incompatible with
TRL's single-sequence-per-rollout assumption. The latent arm uses a custom training loop
in `src/training/grpo_latent.py`, following the same structural pattern as
`src/training/grpo_token_markov.py`.

Rollout phase: `@torch.no_grad()` for all generation. Hidden states extracted during
rollout (also `no_grad`) are re-used as frozen `repr_h` tensors in the learning pass.

Learning phase: backbone unfrozen (Phase 1). Re-run forward pass with grad to get
log-probs for L_RL. `repr_h` tensors from rollout are used as static inputs to the
encoder, transition model, and decoder — not recomputed via backbone forward pass
in the learning phase.

This avoids double-counting the backbone computation and keeps the learning pass clean:
L_RL gradient flows through the token generation path; L_transition and L_VAE gradients
flow through the VAE components using static `repr_h` as inputs.

---

## Implementation Deliverables

Ordered by dependency. Each step is a gate for the next.

| # | Deliverable | File | Gate |
|---|-------------|------|------|
| 1 | Phase 0 dataset: build easier pretraining pool (e.g. MATH Level 1–3); run calibration (`scripts/calibrate_pool.py`) to confirm pass@8 ≥ 20% | `scripts/prepare_easy_pool.py` | Pool must pass calibration before Phase 0 generation; if pass@8 < 5% try easier difficulty slice |
| 2 | `VAEStateEncoder` — encoder, decoder, transition model | `src/models/vae_state_encoder.py` | Core module; all training depends on it |
| 3 | `OutcomeHead` — 2-layer MLP on z_final, used in Phase 0 only | `src/models/vae_state_encoder.py` | Phase 0 training |
| 4 | Hidden state extraction + Phase 0 rollout generation script | `scripts/generate_phase0_rollouts.py` | Must produce `repr_h` + label files before Phase 0 training |
| 5 | `pretrain_vae()` — Phase 0 training loop (L_ELBO + L_transition + L_outcome) | `src/training/grpo_latent.py` | Phase 0 must complete before Phase 1 |
| 6 | Smoke config for Phase 0 (2 steps, 4 problems, 4060) | `configs/train_latent_grpo_smoke.yaml` | Gate: smoke must pass before any A100 use |
| 7 | **NFR6 gate**: t-SNE/UMAP of z_final coloured by correct/incorrect | `scripts/check_latent_structure.py` | Must show separation before Phase 1 |
| 8 | `train_latent()` — Phase 1 custom GRPO loop (L_RL + λ_t × L_transition + L_VAE) | `src/training/grpo_latent.py` | Depends on Phase 0 checkpoint |
| 9 | Latent eval mode in pass@k script | `scripts/eval_passk.py` | Evaluation after Phase 1 training |
| 10 | Full config for Phase 1 (200 steps, A100) | `configs/train_latent_grpo.yaml` | Full run |
| 11 | A100 run: Phase 0 → NFR6 gate → Phase 1 → eval | — | Final result |

**Smoke test requirement (NFR3):** Step 6 must run end-to-end in < 10 minutes on the
local 4060 (QLoRA, 2 steps, 4 problems). This is the gate before any A100 spend.

**NFR6 gate (Step 7):** Run Phase 0 on the 4060 (~30–60 min), produce t-SNE of z_final.
If correct and incorrect trajectories do not separate in latent space, do not proceed
to Phase 1. Diagnose: larger latent dim, longer Phase 0, higher λ_out.

**Not in scope for this arm:** uncertainty bonus (β_t × KL in reward). That is
`latent_grpo_uncertainty` — separate implementation session after this arm is complete.
