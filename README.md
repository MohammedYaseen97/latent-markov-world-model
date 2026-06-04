# Latent Markov World Models for RL Post-Training

*This project is the empirical foundation of a broader research agenda on diffusion-based
epistemic state models for RL post-training. The full proposal is in
[`lossfunk/lossfunk_proposal_draft.md`](lossfunk/lossfunk_proposal_draft.md).*

---

RL post-training (GRPO, PPO, RLVR) improves reasoning models by running them on hard
problems and rewarding correct answers. It works — up to a point. Yue et al. (NeurIPS 2025)
documented a capability ceiling: RLVR gets better at finding paths the base model could
already take, but stops expanding what the model can actually solve. Yuan et al. (2026)
traced the root cause to the state representation. The policy's state is the full token
history — an ever-growing sequence of everything generated so far. This is not a Markov
state. It gives the policy no compact model of *where it is* in the solution process, only
*what it has said*. Yuan et al. showed that introducing an explicit Markov state breaks
the ceiling — but only by constructing it by hand from symbolic task structure. For open-ended
reasoning like mathematics, there is no symbolic structure to construct from. The Markov
state has to be *discovered*.

**This project tests whether a learned latent state, extracted from the model's own hidden
representations, can serve as that Markov state and break the ceiling on hard mathematics.**

---

## The Experiment

The core claim is that if you compress what the model has generated so far into a compact
latent vector `z`, and condition each new reasoning chunk on that vector instead of the
full token history, the policy gains a navigable representation of solution-space position.
To test this claim in isolation from confounds, the project runs four arms — identical
in every dimension except the state representation:

| Arm | State fed to the policy |
|-----|------------------------|
| `baseline_grpo` | Full token history (standard RLVR — the documented ceiling) |
| `token_markov_grpo` | Textual Markov state carryover in token space (Markovian Thinker, ICLR 2026) |
| `latent_grpo` | Learned compact latent `z_h`, injected as a soft prefix at each chunk boundary |
| `latent_grpo_uncertainty` | Same as above, plus a KL-based intrinsic exploration bonus |

The token-Markov arm is the critical ablation: it has the same chunked generation
structure and explicit Markov state as the latent arm, but entirely in text space with
no encoder. Whatever the token-Markov arm improves over baseline is attributable to
chunking and approximate Markov structure. Whatever the latent arm improves beyond that
is attributable to the latent representation alone.

All arms train on Qwen2.5-1.5B-Instruct, evaluated on a pool of MATH Level 5 problems
for which the pretrained model scores `pass@128 = 0` — problems the base model
genuinely cannot solve. **Primary metric: `pass@128`.**

---

## How the Latent Arm Works

Each reasoning trace is broken into chunks. After generating chunk `h`, the model's
final-layer hidden state at the last token is passed through a small learned encoder
that compresses it into a 64-dimensional latent vector `z_h`. At the start of chunk
`h+1`, that vector is projected into a single soft prefix embedding and prepended to
the model's input. The model never sees prior chunk tokens — only the prefix.

```
chunk h:    backbone([sys_prompt | problem | chunk_1 | ... | chunk_h])
              └── last_token(final hidden layer)
                     └── Encoder (1536 → 512 → 128 → 64)
                            └── z_h  [64-dim]
                                   └── ZInjector (64 → 1536)
                                          └── prefix_embed  [1 × 1536]

chunk h+1:  backbone([prefix_embed | generate...])   ← no prior tokens visible
```

This enforces the Markov property structurally at the code level — not as a regularizer
or a loss term, but as a hard constraint on what the policy can condition on. The vector
`z_h` must carry all information needed to continue reasoning; if it doesn't, the policy
fails. This is the hypothesis under test.

Training has two phases:

**Phase 0 — Encoder pretraining:** The frozen teacher (Qwen, full context access)
generates step-by-step reasoning traces on easy problems (MATH L1–L4). The student
trains to predict teacher tokens conditioned on `z_prefix` using cross-entropy loss,
with LoRA adapters (r=16) on the backbone. This initialises the encoder and injector
before any RL signal is available. Without this step, z is random noise and Phase 1
GRPO produces zero-gradient steps from the start.

**Phase 1 — On-policy GRPO:** The fully-initialised model trains on the MATH Level 5
hard pool with strict Markov generation — no prior tokens, only `z_prefix`. No crutch,
no shortcuts.

---

## Current Results

**Baseline GRPO** (Arm 1) is complete: `pass@128 = 16.20%` on the Level 5 hard pool
after 200 training steps. This is the ceiling to beat.

**Latent arm** (Arm 3) is in active development. Phase 0 is the current blocker.

The encoder and LoRA adapters converge cleanly under teacher forcing — cross-entropy
reaches ~1.5 nats on held-out L1–L4 problems. But when the student is switched to
autoregressive generation conditioned on `z_prefix` alone, reward rate collapses to
~1.25% on L1–L4 problems that the teacher solves at ~60%. This is the exposure-bias
gap: the model learns to predict the next token given teacher-provided context, but
has not learned to generate coherently from its own context. CE loss and autoregressive
reward measure fundamentally different capabilities.

### What we've ruled out

Systematic debugging over five training runs has eliminated the following explanations:

| Hypothesis tested | Outcome |
|------------------|---------|
| Gradient dilution — CE averaged over all ~300 tokens per chunk dilutes the z-conditioning gradient, so the encoder never receives a strong training signal | Introduced `z_anchor_tokens`: supervise only the first 32 positions where `z_prefix` dominates. CE still converges (~3 nats at anchor), generation still fails. |
| Under-training — 400 steps insufficient at reduced supervision density | Doubled to 800 steps with `lr_lora=3e-4`. Mid-training probes peaked at 5% then regressed to 0% by step 300. Under-training is not the bottleneck. |
| Over-adaptation — LoRA over-specialises on anchor positions, destroying answer-format generation at late positions | Cosine LR warmdown implemented (3e-4 → 3e-5 over last 200 steps). Under evaluation. |
| Teacher data quality — hard-cut 341-token chunks produce mid-sentence boundaries, poisoning training data | Redesigned teacher to generate one step at a time until model EOS; confirmed clean boundaries at ~60% teacher accuracy. |
| Probe bias — generation probes always tested the same hard problems, masking real progress | Changed to random resample from held-out pool at each probe step. Probe variance confirmed not the issue. |
| Results inflated by crutch — earlier architecture gave backbone access to prior chunk tokens alongside z, making z optional | Removed in v3 (now strictly `z_prefix` only). Chunk-1 crutch rate = 0% confirmed across all recent runs. |

### The key architectural question

The predecessor architecture (v2) — which had `L_trans` (a Markov consistency loss
on latent trajectories), `L_out` (a calibration loss), and a generation-time crutch
— reached `pass@128 = 8.42%`. That result is contaminated: the crutch meant z was
optional, so the backbone likely solved most problems without using z at all. The UMAP
latent geometry and pass@128 numbers from v2 cannot be trusted as evidence of z
encoding anything.

When the crutch was removed and `L_trans` replaced with teacher-forced CE distillation
(v3 onward), performance dropped to ≤1.25% student reward. The open question is whether
`L_trans` was enforcing a structural prior on latent trajectories that CE distillation
alone cannot replicate — not because the capacity is wrong, but because CE provides no
direct signal about temporal consistency of z across chunks.

### What's next

The assumption list is being re-evaluated in dependency order
(see `reports/phase0_scratch_notes.md` for the full A0–A8 list and complete run history).
Primary candidates:

1. Reinstate a lightweight transition objective alongside CE — does latent temporal
   consistency need to be directly supervised, or does CE on z-conditioned generations
   imply it?
2. Increase `latent_dim` beyond 64 — the 1536→64 compression is 24×. If z cannot
   represent solution-space position at 64 dims, no amount of training fixes it.
3. Close the exposure-bias gap explicitly — scheduled sampling (gradually mixing
   student-generated tokens into teacher-forced positions) or on-policy CE mixing.

---

## Scaling to Larger Models

A separate architectural issue has emerged that motivates scaling the entire experiment
to 7B or 14B models. The step-by-step teacher prompt instructs the model to "complete
one reasoning step, then stop" — using the model's own EOS as the chunk boundary.
This is the right mechanism: it avoids hard token-count cuts that split mid-LaTeX and
corrupt training data. But it requires the model to self-regulate step length, which
demands reliable instruction-following capacity. The 1.5B model cannot do this
consistently — it runs into the token limit mid-calculation, reverting to the same
hard-boundary problem we designed around.

This is not a motivation to scale for performance. It is a confound: the latent Markov
hypothesis cannot be cleanly tested if the teacher data is corrupted by boundary
artifacts that a larger model would not produce. The planned shift to 7B/14B applies
uniformly to the baseline and all arms, preserving the ablation's comparative structure.

---

## Repository Layout

```
configs/
  base_model.yaml                  Shared model and hardware config
  train_baseline_grpo.yaml         Baseline arm
  train_token_markov_grpo.yaml     Token-Markov arm
  train_latent_grpo.yaml           Latent arm (Phase 0 + Phase 1)
  train_latent_grpo_smoke.yaml     Smoke test (2 steps, end-to-end validation)

data/
  math_easy_pool.jsonl             MATH L1–L4, Phase 0 pretraining (built by script)
  math_level5_hard_pool.jsonl      MATH L5, pass@128=0 filter (built by script)

reports/
  latent_markov_design.md          Full architecture specification
  phase0_scratch_notes.md          Debugging log: assumptions, run history, all numbers
  ablation_core.md                 Official results table

scripts/
  train_latent.py                  Phase 0 pretraining + Phase 1 GRPO
  train_baseline.py                Baseline GRPO
  train_token_markov.py            Token-Markov GRPO
  eval_passk.py                    pass@k evaluation (all arms)
  run_phase0_sanity.py             Phase 0 gate: CE, z-variance, student vs teacher
  eval_markov_diagnostics.py       z-consistency diagnostics
  check_assumptions.py             Teacher trace inspection, assumption verification
  run_ablation_table.py            Aggregate artifacts/ into results table
  prepare_easy_pool.py             Build Phase 0 pool (MATH L1–L4)
  prepare_math_level5_pool.py      Build eval pool (MATH L5, filtered)

src/
  models/
    vae_state_encoder.py           LatentStateEncoder + ZInjector
    token_markov_state.py          Token-space Markov state
  training/
    grpo_baseline.py               Baseline GRPO loop
    grpo_token_markov.py           Token-Markov GRPO loop
    grpo_latent.py                 Latent arm: Phase 0 distillation + Phase 1 GRPO
    reward_bonus.py                KL intrinsic bonus (stub, arm 4)
  utils/
    config_loader.py               YAML extends + deep merge
    seeding.py                     Deterministic seeding

artifacts/                         Per-run directories (created at runtime)
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Build data pools (one-time, ~2h)
python scripts/prepare_easy_pool.py --levels 1 2 3 4
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output data/math_level5_hard_pool.jsonl

# Smoke test — verify pipeline end-to-end, minimal compute
python scripts/train_latent.py --config configs/train_latent_grpo_smoke.yaml --phase 0

# Inspect teacher traces (what training data looks like)
python scripts/check_assumptions.py --n 20 --step_tokens 341 --max_chunks 10

# Phase 0 pretraining
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 0

# Phase 0 sanity gate (must pass before Phase 1)
python scripts/run_phase0_sanity.py \
    --config configs/train_latent_grpo.yaml \
    --checkpoint artifacts/latent_grpo/<run_id>/phase0

# Phase 1 GRPO
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 1

# Evaluation
python scripts/eval_passk.py \
    --generation-mode latent_markov \
    --train-config configs/train_latent_grpo.yaml \
    --checkpoint artifacts/latent_grpo/<run_id>/phase1/final \
    --arm-name latent_grpo
```

---

## Related Work

| | |
|--|--|
| **Yue et al. NeurIPS 2025** | Documents the RLVR capability ceiling empirically. The problem this experiment is designed to break. |
| **Yuan et al. 2026 — Markov States** | Shows explicit Markov states break the ceiling on symbolic tasks. Motivates learning the state instead of constructing it. |
| **Markovian Thinker (ICLR 2026)** | Textual Markov state carryover via RL. Arm 2 is the direct ablation: same Markov structure, no latent. |
| **Coconut (Meta, 2024)** | Continuous thought tokens for reasoning. Inference-time adaptation only; no world model; no RL exploration signal. |
| **Dreamer / DIAMOND / JEPA** | Latent world models for physical environments. Same architectural DNA — but they model external state; this models the agent's epistemic state over the problem. |
