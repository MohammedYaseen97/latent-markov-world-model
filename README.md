# Latent Markov World Models for RL Post-Training

*This project is the empirical foundation of a broader research agenda on diffusion-based
epistemic state models for RL post-training. The full proposal is in
[`lossfunk/lossfunk_proposal_draft.md`](lossfunk/lossfunk_proposal_draft.md).*

---

RL post-training (GRPO, PPO, RLVR) improves reasoning models — up to a point. Yue et al.
(NeurIPS 2025) documented a capability ceiling: RLVR gets better at finding paths the base
model could already take, but stops expanding what the model can actually solve. Yuan et al.
(2026) traced this to the state representation. The policy's state is the full token history —
an ever-growing sequence with no compact model of *where it is* in the solution. They showed
that introducing an explicit Markov state breaks the ceiling, but only by constructing it by
hand from symbolic task structure. For open-ended reasoning like mathematics, there is no
symbolic structure to extract. The Markov state has to be *discovered*.

**This project tests whether a learned latent state, extracted from the model's own hidden
representations, can serve as that Markov state and break the ceiling on hard mathematics.**

---

## Experimental Design

Four arms, identical in every dimension except the state representation:

| Arm | State fed to the policy |
|-----|------------------------|
| `baseline_grpo` | Full token history (standard RLVR — the documented ceiling) |
| `token_markov_grpo` | Textual Markov state carryover in token space (Markovian Thinker, ICLR 2026) |
| `latent_grpo` | Learned compact latent `z_h`, injected as a soft prefix at each chunk boundary |
| `latent_grpo_uncertainty` | Same as above, plus a KL-based intrinsic exploration bonus (runs after Arm 3 Phase 1) |

The token-Markov arm is the critical ablation: it tests whether Markov structure in text space
alone (no encoder, no latent) accounts for any gains. Whatever the latent arm improves beyond
that is attributable to the latent representation alone.

All arms train on Qwen2.5-1.5B-Instruct, evaluated on a held-out pool of MATH Level 5 problems
for which the base model scores `pass@128 = 0`. **Primary metric: `pass@128`.**

---

## Architecture

Each reasoning trace is broken into chunks. After generating chunk `h`, the model's final-layer
hidden state at the last token is passed through a small learned encoder into a compact vector
`z_h`. At the start of chunk `h+1`, that vector is projected into a single soft prefix embedding
prepended to the model's input. The model never sees prior chunk tokens — only the prefix.

```
chunk h:   backbone([sys_prompt | problem | chunk_1 | ... | chunk_h])
             └── last_token(final hidden layer)
                    └── Encoder (1536 → 512 → 128 → z_dim)
                           └── z_h
                                  └── ZInjector (z_dim → 1536)
                                         └── prefix_embed  [1 × 1536]

chunk h+1: backbone([prefix_embed | generate...])   ← no prior tokens visible
```

This enforces the Markov property structurally — not as a regulariser or a loss term, but as a
hard constraint on what the policy can condition on. `z_h` must carry all information needed to
continue reasoning; if it doesn't, the policy fails. This is the hypothesis under test.

Training has two phases:

**Phase 0 — Encoder pretraining:** A stronger teacher (Qwen3-8B, full context access) generates
semantic step-by-step traces on easy problems (MATH L1–L4). The student (Qwen2.5-1.5B) trains to
predict teacher tokens conditioned on `z_prefix` using cross-entropy loss, with LoRA adapters on
the backbone.

**Phase 1 — On-policy GRPO:** The fully-initialised model trains on the hard Level 5 pool under
strict Markov generation — no prior tokens, only `z_prefix`.

---

## Progress

### Arm 1 — Baseline GRPO (complete)

`pass@128 = 16.19%` on the Level 5 hard pool after 200 GRPO steps. This is the ceiling to beat.

### Arm 2 — Token-Markov GRPO (implemented, not yet run)

Code complete. Serves as the critical ablation once latent Phase 1 results are available.

### Arm 3 — Latent arm (Phase 0 in progress — at a specific diagnosed decision point)

Phase 0 is the current blocker. Here is the full history.

**v1 — VAE-ELBO.** The original architecture used a full VAE with a KL term. KL weight = 1.0
was too aggressive: the encoder learned to output `σ² ≈ 1.0` everywhere (posterior collapse to
prior), making z near-constant. `L_trans = 0.015` — trivially low because z barely changed,
not because transitions were clean. Decision: the decoder is unnecessary machinery. Drop the KL,
switch to a deterministic `z = μ`, and fix the benchmark resolution (40 problems / pass@1024 →
350 problems / pass@128).

**v2 — L_trans + L_out + crutch.** Redesigned Phase 0 with Markov consistency loss (`L_trans`),
an outcome head (`L_out`), and CE distillation. Result: `pass@128 = 8.42%`. On closer
inspection this number is contaminated: the generation-time context was `[z_prefix | prev_chunk |
generate]` — the backbone could attend to prior chunk tokens and ignore `z` entirely. z was
optional. 8.42% is also below the baseline (16.19%), which is expected: without the crutch
removed, the model had no reason to use z, so it essentially ran as a weaker version of
standard generation. Decision: remove the crutch. Force strict Markov: `[z_prefix | generate]`
only. If z doesn't encode enough, the model fails. This is what makes z mandatory and the
hypothesis testable. The drop in reward from crutch to strict Markov is expected — it is the
cost of making the constraint real.

**v3 — Strict Markov, CE distillation (current).** Crutch removed. `L_trans` and `L_out`
dropped — without the crutch, later-chunk representations are uninformative at the start of
training; CE distillation from a reliable teacher provides dense signal regardless. Three Phase 0
runs:

| Run | Config | Student reward | Diagnosis |
|-----|--------|---------------|-----------|
| 1 | Full CE, 400 steps | ~1.25% | **Gradient dilution** — CE averaged over 341 tokens/chunk; z-conditioning gradient diluted ~341× by the LM-from-context signal at later positions |
| 2 | z_anchor=32, 400 steps | 0% | **Under-training** — at z_anchor density, 400 steps × 32 supervised positions is too sparse; probe pool was also fixed (always same problems, masking progress) |
| 3 | z_anchor=32, 800 steps, lr=3e-4, probe resample | peaked 5%, regressed to 0% | **Over-adaptation** — LoRA over-specialises on z_anchor positions, disrupting `\boxed{}` generation at tail positions; LR warmdown implemented |

After three runs failing the sanity gate (`student_reward_rate ≥ 10%`, `coverage_ratio ≥ 30%`
of teacher), the pattern was clear: CE loss converges (~1.5 nats) but autoregressive generation
from `z_prefix` alone fails. The gap between teacher-forced training and autoregressive inference
is not closing. Rather than run again on unvalidated foundations, the decision was made to stop
and verify the ground assumptions of the data pipeline first.

**Assumption verification.** A standalone harness (`scripts/check_assumptions.py`) was built
to test every ground assumption before running again. 90 traces from Qwen3-8B across L1–L5.

| ID | Assumption | Status |
|----|-----------|--------|
| A0 | Teacher generates multiple distinct reasoning steps | ✅ Confirmed |
| A1 | Chunk boundaries fall at natural semantic step-ends | ✅ Confirmed |
| A9 | Teacher accuracy sufficient for data generation | ✅ Confirmed — 97% / 83% (L1–L3 / L4) |
| A-prompt | Teacher reliably follows "one step then stop" | ✅ Confirmed |
| A-cont | Continuation message allows natural termination | ✅ Confirmed |
| A-grader | `answers_equivalent` correctly scores all LaTeX variants | ✅ Confirmed |
| A-filter | Clean traces are identifiable and separable from corrupted ones | ✅ Confirmed |
| A2 | Last-token hidden state `repr_h` encodes position in solution space | ⏳ Pending |
| A3 | Encoder produces diverse z vectors (no collapse) | ⏳ Pending |
| A4 | Student generates coherent step N+1 from `z_prefix` alone | ⏳ Pending — this IS Phase 0 |
| A5 | z_dim sufficient for multi-step reasoning | ⚠️ Partial — `z_transition_gap = -0.027`; 64-dim too compressed |
| A3-pool | Last-token repr is sufficient; mean-pool not needed | ⏳ Pending (low priority) |
| A7 | One virtual prefix token is sufficient | ⏳ Pending (low priority) |
| A8 | Linear Z-injector is expressive enough | ⏳ Pending (low priority) |
| A-enc-ag | Same encoder weights generalise across all chunk transitions | ⏳ Pending (low priority) |
| A-lora | LoRA r=16 has capacity for z-conditioning | ⚠️ Partial — near-zero probe rate across all runs |
| A6 | Exposure bias is manageable; teacher forcing transfers to generation | 🔍 Primary open blocker — Phase 0 |
| A-markov | z is a sufficient statistic; context reset loses nothing | 🔍 Core claim, unproven — Phase 1 |
| A-reset | Context reset is lossless (not destructive) | 🔍 Design choice, unproven — Phase 1 |
| A-repr2 | `repr_h` under teacher forcing matches `repr_h` at inference | 🔍 Known exposure-bias risk — Phase 1 |
| A-mlen | First-order Markov length is sufficient | ⏳ Pending (deferred to Phase 1) |

The confirmed assumptions (A0–A-filter) rule out the data pipeline as the failure source. The
⏳ architecture assumptions (A2, A3, A4, A3-pool, A7, A8, A-enc-ag, A-mlen) are the Phase 0
and Phase 1 test plan — they are pending by design, not by oversight. The ⚠️ and 🔍 rows
(A5, A-lora, A6) point directly to the next run: `latent_dim: 128`, `lora_r: 32`,
`z_tail_tokens: 32`. The teacher was also upgraded to Qwen3-8B for verification, confirming
cleaner semantic traces from a stronger model — justifying it for data generation going forward.

---

## Repository Layout

```
configs/
  base_model.yaml                  Shared model and hardware config
  train_baseline_grpo.yaml         Baseline arm
  train_token_markov_grpo.yaml     Token-Markov arm
  train_latent_grpo.yaml           Latent arm (Phase 0 + Phase 1)

data/
  math_easy_pool.jsonl             MATH L1–L4, Phase 0 pretraining
  math_level5_hard_pool.jsonl      MATH L5, pass@128=0 filter

reports/
  latent_markov_design.md          Full architecture specification
  latent_arm_worklog.md            Full run history, assumptions, root-cause map, next run plan
  ablation_core.md                 Official results table

scripts/
  train_latent.py                  Phase 0 pretraining + Phase 1 GRPO
  train_baseline.py                Baseline GRPO
  check_assumptions.py             Teacher trace verification harness
  eval_passk.py                    pass@k evaluation (all arms)
  run_phase0_sanity.py             Phase 0 gate: CE, z-variance, student vs teacher reward
  prepare_easy_pool.py             Build Phase 0 pool (MATH L1–L4)
  prepare_math_level5_pool.py      Build eval pool (MATH L5, filtered)

src/
  models/
    vae_state_encoder.py           LatentStateEncoder + ZInjector
    token_markov_state.py          Token-space Markov state
  training/
    grpo_baseline.py               Baseline GRPO loop + reward grader
    grpo_token_markov.py           Token-Markov GRPO loop
    grpo_latent.py                 Latent arm: Phase 0 distillation + Phase 1 GRPO
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Build data pools (one-time)
python scripts/prepare_easy_pool.py --levels 1 2 3 4
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output data/math_level5_hard_pool.jsonl

# Verify teacher trace quality before training
python scripts/check_assumptions.py --step traces --n 30 --max-step-tok 512 \
    --model Qwen/Qwen3-8B --levels 1 2 3 4 --out reports/traces_check.json

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
| **Markovian Thinker (ICLR 2026)** | Textual Markov state carryover via RL. Arm 2 is the direct ablation: same structure, no latent. |
| **Coconut (Meta, 2024)** | Continuous thought tokens for reasoning. Inference-time only; no world model; no RL exploration signal. |
| **Dreamer / DIAMOND / JEPA** | Latent world models for physical environments. Same architectural DNA — but they model external state; this models the agent's epistemic state over the problem. |
