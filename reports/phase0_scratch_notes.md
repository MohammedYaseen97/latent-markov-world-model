# Phase 0 — Unofficial Scratch Notes

**Not canonical.** Working notes from debugging Phase 0 distillation (latent Markov arm).
For the official design, see `reports/latent_markov_design.md`.

Last updated: 2026-06-04

**Current status: Phase 0 training giving bad results → revisiting assumptions.**

---

## Published results (from lossfunk_proposal_draft.pdf)

These are from **v2** of the architecture — the last version with `L_trans` (transition loss)
and `L_out` (calibration loss), frozen backbone, and no LoRA. The Phase 1 failure at 8.42%
triggered the v2→v3 redesign (teacher-forced CE distillation), which began this conversation.

| Component | Status | Key finding |
|-----------|--------|-------------|
| Baseline GRPO (200 steps) | complete | **pass@128 = 16.20%** |
| Latent Markov encoder (Phase 0, 400 steps) | complete | Structured latent space with outcome-correlated geometry (UMAP: 1.4× enrichment) |
| Phase 1 joint RL | **complete — fail** | **pass@128 = 8.42%** (below Phase 0 floor; encoder instability diagnosed) |
| Token-Markov arm | pending | — |

### Design timeline

| Design | Phase 0 approach | Phase 1 result |
|--------|-----------------|----------------|
| **v2** — PDF numbers | `L_trans` + `L_out`, frozen backbone, 400 steps | **pass@128 = 8.42%** — below baseline ceiling (16.20%), encoder instability |
| **v3** — teacher-forced CE | Replaced L_trans/L_out with CE distillation, frozen backbone | pass@128 = all 0s; Phase 1 reward 0.4%, 13/20 dead steps |
| **v4** — +LoRA | Added LoRA (r=16), CE distillation | Student reward **1.25%**, gradient dilution |
| **v5** — +z_anchor, +steps | z_anchor=32, 800 steps, lr=3e-4 | Probe peaked at 5%, regressed to 0%; best checkpoint student 1.25% |
| **v6** — +warmdown, step-by-step | LR warmdown, variable-length teacher chunks | **Not yet run** |

### ⚠️ Important caveat on v2 numbers

v2 had a **crutch** in Phase 1: the student could see prior chunk tokens directly
in addition to z_prefix. This made z **optional** — the backbone could ignore z
and read prior chunk text. The v2→v3 changelog explicitly states: "Crutch made z
optional — backbone could ignore z and read prior chunk text directly."

Both v2 results are contaminated:

- **UMAP 1.4× enrichment** — the structured latent space may be an artifact of the
  backbone doing the reasoning while z was along for the ride
- **pass@128 = 8.42%** — not pure latent-arm performance; includes crutch-assisted
  generations where z may have contributed nothing

This means the 8.42% is an **upper bound** on what the crutch-assisted pipeline
could do, not a lower bound on what z alone can do. The true z-only performance
of v2's latent space is unknown, and likely closer to the 1.25% seen in later
crutch-free versions.

---

## Assumptions being tested (chronological, in order of architectural dependency)

### A0 — Step boundaries are clean
The teacher's generation steps end at complete reasoning units, not mid-sentence.
**Status:** Partially confirmed. With the new "one step at a time" prompt and
`step_tokens=341`, most steps end naturally. At `step_tokens=200`, steps were cut
mid-LaTeX. Bumped to 341 — re-verify.

### A1 — z_prefix encodes "position in solution space"
The 64-dim latent vector z, extracted from the last token of a chunk, captures
enough information about where we are in the solution to guide the next step.
**Status:** Not tested directly yet. Negative z-transition gap (-0.027) suggests
it may not. The plan is to increase `latent_dim` if this fails.

### A2 — CE loss on z-anchor positions is sufficient signal
Supervising only the first 32 tokens (z_anchor_tokens) of each chunk provides
enough gradient to train the encoder + LoRA to produce useful z prefixes.
**Status: Failing.** CE passes (~1.5 nats) but student reward is 1.25% — 8× below
the 10% threshold. The model learns teacher-forced next-token prediction but cannot
generalize to autoregressive generation.

### A3 — Teacher-forced CE transfers to autoregressive generation
The model trained with teacher-chunk context at positions 0–31 will produce the
same text when generating autoregressively from z_prefix alone.
**Status: Strongly contradicted.** This is the exposure-bias problem. CE passes
(1.5 nats) but generation fails (1.25% reward). The gap between train-time context
(teacher tokens) and inference-time context (own generated tokens) breaks the pipeline.

### A4 — 64-dim z is expressive enough for L1–L4 math
The 1536→64 compression (24× reduction) preserves enough information to represent
where a solution is in "solution space."
**Status: Questionable.** The negative z-transition gap (-0.027) suggests z is not
tracking solution progress. Increasing latent_dim to 128 or 256 is the planned fix.

### A5 — LoRA rank 16 is sufficient
Backbone adapters at r=16 with 32 z_anchor positions provide enough capacity to
learn the z→text mapping.
**Status: Unclear.** CE passes (1.5 nats) → LoRA can memorize under teacher forcing.
But generation fails → capacity may be sufficient, the bottleneck is elsewhere.

### A6 — min_new_tokens=170 doesn't poison generation
Forcing chunks 2–3 to generate at least 170 tokens, even when the model is lost,
doesn't accumulate errors that kill chunk 3 quality.
**Status: Likely hurting.** The forced 170 tokens on broken z_2 produces garbage
that feeds into z_3 → garbage. Plan: reduce to 32 or remove.

### A7 — Teacher quality is sufficient for distillation
The 1.5B Qwen teacher, with "one step at a time" prompting, produces good-enough
traces for the student to learn from.
**Status: Verified.** Teacher reward rate ~60% on L1–L4 with 2.1 avg steps/problem.
Not perfect but better than pre-redesign (which had hallucinated steps mid-chunk).
The 40% incorrect traces will be learned by the student — Phase 1 RL is expected
to correct this, but it means the Phase 0 student starts as a noisy copy.

### A8 — Variable-length chunks train correctly
The new distill_loss supports 1..max_chunks per problem. Samples with 1 chunk
(problem solved in one step) return 0 loss as expected.
**Status: Not yet run.** Implementation complete but not tested in a training run.

---

## Official numbers — all Phase 0 runs

### Phase 0 v1 (no LoRA, 400 steps, full CE on all 341 tokens)

**Training loss (teacher forcing):**
```
Step 10:   14.64
Step 20:   2.34
Step 30:   1.89
Step 40:   1.89
Step 50:   1.80
Step 60:   1.87
Step 70:   1.88
Step 80:   1.89
...
Step 400:  ~1.60
```
CE converged. z variance healthy (0.64). All architectural gates passed.

**Sanity results:**
```
mean_ce_loss:          1.61   ✅ (threshold 2.0)
mean_z_std:            0.64   ✅ (threshold 0.1)
z_transition_gap:      0.089  ✅ (threshold 0.02)
easy_reward_rate:      7.5%   ✅ (threshold 5%)
overall_pass:          true
```
Passed — but `easy_reward_rate` was entirely chunk-1 crutch (answer visible in
the prompt chunk; z pipeline was never exercised).

**Phase 1 training (post v1):**
```
Steps with non-zero gradient:  7 / 20  (35%)
Steps with zero gradient:     13 / 20  (65%)
Peak reward_rate:            ~0.004   (0.4%)
```
Model produced 0 correct answers for 13/20 steps. Phase 1 dead on arrival.

**Phase 1 eval (L5 hard pool):** All zeros. No correct generations across any rollout.

---

### Phase 0 v2 — LoRA, 400 steps, lr_lora=1e-4, full CE

**Sanity results:**
```
mean_ce_loss:          1.6   ✅
mean_z_std:            0.64  ✅
teacher_reward_rate:   49%      (on 80 rollouts)
student_reward_rate:   1.25%  ❌ (need ≥10%)
coverage_ratio:        2.6%   ❌ (need ≥30% of teacher)
overall_pass:          false
```
Chunk-1 crutch ~0% — LoRA fixed the "garbage chunk" issue, but z pipeline still
broken for chunks 2–3. Diagnosis: **gradient dilution** — CE averaged over 341
positions; teacher context dominates at late positions (>32).

---

### Phase 0 v3 — z_anchor_tokens=32, 400 steps, lr_lora=1e-4

**Training probes:**
```
All generation probes: 0% correct
Loss at step 400:      ~3.07 nats (still declining)
```
Diagnosis: under-training at z_anchor density + fixed biased probe pool
(always sampling the same 10 hard L4 problems).

---

### Phase 0 v4 — z_anchor=32, 800 steps, lr_lora=3e-4, probe resample

**Training probes:**
| Step | Loss  | Probe (20 rollouts) |
|------|-------|---------------------|
| 100  | 3.30  | 5%                  |
| 200  | 3.29  | 5%                  |
| 300+ | ~3.0  | 0%                  |

Loss plateaued at ~3.0 nats by step 500. Probe peaked early then regressed
while CE continued falling. Diagnosis: **over-adaptation** — LoRA over-specialised
on z_anchor positions, disrupting late-position behavior (`\boxed{}`, answer format).

**Sanity on checkpoint-200 (best of v4):**
```
mean_ce_loss:          1.48   ✅ (threshold 2.0)
mean_z_std:            0.40   ✅ (threshold 0.1)
teacher_reward_rate:   32.5%
student_reward_rate:   1.25%  ❌ (need ≥10%)
coverage_ratio:        3.8%   ❌ (need ≥30% of teacher)
chunk1_crutch_rate:    0.0    (pipeline used; not chunk-1 lucky)
z_transition_gap:      -0.027 ❌ (z not tracking solution progress)
overall_pass:          false
```

Qualitative: chunk 1 sometimes starts OK; chunks 2–3 collapse into unrelated math,
broken LaTeX, HTML junk, repetition. **Not Phase 1 ready.**

---

### Phase 0 v5 (current, not yet run) — step-by-step teacher, variable chunks

Changes from v4:
- **Step-based teacher generation** with "Complete one reasoning step, then stop" prompt
- **Variable-length chunk lists** — distill_loss handles 1..max_chunks per problem
- `step_tokens=341`, `max_chunks=10`
- `\boxed{}` detection as the stop signal (no hardcoded `--- STATE:` markers)
- Teacher reward ~60% on L1–L4 with avg 2.1 steps/problem (verified via check_assumptions.py)

**Status:** not yet run. Pending after assumption verification.

---

## Fixes — already implemented

| Fix | Where | Notes |
|-----|-------|-------|
| LoRA on backbone (r=16, all attn+MLP) | `grpo_latent.py`, config | Base frozen; adapters at `lr_lora` |
| Phase 0 saves adapter only; eval merges LoRA | `_save_phase0_checkpoint`, `eval_passk.py`, `eval_markov_diagnostics.py` | Unwrap `torch.compile` before `PeftModel.save_pretrained` |
| `_LATENT_SYSTEM_PROMPT` uniform | `format_prompt()` in `grpo_latent.py` | Teacher, student, probe, sanity, Phase 1 eval all use same prompt |
| Sanity redesign | `run_phase0_sanity.py` | Teacher baseline + student vs teacher + chunk-1 crutch; z-transition info-only |
| `z_anchor_tokens` CE masking | `_distill_loss`, config | Supervise first K positions only; concentrates z-conditioning gradient |
| Mid-training generation probe | `pretrain_distill` loop | Logs `probe_rate` every N steps |
| Probe pool random resample | `pretrain_distill` | Held-out 20% tail; fresh sample each probe (was fixed `problems[-10:]`) |
| `n_steps: 800` | config | More training mass at z_anchor density |
| `lr_lora: 3e-4` | config | Compensate for 32/341 supervised positions |
| Cosine LR warmdown | `grpo_latent.py`, config | Last 200 steps: 3e-4 → 3e-5; addresses over-adaptation |
| `_random` UnboundLocalError | `grpo_latent.py` | Removed inline `import random` inside probe block |
| Step-by-step teacher generation | `_generate_teacher_chunks`, `_LATENT_SYSTEM_PROMPT` | "Complete one reasoning step, then stop"; model EOS = boundary; loop until `\boxed{}` or max_chunks |
| Variable-length distill loss | `_distill_loss` | Handles 1..max_chunks per problem instead of fixed 3 |
| `_truncate_at_boxed` | `generate_latent_traces` | Phase 1 truncation at `\boxed{}` only |

---

## Root-cause map (current best understanding)

```
Teacher forcing CE (positions 0–31)     →  passes (CE ~1.5)
                    ≠
Autoregressive generation (0–341)       →  fails (reward ~1.25%)

Contributing factors:
  1. z too compressed (64-dim; negative z-transition)
  2. Tail positions never supervised (no \boxed{} gradient)
  3. min_new_tokens=170 forces garbage mid-chunk
  4. Exposure bias (teacher tokens in train, own tokens at infer)
  5. Over-adaptation at constant high lr (steps 200–800 regression)
  6. LoRA may lack capacity to condition on prefix (secondary)
```

Chunk-1 crutch 0% → failure is **in the Markov pipeline**, not "answer in chunk 1."

---

## Related files

- Official design: `reports/latent_markov_design.md`
- Official results: `reports/ablation_core.md`
- Training: `src/training/grpo_latent.py`
- Sanity: `scripts/run_phase0_sanity.py`
- Config: `configs/train_latent_grpo.yaml`
- Assumption testing: `scripts/check_assumptions.py`
- Encoder: `src/models/vae_state_encoder.py` (1536→512→128→z_dim)
