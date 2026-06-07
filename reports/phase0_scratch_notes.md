# Phase 0 — Unofficial Scratch Notes

**Not canonical.** Working notes from debugging Phase 0 distillation (latent Markov arm).  
For the official design, see `reports/latent_markov_design.md`.

Last updated: 2026-06-02

---

## Phase 0 success criterion (agreed)

Student (LoRA + encoder + Z-injector) must generate full 3-chunk reasoning on L1–L4 **from `z_prefix` only**, with reward rate comparable to teacher. CE loss alone is insufficient — the final gate is **student vs teacher reward rate** on held-out easy problems.

Sanity thresholds (`phase0_sanity`):
- `coverage_threshold: 0.30` — student rate ≥ 30% of teacher rate
- `min_student_reward: 0.10` — absolute floor 10%
- `chunk1_crutch_rate` — diagnostic only; high crutch = answers from chunk 1, z broken for chunks 2–3

---

## Run history & key observations

### Run 1 — LoRA, full CE, 400 steps, lr_lora=1e-4

- CE passed; z variance healthy
- Student reward ~1.25% vs teacher ~49%
- Chunk-1 crutch ~0% (LoRA fixed “garbage chunk” issue)
- Diagnosis: **gradient dilution** — CE averaged over all 341 tokens; teacher context dominates at late positions; model learns “continue teacher text” not “generate from z”

### Run 2 — z_anchor_tokens=32, 400 steps, lr_lora=1e-4

- All generation probes: **0%**
- Loss still declining at step 400 (~3.07 nats)
- Diagnosis: **under-training** at z_anchor density + fixed biased probe pool

### Run 3 — z_anchor=32, 800 steps, lr_lora=3e-4, probe resample

| Step | Loss | Probe (20 rollouts) |
|------|------|---------------------|
| 100  | 3.30 | 5%                  |
| 200  | 3.29 | 5%                  |
| 300+ | ~3.0 | 0%                  |

- Loss plateau ~3.0 nats by step 500
- Probe **peaked early then regressed** while loss kept falling
- Diagnosis: **over-adaptation** — LoRA over-specialises on z_anchor positions; disrupts late-position behavior (`\boxed{}`, answer format)

### Sanity on `checkpoint-200` (best of Run 3)

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

Qualitative: chunk 1 sometimes starts OK; chunks 2–3 collapse into unrelated math, broken LaTeX, HTML junk, repetition. **Not Phase 1 ready.**

Probe 5% at step 200 was noise on 20 rollouts; sanity 80 rollouts → true rate ~1.25%.

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

---

## Fixes — discussed, not yet implemented

### Training / loss

| Idea | Rationale | Suggested values |
|------|-----------|------------------|
| **`z_tail_tokens`** | z_anchor supervises 0–31 only; tokens 32–340 never trained; `\boxed{}` lives in tail | Supervise last 32 positions of chunks 2–3 in addition to z_anchor |
| **Hybrid CE** | Middle tokens (32–309) unsupervised to avoid diluting z signal; head + tail both matter | `z_anchor_tokens=32`, `z_tail_tokens=32` |
| **Scheduled sampling** | Close teacher-forcing vs autoregressive gap after token K | Gradually mix student tokens into positions > z_anchor |
| **Full CE on chunk 1** | Chunk 1 has full prompt context; no z yet — could stabilize repr_h | Optional; chunk 1 currently no CE |
| **LR warmup** | Stabilize early steps when encoder/LoRA cold-start | e.g. 50 steps 0 → lr |
| **Early stop / best-probe checkpoint** | Best generative quality was ~step 200, not step 800 | Save/copy checkpoint when `probe_rate` is max |
| **z_anchor curriculum** | Start K=16, grow to 32 over training | Softer than jumping to full anchor |

### Architecture / capacity

| Idea | Rationale | Suggested values |
|------|-----------|------------------|
| **`latent_dim` ↑** | 1536→64 compression may lose “position in solution space” for hard L4 math | 128 or 256 (`latent_markov.latent_dim`) |
| **LoRA rank ↑** | Backbone needs capacity to *use* richer z prefix | r=32, alpha=64, **with** higher z_dim |
| **`lora_dropout`** | More capacity → easier over-fit | 0.05 when increasing rank |
| **Lower peak `lr_lora`** | Pair with capacity increase | 2e-4 instead of 3e-4 |
| **Multiple prefix tokens** | Still one virtual token today; richer z → one 1536-dim vector only | Larger arch change; inject K tokens from z |

**Consensus from discussion:** increase `latent_dim` and LoRA rank **together**, not one alone. Pair with regularization (warmdown, dropout, lower peak lr).

### Generation (inference)

| Idea | Rationale | Suggested values |
|------|-----------|------------------|
| **Reduce `min_new_tokens`** | Currently `chunk_tokens // 2` (=170) forces long garbage when model is lost; garbage → bad z_2 → worse chunk 3 | Try 32 or remove once z works |
| **Probe / sanity rollout count** | 20-rollout probes are high-variance | Probes misleading vs 80-rollout sanity |

Location: `generate_latent_traces()` — `min_new_tokens=chunk_tokens // 2` on chunks 2 and 3.

### Eval / ops

| Idea | Notes |
|------|-------|
| Sanity on `checkpoint-200` not final dir | Best artifact from Run 3 may not be `phase0/` root |
| `torch.compile` warning | “reduce-overhead compiled saved, loading default” — likely compile mode mismatch on load; checkpoint fixes address LoRA keys, not necessarily compile mode |
| Phase 1 optimizer not restored from checkpoint | `phase1_latent.pt["optimizer"]` saved but not loaded — noted in audit |

---

## Rejected or deferred (for now)

| Idea | Why deferred |
|------|--------------|
| **Classical simulated annealing** | Doesn’t map to gradient CE training; LR warmdown is the relevant “annealing” |
| **LoRA rank ↑ alone** | Run showed early probe success then regression — dynamics > capacity; rank alone may worsen over-adaptation |
| **LoRA rank ↑ without warmdown/dropout** | Same reason |
| **Increase rank before warmdown run** | Sequential debugging: isolate warmdown first, then capacity |

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

Chunk-1 crutch 0% → failure is **in the Markov pipeline**, not “answer in chunk 1.”

---

## Proposed next run (sketch — not committed)

```yaml
latent_markov:
  latent_dim: 128   # or 256

phase0:
  n_steps: 800
  lora:
    r: 32
    lora_alpha: 64
    lora_dropout: 0.05
  lr_lora: 2.0e-4
  lr_warmdown_steps: 250
  lr_warmdown_final: 0.1
  z_anchor_tokens: 32
  z_tail_tokens: 32          # NOT IMPLEMENTED — add to _distill_loss
  # optional: lr_warmup_steps: 50
  # optional: save best probe checkpoint
```

Also consider lowering `min_new_tokens` in `generate_latent_traces` once tail loss is in place.

---

## Commands (reference)

```bash
# Phase 0 train
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 0

# Sanity — use best intermediate checkpoint if probes peaked early
python scripts/run_phase0_sanity.py \
    --config configs/train_latent_grpo.yaml \
    --checkpoint artifacts/latent_grpo/<run_id>/phase0/checkpoint-200

# Full Phase 0 eval (pass@k, pretrained)
python scripts/eval_passk.py --generation-mode latent_markov_pretrained \
    --train-config configs/train_latent_grpo.yaml \
    --checkpoint artifacts/latent_grpo/<run_id>/phase0 \
    --arm latent_grpo_pretrained
```

---

## Open questions

1. Is 128-dim z enough, or do we need 256 + multi-token injection?
2. Can `z_tail_tokens` alone fix `\boxed{}` without full middle supervision?
3. Should Phase 0 add a small amount of **on-policy** loss (mix own rollouts into CE) before Phase 1?
4. Should sanity gate on **median** problem reward, not mean, given heavy-tailed easy pool?
5. Is z-transition worth reviving as a soft loss once `latent_dim` increases?

---

## Related files

- Official design: `reports/latent_markov_design.md`
- Training: `src/training/grpo_latent.py`
- Sanity: `scripts/run_phase0_sanity.py`
- Config: `configs/train_latent_grpo.yaml`
- Encoder: `src/models/vae_state_encoder.py` (1536→512→128→z_dim)
