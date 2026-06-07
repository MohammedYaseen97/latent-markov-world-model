# Phase 0 — Assumption Tracker & Training Log

Tracks every assumption the design depends on, plus run history and open work.  
For the official architecture, see `reports/latent_markov_design.md`.

Last updated: 2026-06-08

---

## Master Assumption Table

| ID | Assumption | Group | Status | Blocks next run? |
|----|------------|-------|--------|-----------------|
| A0 | Teacher generates multiple distinct reasoning steps | Teacher | ✅ Verified (87% clean; 13% mild restatement/packing, filterable) | — |
| A1 | Chunk boundaries fall at natural semantic step-ends | Teacher | ✅ Verified for L1-L4 (max_tokens noise ~5%, filterable); degrades at L5 | — |
| A9 | Teacher accuracy sufficient for data generation (~85% on L1-L4) | Teacher | ✅ Verified (~82% real on L1-L4; L5 ~57% — marginal) | — |
| A-prompt | Qwen3-8B reliably follows "one step then stop" on L1-L4 | Teacher | ✅ Verified on L1-L4. Loop traps at L4-L5 on enumeration problems (filterable) | — |
| A-cont | Continuation message must allow termination, not demand a next step | Teacher | ✅ Fixed — echo bug patched (removed `\boxed{answer}` from continuation text); loop detector added at 95% similarity threshold; max_chunks raised to 15 | — |
| A-grader | `answers_equivalent` correctly scores all LaTeX variants | Data | ✅ Fixed — root cause was `mv_parse` returning `[]` not `None`; fixed `if p and g` check + normalize-before-parse. Known limitation: polynomial factor ordering (1 case/run, unfixable without CAS expansion) | — |
| A-filter | Multi-step chunks and loop-trap traces are rare and filterable | Data | ✅ Verified — loop traps ~5% at L4, max_tokens cuts ~5-10% at L5 | — |
| A2 | Last-token hidden state `repr_h` encodes position in solution space | Architecture | ❌ Not tested | **Yes** |
| A3 | Encoder produces diverse z vectors (no collapse) | Architecture | ❌ Not tested | **Yes** |
| A4 | Student generates coherent step N+1 from z_prefix alone | Architecture | ❌ Not tested (IS Phase 0) | **Yes** |
| A5 | z_dim is sufficient (64 suspected too small; 128 proposed) | Architecture | ⚠️ Partial | Yes — use 128 |
| A3-pool | Last-token repr is sufficient; mean-pool not needed | Architecture | ❌ Not tested | Low priority |
| A7 | One virtual prefix token is sufficient | Architecture | ❌ Not tested | Low priority |
| A8 | Linear Z-injector (no nonlinearity) is expressive enough | Architecture | ❌ Not tested | Low priority |
| A-enc-ag | Same encoder weights generalize across all chunk transitions | Architecture | ❌ Not tested | Low priority |
| A-lora | LoRA r=16 has capacity for z-conditioning (r=32 proposed) | Architecture | ⚠️ Partial | Yes — use r=32 |
| A-markov | z is a sufficient statistic; context reset loses nothing | Architecture | 🔍 Core claim, unproven | Phase 1 |
| A-reset | Context reset is lossless (not destructive) | Architecture | 🔍 Design choice, unproven | Phase 1 |
| A-repr2 | repr_h under teacher forcing matches repr_h at inference | Architecture | 🔍 Known risk (exposure bias) | Phase 1 |
| A-mlen | First-order Markov length is sufficient (no higher-order deps) | Architecture | ❌ Not tested (deferred) | Phase 1 |
| A6 | Exposure bias is manageable; teacher forcing transfers to generation | Architecture | 🔍 Known failure — primary blocker | Phase 0 |

**Legend:** ✅ Verified · ❌ Not tested · ⚠️ Partial evidence · 🔍 Empirical finding (not pre-tested)

**Blocking priority:** "Yes" = must address before next Phase 0 run. "Low priority" = test only if Phase 0 passes. "Phase 1" = deferred until Phase 1.

---

## Detailed Notes

### Group 1 — Teacher trace quality

**A0 — Teacher generates distinct steps**
- Model: Qwen3-8B, atomic-step prompt (`check_assumptions.py`)
- **Official evidence** (`traces_official_l1l3.json` + `traces_official_l4.json` + `traces_official_l5.json`, 30 problems each):
  - 4–6 chunk traces: 100% atomic across all three files — zero violations. This is the ideal range.
  - 1-chunk traces: rare monolithic packing (L1-L3: 1/6 = 17%, e.g. 0974 computing 6 bounces in one chunk)
  - 2-chunk traces: ~25% mild restatement in L1-L3 (model solves in chunk 1 with EOS, chunk 2 just boxes). Borderline violation but trace is still correct.
  - L4: clean — no packing violations. Structural issues only from loop traps (3697) and over-long enumeration (3708).
  - L5: no packing violations; max_tokens cuts on complex derivations.
- **Verdict: ✅ for L1-L4 at ~87% of traces.** Filterable violations in ~13%.
- Note: Old A0 (fixed 3 chunks × 341 tokens) was **invalidated** by atomic-step experiments and retired.

**A1 — Chunk boundaries are semantic**
- **Official evidence:**
  - L1-L3: 1 max_tokens cut (1899), 1 restatement loop (2249). 28/30 clean.
  - L4: 2 max_tokens cuts (3554 recovered, 3780 didn't). 28/30 clean.
  - L5: 8+ problems with at least one max_tokens chunk. L5 derivations regularly exceed 512 tokens.
- **Verdict: ✅ for L1-L4. Degrades at L5** — max_tokens cuts create artificial boundaries. For Phase 0 (trains L1-L4): A1 holds.
- Filter: discard traces with `n_chunks == max_chunks` and no `\boxed{}`, discard traces where any chunk `stopped_by == "max_tokens"` if the content is mid-derivation.
- Note: Old A1 (token-341 boundary is semantic) retired.

**A9 — Teacher accuracy**
- **Official evidence (per level, 30 problems each):**
  - L1 (from l1l3 run): ~91% real accuracy
  - L2: ~89% real accuracy
  - L3: ~80% real accuracy
  - L4: 17/30 scored = 57%; 6 grader bugs → **~77% real accuracy**
  - L5: 13/30 scored = 43%; 3-4 grader bugs → ~57% real accuracy
- **L1-L4 combined real accuracy ≈ 82%** — sufficient for Phase 0 data generation
- **Verdict: ✅ for L1-L4 (Phase 0 training domain). Marginal for L5 (Phase 1 domain).**

**A-prompt — Model follows one-step instruction**
- **Official evidence:**
  - L1-L3: clean for most. 2 critical failures: 2249 (11-way restatement loop), 0388 (continuation echo bug).
  - L4: zero loop traps (improvement from previous), zero packing violations. 3697 is a search-space loop (model repeats "Not valid" iteration). 3708 is capacity-limited (valid work, hits max_chunks). Both filterable.
  - L5: 2 continuation echo bugs (0261, 0496). L5 search problems still loop-trap.
- **Verdict: ✅ for L1-L4 at ~90%+ of traces. Known exception: enumeration/search problems at L4-L5.**

**A-cont — Continuation message**
- v1: `"Next intermediate result."` → imperative, forced new steps → loop traps
- v2: `"Next step — or \boxed{answer} if done."` → fixed loops but introduced echo bug (model echoed `\boxed{answer}` literally in 3/90 cases)
- v3 (current): `"Give the next step. If you are done, state the final answer in a box."` — no literal `\boxed{}` pattern → echo bug eliminated
- Additional fixes applied: loop detector (95% SequenceMatcher threshold), max_chunks raised 12→15
- **Verdict: ✅ Verified and patched.**

---

### Group 2 — Architecture / latent space

**A2 — `repr_h` encodes solution-space position**
- What to check: cosine similarity between chunk-1 and chunk-2 hiddens of the same problem should be lower than between chunk-1 hiddens of two different problems. PCA/UMAP should show trajectory structure.
- How to test: add `--step repr_h` to `check_assumptions.py`; run on `traces_l1l4v2.json` teacher traces

**A3 — Encoder z diversity**
- What to check: mean z std across dims > 0.1; pairwise z cosine distances not clustering near 1.0
- How to test: add `--step z_variance`; run 50+ teacher `repr_h` vectors through encoder
- Note: `checkpoint-200` sanity showed `mean_z_std: 0.40` ✅ but that's mid-training, not a pre-training baseline

**A4 — Student generates from z_prefix**
- This IS Phase 0. Passes if `student_reward_rate ≥ 10%` and `coverage_ratio ≥ 30%` of teacher.
- Failure mode observed: chunks 2–3 collapse to junk LaTeX, HTML noise, repetition

**A5 — z_dim sufficient**
- Phase 0 Run 3: `z_transition_gap: -0.027` ❌ — z not tracking solution progress at z_dim=64
- 1536→64 = 24× compression for complex math; likely too small. Proposed: z_dim=128
- Not directly isolated; compounded with training dynamics issues

**A3-pool — Last-token vs mean-pool**
- Low priority until A2/A3/A4 are resolved. If Phase 0 passes with last-token, no need to test.

**A7 — One prefix token sufficient**
- If z_dim increases to 128+, multiple prefix tokens become more relevant. Config: `latent_markov.n_prefix_tokens` (currently 1, not in YAML).

**A8 — Linear injector expressiveness**
- `Linear(z_dim, 1536)` only. Low priority — test if Phase 0 fails with all other fixes applied.

**A-enc-ag — Encoder generalizes across chunk indices**
- Same encoder for 1→2 and 2→3 transitions. Test: compare L_trans loss between first and second transitions after training.

**A-lora — LoRA capacity**
- Phase 0 runs with r=16 showed near-zero probe_rate. May be confounded by z_dim + exposure bias.
- Proposed: r=32 (with lora_alpha=64, lora_dropout=0.05) in next run

**A-markov — z is a sufficient statistic** *(core project claim)*
- The entire architecture depends on this. z_h must encode everything from chunks 1..h needed for chunk h+1.
- The context reset *enforces* this: without the reset, the backbone can attend to prior tokens and ignore z. The reset makes z mandatory.
- Validated when: student probe_rate from z-only input is comparable to teacher reward rate.

**A-reset — Context reset**
- The reset is not a bug — it's the mechanism that makes z informative. Without it, z is optional.
- "Lossless" means the encoder captures everything relevant; this is the goal state, not a given.

**A-repr2 — repr_h under teacher forcing**
- Teacher-forced chunk-2 hiddens used to compute repr_2 → z_2 during training. At inference, the student generates chunk 2 itself — hiddens will differ.
- This is the exposure bias problem at the representation level. Observed in Phase 0: CE converges but generation fails.

**A-mlen — Markov chain length**
- For ≤3 chunks: "z_2 alone is sufficient for chunk 3." Deferred.
- For variable-length atomic steps (1–12 chunks): higher-order dependencies are more plausible for hard problems. Revisit in Phase 1.

**A6 — Exposure bias manageable**
- Known failure. CE loss converges (~1.5 nats) but probe reward stays near 0%.
- Mitigations proposed: `z_tail_tokens`, scheduled sampling, LR warmdown (warmdown implemented, others pending).

---

### Group 3 — Data pipeline

**A-grader — `answers_equivalent` correctness**
- Fixes applied in `grpo_baseline.py::answers_equivalent`:
  - `mv_parse` None-check before `mv_verify` (handles `\pi`)
  - `\dfrac` / `\tfrac` → `\frac`
  - `\sqrt5` → `\sqrt{5}` (unbraced single-char)
  - Spaces around `+`/`-` collapsed
  - `, ` inside tuples → `,`
- Remaining edge cases: `\frac16` shorthand, factor ordering — deferred

**A-filter — Trace filtering**
- Discard: traces where any chunk contains `### Step N:` or `Step N:` headers
- Discard: traces where `n_chunks == max_chunks` and `reward == 0`
- At 80%+ valid rate on L1-L4, volume is not a bottleneck

---

## Phase 0 Success Criterion

Student (LoRA + encoder + Z-injector) must generate full reasoning from **`z_prefix` only** on L1-L4, with reward rate comparable to teacher. CE loss alone is insufficient — the final gate is student vs teacher reward rate on held-out easy problems.

| Metric | Threshold | Notes |
|--------|-----------|-------|
| `student_reward_rate` | ≥ 10% | Absolute floor |
| `coverage_ratio` | ≥ 30% of teacher | Student/teacher rate ratio |
| `chunk1_crutch_rate` | diagnostic only | High crutch = z broken for chunks 2-3 |
| `mean_ce_loss` | ≤ 2.0 | Sanity check; not a pass criterion |
| `mean_z_std` | ≥ 0.1 | z not collapsed |

---

## Run History

### Run 1 — Full CE, 400 steps, lr=1e-4

- CE passed; z variance healthy
- Student reward ~1.25% vs teacher ~49%
- Diagnosis: **gradient dilution** — CE averaged over all 341 tokens; model learns "continue teacher text" not "generate from z"

### Run 2 — z_anchor=32, 400 steps, lr=1e-4

- All probes: **0%**. Loss declining at step 400 (~3.07 nats)
- Diagnosis: **under-training** — z_anchor density too sparse + fixed biased probe pool

### Run 3 — z_anchor=32, 800 steps, lr=3e-4, probe resample

| Step | CE Loss | Probe rate |
|------|---------|------------|
| 100  | 3.30    | 5%         |
| 200  | 3.29    | 5%         |
| 300+ | ~3.0    | 0%         |

- Probe **peaked then regressed** while loss continued falling
- Diagnosis: **over-adaptation** — LoRA over-specialises on z_anchor positions; disrupts `\boxed{}` / answer format

### Sanity on checkpoint-200 (best of Run 3)

| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| mean_ce_loss | 1.48 | ≤ 2.0 | ✅ |
| mean_z_std | 0.40 | ≥ 0.1 | ✅ |
| teacher_reward_rate | 32.5% | — | — |
| student_reward_rate | 1.25% | ≥ 10% | ❌ |
| coverage_ratio | 3.8% | ≥ 30% | ❌ |
| chunk1_crutch_rate | 0.0 | diagnostic | — |
| z_transition_gap | -0.027 | diagnostic | ❌ |
| overall_pass | **false** | | |

Qualitative: chunk 1 sometimes starts OK; chunks 2–3 collapse into unrelated math, broken LaTeX, HTML, repetition.

---

## Root-Cause Map

```
Teacher forcing CE (positions 0–31)   →  passes (CE ~1.5 nats)
                  ≠
Autoregressive generation (0–341)     →  fails (reward ~1.25%)

Contributing factors (prioritised):
  1. z too compressed (64-dim; negative z-transition)        → A5
  2. Exposure bias (teacher tokens in train, own at infer)   → A6, A-repr2
  3. Tail positions never supervised (no \boxed{} gradient)  → fix: z_tail_tokens
  4. min_new_tokens=170 forces long garbage mid-chunk        → fix: lower min_new_tokens
  5. Over-adaptation at constant high lr                     → fix: warmdown (done)
  6. LoRA may lack capacity for z-conditioning              → A-lora
```

Chunk-1 crutch 0% → failure is in the Markov pipeline, not "answer lucky in chunk 1."

---

## Fixes

### Implemented

| Fix | Location | What it addresses |
|-----|----------|-------------------|
| LoRA on backbone (r=16, all attn+MLP) | `grpo_latent.py`, config | Base frozen; adapters trained at lr_lora |
| Phase 0 saves adapter only; eval merges LoRA | `_save_phase0_checkpoint`, `eval_passk.py` | Unwrap `torch.compile` before `PeftModel.save_pretrained` |
| `_LATENT_SYSTEM_PROMPT` uniform | `format_prompt()` | Teacher, student, probe, sanity, Phase 1 all use same prompt |
| Sanity redesign | `run_phase0_sanity.py` | Teacher baseline + student vs teacher + crutch diagnostic |
| `z_anchor_tokens` CE masking | `_distill_loss`, config | Concentrate gradient on z-prefix positions (first K tokens) |
| Mid-training generation probe | `pretrain_distill` loop | Logs `probe_rate` every N steps |
| Probe pool random resample | `pretrain_distill` | Fresh 20-problem sample per probe; removes fixed-pool bias |
| `n_steps: 800`, `lr_lora: 3e-4` | config | More training mass at z_anchor density |
| Cosine LR warmdown | `grpo_latent.py`, config | Last 200 steps 3e-4→3e-5; addresses over-adaptation |
| `_random` UnboundLocalError | `grpo_latent.py` | Removed duplicate inline `import random` |
| Grader: `\dfrac`, `\pi`, tuple spacing, `\sqrt5` | `grpo_baseline.py::answers_equivalent` | LaTeX normalisation for string fallback |
| Continuation message fix | `check_assumptions.py` | "Next step — or \boxed{}" replaces imperative message |
| Asymptote filter in `_load_pool` | `check_assumptions.py` | Skip problems with `[asy]` diagrams |

### Pending (not yet implemented)

| Fix | Rationale | Target |
|-----|-----------|--------|
| `z_tail_tokens=32` | `\boxed{}` lives at tail; tail never supervised | `_distill_loss` |
| Hybrid CE (head + tail, skip middle) | Middle tokens dilute z signal; head + tail both matter | `_distill_loss` |
| Scheduled sampling | Close teacher-forcing / autoregressive gap after token K | `_distill_loss` or wrapper |
| `latent_dim: 128` | 64-dim too compressed for complex math | config, `vae_state_encoder.py` |
| LoRA r=32, alpha=64, dropout=0.05 | Capacity to condition on richer z | config |
| `lr_lora: 2e-4` | Pair with capacity increase | config |
| Save best-probe checkpoint | Best quality was step ~200, not final | `pretrain_distill` |
| Lower `min_new_tokens` | =170 forces long garbage; try 32 | `generate_latent_traces()` |

### Rejected / Deferred

| Idea | Reason |
|------|--------|
| Classical simulated annealing | Doesn't map to gradient CE training; LR warmdown covers this |
| LoRA rank ↑ alone (without z_dim ↑) | Over-adaptation risk; dynamics > capacity; must pair with z_dim |
| LoRA rank ↑ without warmdown/dropout | Same reason |

---

## Proposed Next Run

```yaml
latent_markov:
  latent_dim: 128             # was 64 — addresses A5

phase0:
  n_steps: 800
  lora:
    r: 32                     # was 16 — addresses A-lora
    lora_alpha: 64
    lora_dropout: 0.05
  lr_lora: 2.0e-4             # was 3e-4 — compensate for higher capacity
  lr_warmdown_steps: 250
  lr_warmdown_final: 0.1
  z_anchor_tokens: 32
  z_tail_tokens: 32           # NOT IMPLEMENTED — add to _distill_loss
  # optional: lr_warmup_steps: 50
  # optional: save best probe checkpoint
```

Also: lower `min_new_tokens` in `generate_latent_traces()` once tail loss is in place.

---

## Commands

```bash
# Phase 0 train
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 0

# Sanity — use best intermediate checkpoint if probes peaked early
python scripts/run_phase0_sanity.py \
    --config configs/train_latent_grpo.yaml \
    --checkpoint artifacts/latent_grpo/<run_id>/phase0/checkpoint-200

# Full Phase 0 eval
python scripts/eval_passk.py --generation-mode latent_markov_pretrained \
    --train-config configs/train_latent_grpo.yaml \
    --checkpoint artifacts/latent_grpo/<run_id>/phase0 \
    --arm latent_grpo_pretrained

# Teacher trace generation (check_assumptions)
python scripts/check_assumptions.py --step traces --n 50 --max-step-tok 512 \
    --model Qwen/Qwen3-8B \
    --pool data/math_easy_pool.jsonl --levels 1 2 3 4 \
    --out reports/traces_l1l4_datagen.json
```

---

## Official Assumption Verification Runs

### Official Run — 2026-06-08
**Files:** `traces_official_l1l3.json`, `traces_official_l4.json`, `traces_official_l5.json`  
**Model:** Qwen3-8B, `check_assumptions.py --step traces --n 30 --max-step-tok 512`

| Level | N | Scored | Grader bugs | Real accuracy | Notes |
|-------|---|--------|-------------|---------------|-------|
| L1 (from l1l3) | 11 | 9 (82%) | 1 | ~91% | |
| L2 (from l1l3) | 9 | 7 (78%) | 1 | ~89% | |
| L3 (from l1l3) | 10 | 8 (80%) | 0 | ~80% | |
| L4 | 30 | 17 (57%) | 6 | ~77% | Many grader format bugs |
| L5 | 30 | 13 (43%) | 3-4 | ~57% | Reasoning errors increase |
| **L1-L4 combined** | **60** | — | — | **~82%** | Sufficient for Phase 0 |

**Structural issues found:**

| Issue | Count | Files | Severity |
|-------|-------|-------|----------|
| Continuation echo (`\boxed{answer}` echoed) | 3 | L1-L3: 0388; L5: 0261, 0496 | **Critical — must fix** |
| Restatement loop (solved in chunk 1, N restates) | 2 | L1-L3: 2249 (×11), 0889 (×2) | Severe (filterable) |
| Same-iteration loop trap (L4 search) | 1 | L4: 3697 | Severe (filterable) |
| max_chunks exhausted on valid work | 2 | L4: 3708; L5: 0587 | Moderate (filterable) |
| max_tokens cut on single chunk | 8 | L4: 2 cases; L5: 6+ cases | Mild–moderate |
| Mild restatement (chunk N boxes prev result) | ~8 | L1-L3 2-chunk problems | Mild (trace still usable) |

**Active grader bugs (post-fix, still failing):**

| Pattern | Example | Why fix failed |
|---------|---------|----------------|
| `\frac{1}{6}` vs `\frac16` shorthand | L4:3081 | _norm doesn't expand `\frac16` |
| Unsimplified fraction (729/24 ≠ 243/8) | L4:3006 | mv_verify returns False despite equality |
| Factor ordering | L4:3001, L5:0499 | String compare after mv_verify fails |
| `75,\!075` format | L4:3266 | `\!` is a negative thin space — not normalized |
| Tuple with space `(18, -24)` | L4:3600 | mv_parse succeeds but mv_verify fails before fallback runs |

**Key finding — continuation echo bug:**
The current continuation message `"Next step — or \boxed{answer} if done."` contains the literal string `\boxed{answer}`. In ~3% of completions the model outputs this string verbatim, causing the grader to extract `answer` as the boxed content. **Fix before data generation:**
```python
# In check_assumptions.py, change:
"Next step — or \\boxed{answer} if done."
# To:
"Give the next step. If you are done, state the final answer in a \\boxed{}."
```

---

## Open Questions

1. Is z_dim=128 enough, or do we need 256 + multi-token injection?
2. Can `z_tail_tokens` alone fix `\boxed{}` without full middle supervision?
3. Should Phase 0 include a small on-policy loss (mix own rollouts into CE) before Phase 1?
4. Should sanity gate on median problem reward instead of mean, given heavy-tailed easy pool?
5. Is z-transition worth reviving as a soft auxiliary loss once `latent_dim` increases?

---

## Related Files

| File | Purpose |
|------|---------|
| `reports/latent_markov_design.md` | Official architecture design |
| `src/training/grpo_latent.py` | Phase 0/1 training loops |
| `scripts/run_phase0_sanity.py` | Post-training sanity checks |
| `scripts/check_assumptions.py` | Teacher trace generation & assumption tests |
| `src/training/grpo_baseline.py` | Reward function, grader |
| `configs/train_latent_grpo.yaml` | Main config |
| `src/models/vae_state_encoder.py` | Encoder (1536→512→128→z_dim) |
