# Phase 0 — Working Log

This is the living debug log for the latent Markov arm. It records every training run,
every assumption tested, every fix applied, and every open question — in the order things
actually happened. For the official architecture spec, see `reports/latent_markov_design.md`.

Last updated: 2026-06-07

---

## The Story So Far

The project has gone through three distinct design generations. Understanding why each
was replaced is essential context for every decision that follows.

### v1 — VAE-ELBO (abandoned ~2026-05)

The original architecture used a full VAE: encoder outputs `(μ_h, σ²_h)`, reparameterised
sample `z = μ + ε·σ`, and a KL term to regularise the latent. Phase 0 trained with ELBO.

**What broke it:**
- KL weight = 1.0 was too aggressive. The encoder learned to output `σ²≈1.0` everywhere
  (posterior collapse to prior), making z near-constant. E1 diagnostic confirmed it:
  `L_trans = 0.015` — trivially low because z barely changes, not because the transitions
  are clean.
- Eval was underpowered: n=40 problems → ±2.5pp resolution. Can't detect effects below 5pp.
- Metric was pass@1024 — 8× more expensive than needed.

**Decision:** Don't fix the KL weight. The decoder is unnecessary machinery that rung 3
would replace with a diffusion model anyway. Drop it. Switch to a deterministic state
tracker (`z = μ` only), scale the benchmark to n≈350 / pass@128, and fix the training recipe.

---

### v2 — L_trans + L_out + crutch (abandoned ~2026-05)

Redesigned Phase 0: `L_trans` (Markov consistency on latent trajectories) + `L_out`
(outcome head BCE on reward) + `L_calib` (uncertainty calibration). Dropped ELBO and the
decoder. Benchmark fixed: MATH Level 5 hard pool, ~350 problems, pass@128.

**Result: pass@128 = 8.42%.** This looked promising.

**What broke it:**
- The generation-time crutch. Context for chunk 2 was `[z_prefix | prev_chunk | generate]` —
  the backbone could attend to prior chunk tokens and ignore z entirely. z was optional.
- This means the 8.42% number is contaminated: most correct answers were probably generated
  by the backbone reading prior tokens, not by conditioning on z.
- UMAP latent geometry from this run cannot be trusted as evidence of z encoding anything.

**Decision:** Remove the crutch. Force strict Markov: `[z_prefix | generate]` only — no
prior tokens. If z doesn't encode enough, the model fails. This is what makes z mandatory
and the hypothesis testable. Also: `L_trans` and `L_out` both depended on the crutch
architecture; without it, the later-chunk representations are uninformative at the start of
training (strict Markov + frozen backbone = garbage repr_2/repr_3 early on). Replace with
teacher-forced CE distillation which gives dense signal from a reliable teacher regardless
of the student's early quality. This is the v3 design.

---

### v3 — Strict Markov, CE distillation (current codebase)

**Architecture changes from v2:**
- Crutch removed: student context is `[z_prefix | chunk_h]` only, never prior tokens
- `L_trans` dropped: Markov property now enforced structurally by context reset, not as a loss
- `L_out` and `OutcomeHead` dropped: reward signal is sparse on hard pool; not needed in Phase 0
- Transition network removed: was only needed for `L_trans`
- Phase 0 loss: CE distillation — student predicts teacher tokens conditioned on z_prefix
- LoRA added to backbone: adapts at lr=1e-4 without touching pretrained base weights

**Baseline arm (complete):** `pass@128 = 16.19%` on the Level 5 hard pool after 200 GRPO steps.
This is the ceiling the latent arm must beat.

**Latent arm Phase 0 — three failed runs:**

#### Run 1 — Full CE, 400 steps, lr=1e-4

CE converged. z variance healthy (`mean_z_std ≈ 0.4`). Student reward: **~1.25%** vs teacher ~49%.

Diagnosis: **gradient dilution.** CE is averaged over all 341 tokens per chunk. At positions
32–341, the dominant signal is "continue teacher text given teacher context" — not
"generate from z." The z-conditioning gradient gets diluted ~341× by the LM-from-context
signal at later positions. The backbone learns to copy teacher text; it does not learn to
generate from z.

#### Run 2 — z_anchor=32, 400 steps, lr=1e-4

Added `z_anchor_tokens=32`: only supervise the first 32 positions where z_prefix dominates.
All generation probes: **0%.** Loss still declining at step 400 (~3.07 nats).

Diagnosis: **under-training.** At z_anchor density, 400 steps × 32 positions = very sparse
signal. Also: probe pool was fixed — always the same hard problems, masking real progress.

#### Run 3 — z_anchor=32, 800 steps, lr=3e-4, probe resample

Doubled steps, raised lr, switched probe pool to random resample per probe step.

| Step | CE Loss | Probe rate |
|------|---------|------------|
| 100  | 3.30    | 5%         |
| 200  | 3.29    | 5%         |
| 300+ | ~3.0    | 0%         |

Probe **peaked at 5% then regressed to 0%** while loss continued falling.

Diagnosis: **over-adaptation.** LoRA over-specialises on z_anchor positions at constant
high lr. This disrupts `\boxed{}` generation and answer formatting at later positions. The
model learns the z_anchor regime but forgets how to produce a valid answer. Cosine LR
warmdown was implemented (3e-4 → 3e-5 over last 200 steps) to address this.

#### Sanity check on checkpoint-200 (best of Run 3)

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

Qualitative: chunk 1 sometimes starts OK; chunks 2–3 collapse into unrelated math, broken
LaTeX, HTML, repetition. Chunk-1 crutch = 0% confirms the failure is in the Markov pipeline
itself, not chunk-1 luck.

#### Root-cause map

```
Teacher forcing CE (positions 0–31)   →  passes (CE ~1.5 nats)
                  ≠
Autoregressive generation (0–341)     →  fails (reward ~1.25%)

Contributing factors (prioritised):
  1. z too compressed (64-dim; negative z-transition gap)       → A5
  2. Exposure bias (teacher tokens in train, own at infer)      → A6, A-repr2
  3. Tail positions never supervised (no \boxed{} gradient)     → fix: z_tail_tokens
  4. min_new_tokens=170 forces long garbage mid-chunk           → fix: lower min_new_tokens
  5. Over-adaptation at constant high lr                        → fix: warmdown (done)
  6. LoRA r=16 may lack capacity for z-conditioning             → A-lora
```

---

### Why we stopped to check assumptions

After three failed runs and a sanity check that couldn't pass even the basic reward gate,
the decision was made to stop running Phase 0 and verify the ground assumptions first.
Training is expensive and the failure modes were compounding. The question was: are the
assumptions the architecture depends on — teacher trace quality, grader correctness, data
protocol — actually holding before we spend more compute?

The assumption verification runs (below) answered that question.

---

## Master Assumption Table

| ID | Assumption | Group | Status | Blocks next run? |
|----|------------|-------|--------|-----------------|
| A0 | Teacher generates multiple distinct reasoning steps | Teacher | ✅ **Confirmed** — post-fix run: 0/90 loop breaks; healthy traces 97%/93%/67% (L1-L3/L4/L5) | — |
| A1 | Chunk boundaries fall at natural semantic step-ends | Teacher | ✅ **Confirmed** — post-fix run: 1%/13%/16% chunk-level max_tokens rate (L1-L3/L4/L5); all L5 hits on wrong-answer traces | — |
| A9 | Teacher accuracy sufficient for data generation | Teacher | ✅ **Confirmed** — post-fix run: L1-L3 97%, L4 83%, L5 53% | — |
| A-prompt | Qwen3-8B reliably follows "one step then stop" | Teacher | ✅ **Confirmed** — post-fix run: 0/90 echo violations, 0/90 loop breaks across all levels | — |
| A-cont | Continuation message must allow termination, not demand a next step | Teacher | ✅ **Confirmed** — new text produces 0 echo violations over 90 problems | — |
| A-grader | `answers_equivalent` correctly scores all LaTeX variants | Data | ✅ **Confirmed** — 15/16 unit tests pass; +7 L4 and +4 L5 problems recovered post-fix | — |
| A-filter | Clean traces are identifiable and separable from corrupted ones | Data | ✅ **Confirmed** — `clean_trace` field in every record; 28/23/14 clean per 30 problems (L1-L3/L4/L5) | — |
| A2 | Last-token hidden state `repr_h` encodes position in solution space | Architecture | ❌ Not tested | **Yes** |
| A3 | Encoder produces diverse z vectors (no collapse) | Architecture | ❌ Not tested | **Yes** |
| A4 | Student generates coherent step N+1 from z_prefix alone | Architecture | ❌ Not tested (IS Phase 0) | **Yes** |
| A5 | z_dim is sufficient (64 suspected too small; 128 proposed) | Architecture | ⚠️ Partial — negative z-transition gap in Run 3 | Yes — use 128 |
| A3-pool | Last-token repr is sufficient; mean-pool not needed | Architecture | ❌ Not tested | Low priority |
| A7 | One virtual prefix token is sufficient | Architecture | ❌ Not tested | Low priority |
| A8 | Linear Z-injector (no nonlinearity) is expressive enough | Architecture | ❌ Not tested | Low priority |
| A-enc-ag | Same encoder weights generalise across all chunk transitions | Architecture | ❌ Not tested | Low priority |
| A-lora | LoRA r=16 has capacity for z-conditioning (r=32 proposed) | Architecture | ⚠️ Partial — near-zero probe_rate with r=16 | Yes — use r=32 |
| A-markov | z is a sufficient statistic; context reset loses nothing | Architecture | 🔍 Core claim, unproven | Phase 1 |
| A-reset | Context reset is lossless (not destructive) | Architecture | 🔍 Design choice, unproven | Phase 1 |
| A-repr2 | repr_h under teacher forcing matches repr_h at inference | Architecture | 🔍 Known risk (exposure bias) | Phase 1 |
| A-mlen | First-order Markov length is sufficient (no higher-order deps) | Architecture | ❌ Not tested (deferred) | Phase 1 |
| A6 | Exposure bias is manageable; teacher forcing transfers to generation | Architecture | 🔍 Known failure — primary blocker | Phase 0 |

**Legend:** ✅ Confirmed · ❌ Not tested · ⚠️ Partial evidence · 🔍 Empirical finding

**Blocking priority:** "Yes" = must address before next Phase 0 run. "Phase 1" = deferred.

---

## Assumption Detail Notes

### Group 1 — Teacher trace quality

**A0 — Teacher generates distinct steps**
- Model: Qwen3-8B, atomic-step prompt (`check_assumptions.py`)
- Post-fix evidence (30 problems each, all fixes applied):
  - L1-L3: 29/30 healthy (97%). Zero loop breaks. Zero max_chunks hits.
  - L4: 28/30 healthy (93%). Zero loop breaks. Zero max_chunks hits.
  - L5: 20/30 healthy (67%). Zero loop breaks. 4/30 max_chunks — all genuine inability to finish, not loops.
- **Verdict: ✅ Confirmed at all levels.**

**A1 — Chunk boundaries are semantic**
- Post-fix evidence:
  - L1-L3: 1/85 chunks hit max_tokens (1%). 1 trace affected.
  - L4: 12/94 chunks hit max_tokens (13%). 2 traces affected.
  - L5: 20/126 chunks hit max_tokens (16%). 8 traces affected — but every single one is on a wrong-answer trace (reward=0) except 2, which are excluded by `clean_trace`.
- `clean_trace` field enforces this: `reward==1 AND no max_tokens AND no loop_break`.
- **Verdict: ✅ Confirmed.** A1 holds for L1-L3 and L4. At L5 the rate is higher but no max_tokens-cut trace with reward=1 reaches training data.

**A9 — Teacher accuracy**
- Post-fix evidence (30 problems each):
  - L1-L3: 29/30 correct (97%)
  - L4: 25/30 correct (83%)
  - L5: 16/30 correct (53%)
- **Verdict: ✅ Confirmed for L1-L4.** L5 accuracy is a teacher model ceiling on hard problems, not a protocol failure. L5 is eval only — only the grader correctness matters there, which is confirmed.

**A-prompt — Model follows one-step instruction**
- Post-fix evidence: 0/30 echo violations, 0/30 loop breaks at every level (90 problems total).
- **Verdict: ✅ Confirmed.**

**A-cont — Continuation message**
- v1: `"Next intermediate result."` → imperative, forced new steps → loop traps
- v2: `"Next step — or \boxed{answer} if done."` → echo bug (model echoed `\boxed{answer}` literally; 3/90 cases)
- v3 (current): `"Give the next step. If you are done, state the final answer in a box."` — no `\boxed{}` in prompt text → physically impossible to echo
- Supporting fixes: loop detector at 95% SequenceMatcher threshold; max_chunks raised 12→15
- **Post-fix evidence: 0/90 echo violations. Verdict: ✅ Confirmed.**

---

### Group 2 — Architecture / latent space

**A2 — `repr_h` encodes solution-space position**
- What to check: cosine similarity between chunk-1 and chunk-2 hiddens of the same problem should be lower than between chunk-1 hiddens of two different problems. PCA/UMAP should show trajectory structure.
- How to test: add `--step repr_h` to `check_assumptions.py`; run on teacher traces

**A3 — Encoder z diversity**
- What to check: mean z std across dims > 0.1; pairwise z cosine distances not clustering near 1.0
- How to test: add `--step z_variance`; run 50+ teacher `repr_h` vectors through encoder
- Note: checkpoint-200 sanity showed `mean_z_std: 0.40` ✅ but that's mid-training, not a pre-training baseline

**A4 — Student generates from z_prefix**
- This IS Phase 0. Passes if `student_reward_rate ≥ 10%` and `coverage_ratio ≥ 30%` of teacher.
- Failure mode observed in Runs 1–3: chunks 2–3 collapse to junk LaTeX, HTML noise, repetition

**A5 — z_dim sufficient**
- Run 3: `z_transition_gap: -0.027` — z not tracking solution progress at z_dim=64
- 1536→64 = 24× compression for complex multi-step math; likely too small
- Proposed: z_dim=128. Confounded with training dynamics; not directly isolated.

**A3-pool — Last-token vs mean-pool**
- Low priority. If Phase 0 passes with last-token, no need to test.

**A7 — One prefix token sufficient**
- If z_dim increases to 128+, multiple prefix tokens become more relevant.

**A8 — Linear injector expressiveness**
- `Linear(z_dim, 1536)` only. Low priority — test if Phase 0 fails with all other fixes applied.

**A-enc-ag — Encoder generalises across chunk indices**
- Same encoder for 1→2 and 2→3 transitions. Test: compare CE loss between first and second transitions after training.

**A-lora — LoRA capacity**
- Runs 1–3 with r=16 showed near-zero probe_rate. Confounded by z_dim + exposure bias but also plausibly a capacity issue.
- Proposed: r=32, lora_alpha=64, lora_dropout=0.05

**A-markov — z is a sufficient statistic** *(core project claim)*
- The entire architecture depends on this. z_h must encode everything from chunks 1..h needed for chunk h+1.
- The context reset enforces this structurally: without the reset, the backbone can attend to prior tokens and ignore z. The reset makes z mandatory.
- Validated when: student probe_rate from z-only input is comparable to teacher reward rate.

**A-reset — Context reset**
- The reset is not a bug — it is the mechanism that makes z informative. Without it, z is optional.
- "Lossless" is the goal state, not a given.

**A-repr2 — repr_h under teacher forcing**
- Teacher-forced chunk-2 hiddens used to compute repr_2 → z_2 during training. At inference, the student generates chunk 2 itself — hiddens will differ.
- This is the exposure bias problem at the representation level. Observed in Phase 0: CE converges but generation fails.

**A-mlen — Markov chain length**
- Deferred. For variable-length atomic steps (1–15 chunks), higher-order dependencies are more plausible for hard problems. Revisit in Phase 1.

**A6 — Exposure bias manageable**
- Known failure. CE loss converges (~1.5 nats) but probe reward stays near 0%.
- Mitigations proposed: `z_tail_tokens`, scheduled sampling, LR warmdown (warmdown implemented; others pending).

---

### Group 3 — Data pipeline

**A-grader — `answers_equivalent` correctness**
- Root cause: `mv_parse()` returns `[]` on failure, not `None`. Old code used `if p is not None and g is not None` — passed `mv_verify(expr, [])` which always returns `False`, silently bypassing the string fallback.
- Fixes applied in `grpo_baseline.py::answers_equivalent`:
  - Truthiness check: `if p and g`
  - `_norm()` applied before `mv_parse`: `\dfrac`/`\tfrac` → `\frac`, unbraced `\frac16` → `\frac{1}{6}`, unbraced `\sqrt5` → `\sqrt{5}`, thin-space `\!`/`\,` removed, thousands-comma removed, whitespace stripped
- Unit test: 15/16 specific failing cases now pass
- Post-fix trace evidence: recovered 3081, 3600, 3737 at L4; 0034, 0111 at L5
- Known unfixable: polynomial factor ordering — `math-verify` cannot expand and reorder; ~1 case/run
- **Verdict: ✅ Confirmed.**

**A-filter — Clean trace identification**
- `clean_trace` boolean field computed per-record in `check_assumptions.py`
- `clean_trace = reward==1 AND no chunk stopped_by in {max_tokens, loop_break}`
- Post-fix counts: L1-L3 28/30, L4 23/30, L5 14/30
- **Verdict: ✅ Confirmed. Volume not a bottleneck for L1-L4.**

---

## Assumption Verification Runs

### Run A — Pre-fix baseline (2026-06-08)

**Files:** `reports/traces_official_l1l3.json`, `reports/traces_official_l4.json`, `reports/traces_official_l5.json`
**Status: Archived. Bugs found and fixed here. See Run B for the final verdict.**

| Level | Correct | Clean | Key issues found |
|-------|---------|-------|-----------------|
| L1-L3 | ~83% | ~90% | Echo bug (0388), restatement loop (2249) |
| L4 | 60% (18/30) | 57% (17/30) | 6 grader bugs, search loop (3697), max_chunks exhausted (3708) |
| L5 | 47% (14/30) | 40% (12/30) | Echo bugs (0261, 0496), grader failures, high max_tokens rate |

Bugs identified and fixed:
- **A-cont echo**: continuation text `"Next step — or \boxed{answer} if done."` contained a literal `\boxed{answer}` → model echoed it verbatim → fixed by rewriting to text-only prompt
- **A-grader**: `mv_parse` returns `[]` not `None` → `if p is not None` always True → `mv_verify(expr, [])` always False → fixed with `if p and g` + full `_norm()` pre-processing
- **A0 loop**: search-space loop in 3697 → fixed by 95% similarity loop detector
- **A1 capacity**: max_chunks=12 too low → raised to 15

---

### Run B — Post-fix verification (2026-06-07)

**Files:** `reports/new traces/traces_official_l1l3.json`, `reports/new traces/traces_official_l4.json`, `reports/new traces/traces_official_l5.json`
**All fixes applied. This is the final verdict.**

| Level | Correct | Clean | Echo | Loop breaks | max_chunks | max_tokens (chunk-level) |
|-------|---------|-------|------|-------------|------------|--------------------------|
| L1-L3 | 29/30 (97%) | 28/30 (93%) | 0 | 0 | 0 | 1/85 (1%) |
| L4 | 25/30 (83%) | 23/30 (77%) | 0 | 0 | 0 | 12/94 (13%) |
| L5 | 16/30 (53%) | 14/30 (47%) | 0 | 0 | 4/30 | 20/126 (16%) |

| Assumption | Verdict | Evidence |
|------------|---------|----------|
| A-cont | ✅ | 0/90 echo violations |
| A-grader | ✅ | 15/16 unit tests pass; +7 L4 +4 L5 recovered vs Run A |
| A0 | ✅ | 0/90 loop breaks; 4 L5 max_chunks are genuine solver failures, not loops |
| A1 | ✅ | max_tokens rate acceptable at L1-L3/L4; all L5 hits on wrong-answer traces or excluded by `clean_trace` |
| A9 | ✅ | L1-L3 97%, L4 83% — sufficient for Phase 0 training data generation |
| A-filter | ✅ | `clean_trace` field computed per-record; 28/23/14 clean per 30 problems |

**Overall verdict: assumptions hold sufficiently at all three levels.** Ready to run Phase 0 again.

---

## check_assumptions.py vs grpo_latent.py — What's Different

`check_assumptions.py` is a standalone verification harness, not the training arm. Every
meaningful difference:

| Dimension | `check_assumptions.py` | `grpo_latent.py` (training arm) |
|-----------|------------------------|----------------------------------|
| **Model** | Qwen3-8B | Qwen2.5-1.5B (from `base_model.yaml`) |
| **Chunk count** | Variable — 1 to 15, driven by natural EOS and `\boxed{}` | Fixed — always 3 chunks (`N_CHUNKS = 3`) |
| **Chunk length** | Variable — ends at EOS; `max_step_tokens=512` is a ceiling | Fixed — `chunk_tokens=341`; no EOS termination |
| **System prompt** | `_ATOMIC_SYSTEM_PROMPT` — explicit one-step rules, STOP instruction | `_LATENT_SYSTEM_PROMPT` — generic "finish current step, box final answer" |
| **Conversation style** | Multi-turn — continuation message each step; context grows | Single-pass — teacher generates 3 chunks sequentially |
| **Temperature** | `0.5` | `1.0` |
| **Qwen3 thinking** | `enable_thinking=False` — suppresses `<think>…</think>` blocks | Not set (Qwen2.5 doesn't have this) |
| **Loop detector** | Yes — 95% SequenceMatcher threshold; sets `stopped_by=loop_break` | No |
| **Asymptote filter** | Yes — skips `[asy]` diagram problems | No |
| **Student context** | N/A | Strict Markov — `[z_prefix \| chunk]` only |
| **min_new_tokens** | `10` (prevents immediate EOS) | `chunk_tokens // 2 ≈ 170` (prevents EOS collapse under z-prefix) |
| **clean_trace flag** | Yes — computed per-record | No |
| **Purpose** | Assumption validation; diagnostic trace data | Actual Phase 0 distillation + Phase 1 GRPO |

**Key implication:** assumptions were verified on a stronger model (8B vs 1.5B) with a
stricter prompt and semantic chunking. The training arm uses Qwen2.5-1.5B with fixed
3×341 hard-cut chunks. A0 and A1 were confirmed in the better setup. Whether the 1.5B
model with the softer prompt produces equally clean chunk boundaries is a known untested
risk — as noted in the README, this is the motivation for eventually moving to 7B/14B.

---

## Phase 0 Success Criterion

Student (LoRA + encoder + Z-injector) must generate full reasoning from `z_prefix` only
on L1-L4, with reward rate comparable to teacher. CE loss convergence is necessary but
not sufficient — the gate is student vs teacher reward rate on held-out problems.

| Metric | Threshold | Notes |
|--------|-----------|-------|
| `student_reward_rate` | ≥ 10% | Absolute floor |
| `coverage_ratio` | ≥ 30% of teacher | Student/teacher rate ratio |
| `chunk1_crutch_rate` | diagnostic only | High crutch = z broken for chunks 2–3 |
| `mean_ce_loss` | ≤ 2.0 | Sanity check; not a pass criterion |
| `mean_z_std` | ≥ 0.1 | z not collapsed |

---

## Proposed Next Run

All ground assumptions confirmed. The three changes below address the three remaining
blockers from the root-cause map.

```yaml
latent_markov:
  latent_dim: 128           # was 64 — addresses A5 (24× compression too aggressive)

phase0:
  n_steps: 800
  lora:
    r: 32                   # was 16 — addresses A-lora
    lora_alpha: 64
    lora_dropout: 0.05
  lr_lora: 2.0e-4           # was 3e-4 — lower to compensate for higher capacity
  lr_warmdown_steps: 250
  lr_warmdown_final: 0.1
  z_anchor_tokens: 32
  z_tail_tokens: 32         # NOT IMPLEMENTED YET — add to _distill_loss
```

Also: lower `min_new_tokens` in `generate_latent_traces()` once tail loss is in place.

---

## Pending Fixes (not yet implemented)

| Fix | Rationale | Target |
|-----|-----------|--------|
| `z_tail_tokens=32` | `\boxed{}` lives at tail; tail never supervised under z_anchor-only regime | `_distill_loss` |
| Hybrid CE (head + tail, skip middle) | Middle tokens dilute z signal; head + tail both matter for answer quality | `_distill_loss` |
| Scheduled sampling | Close teacher-forcing / autoregressive gap after token K | `_distill_loss` or wrapper |
| Save best-probe checkpoint | Best quality was step ~200, not final — don't overwrite it | `pretrain_distill` |
| Lower `min_new_tokens` | =170 forces long garbage output; try 32 | `generate_latent_traces()` |

## Implemented Fixes (since Run 1)

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
| Grader: `if p and g` + full `_norm()` | `grpo_baseline.py::answers_equivalent` | Root-cause fix + pre-normalisation before `mv_parse` |
| Continuation message v3 | `check_assumptions.py` | Removed `\boxed{}` from prompt text; echo physically impossible |
| Loop detector (95% SequenceMatcher) | `check_assumptions.py` | Catches restatement and search-space loops |
| max_chunks raised 12→15 | `check_assumptions.py` | Capacity for longer enumeration problems |
| `clean_trace` field per record | `check_assumptions.py` | `reward==1 AND no max_tokens AND no loop_break`; safe-to-train flag |
| Asymptote filter in `_load_pool` | `check_assumptions.py` | Skip problems with `[asy]` diagrams |

### Rejected / Deferred

| Idea | Reason |
|------|--------|
| Classical simulated annealing | Doesn't map to gradient CE training; LR warmdown covers this |
| LoRA rank ↑ alone (without z_dim ↑) | Over-adaptation risk; dynamics > capacity; must pair with z_dim |
| LoRA rank ↑ without warmdown/dropout | Same reason |

---

## Open Questions

1. Is z_dim=128 enough, or do we need 256 + multi-token injection?
2. Can `z_tail_tokens` alone fix `\boxed{}` without full middle supervision?
3. Should Phase 0 include a small on-policy loss (mix own rollouts into CE) before Phase 1?
4. Should sanity gate on median problem reward instead of mean, given heavy-tailed easy pool?
5. Is a lightweight `L_trans` worth reinstating as a soft auxiliary loss once `latent_dim` increases? The v2 result (8.42%) may partly have been `L_trans` enforcing temporal latent consistency that CE distillation alone cannot replicate.

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

## Related Files

| File | Purpose |
|------|---------|
| `reports/latent_markov_design.md` | Official architecture design (v3) |
| `reports/ablation_core.md` | Official results table (baseline: 16.19% pass@128) |
| `reports/NEXT_STEPS_V2.md` | v2 design decisions and v1→v2 migration rationale |
| `src/training/grpo_latent.py` | Phase 0/1 training loops |
| `scripts/run_phase0_sanity.py` | Post-training sanity checks |
| `scripts/check_assumptions.py` | Teacher trace generation & assumption verification |
| `src/training/grpo_baseline.py` | Reward function, grader |
| `configs/train_latent_grpo.yaml` | Main config |
| `src/models/vae_state_encoder.py` | Encoder (1536→512→128→z_dim) |
