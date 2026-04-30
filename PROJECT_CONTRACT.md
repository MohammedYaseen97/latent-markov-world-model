# Project Contract

Phases, gates, and what to ship. **Vision / hypothesis / VAE design:** `PROJECT_BRIEF.md`. **How MATH-B JSONLs are built:** `reports/DATA_PROTOCOL.md`.

---

## Who owns which doc

| Topic | File |
|--------|------|
| Why, narrative, stack sketch, risks | `PROJECT_BRIEF.md` |
| Phases, repo layout, checklist, Yuan reference | this file |
| Benchmark files, Hub pin, manifest | `reports/DATA_PROTOCOL.md` |
| Future ideas / draft results | `reports/writeup_stubs.md` |

---

## Locked scope

- **Model:** **`Qwen/Qwen2.5-1.5B-Instruct`** + `trl` + `GRPO` (see `configs/base_model.yaml`). Fallback only if blocked: **`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`** (instruct-style distill — use for **all** arms if switched; note in run log). Pool membership still comes from MATH-B-I **base** Hub columns; training checkpoint choice is separate (documented in `DATA_PROTOCOL.md`).
- **Benchmark:** MATH-Beyond **MATH-B-I base** pool — definition and row count in `reports/DATA_PROTOCOL.md`, paths in `configs/eval_math_beyond.yaml` / `configs/final_parity/base_parity.yaml`.
- **Four arms (core table):**  
  1. `baseline_grpo` — history-as-state  
  2. `token_markov_grpo` — **Delethink-style** RL-learned textual carryover (Markovian Thinker, ICLR 2026). Implements Yuan et al.'s Markovian state compression idea for open-ended math where symbolic states are undefined. NOT Yuan's symbolic state predictor — that requires a ground-truth symbolic environment (board games) unavailable for competition math.  
  3. `latent_grpo` — VAE latent, no uncertainty bonus  
  4. `latent_grpo_uncertainty` — latent + KL bonus  
- **Metrics:** report **`pass@1024` for all arms** (primary); also `pass@1`, `pass@16`.
- **Out of scope for this phase:** second model family, second RL algo, diffusion/persona branches, extra benchmarks before the core table.

**Fair comparison:** same data, eval protocol, budgets, and shared hyperparameters across arms unless the field is explicitly method-specific (Markov module, VAE, uncertainty). Yuan’s paper motivates the token-Markov **idea**; we do **not** match their puzzle benchmark or treat their numbers as success criteria.

**Training budget — 200 steps (all four arms):** GRPO on MATH-Beyond is a sparse-reward problem. With P(pass@1) ≈ 10% for the base model and G=8 completions per problem, P(≥1 correct in a group) ≈ 57% per problem. With 40 training problems and batch_size=8, each step sees ~5 problems — roughly half the steps in a short run carry near-zero gradient. At 50–100 steps the expected number of reward-bearing encounters is too small to move the policy reliably. At 200 steps (token-Markov: 5 steps/epoch × 40 epochs = 200 naturally; TRL arms: max_steps=200) the expected reward-bearing encounters reach ~570 — enough for a meaningful gradient signal while keeping per-problem repetition at 40× where stochastic sampling (temperature=1.0) still ensures diversity across encounters. Going beyond 200 risks overfitting on 40 problems without a commensurate gain in signal density. The budget is enforced identically for all arms via the extends config chain (train_baseline_grpo.yaml → train_token_markov_grpo.yaml → future latent configs).

**Yuan (informative only):** KL coef ~0.001, RL LR ~1e-6, batch / mini-batch ~128, group size 8, decode temp 1.0; they use rLLM+VERL and Pass@128 on logic tasks. We use **MATH-B** and **pass@1024**. Deviations are fine; document anything large.

**Frozen run configs:** `configs/final_parity/` (folder name is legacy) + `configs/repro_tolerance.yaml` for same-seed tolerance.

---

## Checklist: trust the 4-arm result

Before treating the core table as final:

- [x] **Data:** `DATA_PROTOCOL` + `prepare_data.py` + `benchmark_manifest.json` + config paths aligned.
- [x] **Token-Markov:** Delethink-style RL-learned carryover implemented; chunk boundary + max carryover length documented; isolated code path (`train_token_markov.py`, `token_markov_state.py`, `grpo_token_markov.py`); RL training budget matches baseline exactly. Eval complete: `pass@1024=7.5%` (3-chunk, m=256). RL training yielded 0 gradient signal (reward sparsity at G=8 — expected given per-sample pass rate ≈ 0.01%; result and story documented in `reports/writeup_stubs.md`).
- [ ] **Markov diagnostic (latent arms):** empirical evidence that `z_h` satisfies the Markov property — latent transition loss (`z_h + a_h → z_{h+1}` without history), last-state-only ablation, latent variance vs. problem difficulty correlation. See `reports/writeup_stubs.md`.
- [ ] **Fairness:** same pretrained checkpoint across arms (`configs/base_model.yaml`), MATH-B pool, reward, train/eval budgets, max length, decode settings (unless documented method-specific).
- [ ] **Metrics:** `pass@1024` all arms; table from `run_ablation_table.py` artifacts, not hand-typed.
- [x] **Repro:** seeds + tolerances in `repro_tolerance.yaml`; seeding stack verified bit-for-bit on smoke (entropy + completion hashes identical across runs); `check_reproducibility.py` in place; checkpoints/log infrastructure confirmed.

**Out of scope:** replicating Yuan’s ablations, matching every Yuan hyperparameter, extra benchmarks before the matrix is done. Park ideas in `reports/writeup_stubs.md`.

---

## Phase 1 ✅ — Baseline + data + eval

**Deliverables:** (1) deps pinned (`requirements.txt`), (2) `prepare_data.py` outputs per `DATA_PROTOCOL`, (3) `train_baseline.py` smoke + A100, (4) `eval_passk.py` for k ∈ {1,16,1024}, (5) artifacts under `artifacts/baseline_grpo/{run_id}/`.

**Implementation status:** all five deliverables implemented and verified. Re-running at 200 steps (budget decision — see Fair Comparison note above). Previous 50-step run retired.

**Pass:** stable A100 baseline; valid pass@ metrics; same-seed rerun within `repro_tolerance.yaml`; README has train/eval commands.

---

## Phase 2 ✅ — Token-Markov arm

**Design decision (post-contract):** Yuan et al.'s symbolic state predictor requires a ground-truth environment to render `s_h` (board configurations). MATH-Beyond competition problems have no analogous symbolic state — reasoning is non-monotonic, non-compositional, and cannot be verified step-by-step. Yuan's mechanism cannot run on this dataset.

**Implementation: Delethink-style RL-learned textual carryover** (Markovian Thinker, ICLR 2026). At each chunk boundary, the policy writes a bounded-length textual state summary. Context is reset; only the carryover is retained. GRPO trains the policy to write summaries sufficient for seamless continuation. No separate transition model. This instantiates the same theoretical idea as Yuan (compact Markov state replaces unbounded history) using a mechanism that generalises to open-ended math.

**Design questions — resolved:**
1. ✅ Chunk boundary: fixed token count (every C tokens). Paragraph-level is ill-defined for MATH-B; per-reasoning-attempt requires a step detector. Fixed tokens matches the paper exactly.
2. ✅ Carryover length: m = C/2 = 256 tokens, derived from the 1024-token budget constraint (C=512, I=3). Unbounded carryover concern addressed — m is strictly bounded and documented in `reports/token_markov_design.md`.
3. ✅ Context reset: implemented via a custom training loop in `src/training/grpo_token_markov.py` that bypasses TRL's `GRPOTrainer`. TRL assumes one continuous sequence per rollout; Delethink's multi-chunk generation is incompatible. Custom loop follows Algorithm 1 from the paper.
4. ✅ Carryover format: freeform (raw last-m tokens of previous chunk). No hand-engineered structure — the model learns what to write via RL reward signal.

**Framing note:** this arm tests "does a textual learned Markov state beat history-as-state on hard math?" It does **not** test Yuan's architecture specifically. In the paper: *"We implement the token-space Markov arm using Delethink's RL-learned textual carryover, which extends Yuan et al.'s Markovian state compression to open-ended reasoning where symbolic states are undefined."*

**Deliverables:** `token_markov_state.py`, `train_token_markov.py`, `grpo_token_markov.py`, `configs/train_token_markov_grpo.yaml`. Re-running at 200 steps (budget decision — see Fair Comparison note above). Previous 50-step run retired.

**Phase 2 implementation complete; re-running training for fair 200-step comparison.**

---

## Phase 3 🔲 — Latent + uncertainty

**Deliverables:** `vae_state_encoder.py`, `grpo_latent.py` (both modes), `reward_bonus.py`, final parity YAMLs for both latent arms.

**Pass:** both arms complete; no NaN blowups; `pass@1024` for **all four** arms; shared knobs match except method-specific parts.

**Additional pass criterion — Markov diagnostic (required):** empirical evidence that `z_h` satisfies the Markov property. Without this, the claim that the VAE learns a Markov state is an assertion, not a result. Minimum bar:
- **Transition sufficiency:** `z_h` + `a_h` predicts `z_{h+1}` without history access (latent transition loss).
- **Policy sufficiency:** policy conditioned only on `z_h` performs comparably to one with full history access (last-state-only ablation, mirroring Yuan et al. Table 4).

Latent variance as exploration signal (`σ_h²`) also requires its own validation: does it correlate with problem difficulty or solution uncertainty, or is it just reconstruction noise?

---

## Phase 4 🔲 — Table + ship

**Deliverables:** `run_ablation_table.py` → `reports/ablation_core.csv` / `.md`; commands, configs, seeds, artifact paths documented.

**Pass:** table from artifacts; `pass@1024` for all four arms; checklist above complete; clear positive **or** negative outcome with evidence.

**Also per brief:** public repo, short paper, blog post; bonus: latent viz.

---

## Repository layout (must exist)

**Configs:** `base_model.yaml`, `train_*_grpo.yaml` (4 arms), `eval_math_beyond.yaml`, `configs/final_parity/*`, `repro_tolerance.yaml`.

**Scripts:** `prepare_data.py`, `train_baseline.py`, `train_token_markov.py`, `train_latent.py`, `eval_passk.py`, `run_ablation_table.py`.

**Packages:** `src/models/{token_markov_state,vae_state_encoder}.py`, `src/training/{grpo_baseline,grpo_token_markov,grpo_latent,reward_bonus}.py`, `src/eval/metrics.py`, `src/utils/{logging,seeding}.py`.

**Dirs:** `configs/`, `data/`, `scripts/`, `src/`, `artifacts/`, `reports/`.

---

## Rubric (optional, mentor-style)

Roughly: framing (15) · baselines + token-Markov (25) · VAE correctness (20) · rigor / matched budgets (20) · honest interpretation (10) · reproducibility (10). Target ≥ 80 for “publishable shape.”

---

## Scope guardrails

No extra model families, algorithms, or benchmark expansion until the core four-arm table is done. Extra ideas → `reports/writeup_stubs.md` (**Future work**).

## Related work positioning (writeup constraint)

**Reasoning Palette** (VAE latent as reasoning strategy prefix) is the closest adjacent paper and a potential reviewer landmine. The differentiator must be explicit: their VAE is sampled once per problem as a strategy prefix before generation; our `z_h` evolves step-wise during the rollout as a Markov state. Different MDP formulation, not just a different architecture.

**The load-bearing frame is the MDP reformulation, not the VAE architecture.** The VAE is the implementation vehicle. The contribution is replacing history-as-state with a compact learned Markov state — grounded in Yuan et al.'s theoretical guarantee. Every section of the writeup should lead with this, not with "we train a VAE."

---

## Timeline

Phase-gated only — no calendar obligation. A loose week sketch lives in `PROJECT_BRIEF.md`; **this file wins** if they disagree.
