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
- [x] **Token-Markov:** Delethink-style RL-learned carryover implemented; chunk boundary + max carryover length documented; isolated code path (`train_token_markov.py`, `token_markov_state.py`, `grpo_token_markov.py`); RL training budget matches baseline exactly. Pretrained instruct model scores 12.5% under both regimes (clean control). After 200 steps GRPO: baseline 12.5%→15.0% (signal received); token-Markov ≈12.5% (zero gradient for all 200 steps — stable fixed point at reference; SHA256 confirms no weight change; apparent 5.0% eval is noise). Full arithmetic in `reports/writeup_stubs.md`.
- [ ] **Markov diagnostic (latent arms):** empirical evidence that `z_h` satisfies the Markov property — latent transition loss (`f(z_h) → z_{h+1}` without history), last-state-only ablation, latent variance vs. problem difficulty correlation. See `reports/writeup_stubs.md`.
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

**Deliverables:** `token_markov_state.py`, `train_token_markov.py`, `grpo_token_markov.py`, `configs/train_token_markov_grpo.yaml`, eval artifacts for pretrained + trained variants of both arms.

**Results (200-step run, final):**
- `baseline_pretrained`: 12.5% — instruct model without GRPO (capability floor)
- `token_markov_pretrained`: 12.5% — same model, Delethink regime (clean control; equal capability at init)
- `baseline_grpo` checkpoint-200: **15.0%** — confirmed across two identical eval runs; genuine +2.5pp from 1–2 gradient events
- `token_markov_grpo` checkpoint-200: **≈12.5%** — SHA256 = pretrained weights; zero gradient for all 200 steps (zero reward → zero ppo_term; curr=ref throughout → zero kl_term — stable fixed point); local-path eval noisy (2–3/40 problems at noise floor); true value = pretrained baseline under Delethink regime

**Phase 2 complete.**

---

## Phase 3 🔄 — Latent Markov arm (`latent_grpo`) — v2 redesign, running from scratch

**v2 redesign (2026-05-11):** Phase 1 produced 7.5% pass@1024 (below baseline 15%). Six root causes identified and fixed before re-running. Full rationale in `reports/latent_markov_design.md §Redesign v2`. Summary:
1. **lambda_vae bug fixed** — was running at 1.0 (dead config), now wired at 0.05 from YAML.
2. **Gradient flow restored** — L_RL now flows through encoder via fresh z_h (not detached trace z). All three loss terms update the encoder jointly.
3. **batch_size 1 → 4** — 4× more problem-encounters per step, closing the reward-density gap vs the baseline.
4. **λ_t schedule flipped** — warmup (0.1 → 0.5) instead of decay (1.0 → 0.1). Avoids locking encoder into Markov structure before z carries signal.
5. **KL annealing added** — kl_weight 0 → 1 over first 50% of Phase 0 steps. Prevents posterior collapse that made σ² useless as uncertainty signal.
6. **Phase 0 pool upgraded** — L1–L5 (was L1–L3). Includes harder trajectories to bridge distributional gap to Phase 1 MATH-B-I problems.

All Phase 3 artifact items below are reset to [ ] and must be re-run in order.

**Design doc:** `reports/latent_markov_design.md` (authoritative). Implementation steps and ordering in that doc's "Implementation Deliverables" table.

**Deliverables — `latent_grpo` arm (v2):**
- [ ] Phase 0 pretraining dataset: `data/math_easy_pool.jsonl` — L1–L5 from `EleutherAI/hendrycks_math` (v2: was L1-L3 only); re-run `scripts/prepare_easy_pool.py` to regenerate
- [ ] Phase 0 rollouts: `data/phase0_rollouts.pt` — re-run `scripts/generate_phase0_rollouts.py` on new pool
- [x] `src/models/vae_state_encoder.py` — `VAEStateEncoder` (encoder, decoder, transition) + `OutcomeHead` (Phase 0 only); `compute_elbo` now accepts `kl_weight`
- [x] `src/training/grpo_latent.py` — all Phase 0 + Phase 1 training loops (v2: kl warmup, lambda_trans warmup, gradient flow fix, lambda_vae wired, batch_size=4)
- [x] `configs/train_latent_grpo_smoke.yaml` — smoke config (needs re-smoke after code changes)
- [x] `configs/train_latent_grpo.yaml` — full Phase 1 config (v2: batch_size=4, kl_warmup_frac=0.5, lambda_vae wired)
- [x] `scripts/check_latent_structure.py` — t-SNE/UMAP of z_final (NFR6 gate)
- [x] Latent generation mode in `scripts/eval_passk.py` (including `latent_markov_pretrained` for controlled baseline)
- [ ] Phase 0 VAE pretraining: VAE checkpoint → `runs/latent_grpo/phase0_vae.pt`
- [ ] NFR6 gate: UMAP of z_final shows structured manifold (re-run after new Phase 0)
- [ ] **Controlled latent baseline** — eval `latent_grpo_pretrained`: Phase 0 VAE + pretrained backbone (no Phase 1 updates). This is the latent-regime capability floor, equivalent to `baseline_pretrained` and `token_markov_pretrained` for the other arms. Must be evaluated before Phase 1 training begins so the Phase 1 improvement is measured against a real baseline, not inferred.
- [ ] Phase 1 training: 200 steps on MATH-B-I, checkpoint → `artifacts/latent_grpo/{run_id}/`
- [ ] Phase 1 eval: pass@k on MATH-B-I holdout; result logged in `reports/ablation_core.md`

**v1 Phase 0 artifacts (A100 run, 2026-05-09 — superseded by v2):**
- Rollouts: `data/phase0_rollouts.pt` — 23,792 trajectories (2974 L1-L3 problems × G=8), reward rate 38.6%
- VAE checkpoint: `runs/latent_grpo/phase0_vae.pt`
- NFR6 plot: `runs/latent_grpo/plots/latent_structure_umap.png`
- These artifacts are from the v1 run. Re-running from scratch with v2 design; these will be overwritten.

**Pass criteria — `latent_grpo` arm (v2):**
- [ ] Smoke test completes end-to-end in < 10 min on 4060 (re-smoke after v2 code changes)
- [ ] **NFR6 gate:** UMAP of z_final shows structured manifold (re-run on v2 Phase 0 checkpoint)
- [ ] `latent_grpo_pretrained` eval (controlled baseline): pass@1024 ≥ pretrained baseline (confirms z injection does not degrade capability before Phase 1)
- [ ] Phase 1 training log shows L_transition non-zero from step 0; L_RL non-zero within first 30 steps; lambda_vae=0.05 confirmed in log
- [ ] `pass@1024` for `latent_grpo` ≥ 18.0% (≥ 3pp above baseline_grpo — required to clear noise floor on 40-problem pool)
- [ ] No NaN blowups under zero-reward stretches (R6.5)
- [ ] Shared hyperparameters (G=8, lr=1e-6, 200 steps, same backbone) match all other arms

**Additional pass criterion — Markov diagnostic (required for paper):** empirical evidence that `z_h` satisfies the Markov property. Without this, the claim is an assertion, not a result:
- **E1 — Transition sufficiency:** held-out transition loss (f(z_h) → z_{h+1} without history) → `scripts/eval_markov_diagnostics.py`
- **E2 — Policy sufficiency:** last-state-only ablation (latent arm pass@1024 vs baseline) → covered by ablation table via `eval_passk.py`; no separate script needed
- **E3 — Uncertainty calibration:** σ_h² correlation with outcome → `scripts/eval_markov_diagnostics.py`

## Phase 3b 🔲 — Uncertainty arm (`latent_grpo_uncertainty`)

**Status:** design not yet started. Begins after `latent_grpo` arm is complete and results are logged.

**Design doc:** to be written. Context preserved in `reports/SCRATCH_latent_design_rough.md` (Uncertainty Arm section).

**Deliverables:** `src/training/reward_bonus.py` — `compute_uncertainty_bonus()`; `configs/train_latent_grpo_uncertainty.yaml`; `train_latent_with_uncertainty()` in `grpo_latent.py`; eval artifacts.

**Pass:** same criteria as `latent_grpo` arm + uncertainty bonus active (β_t schedule documented); E3 diagnostic completed (σ_h² validated as exploration signal, not noise).

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
