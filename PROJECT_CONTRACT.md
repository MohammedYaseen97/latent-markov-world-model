# Project Contract (v2)

Phases, gates, and what to ship. **Vision / hypothesis / architecture:** `PROJECT_BRIEF.md`.
**Data pools, manifest specs:** `reports/DATA_PROTOCOL.md`.
**v2 rationale and research ladder:** `reports/NEXT_STEPS_V2.md`.

---

## Who owns which doc

| Topic | File |
|--------|------|
| Why, narrative, related work | `PROJECT_BRIEF.md` |
| Phases, repo layout, checklist, gates | this file |
| Pool files, manifest spec, mutual exclusivity | `reports/DATA_PROTOCOL.md` |
| v2 rationale, architecture changes, run order | `reports/NEXT_STEPS_V2.md` |
| Latent arm full design, gradient flow, pipeline diagrams | `reports/latent_markov_design.md` |
| Future work, paper draft stubs | `reports/writeup_stubs.md` |

---

## Locked scope (v2)

- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` + TRL + GRPO (see `configs/base_model.yaml`).
  Fallback only if blocked: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — apply to
  ALL arms if switched and document in run log.
- **Benchmark:** MATH Level 5 hard pool — definition and build in `reports/DATA_PROTOCOL.md`,
  path in `configs/eval_math_beyond.yaml`.
- **Three active arms (core table):**
  1. `baseline_grpo` — history-as-state  
  2. `token_markov_grpo` — Delethink-style RL-learned textual carryover (Markovian Thinker, ICLR 2026)
  3. `latent_grpo` — deterministic latent Markov state with calibrated uncertainty (no bonus)
  4. `latent_grpo_uncertainty` — latent + KL variance exploration bonus (stub — not yet implemented)
- **Primary metric:** `pass@128` on the Level 5 hard pool (all arms).
  Also report `pass@1`, `pass@16`.
- **Out of scope for this phase:** second model family, second RL algorithm,
  additional benchmarks, diffusion upgrade (rung 3). Ideas → `reports/writeup_stubs.md`.

**Fair comparison:** same data, eval protocol, training budgets, and shared
hyperparameters across arms unless the field is explicitly method-specific
(Markov module, encoder, uncertainty bonus).

---

## Pass criteria (v2)

| Criterion | Target |
|---|---|
| Easy pool: Level 1–4 only | No Level 5 `problem_id`s in easy pool manifest |
| Hard pool: Level 5 only | All rows have `source_level=5`; pass@128=0 for pretrained model |
| Pools mutually exclusive | `problem_id` intersection = ∅ |
| NFR6 gate | UMAP shows outcome-correlated geometry in z-space |
| `latent_grpo_pretrained` pass@128 | ≥ `baseline_pretrained` pass@128 (controlled regime check) |
| `latent_grpo` pass@128 | ≥ `baseline_grpo` pass@128 + 3pp |
| **E1 + E3 joint (solution-space check)** | E1 < 0.5 AND E3 Pearson r < −0.1 AND Δσ²(correct vs incorrect) > 0.01 |
| No NaN blowups | L_RL, L_trans, L_calib all non-zero throughout Phase 1 |

---

## Phase 0 ☐ — Pool preparation

**Goal:** build both data pools before any training begins.

- [ ] **Easy pool (Level 1–4):** `python scripts/prepare_easy_pool.py --levels 1 2 3 4`
  - Verify: `data/easy_pool_manifest.json` shows `levels_included: [1, 2, 3, 4]`
  - Verify: no Level 5 `problem_id`s in output

- [ ] **Level 5 hard pool:** `python scripts/prepare_math_level5_pool.py`
  - Verify: `data/level5_pool_manifest.json` exists with `source_level: 5`
  - Verify: `row_count` ~350 (expect 300–400 after filtering)
  - Verify: SHA-256 recorded in manifest

- [ ] **Mutual exclusivity:** `problem_id` intersection of both pools = ∅

---

## Phase 1 ☐ — Baseline arm (`baseline_grpo`)

**Deliverables:** train 200 GRPO steps on Level 5 hard pool → eval pass@128.

- [ ] Training: `python scripts/train_baseline.py --config configs/train_baseline_grpo.yaml`
- [ ] Eval: `python scripts/eval_passk.py --generation-mode baseline --arm baseline_grpo ...`
- [ ] Artifact: `artifacts/baseline_grpo/{run_id}/eval_metrics.json`
- [ ] Log: L_RL non-zero within first 30 steps

**Pass:** stable pass@128 with non-zero improvement over pretrained baseline.

---

## Phase 2 ☐ — Token-Markov arm (`token_markov_grpo`)

**Design:** Delethink-style RL-learned textual carryover (Algorithm 1, Markovian Thinker ICLR 2026).
Context reset at each chunk boundary; last m=256 tokens carry forward. C=512, I=3.

**Note on v1 result:** Token-Markov training produced zero gradient for all 200 steps
in v1 (SHA256 checkpoint = pretrained weights). The reward sparsity under the Delethink
regime (per-sample success ≈ 0%) creates a stable fixed point. v2 uses the Level 5 hard
pool which has the same structural sparsity issue. The controlled pretrained eval
(`token_markov_pretrained` pass@128) is the key check — if it matches `baseline_pretrained`,
the regime is sound and the zero-gradient outcome (if it recurs) is attributable to the
method, not implementation.

- [ ] Training: `python scripts/train_token_markov.py --config configs/train_token_markov_grpo.yaml`
- [ ] Eval: `python scripts/eval_passk.py --generation-mode token_markov ...`
- [ ] Artifact: `artifacts/token_markov_grpo/{run_id}/eval_metrics.json`
- [ ] Controlled pretrained eval: `token_markov_pretrained` pass@128 ≥ `baseline_pretrained` pass@128

**Pass:** eval complete; result interpreted honestly (zero improvement accepted if SHA256 proof provided).

---

## Phase 3 ☐ — Latent Markov arm (`latent_grpo`)

**Design doc:** `reports/latent_markov_design.md` (authoritative).

**Deliverables:**

### Phase 0 — Encoder pretraining (400 steps)

- [ ] Run: `python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 0`
- [ ] Checkpoint: `runs/latent_grpo/phase0_encoder.pt`
- [ ] Loss: L_trans declining; L_calib declining (σ² learning to correlate with reward)
- [ ] Log: reward rate ~15–20% on easy pool (calibration check)

### NFR6 gate

- [ ] Run: `python scripts/run_nfr6_gate.py --config configs/train_latent_grpo.yaml --n-problems 200 --n-rollouts 2`
- [ ] **PASS criterion:** UMAP manifold shows outcome-correlated geometry (correct/incorrect not uniformly mixed)
- [ ] Do NOT proceed to Phase 1 on gate failure — diagnose Phase 0 first

### Controlled latent baseline eval

- [ ] Run: `python scripts/eval_passk.py --generation-mode latent_markov_pretrained ...`
- [ ] **PASS criterion:** pass@128 ≥ baseline_pretrained pass@128 (ZInjector near-zero init check)

### Phase 1 — Joint RL (200 steps)

- [ ] Run: `python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 1`
- [ ] Log: L_trans non-zero from step 0; L_RL non-zero within first 30 steps; L_calib non-zero
- [ ] Log: λ_trans=0.3 confirmed (maintenance mode); λ_calib=0.5 confirmed

### Phase 1 eval

- [ ] Run: `python scripts/eval_passk.py --generation-mode latent_markov ...`
- [ ] Artifact: `artifacts/latent_grpo/{run_id}/phase1/final/eval_metrics.json`
- [ ] **Target:** pass@128 ≥ baseline_grpo pass@128 + 3pp

### E1 + E3 Markov diagnostics

- [ ] Run: `python scripts/eval_markov_diagnostics.py --checkpoint $PHASE1_CKPT ...`
- [ ] **E1:** held-out L_trans < 0.5 (Markov property holds on unseen trajectories)
- [ ] **E3 joint:** Pearson r < −0.1 AND Δσ²(correct vs incorrect) > 0.01
  - Both conditions required for solution-space interpretation (see `reports/NEXT_STEPS_V2.md`)

**Pass:** E1 + E3 joint criterion met; pass@128 result interpreted in context of diagnostics.

---

## Phase 3b ☐ — Uncertainty arm (`latent_grpo_uncertainty`)

**Status:** stub — not yet started. Begins after Phase 3 is complete and results logged.

**Deliverables:**
- `src/training/reward_bonus.py` — implement `compute_uncertainty_bonus()`
- `configs/train_latent_grpo_uncertainty.yaml` — fill in null fields
- `train_latent_with_uncertainty()` in `grpo_latent.py`
- Eval artifacts

**Pass:** same criteria as Phase 3 + uncertainty bonus active; E3 diagnostic validates
σ² as an exploration signal.

---

## Phase 4 ☐ — Table + ship

**Deliverables:**
- `scripts/run_ablation_table.py` → `reports/ablation_core.md` (auto-generated from artifacts)
- Commands, configs, seeds, artifact paths documented
- README updated with final run commands

**Pass:** table generated from artifacts (not hand-typed); `pass@128` for all three
active arms; E1 + E3 results for latent arm; honest positive or negative outcome with
evidence; `PROJECT_CONTRACT.md` all checklist items complete.

**Also per brief:** public repo, short paper, blog post; bonus: latent visualisation.

---

## Checklist: trust the result

Before treating the core table as final:

- [ ] **Data:** `DATA_PROTOCOL.md` + manifest files + config paths all consistent
- [ ] **Token-Markov:** pretrained arm matches baseline at initialisation (regime sanity check)
- [ ] **Latent diagnostics:** E1 + E3 joint criterion met (solution-space evidence)
- [ ] **Fairness:** same pretrained checkpoint, same reward, same eval budget, same decode settings
- [ ] **Metrics:** `pass@128` all active arms; table from `run_ablation_table.py` artifacts
- [ ] **Repro:** seeds in `repro_tolerance.yaml`; `check_reproducibility.py` verified

---

## Repository layout

**Configs:** `base_model.yaml`, `train_*_grpo.yaml` (4 arms), `eval_math_beyond.yaml`, `repro_tolerance.yaml`.

**Scripts:** `prepare_easy_pool.py`, `prepare_math_level5_pool.py`, `train_baseline.py`,
`train_token_markov.py`, `train_latent.py`, `eval_passk.py`, `run_ablation_table.py`,
`eval_markov_diagnostics.py`, `run_nfr6_gate.py`, `check_reproducibility.py`, `calibrate_pool.py`.

**Packages:** `src/models/{token_markov_state,vae_state_encoder}.py`,
`src/training/{grpo_baseline,grpo_token_markov,grpo_latent,reward_bonus}.py`,
`src/utils/{config_loader,seeding}.py`.

**Dirs:** `configs/`, `data/`, `scripts/`, `src/`, `artifacts/`, `reports/`.

---

## Scope guardrails

No extra model families, algorithms, or benchmark expansion until the core three-arm
table is done. Extra ideas → `reports/writeup_stubs.md` (Future work).

## Related work positioning (writeup constraint)

**Reasoning Palette** is the closest adjacent paper. Differentiator must be explicit:
their VAE is sampled once per problem as a strategy prefix before generation; our `z_h`
evolves step-wise during the rollout as a Markov state. Different MDP formulation, not
just a different architecture.

**The load-bearing frame is the MDP reformulation, not the encoder architecture.**
The contribution is replacing history-as-state with a compact learned Markov state —
grounded in Yuan et al.'s theoretical guarantee. Every section leads with this, not
with "we train an encoder."

## Timeline

Phase-gated only — no calendar obligation.
Compute estimate (RTX Pro 6000 Blackwell): ~14h total (see `reports/NEXT_STEPS_V2.md`).
