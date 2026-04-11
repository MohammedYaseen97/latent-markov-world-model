# Project Contract: Phase-Gated Implementational Deliverables (Brief-Aligned)

This document is the execution contract.  
`PROJECT_BRIEF.md` is the single source of truth for project vision and implementation intent.  
Timeline is phase-gated, not calendar-gated: do not enter the next phase until current phase criteria pass.
`PROJECT_SCOPE_RUBRIC_CHECKLIST.md` and `YUAN_PARITY_CHECKLIST.md` are enforcement checklists for this contract.

## Document Authority Model (Single-Edit Policy)

Edit exactly one owner document per change type:

- Research vision, hypothesis, scientific narrative -> `PROJECT_BRIEF.md`
- Execution phases, required implementation tasks, phase pass/fail gates -> `PROJECT_CONTRACT.md`
- Yuan parity requirements and claim-eligibility gate -> `YUAN_PARITY_CHECKLIST.md`
- Grading rubric and project completion checklist -> `PROJECT_SCOPE_RUBRIC_CHECKLIST.md`
- MATH-B JSONL construction and benchmark definitions -> `reports/DATA_PROTOCOL.md`
- Future-work / results / paper scratch placeholders -> `reports/writeup_stubs.md`

Non-owner documents must reference owner documents and must not restate policy text unless required for local clarity.

## Locked Scope (Non-Negotiable)

- Base model family: `Qwen2.5-1.5B` track (primary) with `trl` + `GRPO`.
- Allowed fallback (only on explicit blocker): `DeepSeek-R1-Distill-1.5B`, documented in parity and final report.
- Primary benchmark: `MATH-Beyond` **MATH-B-I base** pool (definition and row count: `reports/DATA_PROTOCOL.md`, paths: `configs/eval_math_beyond.yaml`).
- Mandatory experiment matrix (exactly 4 arms for core table):
  1. `baseline_grpo` (history-as-state)
  2. `token_markov_grpo` (Yuan-style token-space Markov state baseline)
  3. `latent_grpo` (VAE latent state, no uncertainty bonus)
  4. `latent_grpo_uncertainty` (VAE latent + uncertainty bonus)
- Core claim metric: `pass@1024` (must be reported for all 4 arms).
- Secondary metrics allowed: `pass@1`, `pass@16`, sample efficiency curves.
- No diffusion, no persona-conditioning, no extra model families in this project phase.

---

## Parameter Parity Rule (Yuan-Equivalent Finals)

Final comparison runs must use Yuan-equivalent settings wherever applicable.

- For `token_markov_grpo`, use Yuan paper setup as faithfully as possible.
- For `baseline_grpo`, keep all shared knobs identical to `token_markov_grpo`.
- For latent methods, keep all shared training/eval knobs identical; only add latent-specific components.

Required before any final run:

- [x] Create `reports/yuan_parity_spec.md` listing:
  - [x] every reproduced parameter from Yuan,
  - [x] every parameter not specified by Yuan and chosen by us,
  - [x] rationale for each unavoidable deviation.
- [x] Create `configs/final_parity/` config files for all 4 arms derived from the same parity base.
- [x] Create `configs/repro_tolerance.yaml` for reproducibility gate thresholds.

No final results are valid without this parity spec.

---

## Repository Deliverables You Must Implement

Required top-level paths:

- `configs/`
- `data/`
- `scripts/`
- `src/`
- `src/models/`
- `src/training/`
- `src/eval/`
- `src/utils/`
- `artifacts/`
- `reports/`
- `README.md`

Required config files:

- `configs/base_model.yaml`
- `configs/train_baseline_grpo.yaml`
- `configs/train_token_markov_grpo.yaml`
- `configs/train_latent_grpo.yaml`
- `configs/train_latent_grpo_uncertainty.yaml`
- `configs/eval_math_beyond.yaml`

Required script entrypoints:

- `scripts/prepare_data.py`
- `scripts/train_baseline.py`
- `scripts/train_token_markov.py`
- `scripts/train_latent.py`
- `scripts/eval_passk.py`
- `scripts/run_ablation_table.py`

Required core modules:

- `src/models/token_markov_state.py`
- `src/models/vae_state_encoder.py`
- `src/training/grpo_baseline.py`
- `src/training/grpo_token_markov.py`
- `src/training/grpo_latent.py`
- `src/training/reward_bonus.py`
- `src/eval/metrics.py`
- `src/utils/logging.py`
- `src/utils/seeding.py`

---

## Phase 1: Baseline + Data + Evaluation Foundation

### Implementational Deliverables

1. Dependency environment locked (`requirements.txt` or `pyproject.toml`).
2. `scripts/prepare_data.py` prepares benchmark artifacts for `MATH-Beyond` per `reports/DATA_PROTOCOL.md`:
   - `data/math_beyond_math_b_i_base.jsonl` (primary claim set, MATH-B-I base gauntlet),
   - `data/math_beyond_hf_strict_all_models.jsonl` (secondary strict intersection),
   - `data/math_beyond_full_181.jsonl` (full `test` split),
   - `data/benchmark_manifest.json` (revision, counts, SHA-256).
3. `scripts/train_baseline.py` runs end-to-end:
   - local smoke profile (RTX 4060),
   - full profile (A100).
4. `scripts/eval_passk.py` supports `k in {1, 16, 1024}` and outputs all three.
5. Baseline artifacts are written to:
   - `artifacts/baseline_grpo/{run_id}/`

**Phase 1 — data track:** Item (2) above is **complete** (protocol, `prepare_data.py`, pins, manifest, config path alignment). Phase 1 **overall** still requires items (1), (3)–(5) and the success criteria below.

### Phase 1 Success Criteria (Pass/Fail)

Pass only if all are true:

- Full A100 baseline run completes without training instability.
- Eval outputs valid `pass@1`, `pass@16`, `pass@1024`.
- Re-run at same seed is reproducible within a pre-defined tolerance recorded in `configs/repro_tolerance.yaml` (to be mirrored into `reports/yuan_parity_spec.md` in Phase 2).
- README commands for baseline train/eval run unedited.

If any fails, Phase 2 is blocked.

---

## Phase 2: Yuan-Style Token Markov Arm (Mandatory, Full Strength)

### Implementational Deliverables

1. `src/models/token_markov_state.py` implements token-space Markov state predictor arm aligned to Yuan method.
2. `scripts/train_token_markov.py` trains `token_markov_grpo` end-to-end on A100.
3. `src/training/grpo_token_markov.py` integrates state predictor into GRPO loop with full logging.
4. Parity package completed:
   - `reports/yuan_parity_spec.md`
   - `configs/final_parity/train_token_markov_grpo.yaml`
   - `configs/final_parity/train_baseline_grpo.yaml`

### Phase 2 Success Criteria (Pass/Fail)

Pass only if all are true:

- Token Markov arm completes full training/eval cycle.
- Token Markov and baseline are parity-matched per parity spec.
- `pass@1024` is produced for both baseline and token Markov arms.
- Any deviations from Yuan are explicitly documented and justified.

If any fails, Phase 3 is blocked.

---

## Phase 3: VAE Latent Arm + Uncertainty Arm

### Implementational Deliverables

1. `src/models/vae_state_encoder.py` implemented per brief:
   - trajectory hidden states input,
   - latent outputs (`mu`, `logvar`, `z`),
   - ELBO training objective.
2. `src/training/grpo_latent.py` supports:
   - `latent_grpo`,
   - `latent_grpo_uncertainty`.
3. `src/training/reward_bonus.py` adds intrinsic bonus:
   - `r_total = r_task + beta_t * KL(q(z|tau) || p(z))`
4. Final parity configs added:
   - `configs/final_parity/train_latent_grpo.yaml`
   - `configs/final_parity/train_latent_grpo_uncertainty.yaml`

### Phase 3 Success Criteria (Pass/Fail)

Pass only if all are true:

- Both latent arms complete full training/eval cycles.
- No NaN/Inf failures in latent pipeline.
- `pass@1024` is produced for all 4 arms.
- Shared knobs are parity-matched across all arms except method-specific components.

If any fails, Phase 4 is blocked.

---

## Phase 4: Core Ablation Table + Publishable Artifacts

### Implementational Deliverables

1. `scripts/run_ablation_table.py` generates:
   - `reports/ablation_core.csv`
   - `reports/ablation_core.md`
2. Core ablation table includes exactly 4 arms:
   - baseline,
   - token Markov (Yuan-style),
   - latent,
   - latent + uncertainty.
3. Reported metrics include mandatory `pass@1024` and secondary `pass@1`, `pass@16`.
4. Final reproducibility package:
   - exact commands,
   - configs,
   - seeds,
   - artifact paths.

### Phase 4 Success Criteria (Pass/Fail)

Pass only if all are true:

- Ablation is generated from artifacts (not manual entry).
- `pass@1024` comparison exists for all 4 arms.
- `YUAN_PARITY_CHECKLIST.md` Section A is fully checked.
- At least one outcome is established clearly:
  - positive: latent method beats strong baselines, or
  - negative: no gain, with rigorous diagnosis and evidence.

If any fails, final delivery is blocked.

---

## Final Deliverables (Exactly per Brief)

By project completion, deliver all of the following:

1. Public GitHub repository (clean, reproducible, documented).
2. arXiv-style preprint (6-8 pages, citable formatting).
3. Blog post version (shorter, visual, shareable).
4. Core result table showing whether latent methods solve what GRPO baseline cannot, with full ablations.
5. Bonus if time: latent space visualization with interpretable structure.

---

## Timeline Policy (Phase-Based)

- There are no fixed week deadlines in this contract.
- Progress is governed strictly by phase gates.
- You do not move forward until current phase criteria pass.

---

## Hard Scope Guardrails

Do NOT do any of the following in this phase:

- No second base model family.
- No second RL algorithm.
- No diffusion replacement.
- No persona-conditioning branch.
- No benchmark expansion before all 4-arm core results are complete.
- No extra ablations outside the mandatory matrix until final deliverables are done.

If tempted to add scope, park it under **Future work** in `reports/writeup_stubs.md` and continue the current phase.
