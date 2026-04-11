# Yuan Parity Spec (Extracted + Project Mapping)

Source paper: **Yuan & Xie (2026), "Breaking the Capability Ceiling of LLM Post-Training by Reintroducing Markov States"** ([arXiv:2603.19987](https://arxiv.org/abs/2603.19987))

**Benchmark scope:** Yuan’s empirical results are on **logic puzzles** (Sudoku / Sokoban / Futoshiki), not MATH-Beyond. This repo’s **MATH-B** evaluation is a separate target (see **Relation to Yuan** in `reports/DATA_PROTOCOL.md`). Yuan parity here means matching their **method and training protocol** where applicable, not matching their Sudoku numbers on math problems.

This file records:
1) parameters and protocol details explicitly recoverable from Yuan,  
2) project-side choices where Yuan is not directly transferable,  
3) deviations and why they do not invalidate our core comparison.

---

## 1) Extracted Yuan Specs (From Paper)

## 1.1 Experimental setup and method

- Framework: `rLLM` backed by `VERL`.
- RL algorithm: `GRPO`.
- Comparator structure: action-sequence baseline vs Markov-state-based approaches.
- Markov pipeline includes a separate state transition/state prediction model.

## 1.2 Tasks / evaluation domain used by Yuan

- Tasks: `Sudoku`, `Sokoban`, `Futoshiki` (Reasoning-Gym-style logic tasks).
- Evaluation split style: in-distribution (ID) and out-of-distribution (OOD).

## 1.3 Training details explicitly stated

- KL divergence coefficient: `0.001`.
- Learning rate (RL post-training): `1e-6`.
- Batch size: `128`.
- Group responses per prompt: `8`.
- Mini-batch size: `128` (one reported exception: `64` for one Sokoban setting).
- SFT warm-up learning rate: `5e-6`.
- SFT warm-up batch size: `256`.

## 1.4 Data scale details explicitly stated

- RL post-training dataset sizes:
  - Sudoku: `10,000`
  - Sokoban: `6,000`
  - Futoshiki: `6,000`
- Test set: `100` synthesized problems per task.
- State prediction model training samples:
  - Sudoku: `174k`
  - Sokoban: `91k`
  - Futoshiki: `108k`

## 1.5 Evaluation details explicitly stated

- Sampling count per question for main table: `k = 128`.
- Decoding temperature: `1.0`.
- Reported metrics include `Avg@128` and `Pass@128`.

---

## 2) Parameters Not Directly Transferable to Our Project

These are intentionally not copied 1:1 because our benchmark claim is on `MATH-Beyond` and `pass@1024`.

- Yuan task generators and board-state transition specifics (Sudoku/Sokoban/Futoshiki) are not directly applicable to MATH-Beyond.
- Yuan exact prompt templates and environment wrappers are task-specific to puzzle environments.
- Yuan uses `Pass@128` as core reporting in the main experiments; our claim metric is `pass@1024` per project brief.
- Yuan uses larger base models in reported experiments (`Qwen3-4B`, `Qwen2.5-3B-Instruct`), while our primary track is `Qwen2.5-1.5B`.

---

## 3) Our Chosen Project Values (Where Yuan Is Unspecified/Non-Transferable)

- Benchmark: `MATH-Beyond` MATH-B-I base pool (`reports/DATA_PROTOCOL.md`).
- Core claim metric: `pass@1024` (mandatory across all 4 core arms).
- Core experiment matrix:
  - `baseline_grpo`
  - `token_markov_grpo`
  - `latent_grpo`
  - `latent_grpo_uncertainty`
- Phase-gated execution and parity gating: per `PROJECT_CONTRACT.md` and `YUAN_PARITY_CHECKLIST.md`.

Specific final-run knobs (steps, rollout budget, decode policy beyond temperature, etc.) will be frozen in:

- `configs/final_parity/*.yaml`

These configs must keep all shared knobs identical across all 4 arms.

---

## 4) Deviations From Yuan and Why They Are Acceptable

1. **Benchmark deviation (logic puzzles -> MATH-Beyond MATH-B-I)**
   - Reason: project brief defines this MATH-B pool as the target capability-ceiling gauntlet (`reports/DATA_PROTOCOL.md`).
   - Why acceptable: our claim is comparative within a controlled 4-arm setup under matched budgets; this is a new-domain extension, not an exact paper replication.

2. **Primary metric deviation (Pass@128 -> pass@1024)**
   - Reason: brief requires stronger high-k claim.
   - Why acceptable: using higher-k makes the claim stricter, not easier.

3. **Base-model size deviation**
   - Reason: project compute and scope constraints.
   - Why acceptable: all arms use the same base model family and matched settings; comparison fairness is preserved.

---

## 5) Parity Readiness Status (Current)

- Paper retrieval and spec extraction: **complete**.
- Comparable-core parity mapping to our project: **complete at documentation level**.
- Final-run parity lock-in (`configs/final_parity/*.yaml`): **pending implementation**.
- End-to-end parity verification from run artifacts: **pending experiments**.

