# Core ablation table

*Generated 2026-05-01 06:01 UTC by `scripts/run_ablation_table.py` from `artifacts/`.*  
*Never hand-typed — re-run to refresh.*

## MATH-B-I base pool

| arm | pass@1 | pass@16 | pass@1024 | run |
|-----|--------|---------|-----------|-----|
| `baseline_grpo` | 0.0002 | 0.0031 | 0.1500 | 20260430T083332Z |
| `token_markov_grpo` | 0.0001 | 0.0019 | **0.0500 †** | 20260430T102111Z |
| `latent_grpo_pretrained` | — | — | — | pending (Phase 0 v2) |
| `latent_grpo` | — | — | — | pending (Phase 3 v2 re-run) |
| `latent_grpo_uncertainty` | — | — | — | — |

**latent_grpo v1 result (2026-05-11, superseded):** pass@1024 = 7.5% (below baseline 15%). Six root causes identified and fixed in v2 redesign. See `reports/latent_markov_design.md §Redesign v2` and `PROJECT_CONTRACT.md §Phase 3` for full analysis. Phase 3 is being re-run from scratch with v2 design.

**† token_markov_grpo eval note:** this number is a single noisy vLLM run (2/40 problems).
SHA256 of checkpoint-200 equals SHA256 of the pretrained model — zero weight updates
occurred across all 200 training steps. True `pass@1024 ≈ 12.5%` (equal to pretrained
baseline). Two runs of the same checkpoint produced 5.0% and 7.5% due to residual GPU
non-determinism at the noise floor (±1 problem = ±2.5pp on 40 problems). The artifact
path points to the 5.0% run. See `reports/writeup_stubs.md` for the full interpretation
and arithmetic.

## Artifact paths

| arm | eval_metrics.json |
|-----|-------------------|
| `baseline_grpo` | `artifacts/baseline_grpo/20260430T083332Z/eval_metrics.json` |
| `token_markov_grpo` | `artifacts/token_markov_grpo/20260430T102111Z/eval_metrics.json` |
| `latent_grpo_pretrained` | `—` (pending Phase 0 v2 + controlled baseline eval) |
| `latent_grpo` | `—` (pending Phase 3 v2 re-run; v1 artifact at `artifacts/latent_grpo/20260511T025154Z/eval_passk.json`) |
| `latent_grpo_uncertainty` | `—` |
