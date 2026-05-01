# Core ablation table

*Generated 2026-05-01 06:01 UTC by `scripts/run_ablation_table.py` from `artifacts/`.*  
*Never hand-typed — re-run to refresh.*

## MATH-B-I base pool

| arm | pass@1 | pass@16 | pass@1024 | run |
|-----|--------|---------|-----------|-----|
| `baseline_grpo` | 0.0002 | 0.0031 | 0.1500 | 20260430T083332Z |
| `token_markov_grpo` | 0.0001 | 0.0019 | 0.0500 | 20260430T102111Z |
| `latent_grpo` | — | — | — | — |
| `latent_grpo_uncertainty` | — | — | — | — |

## Artifact paths

| arm | eval_metrics.json |
|-----|-------------------|
| `baseline_grpo` | `artifacts/baseline_grpo/20260430T083332Z/eval_metrics.json` |
| `token_markov_grpo` | `artifacts/token_markov_grpo/20260430T102111Z/eval_metrics.json` |
| `latent_grpo` | `—` |
| `latent_grpo_uncertainty` | `—` |
