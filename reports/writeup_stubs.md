# Write-up stubs (single file)

Short placeholders for later phases. Keeps the repo to one scratch surface instead of many tiny markdowns.

## Future work

Use this section to park out-of-scope ideas without expanding current project scope.

## Main results

TODO: Populate after core table (`run_ablation_table.py`). Partial results below as arms complete.

**Claim shape (pick honestly):** latent beats **both** baselines on `pass@1024` · beats history only · ties / null — each is publishable if methods and budgets are fair (`PROJECT_CONTRACT.md`).

### Arm results (MATH-B-I base pool, 40 problems, 1024 samples)

| arm | pass@1 | pass@16 | pass@1024 | run |
|-----|--------|---------|-----------|-----|
| `baseline_grpo` | 0.012% | 0.19% | **10.0%** | `20260421T131720Z` / `checkpoint-100` |
| `token_markov_grpo` | — | — | — | pending |
| `latent_grpo` | — | — | — | pending |
| `latent_grpo_uncertainty` | — | — | — | pending |

**Interpretation:** baseline `pass@1024 = 0.10` on the hard capability-ceiling pool. Base models score 0 by construction (MATH-B-I filter). 100 steps of standard GRPO finds solutions for ~4/40 problems given 1,024 attempts but `pass@1 ≈ 0`, consistent with sampling redistribution rather than capability expansion (Yue et al. NeurIPS 2025).

## Limitations

TODO: Document methodological and empirical limitations.

## Paper draft

TODO: Write arXiv-style manuscript draft (6–8 pages equivalent).
