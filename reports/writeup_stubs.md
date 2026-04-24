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

### Framing constraint (do not lose this)

**The load-bearing frame is the MDP reformulation, not the VAE architecture.** Every section leads with: "We replace history-as-state with a compact learned Markov state." The VAE is mentioned as the implementation. This distinction determines the abstract, the intro hook, and the contribution bullets.

The abstract shape (from PROJECT_BRIEF.md): current RL post-training fails because token history is not a valid MDP state → we replace it with a learned latent Markov state → this breaks the capability ceiling where standard RLVR fails.

### Related work — Reasoning Palette (must handle explicitly)

Reasoning Palette (VAE latent modulating reasoning strategy via token prefix) is the closest adjacent paper. Differentiator that must appear clearly in related work:

- **Theirs:** VAE sampled *once per problem* as a strategy prefix before generation begins. Static per problem.
- **Ours:** `z_h` evolves *step-wise during the rollout* as a Markov state. Dynamic through reasoning.
- Different MDP formulation, not just a different architecture. They never question the history-as-state MDP. We do.

### Markov diagnostic results (TODO after Phase 3)

Must include empirical evidence that `z_h` satisfies the Markov property:
- Latent transition loss: `z_h + a_h → z_{h+1}` without history
- Last-state-only ablation: policy on `z_h` only vs full history access
- Latent variance analysis: does `σ_h²` correlate with problem difficulty / solution uncertainty?
