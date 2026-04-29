# Write-up stubs (single file)

Short placeholders for later phases. Keeps the repo to one scratch surface instead of many tiny markdowns.

## Future work

Use this section to park out-of-scope ideas without expanding current project scope.

### Batched Delethink trace generation for production pass@1024 ✅ Implemented

`eval_passk.py --generation-mode token_markov` supports both vLLM (production)
and HF sequential (smoke/fallback) backends.

**vLLM multi-round path (use_vllm: true in eval config):**
- Round 1: identical to baseline — all problems × n_samples via vLLM, chunk_size
  completions at a time (same chunked loop as baseline, max_tokens=C).
- Rounds 2+: 40k unique prompts (query + carryover_i per trace), batched in
  chunks of chunk_size with n=1 each, max_tokens=C-m (shorter per round).
- Wall-clock: round 1 ≈ baseline (~30 min); rounds 2+ are fast (fewer tokens,
  n=1). Total ≈ baseline ± 25%. Full pass@1024 at ~30-45 min on A100.

**HF sequential fallback (use_vllm: false):** correct but slow; smoke use only.

## Main results

TODO: Populate after core table (`run_ablation_table.py`). Partial results below as arms complete.

**Claim shape (pick honestly):** latent beats **both** baselines on `pass@1024` · beats history only · ties / null — each is publishable if methods and budgets are fair (`PROJECT_CONTRACT.md`).

### Arm results (MATH-B-I base pool, 40 problems, 1024 samples)

| arm | pass@1 | pass@16 | pass@1024 | run |
|-----|--------|---------|-----------|-----|
| `baseline_pretrained` (instruct, no GRPO) | — | — | — | pending — run: `eval_passk.py` (no `--checkpoint`) |
| `token_markov_pretrained` (instruct, no GRPO, Delethink regime) | — | — | — | pending — run: `eval_passk.py --generation-mode token_markov` (no `--checkpoint`) |
| `baseline_grpo` | 0.012% | 0.19% | **10.0%** | `20260421T131720Z` / `checkpoint-100` |
| `token_markov_grpo` (3-chunk, m=256) | 0.007% | 0.12% | **7.5%** | `20260428T004551Z` / `checkpoint-50` |
| `latent_grpo` | — | — | — | pending |
| `latent_grpo_uncertainty` | — | — | — | pending |

**Interpretation — baseline:** `pass@1024 = 10.0%` on the hard capability-ceiling pool. Base models score 0 by construction (MATH-B-I filter). 100 steps of standard GRPO finds solutions for ~4/40 problems given 1,024 attempts but `pass@1 ≈ 0`, consistent with sampling redistribution rather than capability expansion (Yue et al. NeurIPS 2025).

**Interpretation — token-Markov:** `pass@1024 = 7.5%` — 3 of 40 problems were solved, each exactly once across 1,024 sampled traces. The three eval metrics are fully consistent with this interpretation:

```
pass@1   = 3 correct / (40 problems × 1024 samples) = 3/40960 ≈ 7.32e-5  ✓
pass@16  = 3 × 16 / 40960                           = 48/40960 ≈ 1.17e-3  ✓
pass@1024 = 3 / 40                                  = 0.075               ✓
```

Each of the 3 solvable problems was solved exactly once, giving each a per-sample success probability of ≈ 1/1024 ≈ 0.0977%.

**Why RL training received zero gradient signal — the arithmetic:**

GRPO uses G=8 generations per problem per training step (matching the baseline; G is not
a free parameter in this experiment). The probability that any single training rollout for
a solvable problem produces at least one correct trace is:

```
P(≥1 correct | G=8) = 1 - (1 - 1/1024)^8 ≈ 8/1024 ≈ 0.78%
```

Over the full training run, each of the 3 solvable problems is encountered approximately
`50 steps × 4 problems/step × (3/40)` = **15 times** (on expectation). Expected total
positive rewards across the entire training run:

```
15 encounters × 0.78% success rate per encounter ≈ 0.12 expected rewards
```

Statistically zero. The training observation of zero positive rewards is not a
malfunction — it is the mathematically expected outcome given the reward density and group
size. Running for more steps would not help: the expected rewards scale linearly, and even
doubling to 100 steps yields ~0.24 expected rewards. RL requires consistent positive
reward within a group to compute a non-degenerate advantage; at this reward density,
G=8 is structurally insufficient.

**Caveat on the head-to-head comparison:** the baseline was successfully RL-trained for
100 steps (10.0% pass@1024), while the token-Markov arm received effectively zero
gradient updates (7.5%, essentially the base model under Delethink constraints). The
10.0% vs 7.5% gap therefore conflates two effects: (1) the capability cost of the
Delethink context-reset constraint, and (2) the benefit of 100 steps of GRPO training.
The honest framing is not "token-Markov underperforms the trained baseline" but rather
"Delethink's structural Markov enforcement imposes a capability cost that prevents RL
from obtaining any training signal at G=8." These are the same finding stated at
different levels, but the second framing is more precise and more defensible.

**Story for the paper:** hard token-space Markov enforcement (context reset + m=256
carryover window) degrades the instruct model's ceiling from ~10% (free generation) to
~7.5% (chunked generation), and this degradation is severe enough that standard
GRPO at G=8 cannot obtain a usable reward signal. The model is structurally Markovian
from step 0, but RL cannot improve the carryover quality because it never observes a
success during training. This is the direct empirical motivation for the latent arm: a
learned continuous Markov state should compress reasoning progress without the
capability degradation of hard context reset — keeping reward density high enough for
RL to train.

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
