# Token-Markov Arm: Design Document

Implementation reference: Delethink / Markovian Thinker (Aghajohari et al., ICLR 2026)
GitHub: https://github.com/McGill-NLP/the-markovian-thinker

---

## The Problem This Arm Solves

Standard GRPO feeds the model the full token history at each step:

```
Input to model = [system prompt] + [question] + [everything generated so far]
```

This grows unboundedly. It's not a Markov state — future decisions depend on an
ever-growing sequence, not a compact sufficient statistic. Yuan et al. proved this
is why RLVR hits a capability ceiling: the policy has to cover an exponentially
large history space.

Delethink fixes this by enforcing a strict information bottleneck: after each chunk,
old tokens are deleted. Only the last m tokens carry forward. The model is forced
to compress its reasoning state into that window.

---

## The Mechanism (from scratch)

A single rollout in the Delethink arm proceeds in up to 3 chunks.

### Chunk 1

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT (x1)                                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ [system prompt] + [question]                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│                     model generates                         │
│                    up to C=512 tokens                       │
│                          │                                  │
│  OUTPUT (y1)             ▼                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ token_1 token_2 ... token_100 ... token_511 token_512  │ │
│  │ └──── planning prefix ────┘    └──── rest of chunk ──┘ │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

After chunk 1:
  - First 100 tokens of y1 are appended to query permanently ("planning prefix")
  - These capture the model's initial problem framing / approach
  - They stay in context for ALL subsequent chunks
```

### Chunk 2

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT (x2) — OLD CONTEXT IS DELETED                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ [system prompt] + [question]  ← original query      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ token_1 ... token_100         ← planning prefix      │    │
│  │ (first 100 tokens of chunk 1, kept permanently)      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ token_257 ... token_512       ← MARKOV STATE         │    │
│  │ (last m=256 tokens of chunk 1)                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  tokens_101 to token_256 of chunk 1: DELETED ✗              │
│                          │                                  │
│                          ▼                                  │
│                     model generates                         │
│                  up to C-m = 256 tokens                     │
│                          │                                  │
│  OUTPUT (y2)             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ token_1 ... token_256                               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Chunk 3

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT (x3) — CHUNK 2 CONTEXT IS DELETED                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ [system prompt] + [question]  ← original query      │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ token_1 ... token_100         ← planning prefix      │    │
│  │ (still here — permanent)                             │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ last 256 tokens of y2         ← MARKOV STATE         │    │
│  │ (chunk 2 output replaces chunk 1 carryover)          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  All of chunk 2 except last 256 tokens: DELETED ✗           │
│                          │                                  │
│                          ▼                                  │
│                     model generates                         │
│                  up to C-m = 256 tokens                     │
│  (stops early if \boxed{answer} + [EOS] is produced)        │
│                          │                                  │
│  OUTPUT (y3)             ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ... reasoning ... \boxed{42}  [EOS]                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### The full rollout end-to-end

```
ROLLOUT FOR ONE PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 query q ──────────────────────────────────────────────────┐
                                                           │
         ┌─────────────────────────────────────────────┐  │
         │ CHUNK 1                                     │  │
         │  input:  q                                  │◄─┘
         │  output: y1 (up to 512 tokens)              │
         │  budget used: 512 tokens                    │
         └──────────────┬──────────────────────────────┘
                        │
          q ← q + y1[0:100]  (planning prefix folded in permanently)
                        │
          carryover ← y1[-256:]
                        │
         ┌──────────────▼──────────────────────────────┐
         │ CHUNK 2                                     │
         │  input:  q + carryover                      │
         │  output: y2 (up to 256 tokens)              │
         │  budget used: 256 tokens                    │
         │  stop here if [EOS] generated               │
         └──────────────┬──────────────────────────────┘
                        │
          carryover ← y2[-256:]
                        │
         ┌──────────────▼──────────────────────────────┐
         │ CHUNK 3                                     │
         │  input:  q + carryover                      │
         │  output: y3 (up to 256 tokens)              │
         │  budget used: 256 tokens                    │
         │  stop here if [EOS] generated               │
         └──────────────┬──────────────────────────────┘
                        │
                        ▼
              TRACE τ = [(x1,y1), (x2,y2), (x3,y3)]

TOTAL NEW TOKENS: 512 + 256 + 256 = 1024  ←  matches baseline exactly ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Reward and Advantage

The reward is evaluated on the **full trace** τ — specifically on the final answer
extracted from the last chunk's output. Same binary math reward as baseline:
correct `\boxed{}` answer → 1, else → 0.

G=8 traces are generated per problem (same as baseline). Advantages are normalized
across the group:

```
A(τ_i) = (R(τ_i) - mean(R)) / std(R)     across i = 1..8
```

The advantage for ALL tokens in ALL chunks of trace τ_i gets the same value A(τ_i).
This is the GRPO formulation extended to multi-chunk traces (directly from Algorithm 1
in the paper).

---

## What the Model Learns (and what it doesn't need to learn)

**The Markov property is structural, not learned.** The context reset happens
regardless of what the model writes. From the first rollout, the model is already
operating as a Markovian thinker — it cannot attend to tokens outside the
carryover window even if it tries. This is the core contribution of this arm:
enforcing the Markov property by construction and measuring whether that alone
helps on hard math.

**What RL does** is progressively improve the *quality* of the carryover as a
sufficient statistic. If a trace reaches the correct answer, every token in all
three chunks gets upweighted — including the tokens that ended up in the carryover
window. Over training, the model learns to write more useful state into that window.
There is no structured format, no prompt engineering for this — but the learning
happens on top of an already-Markovian inference regime, not as the mechanism that
creates it.

**Important caveat for our setting.** The Delethink paper demonstrated this on
R1-Distill-Qwen-1.5B, which already exhibits natural Markovian reasoning behaviour
zero-shot (last-m tokens of its output happen to contain useful state summaries
because of R1-style CoT pretraining). We use Qwen2.5-1.5B-Instruct, which has no
such prior. Our m=256 window in the default 3-chunk config is also 8-16× smaller
than the paper's 2K-4K, which limits carryover richness.

To address the window concern without breaking the 1024-token budget or requiring
a code change, we run a **2-chunk ablation** (`train_token_markov_grpo_2chunk.yaml`)
with I=2, C=768, m=512:
```
budget: 768 + (768-512)*1 = 768 + 256 = 1024 tokens  ✓  (identical to baseline)
carryover: 512 tokens (2× the 3-chunk variant)
```
This doubles the carryover window within the same budget. The result directly
answers the small-window concern: if 2-chunk beats 3-chunk, larger carryover
matters; if they're similar, the 256-token window is sufficient.

Two honest outcomes for the token-Markov arm overall:

- Beats baseline → enforced Markov structure helps even on a non-reasoning-model
  base with a small carryover window.
- ≈ or underperforms baseline → motivates the latent arm: structured token carryover
  is not sufficient; a learned continuous compression (VAE) is needed.

Either result is informative. This arm is a controlled test of what structural
Markov enforcement contributes — not a validation of Delethink's claim on our dataset.

---

## Key Parameters

### Default: 3-chunk config (`train_token_markov_grpo.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| C (chunk size) | 512 tokens | Budget: 512 + 256 + 256 = 1024 ✓ |
| m (carryover) | 256 tokens | Paper's ratio m = C/2 |
| planning_prefix | 100 tokens | Paper's Algorithm 1 exactly |
| iteration_cap (I) | 3 chunks | 2 context resets per rollout |
| carryover_format | freeform | Raw last-m tokens, no structure |
| context_reset | true | Old tokens deleted each chunk boundary |

### Ablation: 2-chunk config (`train_token_markov_grpo_2chunk.yaml`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| C (chunk size) | 768 tokens | Budget: 768 + 256 = 1024 ✓ |
| m (carryover) | 512 tokens | **2× the default** — addresses small-window concern |
| planning_prefix | 100 tokens | Same as default |
| iteration_cap (I) | 2 chunks | 1 context reset per rollout |
| carryover_format | freeform | Inherited from default |
| context_reset | true | Inherited from default |

**Ablation question answered:** does doubling the carryover window (256→512 tokens)
improve `pass@1024`, given the same total token budget?

### Budget derivation formula (both configs)

```
Total new tokens = C + (C - m) * (I - 1)

3-chunk:  512 + (512-256)*2 = 512 + 512   = 1024  ✓
2-chunk:  768 + (768-512)*1 = 768 + 256   = 1024  ✓
baseline: (single sequence, 1024 tokens)           1024  ✓
```

m is derived from the budget formula — it was never an independent choice.
The only free parameter is I (number of chunks).

---

## The Engineering Problem

Standard TRL `GRPOTrainer` assumes one continuous sequence per rollout:

```
WHAT TRL EXPECTS:
  prompt → [single uninterrupted generation of up to max_completion_length tokens]
  result → one (prompt, completion) pair per rollout
```

Delethink needs:

```
WHAT DELETHINK NEEDS:
  prompt → chunk 1 generation (512 tokens)
         → context reset
         → chunk 2 generation (256 tokens, different prompt)
         → context reset
         → chunk 3 generation (256 tokens, different prompt)
  result → three (prompt, completion) pairs, treated as one trace for advantage

```

The mismatch is in two places:
1. **Generation loop** — TRL generates one sequence; we need a sequential
   multi-round loop with context manipulation between rounds.
2. **Loss computation** — TRL computes loss over one (prompt, completion) pair;
   we need loss over all chunks jointly, normalized by total trace length.

These are not trivial to patch. Options to consider next.

---

## Base Model Choice (why we do NOT switch to DeepSeek R1-Distill)

The Delethink paper used R1-Distill-Qwen-1.5B, which already behaves Markovian
zero-shot. One might ask: should we switch to R1-Distill to match their setting?

**No.** For two reasons:

1. **Fairness across all four arms.** The contract requires the same base model
   across all arms. Switching to R1-Distill for only the token-Markov arm confounds
   the comparison — you'd be measuring base model effect, not Markov structure effect.
   Switching for all arms would require re-running the baseline and is out of scope.

2. **Using a non-R1 model is the harder, more informative test.** R1-Distill
   already internalises chain-of-thought summarisation, so it has a natural affinity
   for Delethink's carryover mechanism. Using Qwen2.5-1.5B-Instruct removes that
   confound. If the token-Markov arm shows benefit here, the result is stronger
   (Markov structure helps even without a reasoning-model prior). If it doesn't,
   the result is informative about prerequisites (you need a reasoning-model base
   or the latent VAE approach to generalise).

   Testing on the "harder" model is the right scientific choice — it avoids
   inflating the token-Markov arm's result through base model affinity and gives
   a cleaner signal about what structural Markov enforcement actually contributes.
