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

## What the Model Learns

The reward signal propagates back through all chunks. If a trace gets the right
answer, every token in chunks 1, 2, and 3 gets upweighted. The model gradually
learns that the last 256 tokens of each chunk (what becomes the carryover) should
contain whatever information is needed to continue reasoning in the next chunk.

Nobody tells the model what to write in the carryover. There is no structured
format, no prompt engineering. The Markov state representation emerges from the
reward signal alone. That's the "RL-learned" part.

---

## Key Parameters (our config)

| Parameter | Value | Source |
|-----------|-------|--------|
| C (chunk size) | 512 tokens | Scaled from paper's 4K to fit 1024-token total budget |
| m (carryover) | 256 tokens | Paper's fixed ratio: m = C/2 |
| planning_prefix | 100 tokens | Paper's Algorithm 1 exactly |
| iteration_cap (I) | 3 chunks | Budget: 512 + 256 + 256 = 1024 = baseline ✓ |
| carryover_format | freeform | Raw last-m tokens, no structure |
| context_reset | true | Old tokens deleted — enforces Markov property structurally |

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
