# Data protocol (authoritative)

This document is the **single place** to defend how pool JSONLs in `data/` are built.
Machine-readable provenance lives in `data/level5_pool_manifest.json` and
`data/easy_pool_manifest.json`.

---

## Pool architecture (mutually exclusive by design)

```
EleutherAI/hendrycks_math
│
├── Level 1–4  →  Phase 0 easy pool  (encoder pretraining only)
│               ~4100 problems, guaranteed solvable fraction
│               File: data/math_easy_pool.jsonl
│               Built by: scripts/prepare_easy_pool.py --levels 1 2 3 4
│
└── Level 5    →  Hard pool  (ALL arms: training + eval)
                ~500 raw Level 5 problems
                → filter: keep problems where pretrained model scores pass@128 = 0
                → ~350 problems remain
                File: data/math_level5_hard_pool.jsonl
                Built by: scripts/prepare_math_level5_pool.py
```

**Mutual exclusivity guarantee:** Level 5 problems are never in the easy pool
(L1–4 only). Easy pool problems are never in the hard pool (Level 5 only).
Verified by `source_level` field in both manifest files. The `problem_id`
namespaces are also distinct (`math_easy_*` vs `math_l5_*`).

---

## Hard pool — MATH Level 5 filtered

**File:** `data/math_level5_hard_pool.jsonl`

**Definition:** MATH Level 5 problems (EleutherAI/hendrycks_math) for which the
pretrained model (`Qwen/Qwen2.5-1.5B-Instruct`) scores `pass@128 = 0` at
temperature 1.0, seed 42.

**Why filter by pass@128 = 0:** if the pretrained model can already solve a problem
at 128 samples, it is not hard enough to test the capability-ceiling hypothesis.
This filter is model-specific but fully reproducible given a fixed checkpoint and seed.

**Row count:** ~350 (exact count in `data/level5_pool_manifest.json` after build).

**Build command:**
```bash
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --n-samples 128 \
    --output data/math_level5_hard_pool.jsonl
```

**Schema (one JSON object per line):**
```
problem_id   : "math_l5_{i:04d}"
source_index : i (sequential, 0-based within filtered set)
prompt       : problem text
ground_truth : answer string extracted from LaTeX \boxed{}
data_source  : "hendrycks_math"
topic        : subject category (Algebra, Geometry, etc.)
difficulty   : 5
source_level : 5
```

**Reproducibility:** `data/level5_pool_manifest.json` records the HF revision,
model ID, seed, n_samples, raw count, solvable excluded count, final row count,
and SHA-256 of the output JSONL.

---

## Easy pool — MATH Level 1–4

**File:** `data/math_easy_pool.jsonl`

**Definition:** All Level 1–4 problems from EleutherAI/hendrycks_math (train + test
splits, all 7 subject configs). Filtered to remove problems with no `\boxed{}`
answer or text-form answers (`\text{...}`).

**Purpose:** Phase 0 encoder pretraining only. The encoder is trained on this pool
while the backbone is frozen to orient the latent space toward solution-quality
information before RL begins on the hard pool.

**Build command:**
```bash
python scripts/prepare_easy_pool.py --levels 1 2 3 4
```

**Reproducibility:** `data/easy_pool_manifest.json` records the HF revision,
levels, seed, row count per level, and SHA-256 of the output JSONL.

---

## Primary eval metric

**Metric:** `pass@128` on `data/math_level5_hard_pool.jsonl`.

pass@128 is 8× cheaper than pass@1024 while the larger pool (~350 problems)
compensates for lower k — net statistical resolution is comparable to v1's
pass@1024 on 40 problems, at lower compute cost.

---

## Relation to Yuan & Xie (2026)

| Work | Empirical tasks |
|------|-----------------|
| **Yuan & Xie (2026)** — Markov states | Logic environments (Sudoku, Sokoban, Futoshiki) |
| **This project** | MATH Level 5 hard problems (competition mathematics) |

Yuan ran logic puzzles; we evaluate on MATH Level 5. Their work motivates the
token-Markov comparator arm, not a replication. Fairness across all arms is in
`PROJECT_CONTRACT.md`.

---

## How to regenerate

```bash
# Step 0a: easy pool (encoder pretraining, Level 1-4 only)
python scripts/prepare_easy_pool.py --levels 1 2 3 4

# Step 0b: Level 5 hard pool (training + eval for all arms, ~2h)
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output data/math_level5_hard_pool.jsonl
```

Inspect `data/level5_pool_manifest.json` and `data/easy_pool_manifest.json`
for counts, revisions, and SHA-256 checksums.
