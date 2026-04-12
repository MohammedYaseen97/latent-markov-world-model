# MATH-Beyond data protocol (authoritative)

This document is the **single place** to defend how benchmark JSONLs in `data/` are built. Training and eval code should treat **`data/benchmark_manifest.json`** as machine-readable provenance for a given run of `scripts/prepare_data.py`.

## What we evaluate on (primary)

**File:** `data/math_beyond_math_b_i_base.jsonl`

**Definition:** **MATH-B-Intersection (MATH-B-I)** from Mayilvahanan et al., *MATH-Beyond* ([arXiv:2510.11653](https://arxiv.org/abs/2510.11653)), **Section 3.2.3 “Base Models”**.

- The paper lists **Base Models** separately from **Supplementary Models**.
- **MATH-B-I** is the set of problems for which **every Base Model** failed under the authors’ **pass@1024** evaluation.
- We operationalize that on the Hugging Face release by requiring these Hub columns to be **0** for the same row:

  `qwen2.5-1.5b_pass@1024`, `qwen2.5-7b_pass@1024`, `qwen2.5-math-1.5b_pass@1024`, `qwen2.5-math-7b_pass@1024`, `qwen3-4b-base_pass@1024`, `qwen3_8b_base_pass@1024`, `r1-1.5b_pass@1024`, `r1-7b_pass@1024`, `olmo_7b_pass@1024`, `olmo_2_1124_7b_pass@1024`, `llama3.1_8b_pass@1024`

**Row count:** The paper reports **41** problems. On the **pinned** Hub revision in `configs/math_beyond_hf_revision.txt`, this filter currently yields **40** rows. The manifest records the exact `hf_revision` and `row_count`. Any writeup should state **observed N** from the manifest, not assume 41.

**Why not “all 21 models unsolved”?** That is a **different** (stricter) subset: it requires **Supplementary** models (e.g. instruct variants, some post-trained columns on the Hub) to fail as well. That is **not** the paper’s definition of MATH-B-I.

## Pool definition vs training checkpoint (not a contradiction)

The Hub columns above are **base** (non–chat-tuned) checkpoints used by Mayilvahanan et al. to **score** each row. That procedure answers only: **which `test` rows count as “hard” for this benchmark story?**

**It does not require** your GRPO runs to load those same base weights. This repo trains **instruction-tuned** (chat) checkpoints for rollout quality and stability — see `configs/base_model.yaml` (primary `Qwen/Qwen2.5-1.5B-Instruct`, fallback `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`). All four arms still use the **same** chosen checkpoint for fair comparison.

In writeups, one honest sentence is enough, e.g.: *Problems are MATH-B-I (base-model intersection on the Hub table); we fine-tune **Qwen2.5-1.5B-Instruct** (or the documented fallback) for trainability.*

### Is this still “hard” if we train instruct weights?

**What we are *not* claiming:** that every row is **unsolvable** by *your* instruct checkpoint at **pass@1024** out of the box. MATH-B-I was **defined** using **base** Hub scores (`pass@1024 == 0` for every listed base column). That is a **documented, third-party** notion of “hard in the paper’s base-model sense,” not “provably impossible for Qwen2.5-1.5B-Instruct.”

**What we *are* claiming — and can defend:**

1. **External difficulty filter:** every item in the primary JSONL survived a **strict public filter**: *no* listed **base** model solved it at **pass@1024** on the pinned Hub table. That is a **real capability-ceiling style gauntlet** relative to a multi-model evaluation protocol, not an ad-hoc “we picked tricky prompts.”

2. **Instruct is often stronger than base** — so some rows *might* become reachable for a frozen instruct policy more often than for those base checkpoints. That does **not** invalidate the experiment: the core table is **within-model**, **same instruct backbone**, **same items**, **same eval**. If instruct already clears many items, the run reports that honestly; if not, the items still stress **high‑k** reasoning under **verifiable reward**.

3. **pass@1024 stays demanding:** even when the policy can occasionally succeed, estimating **pass@1024** well is **variance-heavy** and **sample-expensive**; the benchmark is chosen so that **naive RLVR / history-as-state** is a meaningful baseline to beat on **left-hard** mass, not so that zero-shot instruct must score zero.

4. **If you need a stricter “everyone failed” story** (including instruct-style columns on the Hub), use the **secondary** JSONL (`math_beyond_hf_strict_all_models.jsonl`, AND of all `*_unsolved`) — smaller, harder — for appendix or sensitivity, not as the default claim unless you justify it.

**Bottom line for reviewers:** we use MATH-B-I to fix **which problems** enter the gauntlet using a **published, reproducible** rule; we use **instruct** weights to make **RL training and rollouts** workable. The scientific question is **comparative** (latent vs baselines on that fixed list under matched budgets), not “re-prove base models cannot solve them.”

## Secondary pool (appendix / robustness)

**File:** `data/math_beyond_hf_strict_all_models.jsonl`

**Definition:** Logical **AND** of all **21** `*_unsolved` columns on the Hub (equivalent to all `*_pass@1024 == 0` for every model column on this revision). Typically **13** rows — **harder** than MATH-B-I base-only.

Use this for sensitivity analysis, not as the main claim unless explicitly justified.

## Full pool

**File:** `data/math_beyond_full_181.jsonl` — entire `test` split of `brendel-group/MATH-Beyond` at the pinned revision (181 rows).

## Reproducibility

1. **Pin the dataset:** `configs/math_beyond_hf_revision.txt` contains the Hub dataset repo **git commit SHA**. Override with `--hf-revision` or `MATH_BEYOND_HF_REVISION`.
2. **Pin the client libraries:** `requirements.txt` fixes **`datasets`** and **`huggingface_hub`** versions so Hub download, caching, and Arrow decoding stay aligned with the revision pin. The manifest stores **`library_versions_at_build`** for whatever environment last ran `prepare_data.py`.
3. **Stable row IDs:** Each JSONL line includes `source_index`, the 0-based index into `load_dataset(..., split="test", revision=...)`. `problem_id` is `math_b_{source_index:04d}`.
4. **Checksums:** After generation, `data/benchmark_manifest.json` includes SHA-256 of the written JSONL files. Re-running with the same revision and matching library versions should reproduce those hashes (verified in CI or locally).

## Relation to Yuan & Xie (2026)

| Work | Empirical tasks |
|------|-----------------|
| **Yuan & Xie (2026)** — Markov states | Logic environments (e.g. Sudoku, Sokoban, Futoshiki in their paper) |
| **Mayilvahanan et al. — MATH-Beyond** | High-school competition **math** (Hub: `brendel-group/MATH-Beyond`) |

**Yuan** ran logic puzzles; **we** evaluate on MATH-B. Their work motivates a **token-Markov comparator arm**, not a replication. Fairness across the four arms is in **`PROJECT_CONTRACT.md`**.

## How to regenerate

```bash
python scripts/prepare_data.py --output-dir data
```

Inspect `data/benchmark_manifest.json` for counts, revision, and hashes.
