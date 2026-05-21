# Data directory

**Authoritative protocol:** [`reports/DATA_PROTOCOL.md`](../reports/DATA_PROTOCOL.md)

## Generate pools

```bash
# Easy pool (Phase 0 encoder pretraining, Level 1-4 only)
python scripts/prepare_easy_pool.py --levels 1 2 3 4

# Level 5 hard pool (training + eval for all arms, ~2h)
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output data/math_level5_hard_pool.jsonl
```

## Files (after running the scripts)

| File | Role |
|------|------|
| `math_level5_hard_pool.jsonl` | **Primary** training + eval pool — MATH Level 5, pass@128=0 filter. |
| `math_easy_pool.jsonl` | Phase 0 encoder pretraining pool — MATH Level 1–4. |
| `level5_pool_manifest.json` | Revision, model used for filtering, row counts, SHA-256. |
| `easy_pool_manifest.json` | Revision, levels, row counts, SHA-256. |

## JSONL schema

- `problem_id`, `source_index`, `prompt`, `ground_truth`, `data_source`, `topic`, `difficulty`, `source_level`

Large JSONLs are gitignored; commit manifest files after a pinned run to record hashes.

## Mutual exclusivity

Easy pool uses Level 1–4 only. Hard pool uses Level 5 only. Intersection = ∅.
Verified by `source_level` field in both manifest files.
