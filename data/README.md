# Data directory

**Authoritative protocol:** [`reports/DATA_PROTOCOL.md`](../reports/DATA_PROTOCOL.md)

## Generate artifacts

```bash
python scripts/prepare_data.py --output-dir data
```

Requires pinned Hub revision (`configs/math_beyond_hf_revision.txt`) and the **`datasets` / `huggingface_hub` versions** in `requirements.txt` (also recorded under `reproducibility.library_versions_at_build` in `benchmark_manifest.json`).

## Files (after running the script)

| File | Role |
|------|------|
| `math_beyond_math_b_i_base.jsonl` | **Primary** eval pool — paper MATH-B-I (Base Models, pass@1024). |
| `math_beyond_hf_strict_all_models.jsonl` | **Secondary** — all 21 Hub models unsolved (stricter; usually 13 rows). |
| `math_beyond_full_181.jsonl` | Full `test` split (181 rows). |
| `benchmark_manifest.json` | Revision, definitions, row counts, SHA-256. |

## JSONL schema

- `problem_id`, `source_index`, `prompt`, `ground_truth`; optional `data_source`, `topic`, `difficulty`.

Large JSONLs are gitignored; commit **`benchmark_manifest.json`** after a pinned run if you want CI/docs to reference exact hashes.
