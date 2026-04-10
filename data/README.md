# Data Artifacts

This project treats MATH-Beyond as a benchmark/environment pool, not a supervised train/test dataset.

Required benchmark files:

- `math_beyond_ceiling_41.jsonl`
  - Primary claim set used for core comparisons and `pass@1024` reporting.

Optional support file:

- `math_beyond_full_181.jsonl`
  - Full benchmark collection for debugging/progress checks (non-primary claim set).

## Minimal expected JSONL schema

Each line should contain at least:

- `problem_id` (string)
- `prompt` (string)
- `ground_truth` (string)

Additional metadata fields can be included if needed.
