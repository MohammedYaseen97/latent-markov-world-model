# Latent Markov World Model

Research project exploring whether **learned latent state representations** can improve RL post-training for mathematical reasoning compared to standard **history-as-state GRPO** and a **token-Markov comparator**.

The benchmark target is **MATH-Beyond**, with the primary claim metric set to **pass@1024**.

## Why this project exists

Most RL post-training setups for LLM reasoning treat the full generated token history as the "state". This repository investigates a different hypothesis:

- Learn a compact latent state from trajectories (VAE-style encoder),
- condition policy learning on that latent,
- and test whether uncertainty in the latent can provide a useful exploration signal.

In short: can a learned state representation help move beyond simple trajectory reweighting and improve hard-reasoning performance?

## Project goals

The core experiment is a four-arm ablation:

1. `baseline_grpo` - history-as-state GRPO
2. `token_markov_grpo` - token-space Markov-style comparator
3. `latent_grpo` - learned latent state (no uncertainty bonus)
4. `latent_grpo_uncertainty` - latent state + uncertainty bonus

Primary metric: `pass@1024`  
Secondary metrics: `pass@1`, `pass@16`

Scope, phase gates, and fairness constraints are defined in `PROJECT_CONTRACT.md`.

## Current status (honest snapshot)

This repository is **in active development**.

### Implemented

- Reproducible MATH-Beyond data preparation pipeline:
  - `scripts/prepare_data.py`
  - `data/benchmark_manifest.json`
- YAML config inheritance utility:
  - `src/utils/config_loader.py`
- Deterministic seeding helper:
  - `src/utils/seeding.py`
- Baseline/eval script plumbing (CLI, config loading, run artifact output):
  - `scripts/train_baseline.py`
  - `scripts/eval_passk.py`

### Not yet implemented

- Full GRPO training loops for all arms
- pass@k sampling + answer grading engine
- Token-Markov and latent model/training internals
- End-to-end ablation table generation from completed runs

This transparency is intentional: the repo is structured for reproducible research execution, with core training/evaluation internals being filled in next.

## Repository structure

```text
configs/     # model, training, eval, parity/fairness, reproducibility configs
data/        # benchmark JSONL artifacts + manifest
scripts/     # executable entry points (prepare/train/eval/ablation)
src/         # models, training logic, eval logic, utilities
artifacts/   # run outputs (manifests, resolved configs, checkpoints/logs)
reports/     # protocol docs, writeup notes, generated tables
```

## Quick start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Build benchmark artifacts

```bash
python scripts/prepare_data.py --output-dir data
```

This writes:

- `data/math_beyond_math_b_i_base.jsonl` (primary claim pool),
- `data/math_beyond_hf_strict_all_models.jsonl` (stricter intersection pool),
- `data/math_beyond_full_181.jsonl` (full test split),
- `data/benchmark_manifest.json` (reproducibility metadata + hashes).

### 3) Smoke-run baseline script wiring

```bash
python scripts/train_baseline.py --profile smoke
```

Today this validates config/seeding/artifact plumbing and then exits with `NotImplementedError` where the actual GRPO loop will be integrated.

### 4) Smoke-run evaluation script wiring

```bash
python scripts/eval_passk.py --pool primary
```

Today this validates config/pool loading and emits machine-readable output with implementation status.

## Reproducibility and research hygiene

This project prioritizes reproducibility from day one:

- Pinned benchmark revision in `configs/math_beyond_hf_revision.txt`
- Dataset manifest with row counts and SHA-256 hashes
- Config inheritance and resolved run configs saved per run
- Shared parity configs in `configs/final_parity/`
- Seed policy and tolerance controls in `configs/repro_tolerance.yaml`

## Design docs

If you want the full technical and experimental rationale:

- `PROJECT_BRIEF.md` - problem framing, hypothesis, and method sketch
- `PROJECT_CONTRACT.md` - locked scope, phase gates, and comparison rules
- `reports/DATA_PROTOCOL.md` - benchmark definition and data provenance details

## Roadmap

Near-term milestones:

1. Complete baseline GRPO training loop and pass@k evaluator
2. Implement token-Markov comparator
3. Implement latent and latent+uncertainty arms
4. Generate the four-arm ablation table from artifacts
5. Publish concise writeup with results and limitations

## For recruiters / collaborators

This repository demonstrates:

- End-to-end research engineering workflow design
- Reproducible experiment/data protocol practices
- Clean project organization for multi-arm ML experimentation
- Thoughtful scientific framing with explicit fairness constraints

Even pre-results, the structure reflects how I build and ship ML research systems: scoped, auditable, and iteration-friendly.
