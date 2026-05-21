# Latent Markov World Models for RL Post-Training

**Can a learned latent state break the capability ceiling of RL post-training on hard mathematics?**

This project trains and compares four RL post-training arms on a benchmark where standard
RLVR is documented to fail — testing whether replacing token history with a learned latent
representation of reasoning progress enables genuine capability expansion.

---

## Motivation

RL post-training (GRPO, PPO, RLVR) has a structural problem. The state fed to the policy
is the full concatenation of all tokens generated so far — an unbounded, redundant,
non-Markovian object that gives the model no compact summary of *where it actually is*
in the solution space. The consequence is well-documented: RLVR improves sampling
efficiency toward paths the base model could already take, but does not expand the
reasoning frontier.

Yue et al. (NeurIPS 2025) proved this formally. Yuan & Xie (March 2026) confirmed it and
showed that introducing explicit Markov states breaks the ceiling — using a token-space
state predictor. This project asks: **what if the state was learned end-to-end from
reasoning trajectories — a compact latent belief over solution-space position, with
epistemic uncertainty built in?**

---

## The Four Arms

All arms share the same pretrained checkpoint, benchmark pool, reward function, and
decoding budget. Only the state representation differs.

| Arm | State representation |
|-----|---------------------|
| `baseline_grpo` | Full token history (standard RLVR) |
| `token_markov_grpo` | Delethink-style RL-learned textual carryover (Markovian Thinker, ICLR 2026) |
| `latent_grpo` | Deterministic latent Markov state with calibrated uncertainty |
| `latent_grpo_uncertainty` | Latent state + KL-based intrinsic exploration bonus (stub) |

**Primary metric:** `pass@128` on MATH Level 5 hard pool (~350 problems).

---

## Architecture (v2 — rung 2)

```
repr_h  =  mean_pool(final-layer hidden states of chunk h)

Encoder:   repr_h (1536-dim)  →  (μ_h, log_σ²_h)   [64-dim each]
           z_h = μ_h          (deterministic tracking state)
           σ²_h               (calibrated uncertainty — high on incorrect trajectories)

Transition: z_h → z_{h+1}_predicted   (Markov consistency loss)

ZInjector: z_h (64-dim) → prefix_embedding (1536-dim)
           prepended to chunk h+1 via inputs_embeds
```

`σ²` is trained via an explicit calibration loss (`L_calib = BCE(sigmoid(mean_logvar), 1−reward)`),
not as a KL side-effect. This is the key architectural change from v1.

See `reports/latent_markov_design.md` for the full design and `reports/NEXT_STEPS_V2.md`
for the research ladder (rung 1 → rung 2 → rung 3 diffusion).

---

## Benchmark

**Hard pool:** MATH Level 5 problems for which the pretrained model scores `pass@128 = 0`
at temperature 1.0. ~350 problems after filtering. Mutual exclusivity with the Phase 0
easy pool (Level 1–4) is enforced by `source_level` field in both manifests.

**Why not MATH-Beyond?** v1 used MATH-B-I (40 problems, pass@1024). n=40 gives ±2.5pp
resolution per problem — too coarse to detect the 3–5pp improvements expected from a
latent Markov state at this model scale. Level 5 hard pool gives ~350 problems at
pass@128 — ±1.6pp per problem, at 8× lower compute cost.

---

## Model and Training Stack

| Component | Choice |
|-----------|--------|
| Policy backbone | `Qwen/Qwen2.5-1.5B-Instruct` (primary) |
| RL algorithm | GRPO via TRL (baseline) / custom loop (token-Markov, latent) |
| Encoder | Small MLP (~3M params); input: mean-pooled final-layer hidden states |
| Reward | Verifiable correctness vs. `ground_truth` (symbolic/numeric equivalence) |
| Evaluation decode | Temperature 1.0; k ∈ {1, 16, 128} |

---

## Repository Layout

```
configs/          YAML config tree (base_model, per-arm, eval, repro_tolerance)
data/             Pool JSONLs + manifests (built by prepare_*.py scripts)
reports/          DATA_PROTOCOL.md, latent_markov_design.md, NEXT_STEPS_V2.md, etc.
scripts/
  prepare_easy_pool.py         Build Phase 0 easy pool (MATH L1–L4)
  prepare_math_level5_pool.py  Build hard pool (MATH L5, pass@128=0 filter)
  train_baseline.py            Baseline GRPO training
  train_token_markov.py        Token-Markov arm training
  train_latent.py              Latent GRPO arm (Phase 0 and Phase 1)
  eval_passk.py                pass@k evaluation (all arms)
  run_ablation_table.py        Aggregate artifacts → ablation table
  eval_markov_diagnostics.py   E1 + E3 Markov diagnostics
  run_nfr6_gate.py             NFR6 UMAP gate (latent arm)
  calibrate_pool.py            Pool calibration utility
  check_reproducibility.py     Seed + hash reproducibility check
src/
  models/
    vae_state_encoder.py       Encoder, transition, ZInjector, OutcomeHead
    token_markov_state.py      Token-space Markov state (re-exports)
  training/
    grpo_baseline.py           Baseline GRPO loop (TRL)
    grpo_token_markov.py       Token-Markov GRPO loop (custom)
    grpo_latent.py             Latent GRPO loop (Phase 0 + Phase 1)
    reward_bonus.py            KL intrinsic reward bonus (stub — arm 4)
  utils/
    config_loader.py           YAML extends + deep merge
    seeding.py                 Deterministic seeding
artifacts/        Per-run directories (created by training scripts)
```

---

## Running the Experiments

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 0 — Build data pools (one-time, ~2h total):**
```bash
# Easy pool (Phase 0 encoder pretraining, Level 1-4)
python scripts/prepare_easy_pool.py --levels 1 2 3 4

# Level 5 hard pool (training + eval for all arms)
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output data/math_level5_hard_pool.jsonl
```

**Arm 1 — Baseline GRPO:**
```bash
# Training (~2h)
python scripts/train_baseline.py --config configs/train_baseline_grpo.yaml

# Eval
python scripts/eval_passk.py --generation-mode baseline \
    --eval-config configs/eval_math_beyond.yaml \
    --checkpoint artifacts/baseline_grpo/<run_id>/checkpoint-200 \
    --arm baseline_grpo \
    --output artifacts/baseline_grpo/<run_id>/eval_metrics.json
```

**Arm 2 — Token-Markov GRPO:**
```bash
python scripts/train_token_markov.py --config configs/train_token_markov_grpo.yaml

python scripts/eval_passk.py --generation-mode token_markov \
    --train-config configs/train_token_markov_grpo.yaml \
    --eval-config configs/eval_math_beyond.yaml \
    --checkpoint artifacts/token_markov_grpo/<run_id>/checkpoint-200 \
    --arm token_markov_grpo \
    --output artifacts/token_markov_grpo/<run_id>/eval_metrics.json
```

**Arm 3 — Latent GRPO (full sequence):**
```bash
# Phase 0 encoder pretraining (~4h for 400 steps)
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 0

# NFR6 gate — MUST pass before Phase 1
python scripts/run_nfr6_gate.py --config configs/train_latent_grpo.yaml \
    --n-problems 200 --n-rollouts 2

# Controlled baseline eval
python scripts/eval_passk.py --generation-mode latent_markov_pretrained \
    --train-config configs/train_latent_grpo.yaml --arm latent_grpo_pretrained \
    --output runs/latent_grpo/eval_pretrained.json

# Phase 1 joint RL (~2.5h for 200 steps)
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 1

# Phase 1 eval
PHASE1_CKPT=$(ls -td artifacts/latent_grpo/*/phase1/final | head -1)
python scripts/eval_passk.py --generation-mode latent_markov \
    --train-config configs/train_latent_grpo.yaml --checkpoint "$PHASE1_CKPT" \
    --arm latent_grpo --output runs/latent_grpo/eval_phase1.json

# E1 + E3 Markov diagnostics
python scripts/eval_markov_diagnostics.py \
    --checkpoint "$PHASE1_CKPT" \
    --config configs/train_latent_grpo.yaml \
    --n-problems 200 --n-rollouts 4 \
    --output-dir runs/latent_grpo/diagnostics
```

**Smoke tests (RTX 4060 8 GB — pipeline verification only):**
```bash
python scripts/train_baseline.py --config configs/train_baseline_grpo_smoke.yaml
python scripts/train_token_markov.py --config configs/train_token_markov_grpo_smoke.yaml
python scripts/train_latent.py --config configs/train_latent_grpo_smoke.yaml --phase 0
```

---

## Reproducibility

- Same pretrained checkpoint across all arms (or documented fallback, applied uniformly)
- Same benchmark pool, reward, eval budget, decode settings
- Seeds and tolerances in `configs/repro_tolerance.yaml`
- Final result table generated from `artifacts/` by `run_ablation_table.py` — not hand-typed
- Full spec: `PROJECT_CONTRACT.md`

---

## Related Work

| Paper | What it does | How this differs |
|-------|-------------|-----------------|
| Yuan & Xie (2026) — Markov States | Token-space state predictor breaks capability ceiling | State here is learned via encoder, not constructed; adds calibrated uncertainty |
| Yue et al. NeurIPS 2025 — Capability Ceiling | Proves RLVR doesn't expand reasoning frontier | Empirical motivation — the problem this work solves |
| Reasoning Palette | VAE latent as strategy prefix (sampled once per problem) | z_h evolves step-wise as a Markov state; different MDP formulation |
| Coconut (Meta, 2024) | Reasoning in latent space via continuous thought tokens | Inference-time only; no learned world model; no RL |
