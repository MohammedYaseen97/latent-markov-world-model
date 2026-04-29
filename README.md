# Latent Markov World Models for RL Post-Training

**Can a learned latent state break the capability ceiling of RL post-training on hard mathematics?**

This project trains and compares four RL post-training arms on a benchmark where standard RLVR is documented to fail — testing whether replacing token history with a learned latent representation of reasoning progress enables genuine capability expansion, not just sampling redistribution.

---

## Motivation

Reinforcement learning post-training (GRPO, PPO, RLVR) has a structural problem. The state fed to the policy is the full concatenation of all tokens generated so far — an unbounded, redundant, non-Markovian object that gives the model no compact summary of *where it actually is* in the solution space. The consequence is well-documented: RLVR improves sampling efficiency toward paths the base model could already take, but does not expand the reasoning frontier. The capability ceiling is a direct artifact of this degenerate state representation.

Yue et al. (NeurIPS 2025) proved this formally. Yuan & Xie (March 2026) confirmed it and showed that introducing explicit Markov states breaks the ceiling — using an external state predictor operating in token space.

Token space is still not first principles. A token-space summarizer compresses *language about* the solution space. It doesn't discover the structure of that space, and it carries no uncertainty signal.

**This project asks: what if the state was learned end-to-end from reasoning trajectories — a compressed latent belief over solution-space position, discovered from data, with epistemic uncertainty built in?**

---

## Core Hypothesis

A VAE trained on reasoning trajectories can learn a compact latent representation of where a reasoning agent currently is in the solution space. When an RL policy conditions on this latent state instead of raw token history, two things happen:

1. **The capability ceiling breaks** — the policy operates on a proper Markov state, enabling genuine discovery rather than redistribution of existing paths.
2. **Uncertainty drives exploration** — the variance of the VAE posterior is an epistemic signal: high variance means the model doesn't know where it is (explore); low variance means it's confident (exploit).

This is a minimal learned latent world model for abstract reasoning. The "world" being modeled is not a physical environment but an *epistemic state* — where the agent is in its understanding of the problem.

---

## The Four Arms

All arms share the same pretrained checkpoint, benchmark pool, reward function, and decoding budget. Only the state representation differs.

| Arm | State representation |
|-----|---------------------|
| `baseline_grpo` | Full token history (standard RLVR) |
| `token_markov_grpo` | Delethink-style RL-learned textual carryover — chunked generation with context reset (Markovian Thinker, ICLR 2026) |
| `latent_grpo` | VAE latent state — learned, no uncertainty bonus |
| `latent_grpo_uncertainty` | VAE latent state + KL-based intrinsic exploration bonus |

**Primary metric:** `pass@1024` (also `pass@1`, `pass@16`).

---

## VAE Design

**Input:** Final-layer hidden states from the backbone LM's reasoning trajectory — the model's internal representations averaged over the last N tokens at each reasoning step.

**Encoder:** MLP → (μ, σ²) in latent space **z**. Architecture: 2–3 layers, latent dim 64–128.

**Decoder:** MLP mapping **z** back to trajectory representation. Trained with standard ELBO: reconstruction loss + KL divergence.

**Policy conditioning:** The RL policy receives **z** (sampled from the encoder posterior) concatenated with the current token embedding before the policy head.

**Uncertainty bonus:** The intrinsic reward term is β_t × KL(q(**z**|τ) ∥ p(**z**)). High KL = high uncertainty = explore. β is annealed over training so exploration dominates early; exploitation dominates late.

---

## Benchmark

**Primary pool:** MATH-Beyond MATH-B-I (base model intersection) from Mayilvahanan et al. ([arXiv:2510.11653](https://arxiv.org/abs/2510.11653)). These are problems for which *every* listed base model scored `pass@1024 = 0` on the published Hub evaluation — 11 base model columns, pinned Hub revision.

This defines a hard capability-ceiling gauntlet using a published, reproducible, third-party difficulty filter. The scientific question is comparative: does the latent arm outperform the baselines on this fixed list under matched budgets?

> Problems are MATH-B-I (base-model intersection on the Hub table, N from `data/benchmark_manifest.json`); training uses **Qwen2.5-1.5B-Instruct** for rollout quality. The pool membership and the training checkpoint are separate decisions — documented in `reports/DATA_PROTOCOL.md`.

**Why not GSM8K or MATH?** Saturated. RLVR already works there. The capability ceiling is documented on MATH-Beyond and that's where the experiment is meaningful.

**Secondary pool:** AND of all 21 `*_unsolved` Hub columns (13 rows, stricter). Used for appendix / robustness.

---

## Model and Training Stack

| Component | Choice |
|-----------|--------|
| Policy backbone | `Qwen/Qwen2.5-1.5B-Instruct` (primary); `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (fallback) |
| RL algorithm | GRPO via TRL |
| VAE | Custom PyTorch (~200 lines); input: trajectory hidden states |
| Reward | Verifiable correctness vs. `ground_truth` (symbolic/numeric equivalence) |
| Evaluation decode | Temperature 1.0; k ∈ {1, 16, 1024} |

GRPO hyperparameters follow Yuan & Xie (2026) as a starting reference (LR 1e-6, KL coef 0.001, batch 128, group size 8) and are frozen identically across all four arms.

---

## Repository Layout

```
configs/                  YAML config tree (base_model, per-arm, eval, final_parity, repro_tolerance)
data/                     MATH-Beyond JSONLs + benchmark_manifest.json (built by prepare_data.py)
reports/                  DATA_PROTOCOL.md, writeup_stubs.md
scripts/
  prepare_data.py         Build MATH-Beyond JSONLs from HF (reproducible, SHA-256 checksummed)
  train_baseline.py       Baseline GRPO training entrypoint
  train_token_markov.py   Token-Markov arm entrypoint
  train_latent.py         Latent GRPO arm entrypoint (both modes)
  eval_passk.py           pass@k evaluation entrypoint
  run_ablation_table.py   Aggregate artifacts → ablation table
src/
  models/
    vae_state_encoder.py       VAE encoder/decoder
    token_markov_state.py      Token-space Markov state predictor
  training/
    grpo_baseline.py           Baseline GRPO loop
    grpo_token_markov.py       Token-Markov GRPO loop
    grpo_latent.py             Latent GRPO loop (with/without uncertainty)
    reward_bonus.py            KL intrinsic reward bonus
  eval/
    metrics.py                 pass@k computation
  utils/
    config_loader.py           YAML extends + deep merge
    seeding.py                 Deterministic seeding (Python, NumPy, PyTorch, CUDA)
    logging.py                 Experiment logger
artifacts/                Per-run directories: manifests, resolved configs, checkpoints, logs
```

---

## Reproducing the Data

The benchmark JSONLs in `data/` are fully reproducible from the pinned HF revision:

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --output-dir data
```

The Hub dataset revision is pinned in `configs/math_beyond_hf_revision.txt`. `data/benchmark_manifest.json` records the exact revision, row counts, column filters, and SHA-256 checksums of all three JSONL files.

---

## Connection to the Broader Vision

LeWM, JEPA, and Dreamer model what the *physical world* will look like next. This project models what the agent's *understanding of a problem* looks like right now, and how it should evolve. Same philosophical DNA — predict and plan in latent space, not raw observation space — applied to a completely different domain.

If a learned latent state over epistemic position works here, the generalization is direct:
- Replace VAE with a diffusion model → richer, progressive denoising of belief state
- Add identity conditioning → persona-conditioned belief initialization
- Scale beyond RL post-training → any domain where an agent tracks its own epistemic state

This project earns the right to make that larger claim.

---

## Related Work

| Paper | What it does | How this differs |
|-------|-------------|-----------------|
| Yuan & Xie (2026) — Markov States | Introduces external token-space state predictor; breaks the capability ceiling | State here is learned via VAE, not constructed; adds uncertainty signal |
| Yue et al. NeurIPS 2025 — Capability Ceiling | Proves RLVR doesn't expand the reasoning frontier | Empirical motivation — the problem this work solves |
| LeWM / JEPA (LeCun, 2026) | Latent world model for physical environments from pixels | Same philosophy, different domain: epistemic space, not physical space |
| Dreamer / DIAMOND | Diffusion/latent world models for pixel-based RL | Physical world, not abstract reasoning |
| Coconut (Meta, 2024) | Reasoning in latent space via continuous thought tokens | Inference-time only; no learned world model; no RL |

**Novel claim:** A VAE trained on reasoning trajectories learns a compact Markov state over solution-space position. An RL policy conditioned on this latent — with variance as an exploration bonus — breaks the capability ceiling on benchmarks where standard RLVR fails. Three things new together: (1) state is learned end-to-end, not constructed; (2) uncertainty is a first-class signal; (3) it instantiates a latent world model for abstract reasoning, a domain this has not been built for.

---

## Running the Experiments

**Install dependencies (exact versions pinned):**
```bash
pip install -r requirements.txt
```

**Smoke test (RTX 4060 8 GB — pipeline verification only):**
```bash
# Training
python scripts/train_baseline.py --config configs/train_baseline_grpo_smoke.yaml

# Eval
python scripts/eval_passk.py \
    --eval-config configs/eval_math_beyond_smoke.yaml \
    --pool-path data/math_beyond_smoke.jsonl \
    --arm-name baseline_grpo_smoke

# Reproducibility check (run twice, compare)
python scripts/check_reproducibility.py \
    artifacts/baseline_grpo/<run_1>/checkpoint-4/trainer_state.json \
    artifacts/baseline_grpo/<run_2>/checkpoint-4/trainer_state.json
```

**Production (A100 80 GB):**
```bash
# Training
python scripts/train_baseline.py --config configs/train_baseline_grpo.yaml

# Eval (uses vLLM for fast pass@1024 — 40 k completions)
python scripts/eval_passk.py \
    --eval-config configs/eval_math_beyond.yaml \
    --checkpoint artifacts/baseline_grpo/<run_id>/checkpoint-<N> \
    --arm-name baseline_grpo \
    --output artifacts/baseline_grpo/<run_id>/eval_metrics.json
```

**Token-Markov arm — smoke (RTX 4060 8 GB):**
```bash
python scripts/train_token_markov.py \
    --config configs/train_token_markov_grpo_smoke.yaml
```

**Token-Markov arm — production (A100 80 GB):**
```bash
# Training
python scripts/train_token_markov.py \
    --config configs/train_token_markov_grpo.yaml

# Eval — full pass@1024 (~30-45 min on A100 via vLLM multi-round)
python scripts/eval_passk.py \
    --generation-mode token_markov \
    --train-config configs/train_token_markov_grpo.yaml \
    --eval-config configs/eval_math_beyond.yaml \
    --checkpoint artifacts/token_markov_grpo/<run_id>/checkpoint-<N> \
    --arm-name token_markov_3chunk \
    --output artifacts/token_markov_grpo/<run_id>/eval_metrics.json
```

---

## Reproducibility Contract

- Same pretrained checkpoint across all four arms (or documented fallback, applied uniformly)
- Same benchmark pool, reward function, eval budget, and decode settings
- Seeds and tolerances in `configs/repro_tolerance.yaml`
- Final result table generated from `artifacts/` by `run_ablation_table.py` — not hand-typed
- Full spec: `PROJECT_CONTRACT.md`
