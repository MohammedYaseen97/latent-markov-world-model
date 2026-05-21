# Experiment Reset — v2 Plan

**Decision date:** 2026-05-20  
**Reason:** v1 experiment (MATH-B-I, pass@1024, VAE-ELBO) was under-powered and
over-regularised. n=40 gives ±2.5pp resolution; KL weight=1.0 squashed σ²→1.0 killing
the uncertainty signal. Diagnostics (E1=0.015, E3 r=-0.233) confirmed the machinery
conceptually works. The recipe and benchmark need to be fixed.

---

## The Research Ladder (read this first)

This project sits on rung 2 of a deliberate 3-rung progression. Every architecture
decision in v2 is made to correctly occupy rung 2 — no more, no less.

```
RUNG 1 — Plain latent layer
  repr_h → Encoder → z_h (single point estimate)
  Proves: compact learned state > raw token history (Markov property holds)
  Missing: no uncertainty, no distribution, no path forward
  Contribution level: engineering

RUNG 2 — THIS PROJECT (v2)
  repr_h → Encoder → (μ_h, σ²_h)
  z_h = μ_h  (deterministic — used for Markov tracking)
  σ²_h       (calibrated — used for uncertainty signal)
  Proves: compact state + calibrated uncertainty breaks the capability ceiling
  Adds: σ² as epistemic signal (high = lost, low = confident)
  Contribution level: research (two claims: Markov state + calibrated uncertainty)

RUNG 3 — NEXT PROJECT (diffusion upgrade)
  repr_h → Encoder → (μ_h, σ²_h)            ← identical to rung 2
  z_h = μ_h                                   ← identical to rung 2
  + Diffusion model conditioned on z_h         ← new: generates expected repr_{h+1}
    → denoises toward the "next corridor"
    → when σ² is high, diffusion produces diverse futures (uncertain planning)
    → when σ² is low, diffusion produces concentrated futures (confident planning)
  Proves: imagination (visualising future states) improves reasoning over tracking alone
  Contribution level: full latent world model for abstract reasoning
```

**Why rung 2 is not a VAE:** A VAE adds a decoder (to reconstruct repr_h from z_h) and
sampling noise (`z = μ + ε·σ`) so you can generate novel repr_h values. Neither is
needed here — we are not generating anything, we are tracking. The decoder would be
replaced by the diffusion model in rung 3 anyway, so adding it now creates machinery to
immediately throw away. Rung 2 is a deterministic state tracker with calibrated
uncertainty. That is the right minimal architecture.

**Why rung 2 is not JEPA:** JEPA has no uncertainty output (no σ²). It predicts forward
but doesn't know how confident it is about the current state. Rung 2 needs σ² because
the uncertainty arm (Phase 3b) uses it as an exploration bonus, and rung 3 uses it to
modulate diffusion diversity. σ² is load-bearing for everything downstream.

---

## What Changes and Why

| Dimension | v1 (done) | v2 (next) | Why |
|---|---|---|---|
| **Eval benchmark** | MATH-B-I base, 40 problems | MATH Level 5, filtered, ~350 problems | n=40 → ±2.5pp/problem; can't detect effects below 5pp |
| **Primary metric** | pass@1024 | pass@128 | 8× cheaper; more problems compensates for lower k |
| **Easy pool (Phase 0)** | MATH L1–L5, 4974 problems | MATH L1–L4 only, ~4100 problems | Exclude Level 5 — pools must be mutually exclusive |
| **Latent architecture** | VAE-ELBO (decoder + KL + sampling) | Deterministic encoder + calibrated σ² | No generation needed; σ² trained explicitly not via KL side-effect |
| **Phase 0 steps** | 200 | 400 | Losses still declining at step 200 |
| **Phase 1 λ_trans** | 1.0 (40% gradient budget) | 0.3 (maintenance only) | E1=0.015: transition already well-trained in Phase 0 |
| **Arms re-run** | All at v1 settings | All at v2 settings | Can't mix benchmarks across arms |
| **Compute** | A100 80GB | RTX Pro 6000 Blackwell (96GB, ~2–2.5× faster BF16) | Shorter wall-clock |

---

## Pool Architecture (Mutually Exclusive)

```
MATH dataset (EleutherAI/hendrycks_math)
│
├── Level 1–4  →  Phase 0 easy pool  (VAE pretraining only)
│                 ~4100 problems, mixed difficulty, guaranteed solvable fraction
│                 File: data/math_easy_pool.jsonl
│
└── Level 5    →  Hard pool  (ALL arms: training + eval)
                  ~500 raw
                  → filter: keep problems where pretrained model scores pass@128 = 0
                  → ~350 problems remain
                  File: data/math_level5_hard_pool.jsonl
```

**Why filter by pass@128 = 0:** If the pretrained model can already solve a problem at
128 samples, it's not hard enough to test the ceiling-breaking hypothesis. This filter is
model-specific but reproducible given a fixed checkpoint and seed.

**Mutual exclusivity guarantee:** Level 5 problems are never in the easy pool (L1–4 only).
Easy pool problems are never in the hard pool (Level 5 only). Verified by source_level
field in both manifest files.

---

## Architecture: Deterministic State Encoder with Calibrated Uncertainty

### What this is

A small MLP that takes the backbone's hidden state summary for each chunk and produces:
- **μ_h** — where the model thinks it is in solution space after chunk h
- **σ²_h** — how confident the model is about that position

`z_h = μ_h` is used everywhere in the pipeline (transition, outcome head, ZInjector).  
`σ²_h` is trained separately to be high when the trajectory is incorrect and low when correct.

### What gets removed from v1

| Removed | Reason |
|---|---|
| `VAEStateEncoder.decode()` — decoder MLP | Not needed; we are not generating repr_h |
| `VAEStateEncoder.compute_elbo()` — ELBO loss | No reconstruction + no KL |
| Reparameterization trick `z = μ + ε·σ` | No sampling; z is deterministic = μ |
| `kl_warmup_frac`, `lambda_elbo`, `lambda_kl` | No KL term |
| `lambda_recon` | No reconstruction term |

### What stays / what changes

| Component | v1 | v2 |
|---|---|---|
| `encode()` → (μ, logvar) | ✓ unchanged | ✓ unchanged |
| `reparameterize()` | used in training | **only used at inference** (optional noise for diversity) |
| `transition()` — f(μ_h) → μ_{h+1} | ✓ unchanged | ✓ unchanged |
| `compute_transition_loss()` | ✓ unchanged | ✓ unchanged |
| `OutcomeHead` | z_3 → P(correct) | ✓ unchanged — still uses μ_3 |
| `ZInjector` | near-zero init | ✓ unchanged |
| `logvar_head` | trained by KL side-effect | **trained by explicit calibration loss** |

### New Phase 0 loss

```
L_phase0 = λ_trans × L_trans
         + λ_out   × L_out
         + λ_calib × L_calib

L_trans = Σ_h  MSE( transition(μ_h),  μ_{h+1}.detach() )
L_out   = BCE( outcome_head(μ_3),  reward )
L_calib = BCE( sigmoid( mean_logvar ),  1 − reward )

where mean_logvar = mean over chunks and latent dims of logvar_h
```

`L_calib` trains `logvar_head` directly: high logvar (high σ²) should predict incorrect
trajectories; low logvar should predict correct ones. No KL, no reconstruction, no
annealing schedule.

**Approximate gradient budget at Phase 0 peak:**
```
L_trans × 3.0  → ~50%   primary: Markov structure
L_out   × 5.0  → ~23%   primary: task-relevant z
L_calib × 1.0  → ~10%   explicit: σ² calibration
(remainder: variance across steps)
```

### Phase 0 config changes
- Remove: `kl_warmup_frac`, `lambda_elbo`, `lambda_kl`, `lambda_recon`
- Add: `lambda_calib: 1.0`
- Change: `phase0.n_steps`: 200 → **400**
- Keep: `lambda_trans_peak: 3.0`, `lambda_out: 5.0`

### Phase 1 loss

```
L_phase1 = L_RL  +  λ_trans × L_trans  +  λ_calib × L_calib
```

- `λ_trans = 0.3` (maintenance; transition already trained in Phase 0)
- `λ_calib = 0.5` (lighter anchor; σ² should stay calibrated but not dominate)
- `λ_vae` (the old VAE anchor term) removed — no VAE reconstruction to anchor

### Path to rung 3 (diffusion — next project)

The encoder architecture is identical in rung 3. The diffusion model adds ON TOP:

```python
# Rung 3 additional component (not implemented here — for reference only)
class ReasoningDiffusion(nn.Module):
    """Given z_h = μ_h, generates the expected repr_{h+1} by denoising.
    σ²_h controls diversity: high uncertainty → diverse futures.
    Trained with: L_diffusion = denoising_score_matching(repr_{h+1}, z_h)
    """
```

Nothing in the rung 2 architecture needs to change. The diffusion model is a new module
that reads z_h and writes a distribution over future states. Rung 2 must be completed
and validated before rung 3 begins.

---

## Implementation Checklist

### Step 0 — Pool preparation (one-time, before anything else)

- [ ] **`scripts/prepare_easy_pool.py`**: change default `--levels` to `1 2 3 4`.
  ```bash
  python scripts/prepare_easy_pool.py --levels 1 2 3 4
  ```

- [ ] **New `scripts/prepare_math_level5_pool.py`**: load MATH Level 5, run pretrained
  model at pass@128, keep problems where pass@128=0.
  ```bash
  python scripts/prepare_math_level5_pool.py \
      --model-id Qwen/Qwen2.5-1.5B-Instruct \
      --n-samples 128 \
      --output data/math_level5_hard_pool.jsonl
  ```

### Step 1 — Architecture update

- [ ] **`src/models/vae_state_encoder.py`**:
  - Remove `decode()` and `compute_elbo()`
  - Add `compute_calibration_loss(logvar_list, rewards)` — BCE of sigmoid(mean_logvar) vs (1−reward)
  - Keep `encode()`, `reparameterize()` (used at inference only), `transition()`, `compute_transition_loss()`

- [ ] **`src/training/grpo_latent.py`**:
  - `pretrain_vae_online`: replace ELBO call with `L_trans + L_out + L_calib`. Remove all kl_weight/kl_warmup logic.
  - `latent_training_step`: replace VAE anchor with `L_calib`. Update `lambda_trans` to 0.3.
  - Update logging: replace `elbo/recon/kl` with `calib`.

### Step 2 — Config updates

- [ ] **`configs/train_latent_grpo.yaml`**:
  - `phase0.n_steps`: 200 → 400
  - Remove `kl_warmup_frac`, `lambda_elbo`
  - Add `lambda_calib: 1.0`
  - `phase1_loss.lambda_trans_peak`: 1.0 → 0.3
  - `phase1_loss.lambda_calib`: 0.5 (new)
  - Remove `phase1_loss.lambda_vae`
  - `training.pool_path` → `data/math_level5_hard_pool.jsonl`

- [ ] **`configs/train_baseline_grpo.yaml`** and token-markov config:
  - `training.pool_path` → `data/math_level5_hard_pool.jsonl`

- [ ] **Eval config**: pool path → Level 5 hard pool; `n_samples` → 128

### Step 3 — Design documentation update

- [ ] **`reports/latent_markov_design.md`**: update architecture section, loss diagrams,
  parameter table. Remove ELBO/KL/decoder. Add calibration loss. Update Phase 0 steps to 400.

- [ ] **`PROJECT_CONTRACT.md`**: reset Phase 3 checklist for v2 deliverables and pass criteria.

- [ ] **`reports/DATA_PROTOCOL.md`**: add MATH Level 5 hard pool section; document mutual
  exclusivity with easy pool.

- [ ] **`reports/ablation_core.md`**: add v2 results table (v1 table stays as prior experiment).

---

## Run Order (v2, all commands in sequence)

```bash
# 0a. Rebuild easy pool (Level 1-4 only, no Level 5)
python scripts/prepare_easy_pool.py --levels 1 2 3 4

# 0b. Build MATH Level 5 hard pool (filter by pass@128=0 on pretrained model)
python scripts/prepare_math_level5_pool.py \
    --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --output data/math_level5_hard_pool.jsonl

# 1. Baseline arm (new benchmark + metric)
python scripts/train_baseline.py --config configs/train_baseline_grpo.yaml
python scripts/eval_passk.py --generation-mode baseline --arm baseline_grpo \
    --output runs/baseline_grpo/eval_v2.json

# 2. Token-Markov arm
python scripts/train_token_markov.py --config configs/train_token_markov_grpo.yaml
python scripts/eval_passk.py --generation-mode token_markov --arm token_markov_grpo \
    --output runs/token_markov_grpo/eval_v2.json

# 3a. Latent arm — Phase 0 (400 steps, β-VAE recipe)
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 0

# 3b. NFR6 gate (UMAP of z_final — must pass before Phase 1)
python scripts/run_nfr6_gate.py --config configs/train_latent_grpo.yaml \
    --n-problems 200 --n-rollouts 2

# 3c. Controlled latent baseline eval (pretrained backbone + Phase 0 encoder)
python scripts/eval_passk.py --generation-mode latent_markov_pretrained \
    --train-config configs/train_latent_grpo.yaml --arm latent_grpo_pretrained \
    --output runs/latent_grpo_v2/eval_pretrained.json

# 3d. Latent arm — Phase 1 (200 steps GRPO)
python scripts/train_latent.py --config configs/train_latent_grpo.yaml --phase 1

# 3e. Phase 1 eval
PHASE1_CKPT=$(ls -td artifacts/latent_grpo/*/phase1/final | head -1)
python scripts/eval_passk.py --generation-mode latent_markov \
    --train-config configs/train_latent_grpo.yaml --checkpoint "$PHASE1_CKPT" \
    --arm latent_grpo --output runs/latent_grpo_v2/eval_phase1.json

# 3f. E1 + E3 Markov diagnostics
python scripts/eval_markov_diagnostics.py \
    --checkpoint "$PHASE1_CKPT" \
    --config configs/train_latent_grpo.yaml \
    --n-problems 200 --n-rollouts 4 \
    --output-dir runs/latent_grpo_v2/diagnostics
```

---

## Compute Estimate (RTX Pro 6000 Blackwell)

| Stage | v2 estimate |
|---|---|
| Pool prep + Level 5 filtering | ~2h |
| Baseline training + eval | ~2h + 5min |
| Token-Markov training + eval | ~2h + 5min |
| Latent Phase 0 (400 steps) | ~4h |
| Latent Phase 1 (200 steps) | ~2.5h |
| Latent eval + diagnostics | ~45min |
| **Total** | **~14h** |

---

## The Central Diagnostic Question: Token Space or Solution Space?

The most important question this experiment must answer is not "did pass@128 go up?" —
it is: **is the encoder operating in solution space or token space?**

- **Token space**: z_h is a compressed summary of the tokens generated so far — a lossy
  hash of the text. The transition f(z_h)→z_{h+1} predicts "what will the next text
  summary look like," which depends on surface vocabulary and phrasing, not reasoning
  position. Token-space representations do not generalise across unseen problems because
  phrasing varies too much.

- **Solution space**: z_h encodes reasoning *position* — how far along the solution path
  the agent is, which subgoals have been resolved, which constraints remain. The transition
  predicts "where will the reasoning go next given where it is now." Solution-space
  representations generalise across problems because the underlying logical structure is
  shared even when surface text differs.

**The metric that distinguishes them: E1 × E3 together.**

| E1 held-out L_trans | E3 σ² spread | Interpretation |
|---|---|---|
| Low | Low (collapsed, σ²≈1.0) | **Degenerate** — z near-constant, trivially predictable (v1 failure mode) |
| Low | High (spread across outcomes) | ✅ **Solution space** — z encodes structure that generalises to unseen problems |
| High | High | Token space — z varies but doesn't predict meaningfully |
| High | Low | Degenerate — shouldn't occur |

E1 alone is ambiguous (v1 had E1=0.015 but σ²≈1.0 — trivially low because z barely
varied). The joint condition — **E1 low AND σ² genuinely spread (E3 r < −0.1 with real
Δσ²)** — is the only strong evidence for solution-space operation.

If both conditions hold in v2, the experiment has succeeded conceptually regardless of
the pass@128 number. If pass@128 also improves, the concept has practical effect at this
scale. If pass@128 doesn't improve despite both E1 and E3 passing, scale (model size,
training steps) is the remaining variable — not the architecture.

---

## Pass Criteria (v2)

| Criterion | Target |
|---|---|
| Easy pool: Level 1–4 only | No Level 5 problem_ids in manifest |
| Hard pool: Level 5 only | No Level 1–4 problem_ids; source_level=5 for all rows |
| Pools mutually exclusive | problem_id intersection = ∅ |
| NFR6 gate | UMAP shows outcome-correlated geometry in z-space |
| `latent_grpo_pretrained` pass@128 | ≥ baseline_pretrained pass@128 (controlled check) |
| `latent_grpo` pass@128 | ≥ baseline_grpo pass@128 + 3pp |
| **E1 + E3 joint (solution-space check)** | E1 < 0.5 AND E3 Pearson r < −0.1 AND Δσ²(correct vs incorrect) > 0.01 |
| No NaN blowups | L_RL, L_trans, L_calib all non-zero throughout Phase 1 |

---

## What v1 Results Become

v1 (MATH-B-I, pass@1024) is not discarded. It goes in the appendix / ablation notes:
- Establishes: concept machinery works (E1, E3, NFR6 all passed)
- Establishes: v1 recipe was under-calibrated (KL over-regularisation, σ²≈1.0)
- Establishes: v1 eval was under-powered (n=40, ±2.5pp/problem)

The v2 table is the main result. v1 is the motivation for v2.
