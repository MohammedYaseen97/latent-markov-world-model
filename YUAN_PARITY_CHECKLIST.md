# Yuan Parity Checklist (Scope-Safe)

Use this checklist to ensure our `token_markov_grpo` comparator is strong and defensible without expanding project scope.

## Goal

Reproduce Yuan's core comparator behavior on claim-critical dimensions only, then compare fairly against our latent-state methods.

Success standard:

> Comparator is faithful on claim-critical dimensions; differences are documented; results are artifact-backed.

---

## Section A: Must Match (Non-Negotiable)

All items in this section must be checked before final results are accepted.

Status note (current): only documentation-level parity extraction is complete; implementation and final-run checks remain pending.

### A1) Comparator Definition

- [ ] `token_markov_grpo` is implemented as a token-space Markov state predictor integrated into GRPO.
- [ ] Implementation reflects Yuan's method intent (not a toy or weakened variant).
- [ ] Comparator code path is isolated and reproducible (`scripts/train_token_markov.py`).

### A2) Shared Experimental Substrate

- [ ] Same base model family as our other arms for fair comparison.
- [ ] Same benchmark task and same train/eval split definitions.
- [ ] Same reward correctness definition for final reported runs.
- [ ] Same training-step budget / rollout budget across all core arms.
- [ ] Same maximum generation length across all core arms.
- [ ] Same evaluation decode settings across all core arms.

### A3) Core Metric and Reporting

- [ ] `pass@1024` is reported for baseline, token Markov, latent, latent+uncertainty.
- [ ] Secondary metrics (`pass@1`, `pass@16`) are reported but not used as the primary claim metric.
- [ ] Core ablation table is generated from artifacts, not manually assembled.

### A4) Reproducibility + Fairness

- [ ] Seed policy is identical across all core arms (or explicitly justified if not feasible).
- [x] Final configs are captured in versioned files under `configs/final_parity/`.
- [ ] Run artifacts (logs, checkpoints, eval outputs) are preserved for all core runs.

### A5) Explicit Parity Documentation

- [x] `reports/yuan_parity_spec.md` exists and includes:
  - [x] Parameters directly matched to Yuan.
  - [x] Parameters Yuan does not specify and our chosen values.
  - [x] Any unavoidable deviations from Yuan with rationale.
  - [x] Why each deviation does not invalidate the comparison.

---

## Section B: Out of Scope (Explicitly Excluded)

These are not required for this project and must not be added before core deliverables are complete.

- [ ] Reproducing all ablations from Yuan's paper.
- [ ] Reproducing Yuan across multiple model families.
- [ ] Reproducing Yuan across additional benchmarks before core 4-arm result is complete.
- [ ] Reproducing every diagnostic figure from Yuan.
- [ ] Adding new algorithmic branches (diffusion, persona conditioning, extra RL algorithms) in this phase.

If any item above is requested mid-project, record it in `reports/future_work.md` and continue core plan.

---

## Acceptance Gate for "We Beat Yuan-Style Baseline"

You may claim a strong comparative result only if all conditions hold:

- [ ] Section A is fully complete.
- [ ] Core table shows `pass@1024` comparison across all 4 arms.
- [ ] Compute and decoding parity are documented.
- [ ] Result is reproducible from scripts/configs in repo.

If these are not all true, use a weaker claim phrasing in paper/blog.

---

## Allowed Claim Templates

Use one of these templates in `reports/main_results.md` and paper draft:

1. **Strong claim (all gates passed)**
   - "Under Yuan-faithful token-Markov parity conditions, our latent-state method outperforms strong baselines on `pass@1024`."

2. **Qualified claim (minor parity gaps)**
   - "Our latent-state method outperforms our implemented token-Markov baseline on `pass@1024`; parity deviations are documented and likely non-decisive."

3. **Negative result claim**
   - "Under parity-controlled comparisons, latent state did not outperform token-Markov baseline; we provide a diagnosis of failure modes."

---

## Quick Decision Rule (When Unsure)

Ask:

1. Does this change improve claim-critical fairness with Yuan?
2. Is it required for `pass@1024` core-table credibility?
3. Can we do it without delaying core 4-arm completion?

If any answer is "no", do not add it now.
