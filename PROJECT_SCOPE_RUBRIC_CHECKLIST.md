# Project Rubric and Completion Checklist

This file owns only grading and completion tracking.

Authority references:

- Vision and scientific scope: `PROJECT_BRIEF.md`
- Execution phases and implementation gates: `PROJECT_CONTRACT.md`
- Yuan parity and claim eligibility: `YUAN_PARITY_CHECKLIST.md`
- MATH-B benchmark construction: `reports/DATA_PROTOCOL.md`

Do not duplicate or edit scope/policy text here; edit the owner document instead.

---

## Rubric (Mentor-Style Grading, 100 Points)

Publishable target: >= 80.

- **Problem framing and hypothesis clarity (15)**
  - Claim is precise, falsifiable, and aligned with capability-ceiling motivation.

- **Baseline and Yuan-comparator quality (25)**
  - Strong, stable `baseline_grpo` and faithful `token_markov_grpo` implementation with parity evidence.

- **Method implementation correctness (20)**
  - VAE latent integration is correct; uncertainty bonus is implemented as specified; no leakage/instability pathologies.

- **Experimental rigor and fairness (20)**
  - Matched budgets/decoding/splits, artifact-generated tables, reproducibility checks, seed policy followed.

- **Result quality and interpretation (10)**
  - Claims match evidence; positive or negative outcome is argued honestly with diagnostics.

- **Reproducibility and packaging (10)**
  - Commands, configs, artifacts, and documentation allow independent reruns.

---

## Completion Checklist (Project-Level)

- [x] Pre-implementation parity spec drafted (`reports/yuan_parity_spec.md`).
- [x] Pre-implementation final parity config scaffold created (`configs/final_parity/`).
- [x] Pre-implementation reproducibility tolerance policy created (`configs/repro_tolerance.yaml`).
- [x] MATH-B data protocol documented (`reports/DATA_PROTOCOL.md`).
- [x] MATH-B benchmark pipeline reproducible: `scripts/prepare_data.py`, pinned Hub revision (`configs/math_beyond_hf_revision.txt`), pinned `datasets` / `huggingface_hub` (`requirements.txt`), and committed `data/benchmark_manifest.json` (SHA-256 + `library_versions_at_build`). Config paths match the protocol (`configs/eval_math_beyond.yaml`, `configs/final_parity/base_parity.yaml`).
- [ ] `PROJECT_CONTRACT.md` Phase 1 passed.
- [ ] `PROJECT_CONTRACT.md` Phase 2 passed.
- [ ] `PROJECT_CONTRACT.md` Phase 3 passed.
- [ ] `PROJECT_CONTRACT.md` Phase 4 passed.
- [ ] `YUAN_PARITY_CHECKLIST.md` Section A fully checked.
- [ ] Core 4-arm ablation table produced from artifacts.
- [ ] Mandatory claim metric (`pass@1024`) reported for all 4 arms.
- [ ] Public GitHub repo delivered.
- [ ] arXiv-style preprint delivered.
- [ ] Blog post delivered.
- [ ] Bonus (optional): latent space interpretability visualization.
