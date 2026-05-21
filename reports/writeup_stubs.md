# Write-up stubs

Short placeholders for later phases.

## Future work

### Batched Delethink trace generation for production pass@128 ✅ Implemented

`eval_passk.py --generation-mode token_markov` supports both vLLM (production)
and HF sequential (smoke/fallback) backends.

**vLLM multi-round path (use_vllm: true in eval config):**
- Round 1: identical to baseline — all problems × n_samples via vLLM, chunk_size
  completions at a time (same chunked loop as baseline, max_tokens=C).
- Rounds 2+: unique prompts (query + carryover_i per trace), batched in
  chunks of chunk_size with n=1 each, max_tokens=C-m (shorter per round).

**HF sequential fallback (use_vllm: false):** correct but slow; smoke use only.

### Rung 3 — Diffusion upgrade (next project)

After rung 2 (this project) is validated, rung 3 adds a diffusion model on top:

```python
class ReasoningDiffusion(nn.Module):
    """Given z_h = μ_h, generates expected repr_{h+1} by denoising.
    σ²_h controls diversity: high uncertainty → diverse futures.
    Trained with: L_diffusion = denoising_score_matching(repr_{h+1}, z_h)
    """
```

Nothing in the rung 2 architecture needs to change. The diffusion model reads z_h
and writes a distribution over future states. Rung 2 must be completed and
validated (E1 + E3 joint criterion) before rung 3 begins.

### Related work — Reasoning Palette (must handle explicitly in paper)

Reasoning Palette (VAE latent modulating reasoning strategy via token prefix) is
the closest adjacent paper. Differentiator that must appear clearly in related work:

- **Theirs:** VAE sampled *once per problem* as a strategy prefix before generation begins. Static per problem.
- **Ours:** `z_h` evolves *step-wise during the rollout* as a Markov state. Dynamic through reasoning.
- Different MDP formulation, not just a different architecture. They never question the history-as-state MDP. We do.

### Paper framing constraint (do not lose this)

**The load-bearing frame is the MDP reformulation, not the encoder architecture.**
Every section leads with: "We replace history-as-state with a compact learned Markov
state." The encoder is mentioned as the implementation. This distinction determines
the abstract, the intro hook, and the contribution bullets.

### Markov diagnostic results (TODO after runs complete)

Must include empirical evidence that `z_h` satisfies the Markov property:
- E1: held-out transition loss `f(z_h) → z_{h+1}` without history
- E3 joint: E1 low AND σ² genuinely spread (r < −0.1, Δσ² > 0.01)

## Main results

TODO: Populate after core table (`run_ablation_table.py`).

## Limitations

TODO: Document methodological and empirical limitations.

## Paper draft

TODO: Write arXiv-style manuscript draft (6–8 pages equivalent).
