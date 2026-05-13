"""Latent Markov GRPO — v3 design: online Phase 0 + on-policy Phase 1.

Design reference: reports/latent_markov_design.md

v3 design
─────────
Both phases share the same generation engine (generate_latent_traces) and the
same backbone-forward helper (_run_pipeline_with_grad).  Traces carry only
chunk_ids, prompt_ids, and reward — no stored repr_h, z_h, μ, logvar,
or log_π_old.

Phase 0 — Online VAE Pretraining
─────────────────────────────────
Backbone FROZEN for weight updates; gradients still pass through activations.
VAE + ZInjector + OutcomeHead optimised jointly.
Each step:
  1. [no_grad] generate G=8 rollouts per problem → chunk_ids + reward
  2. [with_grad] re-run full 3-chunk pipeline → live repr_h, z_h
  3. losses: λ_elbo × L_ELBO + λ_trans × L_trans + λ_out × L_out
  4. step VAE+ZInjector+OutcomeHead optimizer; backbone grads zeroed, NOT stepped.

Gradient chain for L_trans through ZInjector:
  L_trans → z_{h+1} → repr_{h+1}[LIVE] → backbone → prefix_h → ZInjector

Phase 1 — Joint RL Training
─────────────────────────────
Backbone UNFROZEN. VAE + ZInjector loaded from Phase 0 checkpoint.
On-policy GRPO loop (200 steps): every step =
  1. [no_grad] collect G=8 fresh rollouts → chunk_ids + reward
  2. compute GRPO advantages from group rewards
  3. [with_grad] re-run full 3-chunk pipeline with same chunk_ids → live repr_h, z_h, log_π
  4. losses: L_RL + λ_t × L_trans + λ_vae × L_VAE
  5. step all optimizers (backbone + VAE + ZInjector)

IS = 1 exactly (same policy, no intermediate update).  No IS correction needed.
z prefixes in training forward pass use fresh ε ~ N(0,I) — this is an unbiased
REINFORCE-style gradient estimate; diversity comes from autoregressive sampling.

See reports/latent_markov_design.md for full design rationale.
"""
from __future__ import annotations

import json
import logging
import random as _random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.vae_state_encoder import (
    HIDDEN_DIM,
    LATENT_DIM,
    N_CHUNKS,
    OutcomeHead,
    VAEStateEncoder,
    ZInjector,
)
from src.training.grpo_baseline import SYSTEM_PROMPT, answers_equivalent, extract_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — unchanged from v2
# ---------------------------------------------------------------------------

def lambda_trans_schedule(
    step: int, max_steps: int = 200, floor: float = 0.1, peak: float = 3.0
) -> float:
    """Return the transition loss weight λ_t for the current training step.

    Schedule: linear warmup floor → peak over steps 0 … max_steps//2,
    then constant at peak.

    Rationale: at step 0 L_RL (Phase 1) or L_out (Phase 0) is near-zero.
    Starting at floor=0.1 lets early reward events shape z before the Markov
    constraint locks structure in.  Held at peak for the second half.

    Peak default of 3.0 is calibrated so that L_trans contributes ~20% of the
    total loss signal when L_ELBO ≈ 20 (observed Phase 0 scale with the 1536-dim
    ELBO summed over 3 chunks).  The original peak of 0.5 gave L_trans only ~4%
    of the gradient budget, leaving the transition model with insufficient signal.
    Both phases share this schedule; the peak is overridable via config.
    """
    halfway = max_steps // 2
    if step >= halfway:
        return peak
    return floor + (peak - floor) * (step / halfway)


def compute_grpo_advantages(
    rewards: list[float],
    group_size: int,
    eps: float = 1e-8,
) -> list[float]:
    """Normalise rewards into GRPO advantages within each group of G rollouts.

    For each group of `group_size` consecutive rollouts belonging to the same
    problem:  A_i = (r_i − μ_group) / (σ_group + eps)

    Args:
        rewards:    flat list of scalar rewards, length = n_problems × group_size.
                    Rollouts for the same problem must be contiguous.
        group_size: G — number of rollouts per problem.
        eps:        numerical stability floor for the std.

    Returns:
        List of advantages, same length and ordering as `rewards`.
    """
    assert len(rewards) % group_size == 0, (
        f"len(rewards)={len(rewards)} not divisible by group_size={group_size}"
    )
    advantages: list[float] = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        mu  = sum(group) / group_size
        var = sum((r - mu) ** 2 for r in group) / group_size
        sig = var ** 0.5
        for r in group:
            advantages.append((r - mu) / (sig + eps))
    return advantages


def format_prompt(problem: dict, tokenizer: AutoTokenizer) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem["prompt"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


# ---------------------------------------------------------------------------
# Backbone forward helper — hook-based last-hidden extraction
# ---------------------------------------------------------------------------

class _HiddenCapture:
    """Forward hook that captures the final layer-norm output.

    Registered on model.model.norm so that only the last layer's hidden states
    are held as a tensor reference — avoids materialising all 28 intermediate
    layers' activations via output_hidden_states=True.
    """
    __slots__ = ("val",)

    def __init__(self) -> None:
        self.val: torch.Tensor | None = None

    def __call__(self, module: Any, inp: Any, out: torch.Tensor) -> None:  # noqa: ARG002
        self.val = out


def _fwd_with_hidden(
    model: AutoModelForCausalLM,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model forward that returns (logits, last_hidden_states).

    Uses a hook on the final layer norm — efficient in both no_grad (generation)
    and with_grad (training) contexts.  Handles PEFT wrapping:

      Non-PEFT: model.model → Qwen2Model          → .norm
      PEFT:     model.model → Qwen2ForCausalLM    → .model.norm

    Returns:
        logits:      (B, seq_len, vocab_size)
        last_hidden: (B, seq_len, hidden_dim)   — output of final layer norm
    """
    cap = _HiddenCapture()
    inner = model.model
    if hasattr(inner, "norm"):
        norm_layer = inner.norm                      # plain Qwen2Model
    elif hasattr(inner, "model") and hasattr(inner.model, "norm"):
        norm_layer = inner.model.norm                # PEFT: LoRA-wrapped Qwen2ForCausalLM
    else:
        raise AttributeError(
            f"Cannot locate final layer norm. model.model is {type(inner).__name__}. "
            "Expected Qwen2Model (norm) or Qwen2ForCausalLM (model.norm)."
        )
    handle = norm_layer.register_forward_hook(cap)
    try:
        out = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
    finally:
        handle.remove()
    return out.logits, cap.val


# ---------------------------------------------------------------------------
# Rollout generation — shared by Phase 0 and Phase 1
# ---------------------------------------------------------------------------

def generate_latent_traces(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    problems: list[dict],
    n_rollouts: int,
    chunk_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    n_chunks: int = N_CHUNKS,
) -> list[dict]:
    """Generate G chunked rollouts per problem with z_h prefix injection.

    Runs entirely under torch.no_grad().  Computes repr_h and z_h internally
    to inject z prefixes between chunks, but does NOT store them in the
    returned traces.  Only chunk_ids, prompt_ids, and reward are stored.

    The training step (_run_pipeline_with_grad) re-runs the full pipeline with
    gradient to obtain live repr_h, z_h, and log_π from the same chunk_ids.

    Generation per rollout:
        Chunk 1: generate(prompt)  →  chunk1_ids
                 forward([prompt|chunk1]) → repr_1 → z_1 → prefix_1
        Chunk 2: generate([prefix_1|chunk1]) → chunk2_ids
                 forward([prefix_1|chunk1|chunk2]) → repr_2 → z_2 → prefix_2
        Chunk 3: generate([prefix_2|chunk2]) → chunk3_ids
                 grade(chunk1+chunk2+chunk3) → reward
        (No chunk-3 forward pass — z_3 not needed for prefix injection.)

    Args:
        model:       backbone in eval mode.
        tokenizer:   Qwen tokenizer (padding_side="left").
        vae:         VAEStateEncoder in eval mode.
        z_injector:  ZInjector in eval mode.
        problems:    list of problem dicts (keys: "prompt", "ground_truth").
        n_rollouts:  G — rollouts per problem.
        chunk_tokens: new tokens per chunk (config value).
        temperature: sampling temperature.
        top_p:       nucleus sampling cutoff.
        device:      CUDA device.
        n_chunks:    always 3 (forward-compat arg).

    Returns:
        Flat list of trace dicts, length = len(problems) × n_rollouts.
        Rollouts for the same problem are contiguous (required by
        compute_grpo_advantages).

        Each dict:
            "problem_id"  : str
            "rollout_idx" : int  (0 … G-1)
            "ground_truth": str
            "completion"  : str  (full decoded, all 3 chunks)
            "reward"      : int  (0 or 1)
            "prompt_ids"  : Tensor (prompt_len,)  on CPU
            "chunk_ids"   : list[Tensor]  — [c1, c2, c3] on CPU
    """
    pad_id      = tokenizer.eos_token_id
    embed_layer = model.get_input_embeddings()
    model_dtype = embed_layer.weight.dtype
    hidden_dim  = vae.hidden_dim

    # Build flat list: each problem repeated n_rollouts times (contiguous).
    all_prompt_ids:   list[list[int]] = []
    all_gt:           list[str]       = []
    all_problem_ids:  list[str]       = []
    all_rollout_idxs: list[int]       = []

    for prob in problems:
        pids = format_prompt(prob, tokenizer)
        for r in range(n_rollouts):
            all_prompt_ids.append(pids)
            all_gt.append(prob["ground_truth"])
            all_problem_ids.append(prob.get("problem_id", "unknown"))
            all_rollout_idxs.append(r)

    B              = len(all_prompt_ids)
    prompt_lengths = [len(p) for p in all_prompt_ids]
    max_prompt_len = max(prompt_lengths)

    # ── Chunk 1 generation — left-padded token IDs ────────────────────────
    input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
    for i, pids in enumerate(all_prompt_ids):
        off = max_prompt_len - len(pids)
        input_ids[i, off:] = torch.tensor(pids, dtype=torch.long, device=device)
        attn_mask[i, off:] = 1

    gen1 = model.generate(
        input_ids, attention_mask=attn_mask,
        max_new_tokens=chunk_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
    )
    chunk1_ids_list = [gen1[i, max_prompt_len:].cpu() for i in range(B)]
    del gen1, input_ids, attn_mask
    torch.cuda.empty_cache()

    # Forward [prompt | chunk1] for repr_1 → z_1 → prefix_1 (right-padded)
    full_seqs1 = [
        torch.cat([
            torch.tensor(all_prompt_ids[i], dtype=torch.long, device=device),
            chunk1_ids_list[i].to(device),
        ])
        for i in range(B)
    ]
    max_full1 = max(s.shape[0] for s in full_seqs1)
    fi1 = torch.full((B, max_full1), pad_id, dtype=torch.long, device=device)
    fa1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs1):
        L = seq.shape[0]; fi1[i, :L] = seq; fa1[i, :L] = 1

    _, hidden1 = _fwd_with_hidden(model, input_ids=fi1, attention_mask=fa1)

    repr_1_list: list[torch.Tensor] = []
    for i in range(B):
        pl = prompt_lengths[i]; rl = chunk1_ids_list[i].shape[0]
        repr_1_list.append(hidden1[i, pl:pl + rl, :].mean(0))
    del fi1, fa1, hidden1
    torch.cuda.empty_cache()

    # VAE → z_1 (deterministic: vae.eval() so reparameterize returns μ)
    # .float(): backbone runs in bf16; VAE MLP weights are fp32.
    repr_1_batch = torch.stack(repr_1_list).float()   # (B, hidden) fp32
    mu_1, logvar_1 = vae.encode(repr_1_batch)
    z_1_batch = vae.reparameterize(mu_1, logvar_1)    # (B, latent)
    del repr_1_batch, mu_1, logvar_1
    torch.cuda.empty_cache()

    # ── Chunk 2 generation — left-padded inputs_embeds [z_pfx | chunk1] ──
    z_pfx1   = z_injector.get_prefix_embedding(z_1_batch)   # (B, 1, H)
    c1_lens  = [c.shape[0] for c in chunk1_ids_list]
    max_c1   = max(c1_lens)
    emb_len2 = 1 + max_c1
    pad_emb  = embed_layer(torch.tensor([pad_id], dtype=torch.long, device=device))

    ie2 = torch.zeros(B, emb_len2, hidden_dim, dtype=model_dtype, device=device)
    am2 = torch.zeros(B, emb_len2, dtype=torch.long, device=device)
    for i in range(B):
        L1 = c1_lens[i]; off = max_c1 - L1
        if off > 0:
            ie2[i, :off, :] = pad_emb
        ie2[i, off, :]            = z_pfx1[i, 0, :]
        ie2[i, off + 1:off + 1 + L1, :] = embed_layer(chunk1_ids_list[i].to(device))
        am2[i, off:] = 1

    gen2 = model.generate(
        inputs_embeds=ie2, attention_mask=am2,
        max_new_tokens=chunk_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
    )
    chunk2_ids_list = [
        (gen2[i, emb_len2:] if gen2.shape[1] > chunk_tokens else gen2[i]).cpu()
        for i in range(B)
    ]
    del gen2, ie2, am2
    torch.cuda.empty_cache()

    # Forward [z_pfx1 | chunk1 | chunk2] for repr_2 → z_2 → prefix_2 (right-padded)
    c2_lens  = [c.shape[0] for c in chunk2_ids_list]
    max_fwd2 = max(1 + c1_lens[i] + c2_lens[i] for i in range(B))
    fe2 = torch.zeros(B, max_fwd2, hidden_dim, dtype=model_dtype, device=device)
    fa2 = torch.zeros(B, max_fwd2, dtype=torch.long, device=device)
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]; tot = 1 + L1 + L2
        fe2[i, 0, :]          = z_pfx1[i, 0, :]
        fe2[i, 1:1 + L1, :]   = embed_layer(chunk1_ids_list[i].to(device))
        fe2[i, 1 + L1:tot, :] = embed_layer(chunk2_ids_list[i].to(device))
        fa2[i, :tot] = 1

    _, hidden2 = _fwd_with_hidden(model, inputs_embeds=fe2, attention_mask=fa2)

    repr_2_list: list[torch.Tensor] = []
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]
        repr_2_list.append(hidden2[i, 1 + L1:1 + L1 + L2, :].mean(0))
    del fe2, fa2, hidden2, z_pfx1
    torch.cuda.empty_cache()

    # VAE → z_2
    repr_2_batch = torch.stack(repr_2_list).float()   # bf16→fp32 for VAE
    mu_2, logvar_2 = vae.encode(repr_2_batch)
    z_2_batch = vae.reparameterize(mu_2, logvar_2)
    del repr_2_batch, mu_2, logvar_2
    torch.cuda.empty_cache()

    # ── Chunk 3 generation — left-padded inputs_embeds [z_pfx2 | chunk2] ─
    z_pfx2   = z_injector.get_prefix_embedding(z_2_batch)   # (B, 1, H)
    max_c2   = max(c2_lens)
    emb_len3 = 1 + max_c2

    ie3 = torch.zeros(B, emb_len3, hidden_dim, dtype=model_dtype, device=device)
    am3 = torch.zeros(B, emb_len3, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; off = max_c2 - L2
        if off > 0:
            ie3[i, :off, :] = pad_emb
        ie3[i, off, :]            = z_pfx2[i, 0, :]
        ie3[i, off + 1:off + 1 + L2, :] = embed_layer(chunk2_ids_list[i].to(device))
        am3[i, off:] = 1

    gen3 = model.generate(
        inputs_embeds=ie3, attention_mask=am3,
        max_new_tokens=chunk_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
    )
    chunk3_ids_list = [
        (gen3[i, emb_len3:] if gen3.shape[1] > chunk_tokens else gen3[i]).cpu()
        for i in range(B)
    ]
    del gen3, ie3, am3, z_pfx2, z_1_batch, z_2_batch
    torch.cuda.empty_cache()

    # ── Grade and assemble traces ──────────────────────────────────────────
    trajectories: list[dict] = []
    for i in range(B):
        all_chunk_ids = torch.cat([
            chunk1_ids_list[i], chunk2_ids_list[i], chunk3_ids_list[i]
        ])
        completion = tokenizer.decode(all_chunk_ids, skip_special_tokens=True)
        pred   = extract_answer(completion)
        reward = int(pred is not None and answers_equivalent(pred, all_gt[i]))

        trajectories.append({
            "problem_id":   all_problem_ids[i],
            "rollout_idx":  all_rollout_idxs[i],
            "ground_truth": all_gt[i],
            "completion":   completion,
            "reward":       reward,
            "prompt_ids":   torch.tensor(all_prompt_ids[i], dtype=torch.long),
            "chunk_ids":    [chunk1_ids_list[i], chunk2_ids_list[i], chunk3_ids_list[i]],
        })

    return trajectories


# ---------------------------------------------------------------------------
# Shared training-time forward pass — Phase 0 and Phase 1
# ---------------------------------------------------------------------------

def _run_pipeline_with_grad(
    model: AutoModelForCausalLM,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    traces: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    """Re-run the full 3-chunk pipeline WITH gradient for all traces.

    Used by both Phase 0 training step (no L_RL) and Phase 1 training step.
    All repr_h and z_h are LIVE in the computation graph.

    Chunk 1: right-padded [prompt|chunk1] → repr_1 LIVE → z_1 → prefix_1
    Chunk 2: right-padded [prefix_1|chunk1|chunk2] → repr_2 LIVE → z_2 → prefix_2
    Chunk 3: right-padded [prefix_2|chunk2|chunk3] → repr_3 LIVE → z_3

    log_π per chunk: gathered from logits at the causal-LM shifted positions.

    Args:
        model, vae, z_injector: all in training mode.
        traces: output of generate_latent_traces() for this step.
        device: CUDA device.

    Returns dict with:
        "repr_list"    : list[3 × Tensor (B, hidden)]  — LIVE
        "z_list"       : list[3 × Tensor (B, latent)]  — LIVE
        "mu_list"      : list[3 × Tensor (B, latent)]  — LIVE
        "logvar_list"  : list[3 × Tensor (B, latent)]  — LIVE
        "log_pi_chunks": list[3 × list[B × Tensor(chunk_len,)]]  per-token log-prob
    """
    pad_id      = 0  # pad value for token-id inputs; actual vocab doesn't matter
    embed_layer = model.get_input_embeddings()
    model_dtype = embed_layer.weight.dtype
    B           = len(traces)

    prompt_ids_list  = [t["prompt_ids"]    for t in traces]
    chunk1_ids_list  = [t["chunk_ids"][0]  for t in traces]
    chunk2_ids_list  = [t["chunk_ids"][1]  for t in traces]
    chunk3_ids_list  = [t["chunk_ids"][2]  for t in traces]
    prompt_lengths   = [p.shape[0] for p in prompt_ids_list]
    c1_lens          = [c.shape[0] for c in chunk1_ids_list]
    c2_lens          = [c.shape[0] for c in chunk2_ids_list]
    c3_lens          = [c.shape[0] for c in chunk3_ids_list]

    # ── Chunk 1: [prompt | chunk1] right-padded ────────────────────────────
    full_seqs1 = [
        torch.cat([p.to(device), c.to(device)])
        for p, c in zip(prompt_ids_list, chunk1_ids_list)
    ]
    max_f1 = max(s.shape[0] for s in full_seqs1)
    fi1 = torch.zeros(B, max_f1, dtype=torch.long, device=device)
    fa1 = torch.zeros(B, max_f1, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs1):
        L = seq.shape[0]; fi1[i, :L] = seq; fa1[i, :L] = 1

    logits1, hidden1 = _fwd_with_hidden(model, input_ids=fi1, attention_mask=fa1)
    del fi1, fa1

    repr_1_batch = torch.stack([
        hidden1[i, prompt_lengths[i]:prompt_lengths[i] + c1_lens[i], :].mean(0)
        for i in range(B)
    ]).float()                                               # (B, hidden) LIVE fp32
    del hidden1

    log_pi_1: list[torch.Tensor] = []
    for i in range(B):
        pl = prompt_lengths[i]; rl = c1_lens[i]
        sl = logits1[i, pl - 1:pl + rl - 1, :]              # (rl, vocab) — causal shift
        c1 = chunk1_ids_list[i].to(device)
        lp = sl.gather(1, c1.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        log_pi_1.append(lp)
    del logits1

    mu_1, logvar_1 = vae.encode(repr_1_batch)
    z_1_batch = vae.reparameterize(mu_1, logvar_1)           # (B, latent) LIVE
    prefix_1  = z_injector.get_prefix_embedding(z_1_batch)   # (B, 1, H) LIVE

    # ── Chunk 2: [z_pfx | chunk1 | chunk2] right-padded ───────────────────
    max_f2 = max(1 + c1_lens[i] + c2_lens[i] for i in range(B))
    fe2 = torch.zeros(B, max_f2, model.config.hidden_size, dtype=model_dtype, device=device)
    fa2 = torch.zeros(B, max_f2, dtype=torch.long, device=device)
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]; tot = 1 + L1 + L2
        fe2[i, 0, :]          = prefix_1[i, 0, :]
        fe2[i, 1:1 + L1, :]   = embed_layer(chunk1_ids_list[i].to(device))
        fe2[i, 1 + L1:tot, :] = embed_layer(chunk2_ids_list[i].to(device))
        fa2[i, :tot] = 1

    logits2, hidden2 = _fwd_with_hidden(model, inputs_embeds=fe2, attention_mask=fa2)
    del fe2, fa2

    repr_2_batch = torch.stack([
        hidden2[i, 1 + c1_lens[i]:1 + c1_lens[i] + c2_lens[i], :].mean(0)
        for i in range(B)
    ]).float()                                               # (B, hidden) LIVE fp32
    del hidden2

    log_pi_2: list[torch.Tensor] = []
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]
        sl = logits2[i, L1:L1 + L2, :]                      # causal: pos L1 predicts c2[0]
        c2 = chunk2_ids_list[i].to(device)
        lp = sl.gather(1, c2.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        log_pi_2.append(lp)
    del logits2

    mu_2, logvar_2 = vae.encode(repr_2_batch)
    z_2_batch = vae.reparameterize(mu_2, logvar_2)           # (B, latent) LIVE
    prefix_2  = z_injector.get_prefix_embedding(z_2_batch)   # (B, 1, H) LIVE

    # ── Chunk 3: [z_pfx | chunk2 | chunk3] right-padded ───────────────────
    max_f3 = max(1 + c2_lens[i] + c3_lens[i] for i in range(B))
    fe3 = torch.zeros(B, max_f3, model.config.hidden_size, dtype=model_dtype, device=device)
    fa3 = torch.zeros(B, max_f3, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; L3 = c3_lens[i]; tot = 1 + L2 + L3
        fe3[i, 0, :]          = prefix_2[i, 0, :]
        fe3[i, 1:1 + L2, :]   = embed_layer(chunk2_ids_list[i].to(device))
        fe3[i, 1 + L2:tot, :] = embed_layer(chunk3_ids_list[i].to(device))
        fa3[i, :tot] = 1

    logits3, hidden3 = _fwd_with_hidden(model, inputs_embeds=fe3, attention_mask=fa3)
    del fe3, fa3

    repr_3_batch = torch.stack([
        hidden3[i, 1 + c2_lens[i]:1 + c2_lens[i] + c3_lens[i], :].mean(0)
        for i in range(B)
    ]).float()                                               # (B, hidden) LIVE fp32
    del hidden3

    log_pi_3: list[torch.Tensor] = []
    for i in range(B):
        L2 = c2_lens[i]; L3 = c3_lens[i]
        sl = logits3[i, L2:L2 + L3, :]
        c3 = chunk3_ids_list[i].to(device)
        lp = sl.gather(1, c3.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        log_pi_3.append(lp)
    del logits3

    mu_3, logvar_3 = vae.encode(repr_3_batch)
    z_3_batch = vae.reparameterize(mu_3, logvar_3)           # (B, latent) LIVE

    return {
        "repr_list":     [repr_1_batch, repr_2_batch, repr_3_batch],
        "z_list":        [z_1_batch,    z_2_batch,    z_3_batch],
        "mu_list":       [mu_1,         mu_2,         mu_3],
        "logvar_list":   [logvar_1,     logvar_2,     logvar_3],
        "log_pi_chunks": [log_pi_1,     log_pi_2,     log_pi_3],
    }


# ---------------------------------------------------------------------------
# Phase 0 — Online VAE Pretraining
# ---------------------------------------------------------------------------

def pretrain_vae_online(config: dict[str, Any], run_dir: Path) -> None:
    """Train VAEStateEncoder, ZInjector, and OutcomeHead online with frozen backbone.

    Each step:
      1. [no_grad] generate G rollouts per problem → chunk_ids + reward
      2. [with_grad] re-run full pipeline → live repr_h, z_h
      3. losses: λ_elbo × L_ELBO + λ_trans × L_trans + λ_out × L_out
      4. step VAE+ZInjector+OutcomeHead; backbone grads zeroed (not stepped)

    Config keys consumed (under "phase0"):
        model_id / revision / dtype  — backbone (from "primary")
        pool_path                    — data/math_easy_pool.jsonl
        n_steps                      — training steps (default: 200)
        batch_size                   — problems per step (default: 4)
        num_generations              — G rollouts per problem (default: 8)
        learning_rate                — AdamW lr for VAE/ZInj/OutcomeHead (default: 3e-4)
        lambda_elbo, lambda_trans_peak, lambda_out — loss weights / schedule peak
        kl_warmup_frac               — KL annealing fraction (default: 0.5)
        temperature, top_p           — sampling params (from training.*)
        chunk_tokens                 — tokens per chunk (from latent_markov.*)
        checkpoint_path              — where to save phase0_vae.pt
        logging_steps, save_steps
    """
    primary      = config["primary"]
    phase0_cfg   = config["phase0"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]

    model_id  = primary["huggingface_repo_id"]
    revision  = primary.get("revision", "main")
    dtype     = getattr(torch, primary.get("dtype", "bfloat16"))

    n_steps       = int(phase0_cfg.get("n_steps",         200))
    batch_size    = int(phase0_cfg.get("batch_size",        4))
    G             = int(phase0_cfg.get("num_generations",   8))
    lr            = float(phase0_cfg.get("learning_rate",  3e-4))
    lambda_elbo        = float(phase0_cfg.get("lambda_elbo",        1.0))
    lambda_trans_peak  = float(phase0_cfg.get("lambda_trans_peak",  3.0))
    lambda_out         = float(phase0_cfg.get("lambda_out",         1.0))
    kl_warmup_frac     = float(phase0_cfg.get("kl_warmup_frac",     0.5))
    temperature   = float(training_cfg.get("temperature",  1.0))
    top_p         = float(training_cfg.get("top_p",        1.0))
    chunk_tokens  = int(latent_cfg.get("chunk_tokens",     341))
    latent_dim    = int(latent_cfg.get("latent_dim",  LATENT_DIM))
    hidden_dim    = int(latent_cfg.get("hidden_dim",  HIDDEN_DIM))
    log_steps     = int(phase0_cfg.get("logging_steps",    10))
    save_steps    = int(phase0_cfg.get("save_steps",       50))
    pool_path     = Path(phase0_cfg.get("pool_path",
                                        "data/math_easy_pool.jsonl"))
    ckpt_path     = Path(phase0_cfg.get("checkpoint_path",
                                        str(run_dir / "phase0_vae.pt")))
    seed          = int(training_cfg.get("seed", 42))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 0 (online) — device: %s", device)

    # ------------------------------------------------------------------
    # Backbone — loaded but NOT added to any optimizer
    # requires_grad=True so that L_trans → ZInjector gradient flows
    # through backbone activations.  No .step() called on backbone.
    # ------------------------------------------------------------------
    logger.info("Loading backbone %s @ %s ...", model_id, revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("  backbone loaded (grad-enabled, optimizer NOT stepped)")

    # ------------------------------------------------------------------
    # VAE / ZInjector / OutcomeHead
    # ------------------------------------------------------------------
    vae          = VAEStateEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    z_injector   = ZInjector(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    outcome_head = OutcomeHead(latent_dim=latent_dim).to(device)

    vae_params = (
        list(vae.parameters())
        + list(z_injector.parameters())
        + list(outcome_head.parameters())
    )
    optimizer = torch.optim.AdamW(vae_params, lr=lr)

    if training_cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  gradient checkpointing enabled")

    # ------------------------------------------------------------------
    # Problem pool
    # ------------------------------------------------------------------
    with open(pool_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    logger.info("Phase 0 pool: %d problems from %s", len(problems), pool_path)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    log_history: list[dict] = []
    pending:     dict[str, float] = {}
    pool_order:  list[int] = []
    kl_warmup_steps = max(1, int(n_steps * kl_warmup_frac))

    step_bar = tqdm(total=n_steps, desc="phase0", unit="step", dynamic_ncols=True)

    for global_step in range(n_steps):
        kl_weight = min(1.0, global_step / kl_warmup_steps)
        lambda_t  = lambda_trans_schedule(global_step, n_steps, peak=lambda_trans_peak)

        # Sample batch_size problems (reshuffle when pool exhausted).
        while len(pool_order) < batch_size:
            order = list(range(len(problems)))
            _random.shuffle(order)
            pool_order.extend(order)
        step_problems = [problems[pool_order.pop(0)] for _ in range(batch_size)]

        # ── Rollout collection (no gradient) ──────────────────────────────
        with torch.no_grad():
            model.eval(); vae.eval(); z_injector.eval()
            traces = generate_latent_traces(
                model=model, tokenizer=tokenizer,
                vae=vae, z_injector=z_injector,
                problems=step_problems, n_rollouts=G,
                chunk_tokens=chunk_tokens,
                temperature=temperature, top_p=top_p,
                device=device,
            )

        model.train(); vae.train(); z_injector.train(); outcome_head.train()

        # ── Training step (with gradient) ─────────────────────────────────
        # Zero backbone grads from previous step (backbone is NOT in optimizer).
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad()

        pipe = _run_pipeline_with_grad(model, vae, z_injector, traces, device)

        rewards_t = torch.tensor(
            [float(t["reward"]) for t in traces],
            dtype=torch.float32, device=device,
        ).unsqueeze(-1)                                      # (B×G, 1)

        l_elbo, l_recon, l_kl = vae.compute_elbo(
            pipe["repr_list"], pipe["z_list"],
            pipe["mu_list"],   pipe["logvar_list"],
            kl_weight=kl_weight,
            return_decomposed=True,
        )
        l_trans = vae.compute_transition_loss(pipe["z_list"])
        l_out   = F.binary_cross_entropy(
            outcome_head(pipe["z_list"][-1]), rewards_t
        )

        total: torch.Tensor = lambda_elbo * l_elbo + lambda_t * l_trans + lambda_out * l_out

        total.backward()
        torch.nn.utils.clip_grad_norm_(vae_params, max_norm=1.0)
        optimizer.step()
        # Backbone grads accumulated during backward — zero now to prevent
        # stale accumulation across steps (backbone is never stepped).
        model.zero_grad(set_to_none=True)

        # ── Logging ───────────────────────────────────────────────────────
        for k, v in (("loss", total.detach().item()), ("elbo", l_elbo.detach().item()),
                     ("recon", l_recon), ("kl", l_kl),
                     ("trans", l_trans.detach().item()), ("out", l_out.detach().item())):
            pending[k] = pending.get(k, 0.0) + v
        reward_rate = sum(t["reward"] for t in traces) / len(traces)
        pending["reward_rate"] = pending.get("reward_rate", 0.0) + reward_rate

        if (global_step + 1) % log_steps == 0:
            n = log_steps
            entry = {
                "step":       global_step + 1,
                "kl_weight":  round(kl_weight, 4),
                "lambda_t":   round(lambda_t, 4),
                **{k: pending.get(k, 0.0) / n
                   for k in ("loss", "elbo", "recon", "kl", "trans", "out", "reward_rate")},
            }
            log_history.append(entry)
            pending = {}
            logger.info(
                "step %d | kl_w=%.3f λ_t=%.2f | loss=%.4f elbo=%.4f "
                "(recon=%.4f kl=%.4f) trans=%.4f out=%.4f | reward=%.1f%%",
                entry["step"], entry["kl_weight"], entry["lambda_t"],
                entry["loss"], entry["elbo"], entry["recon"], entry["kl"],
                entry["trans"], entry["out"],
                entry["reward_rate"] * 100,
            )

        if (global_step + 1) % save_steps == 0:
            _save_phase0_checkpoint(
                run_dir / f"checkpoint-{global_step + 1}",
                vae, z_injector, outcome_head, global_step + 1, log_history,
            )

        step_bar.set_postfix(
            loss=f"{total.item():.4f}",
            kl=f"{kl_weight:.2f}",
            rwd=f"{reward_rate:.0%}",
        )
        step_bar.update(1)

    step_bar.close()

    _save_phase0_checkpoint(
        ckpt_path.parent, vae, z_injector, outcome_head, n_steps, log_history
    )
    logger.info("Phase 0 complete. Checkpoint → %s", ckpt_path.parent)


def _save_phase0_checkpoint(
    directory: Path,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    outcome_head: OutcomeHead,
    step: int,
    log_history: list[dict],
) -> None:
    """Save VAE + ZInjector + OutcomeHead weights and trainer state."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vae":          vae.state_dict(),
            "z_injector":   z_injector.state_dict(),
            "outcome_head": outcome_head.state_dict(),
            "step":         step,
        },
        directory / "phase0_vae.pt",
    )
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}, indent=2)
    )


# ---------------------------------------------------------------------------
# Phase 1 — Joint RL Training
# ---------------------------------------------------------------------------

def latent_training_step(
    model: AutoModelForCausalLM,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    traces: list[dict],
    advantages: list[float],
    lambda_t: float,
    lambda_vae: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute the combined Phase 1 loss for one training step.

    L_total = L_RL  +  λ_t × L_transition  +  λ_vae × L_VAE

    L_RL (GRPO, no IS correction):
        The full 3-chunk pipeline is re-run with grad for each stored rollout.
        repr_h and z_h are LIVE — no stored constants from the rollout phase.
        z prefixes use fresh ε ~ N(0,I) drawn at training time; this is an
        unbiased REINFORCE-style gradient estimate (IS = 1 exactly at training
        time since the policy hasn't been updated since the rollout).

        L_RL = -mean_{i,h,t} [ advantage_i × log π_θ(token_t | context_{h,i}) ]

    L_VAE and L_transition:
        Computed from the same live forward pass.  All three losses flow
        gradients through backbone → repr_h → encoder → z_h → ZInjector.

    Args:
        model:      backbone in training mode (UNFROZEN — .step() called by caller).
        vae:        VAEStateEncoder in training mode.
        z_injector: ZInjector in training mode.
        traces:     output of generate_latent_traces() for this step.
        advantages: aligned GRPO advantages (output of compute_grpo_advantages()).
        lambda_t:   current transition loss weight (from lambda_trans_schedule()).
        lambda_vae: L_VAE loss weight (config: phase1_loss.lambda_vae).
        device:     CUDA device.

    Returns:
        Dict with backward-able scalar tensor "total" and detached scalars
        "l_rl", "l_vae", "l_trans" for logging.
    """
    pipe = _run_pipeline_with_grad(model, vae, z_injector, traces, device)

    l_vae   = vae.compute_elbo(
        pipe["repr_list"], pipe["z_list"],
        pipe["mu_list"],   pipe["logvar_list"],
    )
    l_trans = vae.compute_transition_loss(pipe["z_list"])

    # L_RL: -mean_{i,h,t} [ adv_i × log_π(token_t | context_{h,i}) ]
    adv = [float(a) for a in advantages]
    rl_sum    = torch.zeros(1, device=device)
    n_tokens  = 0
    for h, lp_chunk in enumerate(pipe["log_pi_chunks"]):   # 3 chunks
        for i, lp in enumerate(lp_chunk):                  # B×G traces
            rl_sum  = rl_sum + (-adv[i] * lp.sum())
            n_tokens += lp.shape[0]
    l_rl = rl_sum / max(n_tokens, 1)

    l_total = l_rl + lambda_t * l_trans + lambda_vae * l_vae

    return {
        "total":   l_total,
        "l_rl":    l_rl.detach(),
        "l_vae":   l_vae.detach(),
        "l_trans": l_trans.detach(),
    }


def train_latent(config: dict[str, Any], run_dir: Path) -> None:
    """Phase 1: joint GRPO training with live backbone and z injection.

    Loads Phase 0 VAE + ZInjector checkpoint, unfreezes backbone, and runs
    the on-policy GRPO loop for max_steps steps.

    Every training step:
      1. [no_grad] collect G=8 fresh rollouts for the current batch
      2. compute GRPO advantages from group rewards
      3. [with_grad] re-run full pipeline → L_RL + λ_t·L_trans + λ_vae·L_VAE
      4. step all optimizers (backbone lr=1e-6, VAE lr=3e-4)

    Config keys consumed:
        primary.*              — backbone model ID, revision, dtype
        phase0.checkpoint_path — path to phase0_vae.pt (VAE + ZInjector)
        latent_markov.*        — latent_dim, hidden_dim, chunk_tokens
        training.*             — seed, learning_rate, num_generations,
                                 batch_size, max_steps, temperature, top_p,
                                 gradient_checkpointing, logging_steps, save_steps
        phase1_loss.*          — lambda_vae, lambda_trans_peak
        evaluation.path        — MATH-B-I JSONL pool for RL training
    """
    primary      = config["primary"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]
    phase0_cfg   = config["phase0"]
    phase1_cfg   = config.get("phase1_loss", {})

    lambda_vae        = float(phase1_cfg.get("lambda_vae",        0.05))
    lambda_trans_peak = float(phase1_cfg.get("lambda_trans_peak", 3.0))

    model_id     = primary["huggingface_repo_id"]
    revision     = primary.get("revision", "main")
    dtype        = getattr(torch, primary.get("dtype", "bfloat16"))
    is_smoke     = (config.get("experiment") or {}).get("profile") == "smoke"

    seed         = int(training_cfg.get("seed", 42))
    lr_backbone  = float(training_cfg.get("learning_rate", 1e-6))
    lr_vae       = 3e-4        # VAE/ZInjector trained at Phase 0 rate throughout
    G            = int(training_cfg.get("num_generations",  8))
    batch_size   = int(training_cfg.get("batch_size",       4))
    max_steps    = int(training_cfg.get("max_steps",       200))
    temperature  = float(training_cfg.get("temperature",   1.0))
    top_p        = float(training_cfg.get("top_p",         1.0))
    log_steps    = int(training_cfg.get("logging_steps",   10))
    save_steps   = int(training_cfg.get("save_steps",      50))
    grad_clip    = 1.0

    chunk_tokens = int(latent_cfg.get("chunk_tokens",  341))
    latent_dim   = int(latent_cfg.get("latent_dim",  LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim",  HIDDEN_DIM))

    vae0_path    = Path(phase0_cfg.get("checkpoint_path", run_dir / "phase0_vae.pt"))
    pool_path    = Path(config["evaluation"]["path"])
    ckpt_path    = run_dir / "phase1"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 1 — device: %s  lambda_vae=%.4f", device, lambda_vae)

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    logger.info("Loading backbone %s @ %s ...", model_id, revision)
    if is_smoke:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            quantization_config=bnb_config, device_map="auto",
        )
        lora_cfg = LoraConfig(
            task_type="CAUSAL_LM", r=8, lora_alpha=16,
            target_modules="all-linear", lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            torch_dtype=dtype, device_map="auto",
        )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("  backbone ready")

    # ------------------------------------------------------------------
    # VAE + ZInjector — loaded from Phase 0 checkpoint
    # ------------------------------------------------------------------
    vae        = VAEStateEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    z_injector = ZInjector(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

    logger.info("Loading Phase 0 checkpoint from %s ...", vae0_path)
    ckpt = torch.load(vae0_path, weights_only=False, map_location=device)
    vae.load_state_dict(ckpt["vae"])
    z_injector.load_state_dict(ckpt["z_injector"])
    vae.train(); z_injector.train()
    logger.info("  VAE + ZInjector loaded (step %d)", ckpt.get("step", 0))

    if training_cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  gradient checkpointing enabled")

    # ------------------------------------------------------------------
    # Optimizer — two learning rates: backbone low, VAE/ZInjector higher
    # ------------------------------------------------------------------
    vae_params = list(vae.parameters()) + list(z_injector.parameters())
    optimizer  = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr_backbone},
        {"params": vae_params,          "lr": lr_vae},
    ])

    # ------------------------------------------------------------------
    # Training pool (MATH-B-I)
    # ------------------------------------------------------------------
    logger.info("Loading training pool from %s ...", pool_path)
    with open(pool_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    logger.info("  %d problems in training pool", len(problems))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    log_history: list[dict] = []
    pending:     dict[str, float] = {}
    pool_order:  list[int] = []

    step_bar = tqdm(total=max_steps, desc="phase1", unit="step", dynamic_ncols=True)

    for global_step in range(max_steps):
        lambda_t = lambda_trans_schedule(global_step, max_steps, peak=lambda_trans_peak)

        # Sample batch_size problems (reshuffle when pool exhausted).
        while len(pool_order) < batch_size:
            order = list(range(len(problems)))
            _random.shuffle(order)
            pool_order.extend(order)
        step_problems = [problems[pool_order.pop(0)] for _ in range(batch_size)]

        # ── Rollout collection (no gradient) ──────────────────────────────
        with torch.no_grad():
            model.eval(); vae.eval(); z_injector.eval()
            traces = generate_latent_traces(
                model=model, tokenizer=tokenizer,
                vae=vae, z_injector=z_injector,
                problems=step_problems, n_rollouts=G,
                chunk_tokens=chunk_tokens,
                temperature=temperature, top_p=top_p,
                device=device,
            )

        model.train(); vae.train(); z_injector.train()

        rewards    = [float(t["reward"]) for t in traces]
        advantages = compute_grpo_advantages(rewards, group_size=G)

        # ── Training step (with gradient) ─────────────────────────────────
        optimizer.zero_grad()

        metrics = latent_training_step(
            model=model, vae=vae, z_injector=z_injector,
            traces=traces, advantages=advantages,
            lambda_t=lambda_t, lambda_vae=lambda_vae,
            device=device,
        )

        loss: torch.Tensor = metrics["total"]
        loss.backward()
        all_params = list(model.parameters()) + vae_params
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        for k in ("total", "l_rl", "l_vae", "l_trans"):
            pending[k] = pending.get(k, 0.0) + metrics[k].item()
        pending["reward_rate"] = (
            pending.get("reward_rate", 0.0) + sum(rewards) / len(rewards)
        )

        if (global_step + 1) % log_steps == 0:
            n = log_steps
            entry = {
                "step":     global_step + 1,
                "lambda_t": round(lambda_t, 4),
                **{k: pending.get(k, 0.0) / n
                   for k in ("total", "l_rl", "l_vae", "l_trans", "reward_rate")},
            }
            log_history.append(entry)
            pending = {}
            logger.info(
                "step %d | λ_t=%.2f λ_vae=%.3f | total=%.4f rl=%.4f "
                "vae=%.4f trans=%.4f | reward=%.1f%%",
                entry["step"], entry["lambda_t"], lambda_vae,
                entry["total"], entry["l_rl"],
                entry["l_vae"], entry["l_trans"],
                entry["reward_rate"] * 100,
            )

        if (global_step + 1) % save_steps == 0:
            _save_phase1_checkpoint(
                ckpt_path / f"checkpoint-{global_step + 1}",
                model, vae, z_injector, optimizer, global_step + 1, log_history,
                tokenizer=tokenizer,
            )

        step_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            rl=f"{metrics['l_rl'].item():.4f}",
            λ_t=f"{lambda_t:.2f}",
        )
        step_bar.update(1)

    step_bar.close()

    _save_phase1_checkpoint(
        ckpt_path / "final", model, vae, z_injector,
        optimizer, max_steps, log_history, tokenizer=tokenizer,
    )
    logger.info("Phase 1 complete. Checkpoint → %s", ckpt_path / "final")


def _save_phase1_checkpoint(
    directory: Path,
    model: "AutoModelForCausalLM",
    vae: "VAEStateEncoder",
    z_injector: "ZInjector",
    optimizer: torch.optim.Optimizer,
    step: int,
    log_history: list[dict],
    tokenizer=None,
) -> None:
    """Save backbone, VAE, ZInjector, and optimizer state."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vae":        vae.state_dict(),
            "z_injector": z_injector.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "step":       step,
        },
        directory / "phase1_latent.pt",
    )
    model.save_pretrained(str(directory / "backbone"))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(directory / "backbone"))
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}, indent=2)
    )
