"""Latent Markov GRPO — v2 design: online Phase 0 + on-policy Phase 1.

Design reference: reports/latent_markov_design.md

Architecture change from v1: VAE-ELBO replaced by deterministic encoder +
explicit calibration loss. z_h = μ_h (no sampling during training). σ² is
trained directly via L_calib = BCE(sigmoid(mean_logvar), 1 − reward).

Phase 0 — Online Encoder Pretraining
──────────────────────────────────────
Backbone FROZEN for weight updates; gradients still pass through activations.
Encoder + ZInjector + OutcomeHead optimised jointly.
Each step:
  1. [no_grad] generate G=128 rollouts per problem → chunk_ids + reward
  2. [with_grad] re-run full 3-chunk pipeline → live repr_h, z_h (= μ_h)
  3. losses: λ_trans × L_trans + λ_out × L_out + λ_calib × L_calib
  4. step Encoder+ZInjector+OutcomeHead optimizer; backbone grads zeroed, NOT stepped.

Phase 1 — Joint RL Training
─────────────────────────────
Backbone UNFROZEN. Encoder + ZInjector loaded from Phase 0 checkpoint.
On-policy GRPO loop (200 steps): every step =
  1. [no_grad] collect G=128 fresh rollouts → chunk_ids + reward
  2. compute GRPO advantages from group rewards
  3. [with_grad] re-run full 3-chunk pipeline → live repr_h, z_h, log_π
  4. losses: L_RL + λ_t × L_trans + λ_calib × L_calib
  5. step all optimizers (backbone + Encoder + ZInjector)

G=128 matches eval pass@128 scale. IS = 1 exactly. z_h = μ_h (deterministic).

See reports/latent_markov_design.md for full design rationale.
"""
from __future__ import annotations

import json
import logging
import math
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
    step: int, max_steps: int = 400, floor: float = 0.1, peak: float = 3.0
) -> float:
    """Return the transition loss weight λ_t for the current training step.

    Schedule: linear warmup floor → peak over steps 0 … max_steps//2,
    then constant at peak.

    Phase 0 default (max_steps=400, peak=3.0): aggressive Markov training.
    Phase 1 default (peak=0.3): maintenance only — transition already well-trained.

    Both phases share this schedule; peak is overridable via config.
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
    need_logits: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Model forward that returns (logits | None, last_hidden_states).

    When need_logits=False (Phase 0 — no L_RL), calls model.model(...) directly,
    completely bypassing lm_head.  For B=128 seqs × vocab=151,936 this skips a
    ~26 GB peak allocation per chunk (3 chunks → ~79 GB saved over one step).

    When need_logits=True (Phase 1 — L_RL needs log π), uses a hook on the final
    layer norm to capture hidden states while the full model forward runs.

    Handles PEFT wrapping (hook path only):
      Non-PEFT: model.model → Qwen2Model          → .norm
      PEFT:     model.model → Qwen2ForCausalLM    → .model.norm

    Returns:
        logits:      (B, seq_len, vocab_size)  or None when need_logits=False
        last_hidden: (B, seq_len, hidden_dim)  — output of final layer norm
    """
    if not need_logits:
        # Call the base transformer directly — lm_head is never invoked.
        # model.model is Qwen2Model which returns last_hidden_state directly.
        out = model.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return None, out.last_hidden_state

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

@torch.no_grad()
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

    Runs entirely under torch.no_grad() (enforced by decorator — callers do
    not need their own context manager).  Computes repr_h and z_h internally
    to inject z prefixes between chunks, but does NOT store them in the
    returned traces.  Only chunk_ids, prompt_ids, and reward are stored.

    The training step (_run_pipeline_with_grad) re-runs the full pipeline with
    gradient to obtain live repr_h, z_h, and log_π from the same chunk_ids.

    Generation per rollout:
        Chunk 1: generate(prompt)  →  chunk1_ids
                 forward([prompt|chunk1]) → repr_1 → z_1 → prefix_1
        Chunk 2: generate([prefix_1|chunk1]) → chunk2_ids        (crutch: chunk1 kept for generation quality)
                 forward([prefix_1|chunk2])       → repr_2 → z_2 → prefix_2  (strict Markov)
        Chunk 3: generate([prefix_2|chunk2]) → chunk3_ids        (crutch: chunk2 kept for generation quality)
                 grade(chunk1+chunk2+chunk3) → reward
        (No chunk-3 repr forward here — z_3 not needed for prefix injection;
         repr_3 → z_3 is computed in _run_pipeline_with_grad with strict Markov
         context [prefix_2|chunk3].)

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

    # need_logits=False: logits are never used during generation — skipping lm_head
    # saves the (B, seq, 151936) tensor (~6.6 GB at B=32) on every forward call.
    _, hidden1 = _fwd_with_hidden(model, input_ids=fi1, attention_mask=fa1,
                                  need_logits=False)

    repr_1_list: list[torch.Tensor] = []
    for i in range(B):
        pl = prompt_lengths[i]; rl = chunk1_ids_list[i].shape[0]
        repr_1_list.append(hidden1[i, pl:pl + rl, :].mean(0))
    del fi1, fa1, hidden1

    # VAE → z_1 (deterministic: vae.eval() so reparameterize returns μ)
    # .float(): backbone runs in bf16; VAE MLP weights are fp32.
    repr_1_batch = torch.stack(repr_1_list).float()   # (B, hidden) fp32
    mu_1, logvar_1 = vae.encode(repr_1_batch)
    z_1_batch = vae.reparameterize(mu_1, logvar_1)    # (B, latent)
    del repr_1_batch, mu_1, logvar_1

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

    # Forward [z_pfx1 | chunk2] for repr_2 → z_2 → prefix_2
    # Strict Markov: same as _run_pipeline_with_grad — chunk-1 raw tokens dropped.
    # The z_2 injected for chunk-3 generation is computed under the same context
    # as the z_2 in the training pass, keeping generation and training consistent.
    c2_lens  = [c.shape[0] for c in chunk2_ids_list]
    max_fwd2 = max(1 + c2_lens[i] for i in range(B))
    fe2 = torch.zeros(B, max_fwd2, hidden_dim, dtype=model_dtype, device=device)
    fa2 = torch.zeros(B, max_fwd2, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; tot = 1 + L2
        fe2[i, 0, :]        = z_pfx1[i, 0, :]
        fe2[i, 1:1 + L2, :] = embed_layer(chunk2_ids_list[i].to(device))
        fa2[i, :tot] = 1

    _, hidden2 = _fwd_with_hidden(model, inputs_embeds=fe2, attention_mask=fa2,
                                  need_logits=False)

    repr_2_list: list[torch.Tensor] = []
    for i in range(B):
        L2 = c2_lens[i]
        repr_2_list.append(hidden2[i, 1:1 + L2, :].mean(0))
    del fe2, fa2, hidden2, z_pfx1

    # VAE → z_2
    repr_2_batch = torch.stack(repr_2_list).float()   # bf16→fp32 for VAE
    mu_2, logvar_2 = vae.encode(repr_2_batch)
    z_2_batch = vae.reparameterize(mu_2, logvar_2)
    del repr_2_batch, mu_2, logvar_2

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
    *,
    compute_log_pi: bool = True,
) -> dict[str, Any]:
    """Re-run the full 3-chunk pipeline WITH gradient for all traces.

    Used by both Phase 0 training step (no L_RL) and Phase 1 training step.
    All repr_h and z_h are LIVE in the computation graph.

    Chunk 1: right-padded [prompt|chunk1] → repr_1 LIVE → z_1 → prefix_1
    Chunk 2: right-padded [prefix_1|chunk2] → repr_2 LIVE → z_2 → prefix_2
    Chunk 3: right-padded [prefix_2|chunk3] → repr_3 LIVE → z_3

    log_π per chunk: gathered from logits at the causal-LM shifted positions.
    When compute_log_pi=False (Phase 0), lm_head is bypassed entirely — saves
    ~26 GB per chunk (B=128 × seq≈683 × vocab=151,936 × 2 B).

    Args:
        model, vae, z_injector: all in training mode.
        traces: output of generate_latent_traces() for this step.
        device: CUDA device.
        compute_log_pi: set False for Phase 0 (L_RL not used; logits not needed).

    Returns dict with:
        "repr_list"    : list[3 × Tensor (B, hidden)]  — LIVE
        "z_list"       : list[3 × Tensor (B, latent)]  — LIVE
        "mu_list"      : list[3 × Tensor (B, latent)]  — LIVE
        "logvar_list"  : list[3 × Tensor (B, latent)]  — LIVE
        "log_pi_chunks": list[3 × list[B × Tensor(chunk_len,)]]  — empty when compute_log_pi=False
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

    logits1, hidden1 = _fwd_with_hidden(
        model, input_ids=fi1, attention_mask=fa1, need_logits=compute_log_pi
    )
    del fi1, fa1

    repr_1_batch = torch.stack([
        hidden1[i, prompt_lengths[i]:prompt_lengths[i] + c1_lens[i], :].mean(0)
        for i in range(B)
    ]).float()                                               # (B, hidden) LIVE fp32
    del hidden1

    log_pi_1: list[torch.Tensor] = []
    if compute_log_pi:
        for i in range(B):
            pl = prompt_lengths[i]; rl = c1_lens[i]
            sl = logits1[i, pl - 1:pl + rl - 1, :]          # (rl, vocab) — causal shift
            c1 = chunk1_ids_list[i].to(device)
            lp = sl.gather(1, c1.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
            log_pi_1.append(lp)
    del logits1

    mu_1, logvar_1 = vae.encode(repr_1_batch)
    z_1_batch = vae.reparameterize(mu_1, logvar_1)           # (B, latent) LIVE
    prefix_1  = z_injector.get_prefix_embedding(z_1_batch)   # (B, 1, H) LIVE

    # ── Chunk 2: [z_pfx1 | chunk2] right-padded ───────────────────────────
    # Strict Markov: chunk-1 raw tokens are dropped. repr_2 is conditioned only
    # on z_1 (the Markov state) + chunk-2 tokens. The backbone cannot attend to
    # raw chunk-1 history, forcing z_1 to actually carry the necessary context.
    # Generation still uses [z_pfx1 | chunk1] as a quality crutch during Phase 0;
    # for Phase 1 generation should be updated to [z_pfx1] only for full consistency.
    max_f2 = max(1 + c2_lens[i] for i in range(B))
    fe2 = torch.zeros(B, max_f2, model.config.hidden_size, dtype=model_dtype, device=device)
    fa2 = torch.zeros(B, max_f2, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; tot = 1 + L2
        fe2[i, 0, :]        = prefix_1[i, 0, :]
        fe2[i, 1:1 + L2, :] = embed_layer(chunk2_ids_list[i].to(device))
        fa2[i, :tot] = 1

    logits2, hidden2 = _fwd_with_hidden(
        model, inputs_embeds=fe2, attention_mask=fa2, need_logits=compute_log_pi
    )
    del fe2, fa2

    repr_2_batch = torch.stack([
        hidden2[i, 1:1 + c2_lens[i], :].mean(0)
        for i in range(B)
    ]).float()                                               # (B, hidden) LIVE fp32
    del hidden2

    log_pi_2: list[torch.Tensor] = []
    if compute_log_pi:
        for i in range(B):
            L2 = c2_lens[i]
            sl = logits2[i, 0:L2, :]   # pos 0 (z_pfx1) predicts c2[0], pos 1 predicts c2[1]…
            c2 = chunk2_ids_list[i].to(device)
            lp = sl.gather(1, c2.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
            log_pi_2.append(lp)
    del logits2

    mu_2, logvar_2 = vae.encode(repr_2_batch)
    z_2_batch = vae.reparameterize(mu_2, logvar_2)           # (B, latent) LIVE
    prefix_2  = z_injector.get_prefix_embedding(z_2_batch)   # (B, 1, H) LIVE

    # ── Chunk 3: [z_pfx2 | chunk3] right-padded ───────────────────────────
    # Same strict Markov logic: chunk-2 raw tokens dropped. repr_3 conditioned
    # only on z_2 + chunk-3 tokens. Symmetric with chunk-2 treatment above.
    max_f3 = max(1 + c3_lens[i] for i in range(B))
    fe3 = torch.zeros(B, max_f3, model.config.hidden_size, dtype=model_dtype, device=device)
    fa3 = torch.zeros(B, max_f3, dtype=torch.long, device=device)
    for i in range(B):
        L3 = c3_lens[i]; tot = 1 + L3
        fe3[i, 0, :]        = prefix_2[i, 0, :]
        fe3[i, 1:1 + L3, :] = embed_layer(chunk3_ids_list[i].to(device))
        fa3[i, :tot] = 1

    logits3, hidden3 = _fwd_with_hidden(
        model, inputs_embeds=fe3, attention_mask=fa3, need_logits=compute_log_pi
    )
    del fe3, fa3

    repr_3_batch = torch.stack([
        hidden3[i, 1:1 + c3_lens[i], :].mean(0)
        for i in range(B)
    ]).float()                                               # (B, hidden) LIVE fp32
    del hidden3

    log_pi_3: list[torch.Tensor] = []
    if compute_log_pi:
        for i in range(B):
            L3 = c3_lens[i]
            sl = logits3[i, 0:L3, :]   # pos 0 (z_pfx2) predicts c3[0], pos 1 predicts c3[1]…
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
    """Train StateEncoder, ZInjector, and OutcomeHead online with frozen backbone.

    Each step:
      1. [no_grad] generate G rollouts per problem → chunk_ids + reward
      2. [with_grad] re-run full pipeline → live repr_h, z_h (= μ_h)
      3. losses: λ_trans × L_trans + λ_out × L_out + λ_calib × L_calib
      4. step Encoder+ZInjector+OutcomeHead; backbone grads zeroed (not stepped)

    Config keys consumed (under "phase0"):
        model_id / revision / dtype  — backbone (from "primary")
        pool_path                    — data/math_easy_pool.jsonl (L1–L4)
        n_steps                      — training steps (default: 400)
        num_generations              — G total rollouts per problem across Phase 0 (default: 128)
        learning_rate                — AdamW lr for Encoder/ZInj/OutcomeHead (default: 3e-4)
        lambda_trans_peak, lambda_trans_warmup_steps — λ_t schedule (peak and initial zero phase)
        lambda_out, lambda_calib — loss weights
        temperature, top_p           — sampling params (from training.*)
        chunk_tokens                 — tokens per chunk (from latent_markov.*)
        checkpoint_path              — where to save phase0_encoder.pt
        logging_steps, save_steps
    """
    primary      = config["primary"]
    phase0_cfg   = config["phase0"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]

    model_id  = primary["huggingface_repo_id"]
    revision  = primary.get("revision", "main")
    dtype     = getattr(torch, primary.get("dtype", "bfloat16"))

    n_steps          = int(phase0_cfg.get("n_steps",          400))
    G                = int(phase0_cfg.get("num_generations",  128))  # total rollouts per problem
    micro_batch_size = int(phase0_cfg.get("micro_batch_size",  32))
    seqs_per_step    = G                                              # 128 seqs/step (memory budget)
    lr               = float(phase0_cfg.get("learning_rate",  3e-4))
    lambda_trans_peak        = float(phase0_cfg.get("lambda_trans_peak",        3.0))
    lambda_trans_warmup_steps = int(phase0_cfg.get("lambda_trans_warmup_steps", 50))
    lambda_out               = float(phase0_cfg.get("lambda_out",               5.0))
    lambda_calib             = float(phase0_cfg.get("lambda_calib",             1.0))
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
                                        str(run_dir / "phase0_encoder.pt")))
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
    attn_impl = primary.get("attn_implementation", "sdpa")
    logger.info("Loading backbone %s @ %s  attn=%s ...", model_id, revision, attn_impl)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
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

    # Phase 0 can override the global gradient_checkpointing setting.
    # Default: off — Phase 0 activation memory fits in 96 GB without GC
    # (~35 GB), and disabling GC keeps model.config.use_cache=True so
    # model.generate() benefits from the KV cache.
    use_gc = phase0_cfg.get(
        "gradient_checkpointing",
        training_cfg.get("gradient_checkpointing", False),
    )
    if use_gc:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  gradient checkpointing enabled")
    else:
        logger.info("  gradient checkpointing disabled (Phase 0: fits in VRAM without GC)")

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

    # ------------------------------------------------------------------
    # Pre-shuffled assignment list
    # n_problems = n_steps × seqs_per_step / G = n_steps (when seqs_per_step = G).
    # Each of the n_steps problems gets exactly G rollout slots spread across
    # all 400 training steps, giving ~110 unique problems per step (maximum
    # diversity for L_out) while every problem accumulates G total rollouts
    # (consistent with the pass@128 evaluation metric).
    # ------------------------------------------------------------------
    n_problems = min(n_steps, len(problems))
    sampled_problems = _random.sample(problems, n_problems)
    flat_assignments: list[dict] = [
        p for p in sampled_problems for _ in range(G)
    ]
    _random.shuffle(flat_assignments)
    logger.info(
        "Phase 0 assignment: %d problems × %d rollouts = %d total "
        "(~%.0f unique problems/step)",
        n_problems, G, len(flat_assignments),
        n_problems * (1 - (1 - 1 / n_problems) ** seqs_per_step),
    )

    log_history: list[dict] = []
    pending:     dict[str, float] = {}

    step_bar = tqdm(total=n_steps, desc="phase0", unit="step", dynamic_ncols=True)

    for global_step in range(n_steps):
        # Warm-start: hold λ_t = 0 for the first `lambda_trans_warmup_steps` steps
        # so L_out (outcome prediction) can establish a signal before the transition
        # loss ramp competes for gradient budget.  After warmup the normal linear
        # ramp resumes, offsetting the step index so the full ramp still completes.
        if global_step < lambda_trans_warmup_steps:
            lambda_t = 0.0
        else:
            effective_step = global_step - lambda_trans_warmup_steps
            effective_max  = n_steps    - lambda_trans_warmup_steps
            lambda_t = lambda_trans_schedule(effective_step, effective_max, peak=lambda_trans_peak)

        # Slice this step's seqs_per_step problem-assignments from the flat list.
        # Each entry is a problem dict; duplicates of the same problem within a
        # step are rare (~1%) and harmless (they count as independent rollouts).
        step_start   = global_step * seqs_per_step
        step_problems = flat_assignments[step_start : step_start + seqs_per_step]

        # ── Rollout collection (no gradient) ──────────────────────────────
        with torch.no_grad():
            model.eval(); vae.eval(); z_injector.eval()
            # n_rollouts=1: one rollout per slot.  B = seqs_per_step = 128 simultaneous
            # sequences — same memory budget as before; contents are now ~110 distinct
            # problems rather than 1 problem × 128.
            traces = generate_latent_traces(
                model=model, tokenizer=tokenizer,
                vae=vae, z_injector=z_injector,
                problems=step_problems, n_rollouts=1,
                chunk_tokens=chunk_tokens,
                temperature=temperature, top_p=top_p,
                device=device,
            )

        model.train(); vae.train(); z_injector.train(); outcome_head.train()

        # Shuffle traces to avoid any ordering artifacts within the step before
        # micro-batching (flat_assignments is pre-shuffled across steps; this
        # ensures uniform mixing within each step's micro-batches as well).
        _random.shuffle(traces)

        # ── Training step (with gradient) ─────────────────────────────────
        # Zero backbone grads from previous step (backbone is NOT in optimizer).
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad()

        # Micro-batch loop: split B×G traces into chunks of micro_batch_size.
        # Each chunk's loss is scaled by 1/n_micro and backward() is called
        # immediately, releasing that chunk's computation graph before the
        # next chunk is processed.  Mathematically identical to one full-batch
        # backward: L_trans / L_out / L_calib are all mean-reduced within each
        # micro-batch, and averaging over n_micro micro-batches preserves the
        # same mean.  Advantages are not used in Phase 0 (no L_RL), so there is
        # no group-normalization dependency across micro-batches.
        n_total = len(traces)
        n_micro = math.ceil(n_total / micro_batch_size)
        l_trans_acc = l_out_acc = l_calib_acc = total_acc = 0.0

        for mb_start in range(0, n_total, micro_batch_size):
            mb_traces  = traces[mb_start : mb_start + micro_batch_size]
            rewards_mb = torch.tensor(
                [float(t["reward"]) for t in mb_traces],
                dtype=torch.float32, device=device,
            ).unsqueeze(-1)                                  # (mb, 1)

            # Phase 0: no L_RL → lm_head bypassed → saves ~26 GB per chunk.
            pipe = _run_pipeline_with_grad(
                model, vae, z_injector, mb_traces, device, compute_log_pi=False
            )

            # pos_weight = n_neg / n_pos balances the 82/18 incorrect/correct split.
            # Clamped to [1, 20]: avoids exploding weights on degenerate micro-batches
            # (e.g. all-zero reward step) while still upweighting rare correct signals.
            pos_rate   = rewards_mb.mean().clamp(min=0.05)
            pos_weight = ((1.0 - pos_rate) / pos_rate).clamp(max=20.0)

            l_trans = vae.compute_transition_loss(pipe["z_list"])
            l_out   = F.binary_cross_entropy_with_logits(
                outcome_head(pipe["z_list"][-1]).view(-1),
                rewards_mb.view(-1),
                pos_weight=pos_weight,
            )
            l_calib = vae.compute_calibration_loss(
                pipe["logvar_list"], rewards_mb, pos_weight=pos_weight
            )

            total_mb: torch.Tensor = (
                lambda_t * l_trans + lambda_out * l_out + lambda_calib * l_calib
            ) / n_micro
            total_mb.backward()  # release this micro-batch's computation graph

            l_trans_acc += l_trans.detach().item() / n_micro
            l_out_acc   += l_out.detach().item()   / n_micro
            l_calib_acc += l_calib.detach().item() / n_micro
            total_acc   += total_mb.detach().item()

        torch.nn.utils.clip_grad_norm_(vae_params, max_norm=1.0)
        optimizer.step()
        # Backbone grads accumulated during backward — zero now to prevent
        # stale accumulation across steps (backbone is never stepped).
        model.zero_grad(set_to_none=True)

        # ── Logging ───────────────────────────────────────────────────────
        for k, v in (("loss",  total_acc),
                     ("trans", l_trans_acc),
                     ("out",   l_out_acc),
                     ("calib", l_calib_acc)):
            pending[k] = pending.get(k, 0.0) + v
        reward_rate = sum(t["reward"] for t in traces) / len(traces)
        pending["reward_rate"] = pending.get("reward_rate", 0.0) + reward_rate

        if (global_step + 1) % log_steps == 0:
            n = log_steps
            entry = {
                "step":     global_step + 1,
                "lambda_t": round(lambda_t, 4),
                **{k: pending.get(k, 0.0) / n
                   for k in ("loss", "trans", "out", "calib", "reward_rate")},
            }
            log_history.append(entry)
            pending = {}
            logger.info(
                "step %d | λ_t=%.2f | loss=%.4f trans=%.4f out=%.4f calib=%.4f | reward=%.1f%%",
                entry["step"], entry["lambda_t"],
                entry["loss"], entry["trans"], entry["out"], entry["calib"],
                entry["reward_rate"] * 100,
            )

        if (global_step + 1) % save_steps == 0:
            _save_phase0_checkpoint(
                run_dir / f"checkpoint-{global_step + 1}",
                vae, z_injector, outcome_head, global_step + 1, log_history,
            )

        step_bar.set_postfix(
            loss=f"{total_acc:.4f}",
            calib=f"{l_calib_acc:.4f}",
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
        directory / "phase0_encoder.pt",
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
    lambda_calib: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute the combined Phase 1 loss for one training step.

    L_total = L_RL  +  λ_t × L_transition  +  λ_calib × L_calib

    L_RL (GRPO, no IS correction):
        The full 3-chunk pipeline is re-run with grad. z_h = μ_h (deterministic).
        IS = 1 exactly (policy not updated between rollout and training step).

        L_RL = -mean_{i,h,t} [ advantage_i × log π_θ(token_t | context_{h,i}) ]

    L_calib:
        BCE( sigmoid(mean_logvar),  1 − reward ) — keeps σ² calibrated to
        trajectory quality during Phase 1. Lighter than Phase 0 (λ_calib=0.5).

    Args:
        model:        backbone in training mode (UNFROZEN — .step() called by caller).
        vae:          VAEStateEncoder in training mode.
        z_injector:   ZInjector in training mode.
        traces:       output of generate_latent_traces() for this step.
        advantages:   aligned GRPO advantages (output of compute_grpo_advantages()).
        lambda_t:     current transition loss weight (from lambda_trans_schedule()).
        lambda_calib: L_calib loss weight (config: phase1_loss.lambda_calib).
        device:       CUDA device.

    Returns:
        Dict with backward-able scalar tensor "total" and detached scalars
        "l_rl", "l_calib", "l_trans" for logging.
    """
    pipe = _run_pipeline_with_grad(model, vae, z_injector, traces, device)

    l_trans = vae.compute_transition_loss(pipe["z_list"])

    rewards_t = torch.tensor(
        [float(t["reward"]) for t in traces],
        dtype=torch.float32, device=device,
    )
    l_calib = vae.compute_calibration_loss(pipe["logvar_list"], rewards_t)

    # L_RL: -mean_{i,h,t} [ adv_i × log_π(token_t | context_{h,i}) ]
    adv = [float(a) for a in advantages]
    rl_sum   = torch.zeros(1, device=device)
    n_tokens = 0
    for lp_chunk in pipe["log_pi_chunks"]:                 # 3 chunks
        for i, lp in enumerate(lp_chunk):                  # B×G traces
            rl_sum   = rl_sum + (-adv[i] * lp.sum())
            n_tokens += lp.shape[0]
    l_rl = rl_sum / max(n_tokens, 1)

    l_total = l_rl + lambda_t * l_trans + lambda_calib * l_calib

    return {
        "total":    l_total,
        "l_rl":     l_rl.detach(),
        "l_calib":  l_calib.detach(),
        "l_trans":  l_trans.detach(),
    }


def train_latent(config: dict[str, Any], run_dir: Path) -> None:
    """Phase 1: joint GRPO training with live backbone and z injection.

    Loads Phase 0 encoder + ZInjector checkpoint, unfreezes backbone, and runs
    the on-policy GRPO loop for max_steps steps.

    Every training step:
      1. [no_grad] collect G=8 fresh rollouts for the current batch
      2. compute GRPO advantages from group rewards
      3. [with_grad] re-run full pipeline → L_RL + λ_t·L_trans + λ_calib·L_calib
      4. step all optimizers (backbone lr=1e-6, encoder lr=3e-4)

    Config keys consumed:
        primary.*              — backbone model ID, revision, dtype
        phase0.checkpoint_path — path to phase0_encoder.pt (encoder + ZInjector)
        latent_markov.*        — latent_dim, hidden_dim, chunk_tokens
        training.*             — seed, learning_rate, num_generations,
                                 batch_size, max_steps, temperature, top_p,
                                 gradient_checkpointing, logging_steps, save_steps
        phase1_loss.*          — lambda_calib, lambda_trans_peak
        evaluation.path        — Level 5 hard pool JSONL for RL training
    """
    primary      = config["primary"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]
    phase0_cfg   = config["phase0"]
    phase1_cfg   = config.get("phase1_loss", {})

    lambda_calib      = float(phase1_cfg.get("lambda_calib",      0.5))
    lambda_trans_peak = float(phase1_cfg.get("lambda_trans_peak", 0.3))

    model_id     = primary["huggingface_repo_id"]
    revision     = primary.get("revision", "main")
    dtype        = getattr(torch, primary.get("dtype", "bfloat16"))
    is_smoke     = (config.get("experiment") or {}).get("profile") == "smoke"

    seed             = int(training_cfg.get("seed", 42))
    lr_backbone      = float(training_cfg.get("learning_rate", 1e-6))
    lr_vae           = 3e-4    # VAE/ZInjector trained at Phase 0 rate throughout
    G                = int(training_cfg.get("num_generations",   128))
    batch_size       = int(training_cfg.get("batch_size",          4))
    micro_batch_size = int(training_cfg.get("micro_batch_size",   32))
    max_steps        = int(training_cfg.get("max_steps",         200))
    temperature  = float(training_cfg.get("temperature",   1.0))
    top_p        = float(training_cfg.get("top_p",         1.0))
    log_steps    = int(training_cfg.get("logging_steps",   10))
    save_steps   = int(training_cfg.get("save_steps",      50))
    grad_clip    = 1.0

    chunk_tokens = int(latent_cfg.get("chunk_tokens",  341))
    latent_dim   = int(latent_cfg.get("latent_dim",  LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim",  HIDDEN_DIM))

    vae0_path    = Path(phase0_cfg.get("checkpoint_path", run_dir / "phase0_encoder.pt"))
    pool_path    = Path(config["evaluation"]["path"])
    ckpt_path    = run_dir / "phase1"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 1 — device: %s  lambda_calib=%.4f  lambda_trans_peak=%.2f",
                device, lambda_calib, lambda_trans_peak)

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    attn_impl = primary.get("attn_implementation", "sdpa")
    logger.info("Loading backbone %s @ %s  attn=%s ...", model_id, revision, attn_impl)
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
            attn_implementation=attn_impl,
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
    # Training pool (Level 5 hard pool)
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

        # Micro-batch loop over the B×G traces.
        # Advantages are pre-computed over the FULL group (above) → GRPO
        # group normalization is correct regardless of how we partition the
        # backward passes.  Scaling each micro-batch's loss by 1/n_micro and
        # calling backward() immediately is mathematically identical to one
        # full-batch backward.
        n_total_p1 = len(traces)
        n_micro_p1 = math.ceil(n_total_p1 / micro_batch_size)
        metrics_acc: dict[str, float] = {
            "total": 0.0, "l_rl": 0.0, "l_calib": 0.0, "l_trans": 0.0
        }

        for mb_start in range(0, n_total_p1, micro_batch_size):
            mb_traces = traces    [mb_start : mb_start + micro_batch_size]
            mb_adv    = advantages[mb_start : mb_start + micro_batch_size]

            metrics_mb = latent_training_step(
                model=model, vae=vae, z_injector=z_injector,
                traces=mb_traces, advantages=mb_adv,
                lambda_t=lambda_t, lambda_calib=lambda_calib,
                device=device,
            )

            (metrics_mb["total"] / n_micro_p1).backward()  # release graph immediately

            for k in ("total", "l_rl", "l_calib", "l_trans"):
                metrics_acc[k] += metrics_mb[k].item() / n_micro_p1

        all_params = list(model.parameters()) + vae_params
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────
        for k in ("total", "l_rl", "l_calib", "l_trans"):
            pending[k] = pending.get(k, 0.0) + metrics_acc[k]
        pending["reward_rate"] = (
            pending.get("reward_rate", 0.0) + sum(rewards) / len(rewards)
        )

        if (global_step + 1) % log_steps == 0:
            n = log_steps
            entry = {
                "step":     global_step + 1,
                "lambda_t": round(lambda_t, 4),
                **{k: pending.get(k, 0.0) / n
                   for k in ("total", "l_rl", "l_calib", "l_trans", "reward_rate")},
            }
            log_history.append(entry)
            pending = {}
            logger.info(
                "step %d | λ_t=%.2f λ_calib=%.3f | total=%.4f rl=%.4f "
                "calib=%.4f trans=%.4f | reward=%.1f%%",
                entry["step"], entry["lambda_t"], lambda_calib,
                entry["total"], entry["l_rl"],
                entry["l_calib"], entry["l_trans"],
                entry["reward_rate"] * 100,
            )

        if (global_step + 1) % save_steps == 0:
            _save_phase1_checkpoint(
                ckpt_path / f"checkpoint-{global_step + 1}",
                model, vae, z_injector, optimizer, global_step + 1, log_history,
                tokenizer=tokenizer,
            )

        step_bar.set_postfix(
            loss=f"{metrics_acc['total']:.4f}",
            rl=f"{metrics_acc['l_rl']:.4f}",
            calib=f"{metrics_acc['l_calib']:.4f}",
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
