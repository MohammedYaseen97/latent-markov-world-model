"""Latent Markov GRPO v3 — teacher-forced Phase 0 + strict Markov Phase 1.

Design reference: reports/latent_markov_design.md

Phase 0 — Teacher-Forced Distillation Pretraining
───────────────────────────────────────────────────
Teacher: original frozen Qwen, full context per chunk.
Student: backbone (low lr) + encoder + injector (higher lr).
Loss: CE(student_logits, teacher_tokens) on chunks 2 and 3.
Context: strict Markov — student sees [z_prefix | teacher_chunk_h], never raw prior chunks.
No reward needed. Dense token-level signal every step.

Phase 1 — On-Policy GRPO (pure L_RL)
──────────────────────────────────────
Strict Markov generation: [z_prefix | generate], no crutch.
Generation and training forward pass use identical context — IS = 1 exactly.
Loss: L_RL = -advantage × log_π (GRPO; G=128 per problem).
Backbone + encoder + injector all receive gradient via live repr_h graph.
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
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from src.models.vae_state_encoder import HIDDEN_DIM, LATENT_DIM, LatentStateEncoder
from src.training.grpo_baseline import SYSTEM_PROMPT, answers_equivalent, extract_answer

logger = logging.getLogger(__name__)

# torch.compile with dynamic=True emits verbose shape-guard warnings from
# symbolic_shapes when Dynamo encounters guards it cannot fully simplify (e.g.
# disjunctive equality expressions from Qwen's attention conditionals). The
# guards are still applied correctly; the messages are purely informational
# noise that flood the terminal and make progress bars unreadable.
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
# Recompile-limit hit messages from convert_frame are similarly noisy after
# the initial warm-up; suppress them at WARNING level too.
logging.getLogger("torch._dynamo.convert_frame").setLevel(logging.ERROR)

# ── Global CUDA performance flags ─────────────────────────────────────────────
# TF32: fp32 matmuls use tensor-core acceleration (Ampere+).
# bf16 reduction: faster reductions with negligible precision impact.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.set_float32_matmul_precision("high")

# Generation sees many distinct sequence lengths (prefill at various prompt
# lengths + single-token decode steps). The default recompile_limit=8 causes
# Dynamo to fall back to eager after 8 shape variants, losing all fusion
# benefits for the rest of the run. 64 covers the realistic shape variety
# while dynamic=True keeps per-variant compile time low.
import torch._dynamo
torch._dynamo.config.recompile_limit = 64

# ── Constants ─────────────────────────────────────────────────────────────────

N_CHUNKS = 3  # fixed; 3 chunks × 341 tokens = 1023 ≈ 1024 token budget


# ── GRPO utilities ─────────────────────────────────────────────────────────────

def compute_grpo_advantages(
    rewards: list[float],
    group_size: int,
    eps: float = 1e-8,
    adv_clip: float = 20.0,
) -> list[float]:
    """Normalise rewards into GRPO advantages within each group of G rollouts.

    A_i = clip((r_i − μ_group) / (σ_group + eps), -adv_clip, adv_clip)

    With G=128, max natural advantage ≈ 11.3 (k=1 correct rollout).
    adv_clip=20 is inert in practice — guards against numerical pathologies only.
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
            raw = (r - mu) / (sig + eps)
            advantages.append(max(-adv_clip, min(adv_clip, raw)))
    return advantages


_LATENT_SYSTEM_PROMPT = (
    "Please reason step by step, completing each reasoning step fully before "
    "moving to the next. Within any token limit, finish the current step before "
    "stopping — never break mid-sentence or mid-calculation. "
    "Put your final answer inside \\boxed{answer}."
)


def format_prompt(problem: dict, tokenizer: AutoTokenizer) -> list[int]:
    messages = [
        {"role": "system", "content": _LATENT_SYSTEM_PROMPT},
        {"role": "user",   "content": problem["prompt"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


# ── Backbone forward helpers ───────────────────────────────────────────────────

def _get_transformer_and_head(
    model: AutoModelForCausalLM,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Return (base_transformer, lm_head) for both PEFT and non-PEFT models.

    Non-PEFT:  model.model = Qwen2Model,           model.lm_head = Linear
    PEFT:      model.model = Qwen2ForCausalLM,     model.model.model = Qwen2Model
               model.model.lm_head = Linear

    Detected by whether model.model itself has an lm_head attribute.
    """
    inner = model.model
    if hasattr(inner, "lm_head"):
        # PEFT: inner is the base CausalLM; go one level deeper for the transformer
        return inner.model, inner.lm_head
    return inner, model.lm_head


def _fwd(
    model: AutoModelForCausalLM,
    *,
    input_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    bypass_lm_head: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Hook-free backbone forward returning (logits | None, last_hidden_states).

    Calls the base transformer directly then optionally applies lm_head.
    No Python-level hooks — fully compatible with torch.compile.

    When bypass_lm_head=True: skips lm_head entirely.
    Saves [B × seq × vocab × 2 bytes] per call — critical for chunk-1 forward
    in Phase 0 where no CE loss is needed (~10 GB at B=64).

    Returns:
        logits:      [B, seq, vocab]  or  None when bypassed
        last_hidden: [B, seq, hidden_dim]
    """
    transformer, lm_head = _get_transformer_and_head(model)
    out         = transformer(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
    )
    last_hidden = out.last_hidden_state
    if bypass_lm_head:
        return None, last_hidden
    return lm_head(last_hidden), last_hidden


from contextlib import contextmanager as _contextmanager


def _apply_lora(model: AutoModelForCausalLM, lora_cfg: dict) -> AutoModelForCausalLM:
    """Wrap model with LoRA adapters for Phase 0 backbone training.

    LoRA targets all attention + MLP projection layers so the backbone learns
    to use z_prefix embeddings at lr=1e-4 without touching the pretrained base
    weights — no catastrophic forgetting risk, 100× more gradient signal than
    full fine-tuning at lr=1e-6.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.0)),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"],
        ),
        bias="none",
    )
    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "  LoRA applied: trainable=%.2fM / total=%.2fM (%.1f%%)",
        trainable / 1e6, total / 1e6, 100 * trainable / total,
    )
    return model


def _clean_state_dict(sd: dict) -> dict:
    """Strip ._orig_mod. prefixes that torch.compile injects into state_dict keys."""
    return {k.replace("._orig_mod.", "."): v for k, v in sd.items()}


@_contextmanager
def _unwrapped_for_save(model: AutoModelForCausalLM):
    """Temporarily restore compiled sub-modules to their originals for save_pretrained.

    _setup_compile replaces model.model and model.lm_head with torch.compile wrappers
    whose parameter names include '_orig_mod', breaking Transformers' tied-weight check.
    This context manager swaps them out, yields, then restores the compiled versions.
    """
    swapped: dict[str, torch.nn.Module] = {}
    for attr in ("model", "lm_head"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "_orig_mod"):
            swapped[attr] = sub
            setattr(model, attr, sub._orig_mod)
    try:
        yield model
    finally:
        for attr, compiled in swapped.items():
            setattr(model, attr, compiled)


def _setup_compile(
    model: AutoModelForCausalLM,
    encoder: LatentStateEncoder | None = None,
) -> None:
    """Apply torch.compile to the backbone transformer, lm_head, and optionally encoder.

    Compiles the sub-modules that are called in the hot path (every forward step
    and every generation decode step). Uses dynamic=True so variable sequence
    lengths don't trigger recompilation.

    model.generate() benefits automatically: HF's generate() calls self.model()
    and self.lm_head() internally — since we replace those with compiled versions
    on the model object, every decode step goes through compiled kernels.

    encoder is optional — pass None for arms that have no latent encoder
    (e.g. baseline, token_markov).
    """
    transformer, lm_head = _get_transformer_and_head(model)
    try:
        # Both backbone and encoder use "default" mode.
        # "reduce-overhead" uses CUDA graphs with static output buffers; the encoder is called
        # 2-4 times per step and all outputs must stay alive simultaneously — subsequent calls
        # would overwrite earlier buffers, corrupting z_1/prefix_1 before they are consumed.
        # "default" still fuses kernels via Triton and avoids this aliasing problem entirely.
        inner = model.model
        if hasattr(inner, "lm_head"):
            inner.model   = torch.compile(transformer, dynamic=True, mode="default")
            inner.lm_head = torch.compile(lm_head,    dynamic=True, mode="default")
        else:
            model.model   = torch.compile(transformer, dynamic=True, mode="default")
            model.lm_head = torch.compile(lm_head,    dynamic=True, mode="default")

        if encoder is not None:
            encoder.encoder  = torch.compile(encoder.encoder,  dynamic=True, mode="default")
            encoder.injector = torch.compile(encoder.injector, dynamic=True, mode="default")

        logger.info("  torch.compile applied (backbone=default, encoder=default, dynamic=True)")
    except Exception as exc:
        logger.warning("  torch.compile not applied (falling back to eager): %s", exc)


def _build_prefix_embeds(
    embed_layer: torch.nn.Embedding,
    z_prefix: torch.Tensor,
    chunk_ids_list: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Build inputs_embeds = [z_prefix | chunk_tokens], right-padded.

    Args:
        embed_layer:    model.get_input_embeddings()
        z_prefix:       [B, 1, hidden_dim]
        chunk_ids_list: list of B CPU tensors (variable length)

    Returns:
        inputs_embeds:  [B, 1 + max_len, hidden_dim]
        attention_mask: [B, 1 + max_len]
        chunk_lens:     list of int — actual lengths of each chunk
    """
    B        = z_prefix.shape[0]
    chunk_lens = [c.shape[0] for c in chunk_ids_list]
    max_len  = max(chunk_lens)
    H        = z_prefix.shape[-1]

    ie = torch.zeros(B, 1 + max_len, H, dtype=dtype, device=device)
    am = torch.zeros(B, 1 + max_len, dtype=torch.long, device=device)

    ie[:, 0, :] = z_prefix[:, 0, :].to(dtype)
    am[:, 0]    = 1

    for i, (cids, L) in enumerate(zip(chunk_ids_list, chunk_lens)):
        ie[i, 1:1 + L, :] = embed_layer(cids.to(device))
        am[i, 1:1 + L]    = 1

    return ie, am, chunk_lens


def _last_token_repr(
    hidden: torch.Tensor,
    last_positions: torch.Tensor,
) -> torch.Tensor:
    """Extract the hidden state at the last real token for each sequence.

    Args:
        hidden:         [B, seq_len, hidden_dim]
        last_positions: [B] int tensor — 0-indexed position of last real token

    Returns:
        repr_h: [B, hidden_dim]  — same dtype as hidden
    """
    B = hidden.shape[0]
    return hidden[torch.arange(B, device=hidden.device), last_positions, :]


# ── Phase 0: teacher-forced distillation pretraining ──────────────────────────

@torch.no_grad()
def _generate_teacher_chunks(
    teacher: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    chunk_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Teacher generates 3 chunks per problem with full context.

    The teacher is the original frozen Qwen with access to all prior tokens:
        chunk1: generate([sys_prompt | problem])
        chunk2: generate([sys_prompt | problem | chunk1])
        chunk3: generate([sys_prompt | problem | chunk1 | chunk2])

    Args:
        teacher:  original frozen backbone in eval mode.
        problems: list of B problem dicts.

    Returns:
        List of B tuples (chunk1_ids, chunk2_ids, chunk3_ids), all CPU tensors.
        Lengths are ≤ chunk_tokens (shorter if EOS fired early).
    """
    pad_id = tokenizer.eos_token_id
    B      = len(problems)

    # ── Chunk 1 ───────────────────────────────────────────────────────────────
    prompt_ids_list = [format_prompt(p, tokenizer) for p in problems]
    max_prompt      = max(len(p) for p in prompt_ids_list)

    input_ids = torch.full((B, max_prompt), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt, dtype=torch.long, device=device)
    for i, pids in enumerate(prompt_ids_list):
        off = max_prompt - len(pids)      # left-pad
        input_ids[i, off:] = torch.tensor(pids, dtype=torch.long, device=device)
        attn_mask[i, off:] = 1

    gen1 = teacher.generate(
        input_ids, attention_mask=attn_mask,
        max_new_tokens=chunk_tokens, do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=pad_id,
    )
    chunk1_ids = [gen1[i, max_prompt:].cpu() for i in range(B)]
    del gen1, input_ids, attn_mask

    # ── Chunk 2: full context [prompt | chunk1] ───────────────────────────────
    prompt_lens = [len(p) for p in prompt_ids_list]
    c1_lens     = [c.shape[0] for c in chunk1_ids]
    full2_lens  = [prompt_lens[i] + c1_lens[i] for i in range(B)]
    max_full2   = max(full2_lens)

    ctx2      = torch.full((B, max_full2), pad_id, dtype=torch.long, device=device)
    ctx2_mask = torch.zeros(B, max_full2, dtype=torch.long, device=device)
    for i in range(B):
        off = max_full2 - full2_lens[i]   # left-pad
        p   = torch.tensor(prompt_ids_list[i], dtype=torch.long, device=device)
        c1  = chunk1_ids[i].to(device)
        ctx2[i, off:off + prompt_lens[i]] = p
        ctx2[i, off + prompt_lens[i]:off + full2_lens[i]] = c1
        ctx2_mask[i, off:off + full2_lens[i]] = 1

    gen2 = teacher.generate(
        ctx2, attention_mask=ctx2_mask,
        max_new_tokens=chunk_tokens, do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=pad_id,
    )
    chunk2_ids = [gen2[i, max_full2:].cpu() for i in range(B)]
    del gen2, ctx2, ctx2_mask

    # ── Chunk 3: full context [prompt | chunk1 | chunk2] ─────────────────────
    c2_lens    = [c.shape[0] for c in chunk2_ids]
    full3_lens = [full2_lens[i] + c2_lens[i] for i in range(B)]
    max_full3  = max(full3_lens)

    ctx3      = torch.full((B, max_full3), pad_id, dtype=torch.long, device=device)
    ctx3_mask = torch.zeros(B, max_full3, dtype=torch.long, device=device)
    for i in range(B):
        off  = max_full3 - full3_lens[i]   # left-pad
        seqi = torch.cat([
            torch.tensor(prompt_ids_list[i], dtype=torch.long, device=device),
            chunk1_ids[i].to(device),
            chunk2_ids[i].to(device),
        ])
        ctx3[i, off:off + full3_lens[i]] = seqi
        ctx3_mask[i, off:off + full3_lens[i]] = 1

    gen3 = teacher.generate(
        ctx3, attention_mask=ctx3_mask,
        max_new_tokens=chunk_tokens, do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=pad_id,
    )
    chunk3_ids = [gen3[i, max_full3:].cpu() for i in range(B)]
    del gen3, ctx3, ctx3_mask

    return list(zip(chunk1_ids, chunk2_ids, chunk3_ids))


def _distill_loss(
    student: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    teacher_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    z_anchor_tokens: int | None = None,
) -> torch.Tensor:
    """Compute CE distillation loss for chunks 2 and 3 (teacher-forced).

    Student context is strictly Markov: chunk h uses [z_{h-1}_prefix | teacher_chunk_h].
    No raw prior chunk tokens are visible to the student.

    z_anchor_tokens (optional):
        If set, only compute CE on the first z_anchor_tokens positions of each
        chunk.  At early positions z_prefix is the primary context; the gradient
        flows almost entirely through z-conditioning.  Without this restriction
        the loss is averaged over all 341 positions, diluting the z-conditioning
        gradient ~341× by the dominant "continue teacher text" signal at later
        positions — causing the backbone to learn LM-from-teacher-context but
        not LM-from-z.  Recommended: 32–64 (roughly 10–20% of a 341-token chunk).

    Gradient flows:
        loss_2 → backbone (chunk 2 logits) → prefix_1 → injector → z_1 → encoder → backbone (chunk 1 hidden)
        loss_3 → backbone (chunk 3 logits) → prefix_2 → injector → z_2 → encoder → backbone (chunk 2 hidden)

    Returns:
        loss: scalar tensor (loss_2 + loss_3), differentiable.
    """
    B           = len(problems)
    dtype       = next(student.parameters()).dtype
    embed_layer = student.get_input_embeddings()
    vocab_size  = student.config.vocab_size

    prompt_ids_list = [format_prompt(p, tokenizer) for p in problems]
    prompt_lens     = [len(p) for p in prompt_ids_list]
    c1_ids_list     = [teacher_chunks[i][0] for i in range(B)]
    c2_ids_list     = [teacher_chunks[i][1] for i in range(B)]
    c3_ids_list     = [teacher_chunks[i][2] for i in range(B)]
    c1_lens         = [c.shape[0] for c in c1_ids_list]
    c2_lens         = [c.shape[0] for c in c2_ids_list]
    c3_lens         = [c.shape[0] for c in c3_ids_list]

    # ── Chunk 1: [prompt | teacher_chunk1] → repr_1 (no CE loss) ─────────────
    full1_lens = [prompt_lens[i] + c1_lens[i] for i in range(B)]
    max_full1  = max(full1_lens)

    fi1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    fa1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    for i in range(B):
        L = full1_lens[i]
        seq = torch.cat([
            torch.tensor(prompt_ids_list[i], dtype=torch.long),
            c1_ids_list[i],
        ]).to(device)
        fi1[i, :L] = seq
        fa1[i, :L] = 1

    # bypass lm_head — no CE loss on chunk 1
    _, hidden1 = _fwd(student, input_ids=fi1, attention_mask=fa1, bypass_lm_head=True)
    del fi1, fa1

    # last real token = last token of teacher_chunk1 (not prompt)
    last1 = torch.tensor([full1_lens[i] - 1 for i in range(B)], device=device)
    repr_1 = _last_token_repr(hidden1, last1)
    del hidden1

    z_1      = encoder.encode(repr_1)
    prefix_1 = encoder.inject(z_1)

    # ── Chunk 2: [prefix_1 | teacher_chunk2] → CE loss + repr_2 ──────────────
    max_c2 = max(c2_lens)
    ie2, am2, _ = _build_prefix_embeds(embed_layer, prefix_1, c2_ids_list, device, dtype)

    logits2, hidden2 = _fwd(student, inputs_embeds=ie2, attention_mask=am2)
    del ie2, am2

    # logits2[:, j, :] predicts teacher_chunk2_token[j]  (j = 0 .. max_c2-1)
    tgt2 = torch.full((B, max_c2), -100, dtype=torch.long, device=device)
    for i in range(B):
        tgt2[i, :c2_lens[i]] = c2_ids_list[i].to(device)
    # z-anchor: only supervise positions where z is the primary context.
    if z_anchor_tokens is not None:
        tgt2[:, z_anchor_tokens:] = -100

    loss_2 = F.cross_entropy(
        logits2[:, :max_c2, :].reshape(B * max_c2, vocab_size),
        tgt2.reshape(B * max_c2),
        ignore_index=-100,
    )
    del logits2

    # repr_2 = last real token of chunk2 in [prefix | chunk2] sequence
    # prefix at position 0, chunk tokens at positions 1..c2_len
    last2  = torch.tensor(c2_lens, device=device)       # position = c2_len (0-indexed)
    repr_2 = _last_token_repr(hidden2, last2)
    del hidden2

    z_2      = encoder.encode(repr_2)
    prefix_2 = encoder.inject(z_2)

    # ── Chunk 3: [prefix_2 | teacher_chunk3] → CE loss ───────────────────────
    max_c3 = max(c3_lens)
    ie3, am3, _ = _build_prefix_embeds(embed_layer, prefix_2, c3_ids_list, device, dtype)

    logits3, _ = _fwd(student, inputs_embeds=ie3, attention_mask=am3)
    del ie3, am3

    tgt3 = torch.full((B, max_c3), -100, dtype=torch.long, device=device)
    for i in range(B):
        tgt3[i, :c3_lens[i]] = c3_ids_list[i].to(device)
    if z_anchor_tokens is not None:
        tgt3[:, z_anchor_tokens:] = -100

    loss_3 = F.cross_entropy(
        logits3[:, :max_c3, :].reshape(B * max_c3, vocab_size),
        tgt3.reshape(B * max_c3),
        ignore_index=-100,
    )
    del logits3

    return loss_2 + loss_3


def pretrain_distill(config: dict[str, Any], run_dir: Path) -> None:
    """Phase 0: teacher-forced distillation pretraining on L1–L4 pool.

    Trains encoder + injector from scratch. Backbone updated at low lr so it
    learns to attend to z prefixes without drifting from pretrained capability.

    Config keys consumed (under "phase0"):
        pool_path          — data/math_easy_pool.jsonl
        n_steps            — training steps (default: 400)
        batch_size         — problems per step (default: 64)
        lr_backbone        — backbone learning rate (default: 1e-6)
        lr_encoder         — encoder/injector learning rate (default: 1e-4)
        temperature        — teacher generation temperature (default: 1.0)
        top_p              — teacher nucleus sampling (default: 1.0)
        logging_steps, save_steps

    run_dir is the phase0 artifact directory
    (artifacts/<arm>/<run_id>/phase0/).  Checkpoints are written directly there;
    no checkpoint_path config key is needed.
    """
    primary      = config["primary"]
    phase0_cfg   = config["phase0"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]

    model_id  = primary["huggingface_repo_id"]
    revision  = primary.get("revision", "main")
    dtype     = getattr(torch, primary.get("dtype", "bfloat16"))
    attn_impl = primary.get("attn_implementation", "sdpa")

    n_steps      = int(phase0_cfg.get("n_steps",         400))
    batch_size   = int(phase0_cfg.get("batch_size",       64))
    lr_backbone  = float(phase0_cfg.get("lr_backbone",    1e-6))
    lr_encoder   = float(phase0_cfg.get("lr_encoder",     1e-4))
    temperature  = float(phase0_cfg.get("temperature",    1.0))
    top_p        = float(phase0_cfg.get("top_p",          1.0))
    chunk_tokens = int(latent_cfg.get("chunk_tokens",     341))
    latent_dim   = int(latent_cfg.get("latent_dim",  LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim",  HIDDEN_DIM))
    log_steps    = int(phase0_cfg.get("logging_steps",    10))
    save_steps   = int(phase0_cfg.get("save_steps",       50))
    pool_path    = Path(phase0_cfg.get("pool_path",        "data/math_easy_pool.jsonl"))
    seed         = int(training_cfg.get("seed", 42))
    # CE anchor: only supervise the first z_anchor_tokens positions of each
    # chunk to force z-conditioning gradient.  None = supervise all tokens.
    _z_anchor_raw = phase0_cfg.get("z_anchor_tokens", None)
    z_anchor_tokens: int | None = int(_z_anchor_raw) if _z_anchor_raw is not None else None

    # Generation probe: every probe_steps steps, run the student in strict
    # Markov mode on probe_problems × probe_rollouts to log reward rate.
    # Off by default (probe_steps=0).
    probe_steps     = int(phase0_cfg.get("generation_probe_steps",    0))
    probe_problems  = int(phase0_cfg.get("generation_probe_problems", 10))
    probe_rollouts  = int(phase0_cfg.get("generation_probe_rollouts",  2))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 0 (distill) — device: %s  batch=%d  lr_bb=%.1e  lr_enc=%.1e",
                device, batch_size, lr_backbone, lr_encoder)

    # ── Problem pool ──────────────────────────────────────────────────────────
    with open(pool_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    logger.info("Phase 0 pool: %d problems from %s", len(problems), pool_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Pre-generate ALL teacher outputs ──────────────────────────────────────
    # Teacher is the only model in VRAM during this phase — student is not
    # loaded yet, so the teacher has the full 96 GB to itself.
    # Chunks are returned as CPU tensors by _generate_teacher_chunks, so
    # buffering all 400 batches costs ~120 MB of system RAM — negligible.
    # After pre-generation the teacher is deleted before the student loads.
    logger.info("Loading teacher %s @ %s ...", model_id, revision)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    _setup_compile(teacher)
    logger.info("  teacher loaded, frozen and compiled")
    pool_order: list[int] = []
    batches: list[tuple[list[dict], list]] = []

    logger.info("Pre-generating teacher chunks for %d steps (batch=%d) …", n_steps, batch_size)
    gen_bar = tqdm(total=n_steps, desc="teacher_gen", unit="step", dynamic_ncols=True)
    for _ in range(n_steps):
        while len(pool_order) < batch_size:
            order = list(range(len(problems)))
            _random.shuffle(order)
            pool_order.extend(order)
        step_problems = [problems[pool_order.pop(0)] for _ in range(batch_size)]
        teacher_chunks = _generate_teacher_chunks(
            teacher, tokenizer, step_problems,
            chunk_tokens=chunk_tokens,
            temperature=temperature, top_p=top_p, device=device,
        )
        batches.append((step_problems, teacher_chunks))
        gen_bar.update(1)
    gen_bar.close()

    del teacher
    torch.cuda.empty_cache()
    logger.info("Teacher done — %d batches cached on CPU.  Teacher VRAM freed.", n_steps)

    # ── Load student + encoder now that teacher VRAM is free ──────────────────
    logger.info("Loading student backbone %s @ %s ...", model_id, revision)
    student = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
    )
    logger.info("  student backbone loaded")

    # Apply LoRA so backbone adapts at lr=1e-4 without touching pretrained
    # base weights — eliminates catastrophic forgetting while giving the model
    # enough gradient signal to learn "generate math after z_prefix".
    lora_cfg = phase0_cfg.get("lora", {})
    student = _apply_lora(student, lora_cfg)
    student.train()

    if training_cfg.get("gradient_checkpointing", False):
        student.config.use_cache = False
        student.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  gradient checkpointing enabled")

    encoder = LatentStateEncoder(hidden_dim=hidden_dim, z_dim=latent_dim).to(device=device, dtype=dtype)
    logger.info("  encoder initialised (%.2fM params)",
                sum(p.numel() for p in encoder.parameters()) / 1e6)

    if config.get("experiment", {}).get("profile") != "smoke":
        _setup_compile(student, encoder)

    _fused = {"fused": True} if torch.cuda.is_available() else {}
    # Only LoRA adapter parameters train — base backbone weights are frozen.
    # lr_lora defaults to lr_encoder so both adapt at the same rate.
    lr_lora = float(phase0_cfg.get("lr_lora", lr_encoder))
    optimizer = torch.optim.AdamW([
        {"params": [p for p in student.parameters() if p.requires_grad], "lr": lr_lora},
        {"params": encoder.parameters(), "lr": lr_encoder},
    ], **_fused)

    # ── Student training ───────────────────────────────────────────────────────
    log_history: list[dict] = []
    pending:     dict[str, float] = {}

    step_bar = tqdm(total=n_steps, desc="phase0_distill", unit="step", dynamic_ncols=True)

    # Held-out slice for generation probes: last 20 % of the pool (never
    # the same problems across probes — resample fresh at each probe step).
    # Using a fixed slice like problems[-10:] risks always hitting the same
    # hard L4 problems, which biases probe_rate low and masks real progress.
    _held_out_size = max(probe_problems * 3, int(len(problems) * 0.2), 1)
    probe_held_out = problems[len(problems) - _held_out_size:] if probe_steps > 0 else []

    if z_anchor_tokens is not None:
        logger.info(
            "  z_anchor_tokens=%d — CE supervised on first %d positions per chunk "
            "(%.0f%% of %d-token chunk)",
            z_anchor_tokens, z_anchor_tokens,
            100.0 * z_anchor_tokens / chunk_tokens, chunk_tokens,
        )

    for global_step, (step_problems, teacher_chunks) in enumerate(batches):
        student.train(); encoder.train()
        optimizer.zero_grad()

        loss = _distill_loss(
            student, encoder, tokenizer,
            step_problems, teacher_chunks, device,
            z_anchor_tokens=z_anchor_tokens,
        )
        loss.backward()

        all_params = list(student.parameters()) + list(encoder.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()

        loss_val = loss.detach().item()

        # ── Logging ───────────────────────────────────────────────────────────
        pending["loss"] = pending.get("loss", 0.0) + loss_val

        if (global_step + 1) % log_steps == 0:
            n = log_steps
            entry: dict = {
                "step": global_step + 1,
                "loss": pending.get("loss", 0.0) / n,
            }
            log_history.append(entry)
            pending = {}
            logger.info("step %d | distill_loss=%.4f", entry["step"], entry["loss"])

        # ── Generation probe ──────────────────────────────────────────────────
        # Periodically run the student in strict Markov mode on held-out
        # problems to measure whether z-conditioning is working.  This gives
        # early warning of z-ignored failure long before the full sanity check.
        if probe_steps > 0 and (global_step + 1) % probe_steps == 0 and probe_held_out:
            probe_sample = _random.sample(
                probe_held_out, min(probe_problems, len(probe_held_out))
            )
            student.eval(); encoder.eval()
            with torch.no_grad():
                traces = generate_latent_traces(
                    model=student, tokenizer=tokenizer, encoder=encoder,
                    problems=probe_sample,
                    n_rollouts=probe_rollouts,
                    chunk_tokens=chunk_tokens,
                    temperature=1.0, top_p=1.0, device=device,
                )
            correct = sum(1 for t in traces if t["reward"] > 0)
            probe_rate = correct / max(len(traces), 1)
            entry_probe = {
                "step":       global_step + 1,
                "probe_rate": probe_rate,
                "probe_n":    len(traces),
            }
            log_history.append(entry_probe)
            logger.info(
                "step %d | [PROBE] generation reward rate: %.4f  (%d/%d correct)",
                global_step + 1, probe_rate, correct, len(traces),
            )
            student.train(); encoder.train()

        if (global_step + 1) % save_steps == 0:
            _save_phase0_checkpoint(
                run_dir / f"checkpoint-{global_step + 1}",
                student, encoder, global_step + 1, log_history, tokenizer,
            )

        step_bar.set_postfix(loss=f"{loss_val:.4f}")
        step_bar.update(1)

    step_bar.close()

    _save_phase0_checkpoint(run_dir, student, encoder, n_steps, log_history, tokenizer)
    logger.info("Phase 0 complete. Checkpoint → %s", run_dir)


# ── Phase 1: on-policy GRPO ────────────────────────────────────────────────────

@torch.no_grad()
def generate_latent_traces(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    encoder: LatentStateEncoder,
    problems: list[dict],
    n_rollouts: int,
    chunk_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[dict]:
    """Collect G rollouts per problem with strict Markov z injection.

    Generation uses [z_prefix | generate] exclusively — no prior chunk tokens
    are visible. z is the only cross-chunk information carrier.

    Chunk 1: generate([prompt])
    Chunk 2: generate([z_1_prefix])   ← strict Markov
    Chunk 3: generate([z_2_prefix])   ← strict Markov

    Repr computation (for prefix injection between chunks):
        repr_h = backbone([z_{h-1}_prefix | chunk_h_tokens])[-1 token hidden state]

    IS guarantee: generation context and training forward context are identical.

    Returns:
        Flat list of trace dicts, length = len(problems) × n_rollouts.
        Rollouts for the same problem are contiguous.

        Each dict keys:
            problem_id, rollout_idx, ground_truth, completion, reward,
            prompt_ids (CPU Tensor), chunk_ids (list of 3 CPU Tensors)
    """
    pad_id      = tokenizer.eos_token_id
    embed_layer = model.get_input_embeddings()
    model_dtype = embed_layer.weight.dtype

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

    # ── Chunk 1: generate from [prompt] ───────────────────────────────────────
    input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
    for i, pids in enumerate(all_prompt_ids):
        off = max_prompt_len - len(pids)     # left-pad
        input_ids[i, off:] = torch.tensor(pids, dtype=torch.long, device=device)
        attn_mask[i, off:] = 1

    gen1 = model.generate(
        input_ids, attention_mask=attn_mask,
        max_new_tokens=chunk_tokens, do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=pad_id,
    )
    chunk1_ids = [gen1[i, max_prompt_len:].cpu() for i in range(B)]
    del gen1, input_ids, attn_mask

    # ── Repr 1: [prompt | chunk1] → last-token hidden → z_1 → prefix_1 ───────
    full1_lens = [prompt_lengths[i] + chunk1_ids[i].shape[0] for i in range(B)]
    max_full1  = max(full1_lens)

    fi1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    fa1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    for i in range(B):
        L   = full1_lens[i]
        seq = torch.cat([
            torch.tensor(all_prompt_ids[i], dtype=torch.long),
            chunk1_ids[i],
        ]).to(device)
        fi1[i, :L] = seq
        fa1[i, :L] = 1

    _, hidden1 = _fwd(model, input_ids=fi1, attention_mask=fa1, bypass_lm_head=True)
    del fi1, fa1

    last1  = torch.tensor([full1_lens[i] - 1 for i in range(B)], device=device)
    repr_1 = _last_token_repr(hidden1, last1)
    del hidden1

    z_1      = encoder.encode(repr_1)
    prefix_1 = encoder.inject(z_1.to(model_dtype))    # [B, 1, H]

    # ── Chunk 2: generate from [z_1_prefix] only — strict Markov ─────────────
    # min_new_tokens prevents the backbone (barely adapted during Phase 0 at
    # lr=1e-6) from sampling EOS immediately when it sees the unfamiliar
    # z_prefix as its only context.  Without this, teacher-forcing exposure
    # bias causes the model to collapse chunk 2 to a single EOS token, making
    # chunk 3 computed from a meaningless repr, and depriving Phase 1 of any
    # non-chunk-1 reward signal.
    am_pfx = torch.ones(B, 1, dtype=torch.long, device=device)
    gen2   = model.generate(
        inputs_embeds=prefix_1, attention_mask=am_pfx,
        max_new_tokens=chunk_tokens,
        min_new_tokens=chunk_tokens // 2,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=pad_id,
    )
    # gen2[:, 0] is placeholder for the prefix embedding position; skip it
    chunk2_ids = [gen2[i, 1:].cpu() for i in range(B)]
    del gen2

    # ── Repr 2: [prefix_1 | chunk2] → last-token hidden → z_2 → prefix_2 ─────
    c2_lens = [c.shape[0] for c in chunk2_ids]
    ie2, am2, _ = _build_prefix_embeds(embed_layer, prefix_1, chunk2_ids, device, model_dtype)

    _, hidden2 = _fwd(model, inputs_embeds=ie2, attention_mask=am2, bypass_lm_head=True)
    del ie2, am2

    last2  = torch.tensor(c2_lens, device=device)
    repr_2 = _last_token_repr(hidden2, last2)
    del hidden2

    z_2      = encoder.encode(repr_2)
    prefix_2 = encoder.inject(z_2.to(model_dtype))    # [B, 1, H]
    del prefix_1

    # ── Chunk 3: generate from [z_2_prefix] only — strict Markov ─────────────
    gen3 = model.generate(
        inputs_embeds=prefix_2, attention_mask=am_pfx,
        max_new_tokens=chunk_tokens,
        min_new_tokens=chunk_tokens // 2,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_p=top_p, pad_token_id=pad_id,
    )
    chunk3_ids = [gen3[i, 1:].cpu() for i in range(B)]
    del gen3, prefix_2, am_pfx

    # ── Grade and assemble traces ──────────────────────────────────────────────
    traces: list[dict] = []
    for i in range(B):
        all_chunk_ids = torch.cat([chunk1_ids[i], chunk2_ids[i], chunk3_ids[i]])
        completion    = tokenizer.decode(all_chunk_ids, skip_special_tokens=True)
        pred          = extract_answer(completion)
        reward        = int(pred is not None and answers_equivalent(pred, all_gt[i]))

        traces.append({
            "problem_id":   all_problem_ids[i],
            "rollout_idx":  all_rollout_idxs[i],
            "ground_truth": all_gt[i],
            "completion":   completion,
            "reward":       reward,
            "prompt_ids":   torch.tensor(all_prompt_ids[i], dtype=torch.long),
            "chunk_ids":    [chunk1_ids[i], chunk2_ids[i], chunk3_ids[i]],
        })

    return traces


def _pipeline_with_grad(
    model: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    traces: list[dict],
    device: torch.device,
) -> dict[str, Any]:
    """Re-run the 3-chunk pipeline WITH gradient.

    Strict Markov: each chunk forward pass uses [z_prefix | chunk_tokens] only.
    Identical context to generate_latent_traces — IS = 1 exactly.

    Returns:
        z_list:         list of 3 tensors [B, latent_dim]  — LIVE
        log_pi_chunks:  list of 3 lists, each B tensors [chunk_len]  — LIVE
    """
    pad_id      = 0
    embed_layer = model.get_input_embeddings()
    model_dtype = embed_layer.weight.dtype
    B           = len(traces)

    prompt_ids_list = [t["prompt_ids"]   for t in traces]
    c1_ids_list     = [t["chunk_ids"][0] for t in traces]
    c2_ids_list     = [t["chunk_ids"][1] for t in traces]
    c3_ids_list     = [t["chunk_ids"][2] for t in traces]
    prompt_lens     = [p.shape[0] for p in prompt_ids_list]
    c1_lens         = [c.shape[0] for c in c1_ids_list]
    c2_lens         = [c.shape[0] for c in c2_ids_list]
    c3_lens         = [c.shape[0] for c in c3_ids_list]

    # ── Chunk 1: [prompt | chunk1] ─────────────────────────────────────────────
    full1_lens = [prompt_lens[i] + c1_lens[i] for i in range(B)]
    max_full1  = max(full1_lens)

    fi1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    fa1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    for i in range(B):
        L   = full1_lens[i]
        seq = torch.cat([prompt_ids_list[i].to(device), c1_ids_list[i].to(device)])
        fi1[i, :L] = seq
        fa1[i, :L] = 1

    logits1, hidden1 = _fwd(model, input_ids=fi1, attention_mask=fa1)
    del fi1, fa1

    # log π for chunk 1 tokens: logits at causal-shifted positions
    log_pi_1: list[torch.Tensor] = []
    for i in range(B):
        pl = prompt_lens[i]; rl = c1_lens[i]
        sl = logits1[i, pl - 1:pl + rl - 1, :]     # (rl, vocab)
        c1 = c1_ids_list[i].to(device)
        lp = sl.gather(1, c1.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        log_pi_1.append(lp)
    del logits1

    last1  = torch.tensor([full1_lens[i] - 1 for i in range(B)], device=device)
    repr_1 = _last_token_repr(hidden1, last1)
    del hidden1

    z_1      = encoder.encode(repr_1)
    prefix_1 = encoder.inject(z_1.to(model_dtype))

    # ── Chunk 2: [prefix_1 | chunk2] ──────────────────────────────────────────
    ie2, am2, _ = _build_prefix_embeds(embed_layer, prefix_1, c2_ids_list, device, model_dtype)

    logits2, hidden2 = _fwd(model, inputs_embeds=ie2, attention_mask=am2)
    del ie2, am2

    # logits2[:, j, :] predicts c2_token[j] (prefix at position 0, tokens at 1..L)
    log_pi_2: list[torch.Tensor] = []
    for i in range(B):
        L  = c2_lens[i]
        sl = logits2[i, :L, :]
        c2 = c2_ids_list[i].to(device)
        lp = sl.gather(1, c2.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        log_pi_2.append(lp)
    del logits2

    last2  = torch.tensor(c2_lens, device=device)
    repr_2 = _last_token_repr(hidden2, last2)
    del hidden2

    z_2      = encoder.encode(repr_2)
    prefix_2 = encoder.inject(z_2.to(model_dtype))

    # ── Chunk 3: [prefix_2 | chunk3] ──────────────────────────────────────────
    ie3, am3, _ = _build_prefix_embeds(embed_layer, prefix_2, c3_ids_list, device, model_dtype)

    logits3, hidden3 = _fwd(model, inputs_embeds=ie3, attention_mask=am3)
    del ie3, am3

    log_pi_3: list[torch.Tensor] = []
    for i in range(B):
        L  = c3_lens[i]
        sl = logits3[i, :L, :]
        c3 = c3_ids_list[i].to(device)
        lp = sl.gather(1, c3.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        log_pi_3.append(lp)
    del logits3

    last3  = torch.tensor(c3_lens, device=device)
    repr_3 = _last_token_repr(hidden3, last3)
    del hidden3

    z_3 = encoder.encode(repr_3)

    return {
        "z_list":        [z_1,     z_2,     z_3],
        "log_pi_chunks": [log_pi_1, log_pi_2, log_pi_3],
    }


def train_latent(config: dict[str, Any], run_dir: Path) -> None:
    """Phase 1: pure L_RL GRPO with live backbone + encoder + z injection.

    Loads Phase 0 backbone and encoder checkpoint, then runs on-policy GRPO
    for max_steps steps on the Level 5 hard pool.

    Every step:
      1. [no_grad]  collect G=128 rollouts for B=4 problems → 512 sequences
      2.            compute GRPO advantages per-problem-group (clipped to ±20)
      3. [with_grad] re-run full 3-chunk pipeline → L_RL
      4.            single global grad clip + step all params

    Config keys consumed:
        primary.*              — backbone model ID, revision, dtype
        latent_markov.*        — latent_dim, hidden_dim, chunk_tokens
        training.*             — seed, learning_rate (backbone), num_generations,
                                 batch_size, micro_batch_size, max_steps,
                                 temperature, top_p, gradient_checkpointing,
                                 logging_steps, save_steps
        phase1_loss.*          — adv_clip, grad_clip
        evaluation.path        — Level 5 hard pool path
    """
    primary      = config["primary"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]
    phase0_cfg   = config["phase0"]
    phase1_cfg   = config.get("phase1_loss", {})

    model_id  = primary["huggingface_repo_id"]
    revision  = primary.get("revision", "main")
    dtype     = getattr(torch, primary.get("dtype", "bfloat16"))
    attn_impl = primary.get("attn_implementation", "sdpa")

    seed             = int(training_cfg.get("seed",              42))
    lr_backbone      = float(training_cfg.get("learning_rate",   1e-6))
    lr_encoder       = float(phase0_cfg.get("lr_encoder",        1e-4))
    G                = int(training_cfg.get("num_generations",   128))
    batch_size       = int(training_cfg.get("batch_size",          4))
    micro_batch_size = int(training_cfg.get("micro_batch_size",  128))
    max_steps        = int(training_cfg.get("max_steps",         200))
    temperature      = float(training_cfg.get("temperature",     1.0))
    top_p            = float(training_cfg.get("top_p",           1.0))
    log_steps        = int(training_cfg.get("logging_steps",      10))
    save_steps       = int(training_cfg.get("save_steps",         50))
    grad_clip        = float(phase1_cfg.get("grad_clip",          1.0))
    adv_clip         = float(phase1_cfg.get("adv_clip",          20.0))
    chunk_tokens     = int(latent_cfg.get("chunk_tokens",         341))
    latent_dim       = int(latent_cfg.get("latent_dim",     LATENT_DIM))
    hidden_dim       = int(latent_cfg.get("hidden_dim",     HIDDEN_DIM))

    # Phase 0 artifacts live at  run_dir/phase0/  (written by pretrain_distill).
    # run_dir is the parent shared by both phases:  artifacts/<arm>/<run_id>/
    phase0_dir = run_dir / "phase0"
    pool_path  = Path(config["evaluation"]["path"])
    ckpt_path  = run_dir / "phase1"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 1 — device: %s  G=%d  adv_clip=%.1f  grad_clip=%.2f",
                device, G, adv_clip, grad_clip)

    # ── Backbone: load from Phase 0 checkpoint, merging LoRA if present ──────
    backbone_dir = phase0_dir / "backbone"
    if backbone_dir.is_dir():
        adapter_cfg = backbone_dir / "adapter_config.json"
        if adapter_cfg.is_file():
            # Phase 0 saved a LoRA adapter — load base model then merge so
            # Phase 1 trains the full merged backbone at lr_backbone.
            logger.info("Phase 0 LoRA adapter found — loading base + merging: %s", backbone_dir)
            base = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision,
                torch_dtype=dtype, device_map="auto",
                attn_implementation=attn_impl,
            )
            model = PeftModel.from_pretrained(base, str(backbone_dir))
            model = model.merge_and_unload()
            logger.info("  LoRA merged into backbone")
        else:
            logger.info("Loading backbone from Phase 0 checkpoint: %s", backbone_dir)
            model = AutoModelForCausalLM.from_pretrained(
                str(backbone_dir),
                torch_dtype=dtype, device_map="auto",
                attn_implementation=attn_impl,
            )
    else:
        logger.info("Phase 0 backbone dir not found; loading from HF: %s", model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            torch_dtype=dtype, device_map="auto",
            attn_implementation=attn_impl,
        )
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        str(backbone_dir) if backbone_dir.is_dir() else model_id,
        revision=None if backbone_dir.is_dir() else revision,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("  backbone ready")

    if training_cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  gradient checkpointing enabled")

    # ── Encoder: load from Phase 0 checkpoint ─────────────────────────────────
    encoder    = LatentStateEncoder(hidden_dim=hidden_dim, z_dim=latent_dim).to(device=device, dtype=dtype)
    enc_ckpt   = phase0_dir / "phase0_encoder.pt"
    if enc_ckpt.is_file():
        ckpt = torch.load(enc_ckpt, weights_only=False, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        logger.info("  encoder loaded from Phase 0 (step %d)", ckpt.get("step", 0))
    else:
        logger.warning("  Phase 0 encoder checkpoint not found at %s; using random init", enc_ckpt)
    encoder.train()

    # ── torch.compile (skip for smoke profiles) ────────────────────────────────
    if config.get("experiment", {}).get("profile") != "smoke":
        _setup_compile(model, encoder)

    # ── Optimiser: two param groups, fused CUDA kernel ─────────────────────────
    _fused = {"fused": True} if torch.cuda.is_available() else {}
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(),   "lr": lr_backbone},
        {"params": encoder.parameters(), "lr": lr_encoder},
    ], **_fused)

    # ── Training pool ─────────────────────────────────────────────────────────
    with open(pool_path, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    logger.info("Phase 1 pool: %d problems from %s", len(problems), pool_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    log_history: list[dict] = []
    pending:     dict[str, float] = {}
    pool_order:  list[int] = []

    step_bar = tqdm(total=max_steps, desc="phase1", unit="step", dynamic_ncols=True)

    for global_step in range(max_steps):
        # Reshuffle pool when exhausted
        while len(pool_order) < batch_size:
            order = list(range(len(problems)))
            _random.shuffle(order)
            pool_order.extend(order)
        step_problems = [problems[pool_order.pop(0)] for _ in range(batch_size)]

        # ── Rollout collection (no gradient, strict Markov) ────────────────────
        with torch.no_grad():
            model.eval(); encoder.eval()
            traces = generate_latent_traces(
                model=model, tokenizer=tokenizer, encoder=encoder,
                problems=step_problems, n_rollouts=G,
                chunk_tokens=chunk_tokens,
                temperature=temperature, top_p=top_p,
                device=device,
            )

        model.train(); encoder.train()

        rewards    = [float(t["reward"]) for t in traces]
        advantages = compute_grpo_advantages(rewards, group_size=G, adv_clip=adv_clip)

        # ── Training step: micro-batched L_RL ──────────────────────────────────
        optimizer.zero_grad()

        n_total = len(traces)
        n_micro = -(-n_total // micro_batch_size)   # ceil division
        l_rl_acc = 0.0

        for mb_start in range(0, n_total, micro_batch_size):
            mb_traces = traces    [mb_start : mb_start + micro_batch_size]
            mb_adv    = advantages[mb_start : mb_start + micro_batch_size]

            pipe = _pipeline_with_grad(model, encoder, mb_traces, device)

            rl_sum = torch.zeros(1, device=device)
            n_tok  = 0
            for lp_chunk in pipe["log_pi_chunks"]:
                for i, lp in enumerate(lp_chunk):
                    rl_sum = rl_sum + (-mb_adv[i] * lp.sum())
                    n_tok += lp.shape[0]
            l_rl = rl_sum / max(n_tok, 1)

            (l_rl / n_micro).backward()
            l_rl_acc += l_rl.detach().item() / n_micro

        all_params = list(model.parameters()) + list(encoder.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        reward_rate = sum(rewards) / len(rewards)
        for k, v in (("total", l_rl_acc), ("l_rl", l_rl_acc), ("reward_rate", reward_rate)):
            pending[k] = pending.get(k, 0.0) + v

        if (global_step + 1) % log_steps == 0:
            n     = log_steps
            entry = {
                "step":        global_step + 1,
                "total":       pending.get("total",       0.0) / n,
                "l_rl":        pending.get("l_rl",        0.0) / n,
                "reward_rate": pending.get("reward_rate", 0.0) / n,
            }
            log_history.append(entry)
            pending = {}
            logger.info(
                "step %d | total=%.4f rl=%.4f | reward=%.3f%%",
                entry["step"], entry["total"], entry["l_rl"],
                entry["reward_rate"] * 100,
            )

        if (global_step + 1) % save_steps == 0:
            _save_phase1_checkpoint(
                ckpt_path / f"checkpoint-{global_step + 1}",
                model, encoder, optimizer, global_step + 1, log_history, tokenizer,
            )

        step_bar.set_postfix(
            rl=f"{l_rl_acc:.4f}",
            rwd=f"{reward_rate:.3%}",
        )
        step_bar.update(1)

    step_bar.close()
    _save_phase1_checkpoint(
        ckpt_path / "final", model, encoder,
        optimizer, max_steps, log_history, tokenizer,
    )
    logger.info("Phase 1 complete. Checkpoint → %s", ckpt_path / "final")


# ── Checkpointing ──────────────────────────────────────────────────────────────

def _save_phase0_checkpoint(
    directory: Path,
    model: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    step: int,
    log_history: list[dict],
    tokenizer: AutoTokenizer | None = None,
) -> None:
    """Save Phase 0 backbone (LoRA adapter) + encoder weights and trainer state."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"encoder": _clean_state_dict(encoder.state_dict()), "step": step},
        directory / "phase0_encoder.pt",
    )
    # If the backbone is a PeftModel (LoRA), save only the adapter weights.
    # Downstream Phase 1 loads the base model + merges adapter before GRPO.
    #
    # torch.compile nuance for PeftModel: _setup_compile replaces the inner
    # base CausalLM's .model and .lm_head (i.e. model.model.{model,lm_head})
    # with compiled wrappers.  PEFT's state_dict walk traverses model.model.*
    # and would see _orig_mod.-prefixed keys for those parameters, causing the
    # saved adapter to be empty or mis-keyed.  We temporarily unwrap those
    # compiled layers (via _unwrapped_for_save on the inner CausalLM) before
    # calling save_pretrained so PEFT sees clean parameter names.
    backbone_dir = str(directory / "backbone")
    if isinstance(model, PeftModel):
        with _unwrapped_for_save(model.model):   # unwrap model.model.{model,lm_head}
            model.save_pretrained(backbone_dir)  # adapter_config.json + adapter weights
    else:
        with _unwrapped_for_save(model) as m:
            m.save_pretrained(backbone_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(backbone_dir)
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}, indent=2)
    )


def _save_phase1_checkpoint(
    directory: Path,
    model: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    log_history: list[dict],
    tokenizer: AutoTokenizer | None = None,
) -> None:
    """Save Phase 1 backbone + encoder + optimizer and trainer state."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder":   _clean_state_dict(encoder.state_dict()),
            "optimizer": optimizer.state_dict(),
            "step":      step,
        },
        directory / "phase1_latent.pt",
    )
    with _unwrapped_for_save(model) as m:
        m.save_pretrained(str(directory / "backbone"))
    if tokenizer is not None:
        tokenizer.save_pretrained(str(directory / "backbone"))
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}, indent=2)
    )
