"""Latent Markov GRPO — Phase 0 VAE pretraining and Phase 1 joint RL training.

Phase 0  (this file, pretrain_vae)
──────────────────────────────────
Trains VAEStateEncoder + OutcomeHead on pre-saved rollout data.
Backbone is FROZEN throughout; only the small VAE modules receive gradients.
Inputs are static (repr_h tensors saved by generate_phase0_rollouts.py) so
there is no backbone forward pass in the training loop.

Total losses:
  L_ELBO       = reconstruction (MSE) + KL divergence   [per chunk, summed]
  L_transition = ‖f(z_h) − z_{h+1}‖²                   [summed h=1→2, 2→3]
  L_outcome    = BCE(outcome_head(z_3), reward_label)    [terminal chunk only]
  L_phase0     = λ_elbo × L_ELBO + λ_trans × L_transition + λ_out × L_outcome

Phase 1  (stub — to be implemented in a future session)
────────────────────────────────────────────────────────
train_latent(): joint GRPO loop with live backbone inference, z injection,
and the combined policy + Markov losses.

See reports/latent_markov_design.md §Training for full design rationale.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.vae_state_encoder import (
    HIDDEN_DIM,
    LATENT_DIM,
    N_CHUNKS,
    OutcomeHead,
    VAEStateEncoder,
    ZInjector,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.training.grpo_baseline import SYSTEM_PROMPT, answers_equivalent, extract_answer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RolloutDataset — loads pre-saved Phase 0 rollout file
# ---------------------------------------------------------------------------

class RolloutDataset(Dataset):
    """Wraps the .pt file produced by generate_phase0_rollouts.py.

    Each item is a dict with tensors for one trajectory:
      repr_1, repr_2, repr_3  — float32 (HIDDEN_DIM,)
      reward                  — int (0 or 1)

    The dataset holds everything in RAM (≈ 440 MB for the full easy pool).
    """

    def __init__(self, rollout_path: Path) -> None:
        super().__init__()
        self.rollout_path = rollout_path
        self.data: list[dict] = torch.load(rollout_path, weights_only=False)
        logger.info(
            "loaded %d trajectories from %s", len(self.data), rollout_path
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]
        return {
            "repr_1": item["repr_1"].float(),
            "repr_2": item["repr_2"].float(),
            "repr_3": item["repr_3"].float(),
            "reward":  torch.tensor(item["reward"], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def vae_training_step(
    batch: dict[str, torch.Tensor],
    vae: VAEStateEncoder,
    outcome_head: OutcomeHead,
    lambda_elbo: float,
    lambda_trans: float,
    lambda_out: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute the combined Phase 0 loss for one mini-batch.

    L_total = λ_elbo × L_ELBO  +  λ_trans × L_transition  +  λ_out × L_outcome

    Args:
        batch:        mini-batch dict from RolloutDataset.
                      Keys: "repr_1", "repr_2", "repr_3" — (B, HIDDEN_DIM);
                            "reward" — (B,) float32.
        vae:          VAEStateEncoder in training mode.
        outcome_head: OutcomeHead in training mode.
        lambda_elbo:  L_ELBO loss weight.
        lambda_trans: L_transition loss weight.
        lambda_out:   L_outcome (BCE) loss weight.
        device:       compute device.

    Returns:
        Dict with scalar tensors:
          "loss"       — total combined loss (backward-able).
          "elbo"       — L_ELBO (detached, for logging).
          "transition" — L_transition (detached, for logging).
          "outcome"    — L_outcome (detached, for logging).
    """
    repr_list = [batch["repr_1"].to(device),
                 batch["repr_2"].to(device),
                 batch["repr_3"].to(device)]
    reward = batch["reward"].to(device).unsqueeze(-1)   # (B, 1) for BCE

    results     = vae.forward(repr_list)
    z_list      = [r[0] for r in results]
    mu_list     = [r[1] for r in results]
    logvar_list = [r[2] for r in results]

    l_elbo  = vae.compute_elbo(repr_list, z_list, mu_list, logvar_list)
    l_trans = vae.compute_transition_loss(z_list)
    l_out   = F.binary_cross_entropy(outcome_head(z_list[-1]), reward)

    total = lambda_elbo * l_elbo + lambda_trans * l_trans + lambda_out * l_out
    return {
        "loss":       total,
        "elbo":       l_elbo.detach(),
        "transition": l_trans.detach(),
        "outcome":    l_out.detach(),
    }

# ---------------------------------------------------------------------------
# pretrain_vae — Phase 0 training loop (scaffolding complete)
# ---------------------------------------------------------------------------

def pretrain_vae(config: dict[str, Any], run_dir: Path) -> None:
    """Train VAEStateEncoder and OutcomeHead on pre-saved Phase 0 rollouts.

    Reads all hyperparameters from `config["phase0"]`. The backbone is NOT
    loaded or used here — training runs entirely on static repr_h tensors.

    Config keys consumed (under "phase0"):
      rollout_path      — path to the .pt file from generate_phase0_rollouts.py
      num_epochs        — training epochs over the rollout dataset
      batch_size        — trajectories per mini-batch
      learning_rate     — AdamW lr
      lambda_elbo       — L_ELBO weight
      lambda_trans      — L_transition weight
      lambda_out        — L_outcome weight
      logging_steps     — log every N steps
      save_steps        — checkpoint every N steps
      checkpoint_path   — where to save VAE + OutcomeHead weights

    After this function returns, the saved checkpoint can be loaded by
    train_latent() (Phase 1) via torch.load().
    """
    phase0_cfg = config["phase0"]

    rollout_path = Path(phase0_cfg["rollout_path"])
    num_epochs   = phase0_cfg.get("num_epochs", 3)
    batch_size   = phase0_cfg.get("batch_size", 256)
    lr           = phase0_cfg.get("learning_rate", 3e-4)
    lambda_elbo  = phase0_cfg.get("lambda_elbo", 1.0)
    lambda_trans = phase0_cfg.get("lambda_trans", 1.0)
    lambda_out   = phase0_cfg.get("lambda_out", 1.0)
    logging_steps = phase0_cfg.get("logging_steps", 10)
    save_steps    = phase0_cfg.get("save_steps", 100)
    ckpt_path     = Path(phase0_cfg.get("checkpoint_path", run_dir / "phase0_vae.pt"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 0 — device: %s", device)

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    dataset = RolloutDataset(rollout_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    logger.info(
        "dataset: %d trajectories | %d steps/epoch | %d total steps",
        len(dataset), steps_per_epoch, total_steps,
    )

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    vae = VAEStateEncoder().to(device)
    outcome_head = OutcomeHead().to(device)
    vae.train()
    outcome_head.train()

    optimizer = torch.optim.AdamW(
        list(vae.parameters()) + list(outcome_head.parameters()),
        lr=lr,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    global_step = 0
    log_history: list[dict] = []
    pending: dict[str, float] = {}

    step_bar = tqdm(total=total_steps, desc="phase0", unit="step", dynamic_ncols=True)

    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            metrics = vae_training_step(
                batch,
                vae,
                outcome_head,
                lambda_elbo=lambda_elbo,
                lambda_trans=lambda_trans,
                lambda_out=lambda_out,
                device=device,
            )

            loss: torch.Tensor = metrics["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(vae.parameters()) + list(outcome_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            global_step += 1

            # Accumulate for logging.
            for k in ("loss", "elbo", "transition", "outcome"):
                pending[k] = pending.get(k, 0.0) + metrics[k].item()

            if global_step % logging_steps == 0:
                n = logging_steps
                entry = {
                    "step":  global_step,
                    "epoch": round(global_step / steps_per_epoch, 2),
                    **{k: pending.get(k, 0.0) / n for k in ("loss", "elbo", "transition", "outcome")},
                }
                log_history.append(entry)
                pending = {}
                logger.info(
                    "step %d | loss %.4f | elbo %.4f | trans %.4f | outcome %.4f",
                    entry["step"], entry["loss"], entry["elbo"],
                    entry["transition"], entry["outcome"],
                )

            if global_step % save_steps == 0:
                _save_phase0_checkpoint(
                    run_dir / f"checkpoint-{global_step}",
                    vae, outcome_head, global_step, log_history,
                )

            step_bar.set_postfix(
                loss=f"{loss.item():.4f}", epoch=epoch, step=global_step
            )
            step_bar.update(1)

    step_bar.close()

    # Final checkpoint.
    _save_phase0_checkpoint(ckpt_path.parent, vae, outcome_head, global_step, log_history)
    logger.info("Phase 0 complete. VAE checkpoint → %s", ckpt_path.parent)


def _save_phase0_checkpoint(
    directory: Path,
    vae: VAEStateEncoder,
    outcome_head: OutcomeHead,
    step: int,
    log_history: list[dict],
) -> None:
    """Save VAE and OutcomeHead weights + trainer state to `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vae": vae.state_dict(),
            "outcome_head": outcome_head.state_dict(),
            "step": step,
        },
        directory / "phase0_vae.pt",
    )
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}, indent=2)
    )


# ---------------------------------------------------------------------------
# Phase 1 helpers — plumbing
# ---------------------------------------------------------------------------

def lambda_trans_schedule(step: int, max_steps: int = 200, floor: float = 0.1) -> float:
    """Return the transition loss weight λ_t for the current training step.

    Schedule (per design doc §Phase 1):
      - steps 0 … max_steps//2 : linear decay from 1.0 → floor
      - steps max_steps//2 … end : constant at floor

    Rationale: high λ_t early provides dense gradient flow while L_RL ≈ 0
    (sparse reward).  Decaying λ_t lets L_RL dominate once rewards start
    appearing.  The floor keeps L_transition active as a Markov regulariser.
    """
    halfway = max_steps // 2
    if step >= halfway:
        return floor
    return 1.0 - (1.0 - floor) * (step / halfway)


def compute_grpo_advantages(
    rewards: list[float],
    group_size: int,
    eps: float = 1e-8,
) -> list[float]:
    """Normalise rewards into GRPO advantages within each group of G rollouts.

    For each group of `group_size` consecutive rollouts (all belonging to the
    same problem), computes:

        A_i = (r_i − μ_group) / (σ_group + eps)

    where μ and σ are the group mean and standard deviation.  The resulting
    advantages have zero mean and unit variance within each group, which is
    the standard GRPO normalisation (DeepSeek-R1, §3.1).

    Args:
        rewards:    flat list of scalar rewards, length = n_problems × group_size.
                    Must be ordered so that rollouts for the same problem are
                    contiguous (i.e. output of generate_latent_traces()).
        group_size: G — number of rollouts per problem.
        eps:        numerical stability floor for the standard deviation.

    Returns:
        List of advantages, same length and ordering as `rewards`.
    """
    assert len(rewards) % group_size == 0, (
        f"len(rewards)={len(rewards)} is not divisible by group_size={group_size}"
    )
    advantages: list[float] = []
    for i in range(0, len(rewards), group_size):
        group = rewards[i : i + group_size]
        mu = sum(group) / group_size
        var = sum((r - mu) ** 2 for r in group) / group_size
        sigma = var ** 0.5
        for r in group:
            advantages.append((r - mu) / (sigma + eps))
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
# Phase 1 core — TODO for you to implement
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
    """Generate G chunked rollouts per problem with z_h injected between chunks.

    This is the inference engine for Phase 1.  Each rollout consists of
    N_CHUNKS=3 sequential generation calls, where chunks 2 and 3 receive the
    previous chunk's latent state z_h as a soft prefix token (via inputs_embeds).

    Generation protocol per rollout
    ────────────────────────────────
    Chunk 1 — no prior state:
        • context  : prompt_ids (system + problem, tokenized)
        • generate : model.generate(input_ids=prompt_ids, max_new_tokens=chunk_tokens)
        • extract  : repr_1 — mean-pool last-layer hidden states over chunk 1 tokens
                     (use a forward hook on model.model, as in generate_phase0_rollouts.py)
        • encode   : mu_1, logvar_1 = vae.encode(repr_1)
                     z_1 = vae.reparameterize(mu_1, logvar_1)   # training mode → stochastic
        • log-probs: forward pass on (prompt_ids + chunk1_ids) → log p(chunk1_ids | prompt)
                     store as old_lp_1 — needed for the GRPO importance ratio in the
                     training step.

    Chunk 2 — z_1 prefix:
        • prefix   : embed = z_injector.get_prefix_embedding(z_1)  # (B, 1, hidden_dim)
        • context  : embed_layer = model.get_input_embeddings()
                     inputs_embeds = torch.cat([embed, embed_layer(chunk1_ids)], dim=1)
                     attention_mask = torch.ones(B, 1 + chunk1_len, ...)
        • generate : model.generate(inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    max_new_tokens=chunk_tokens)
                     NOTE: when inputs_embeds is passed, model.generate does NOT
                     expect input_ids.  The generated token IDs start at index 0
                     of the output (there is no prompt prefix in the returned tensor).
        • extract  : repr_2 from chunk 2 tokens only (same hook trick)
        • encode   : z_2 = vae.reparameterize(*vae.encode(repr_2))
        • log-probs: forward pass on (inputs_embeds + chunk2_ids embedded) to get
                     old_lp_2 for the chunk 2 tokens.

    Chunk 3 — z_2 prefix (same pattern):
        • inputs_embeds = cat([z_injector(z_2), embed_layer(chunk2_ids)], dim=1)
        • generate chunk3_ids
        • repr_3, z_3 (z_3 not used for injection but kept for L_VAE)
        • old_lp_3

    Grading:
        completion = decode(chunk1_ids + chunk2_ids + chunk3_ids)
        reward = grade(completion, ground_truth)   # reuse from generate_phase0_rollouts.py

    Args:
        model:       backbone (unfrozen, in training mode — call model.eval() only
                     during the no_grad generation, then restore model.train()).
        tokenizer:   Qwen tokenizer (padding_side="left" set at load time).
        vae:         VAEStateEncoder (training mode).
        z_injector:  ZInjector (training mode).
        problems:    list of problem dicts (keys: "prompt", "ground_truth",
                     optionally "problem_id").  Length = B (problems per step).
        n_rollouts:  G — rollouts per problem.
        chunk_tokens: new tokens to generate per chunk (from config).
        temperature: sampling temperature.
        top_p:       nucleus sampling cutoff.
        device:      CUDA device.
        n_chunks:    always 3 — kept as argument for forward-compatibility.

    Returns:
        Flat list of trajectory dicts, length = len(problems) × n_rollouts.
        Rollouts for the same problem are contiguous (problem 0 rollouts 0..G-1,
        then problem 1 rollouts 0..G-1, etc.) — required by compute_grpo_advantages().

        Each dict contains:
          "problem_id"   : str
          "rollout_idx"  : int  (0 … G-1)
          "ground_truth" : str
          "completion"   : str  (full decoded text, all 3 chunks concatenated)
          "reward"       : int  (0 or 1)
          "chunk_ids"    : list[Tensor]  — [chunk1_ids, chunk2_ids, chunk3_ids] on CPU
          "old_log_probs": list[Tensor]  — [lp1, lp2, lp3] per-token log-probs on CPU
                           shape of each: (chunk_len,)
          "repr_1", "repr_2", "repr_3": Tensor (hidden_dim,) on CPU, detached
          "z_1",    "z_2",    "z_3"   : Tensor (latent_dim,) on CPU, detached
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

    B             = len(all_prompt_ids)
    prompt_lengths = [len(p) for p in all_prompt_ids]
    max_prompt_len = max(prompt_lengths)

    # ── Chunk 1: left-padded token-ID generation ──────────────────────
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

    # Forward pass over [prompt | chunk1] to get repr_1 and old log-probs.
    full_seqs1 = [
        torch.cat([
            torch.tensor(all_prompt_ids[i], dtype=torch.long, device=device),
            chunk1_ids_list[i].to(device),
        ])
        for i in range(B)
    ]
    max_full1 = max(s.shape[0] for s in full_seqs1)
    full_ids1  = torch.full((B, max_full1), pad_id, dtype=torch.long, device=device)
    full_attn1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs1):
        L = seq.shape[0]; full_ids1[i, :L] = seq; full_attn1[i, :L] = 1

    fwd1     = model(full_ids1, attention_mask=full_attn1, output_hidden_states=True)
    hidden1  = fwd1.hidden_states[-1]                     # (B, max_full1, H)
    lp1_full = torch.log_softmax(fwd1.logits, dim=-1)     # (B, max_full1, vocab)

    repr_1_list: list[torch.Tensor] = []
    old_lp1_list: list[torch.Tensor] = []
    for i in range(B):
        pl = prompt_lengths[i]
        rl = chunk1_ids_list[i].shape[0]
        repr_1_list.append(hidden1[i, pl:pl+rl, :].mean(0).detach().cpu().float())
        c1 = chunk1_ids_list[i].to(device)
        lp = lp1_full[i, pl-1:pl+rl-1, :].gather(1, c1.unsqueeze(1)).squeeze(1)
        old_lp1_list.append(lp.detach().cpu())

    del full_ids1, full_attn1, hidden1, lp1_full, fwd1
    torch.cuda.empty_cache()

    # VAE → z_1
    repr_1_batch = torch.stack(repr_1_list).to(device)
    mu_1, logvar_1 = vae.encode(repr_1_batch)
    z_1_batch = vae.reparameterize(mu_1, logvar_1)        # (B, latent_dim)
    z_1_list  = list(z_1_batch.detach().cpu())
    pad_emb   = embed_layer(torch.tensor([pad_id], dtype=torch.long, device=device))  # (1, H)

    # ── Chunk 2: [z_1 prefix | chunk1] as inputs_embeds ───────────────
    z_pfx1    = z_injector.get_prefix_embedding(z_1_batch)   # (B, 1, H)
    c1_lens   = [c.shape[0] for c in chunk1_ids_list]
    max_c1    = max(c1_lens)
    emb_len2  = 1 + max_c1                                   # z + chunk1 (left-padded)

    ie2 = torch.zeros(B, emb_len2, hidden_dim, dtype=model_dtype, device=device)
    am2 = torch.zeros(B, emb_len2, dtype=torch.long, device=device)
    for i in range(B):
        L1  = c1_lens[i]; off = max_c1 - L1
        ie2[i, :off, :]        = pad_emb
        ie2[i, off, :]         = z_pfx1[i, 0, :]
        ie2[i, off+1:off+1+L1, :] = embed_layer(chunk1_ids_list[i].to(device))
        am2[i, off:] = 1

    gen2 = model.generate(
        inputs_embeds=ie2, attention_mask=am2,
        max_new_tokens=chunk_tokens, do_sample=True,
        temperature=temperature, top_p=top_p,
        pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
    )
    # When inputs_embeds is passed, some HF versions return only new tokens;
    # others append them after the embed length. Handle both.
    chunk2_ids_list = [
        (gen2[i, emb_len2:] if gen2.shape[1] > chunk_tokens else gen2[i]).cpu()
        for i in range(B)
    ]
    del gen2, ie2, am2
    torch.cuda.empty_cache()

    # Forward pass [z_pfx | chunk1 | chunk2] for repr_2 + old log-probs
    c2_lens   = [c.shape[0] for c in chunk2_ids_list]
    max_fwd2  = max(1 + c1_lens[i] + c2_lens[i] for i in range(B))
    fe2 = torch.zeros(B, max_fwd2, hidden_dim, dtype=model_dtype, device=device)
    fa2 = torch.zeros(B, max_fwd2, dtype=torch.long, device=device)
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]; tot = 1 + L1 + L2
        fe2[i, 0, :]         = z_pfx1[i, 0, :]
        fe2[i, 1:1+L1, :]    = embed_layer(chunk1_ids_list[i].to(device))
        fe2[i, 1+L1:tot, :] = embed_layer(chunk2_ids_list[i].to(device))
        fa2[i, :tot] = 1

    fwd2     = model(inputs_embeds=fe2, attention_mask=fa2, output_hidden_states=True)
    hidden2  = fwd2.hidden_states[-1]
    lp2_full = torch.log_softmax(fwd2.logits, dim=-1)

    repr_2_list:  list[torch.Tensor] = []
    old_lp2_list: list[torch.Tensor] = []
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]
        repr_2_list.append(hidden2[i, 1+L1:1+L1+L2, :].mean(0).detach().cpu().float())
        c2 = chunk2_ids_list[i].to(device)
        lp = lp2_full[i, L1:L1+L2, :].gather(1, c2.unsqueeze(1)).squeeze(1)
        old_lp2_list.append(lp.detach().cpu())

    del fe2, fa2, hidden2, lp2_full, fwd2
    torch.cuda.empty_cache()

    # VAE → z_2
    repr_2_batch = torch.stack(repr_2_list).to(device)
    mu_2, logvar_2 = vae.encode(repr_2_batch)
    z_2_batch = vae.reparameterize(mu_2, logvar_2)
    z_2_list  = list(z_2_batch.detach().cpu())
    z_pfx2    = z_injector.get_prefix_embedding(z_2_batch)   # (B, 1, H)

    # ── Chunk 3: [z_2 prefix | chunk2] as inputs_embeds ───────────────
    max_c2   = max(c2_lens)
    emb_len3 = 1 + max_c2

    ie3 = torch.zeros(B, emb_len3, hidden_dim, dtype=model_dtype, device=device)
    am3 = torch.zeros(B, emb_len3, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; off = max_c2 - L2
        ie3[i, :off, :]           = pad_emb
        ie3[i, off, :]            = z_pfx2[i, 0, :]
        ie3[i, off+1:off+1+L2, :] = embed_layer(chunk2_ids_list[i].to(device))
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
    del gen3, ie3, am3
    torch.cuda.empty_cache()

    # Forward pass [z_pfx2 | chunk2 | chunk3] for repr_3 + old log-probs
    c3_lens  = [c.shape[0] for c in chunk3_ids_list]
    max_fwd3 = max(1 + c2_lens[i] + c3_lens[i] for i in range(B))
    fe3 = torch.zeros(B, max_fwd3, hidden_dim, dtype=model_dtype, device=device)
    fa3 = torch.zeros(B, max_fwd3, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; L3 = c3_lens[i]; tot = 1 + L2 + L3
        fe3[i, 0, :]        = z_pfx2[i, 0, :]
        fe3[i, 1:1+L2, :]   = embed_layer(chunk2_ids_list[i].to(device))
        fe3[i, 1+L2:tot, :] = embed_layer(chunk3_ids_list[i].to(device))
        fa3[i, :tot] = 1

    fwd3     = model(inputs_embeds=fe3, attention_mask=fa3, output_hidden_states=True)
    hidden3  = fwd3.hidden_states[-1]
    lp3_full = torch.log_softmax(fwd3.logits, dim=-1)

    repr_3_list:  list[torch.Tensor] = []
    old_lp3_list: list[torch.Tensor] = []
    for i in range(B):
        L2 = c2_lens[i]; L3 = c3_lens[i]
        repr_3_list.append(hidden3[i, 1+L2:1+L2+L3, :].mean(0).detach().cpu().float())
        c3 = chunk3_ids_list[i].to(device)
        lp = lp3_full[i, L2:L2+L3, :].gather(1, c3.unsqueeze(1)).squeeze(1)
        old_lp3_list.append(lp.detach().cpu())

    del fe3, fa3, hidden3, lp3_full, fwd3
    torch.cuda.empty_cache()

    # VAE → z_3 (not used for injection but kept for L_VAE)
    repr_3_batch = torch.stack(repr_3_list).to(device)
    mu_3, logvar_3 = vae.encode(repr_3_batch)
    z_3_batch = vae.reparameterize(mu_3, logvar_3)
    z_3_list  = list(z_3_batch.detach().cpu())

    # ── Grade and assemble trajectories ───────────────────────────────
    trajectories: list[dict] = []
    for i in range(B):
        all_chunk_ids = torch.cat([
            chunk1_ids_list[i], chunk2_ids_list[i], chunk3_ids_list[i]
        ])
        completion = tokenizer.decode(all_chunk_ids, skip_special_tokens=True)
        pred = extract_answer(completion)
        reward = int(pred is not None and answers_equivalent(pred, all_gt[i]))

        trajectories.append({
            "problem_id":    all_problem_ids[i],
            "rollout_idx":   all_rollout_idxs[i],
            "ground_truth":  all_gt[i],
            "completion":    completion,
            "reward":        reward,
            "prompt_ids":    torch.tensor(all_prompt_ids[i], dtype=torch.long),
            "chunk_ids":     [chunk1_ids_list[i], chunk2_ids_list[i], chunk3_ids_list[i]],
            "old_log_probs": [old_lp1_list[i], old_lp2_list[i], old_lp3_list[i]],
            "repr_1": repr_1_list[i], "repr_2": repr_2_list[i], "repr_3": repr_3_list[i],
            "z_1":    z_1_list[i],    "z_2":    z_2_list[i],    "z_3":    z_3_list[i],
        })

    return trajectories




def latent_training_step(
    model: AutoModelForCausalLM,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    traces: list[dict],
    advantages: list[float],
    lambda_t: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Compute the combined Phase 1 loss for one training step.

    L_total = L_RL  +  λ_t × L_transition  +  L_VAE

    ── L_RL (GRPO policy gradient) ──────────────────────────────────────────
    For each chunk h and each rollout i, re-run a differentiable forward pass
    through the backbone (WITH gradient) using the same inputs_embeds structure
    as generate_latent_traces() but with the z values from the trace detached
    (treat them as fixed conditioning, not as part of the gradient path for L_RL).

    For each token t in chunk h of rollout i:
        log_pi_current(t) = log p_θ(token_t | context_h_i)   ← differentiable
        log_pi_old(t)     = traces[i]["old_log_probs"][h][t]  ← constant from trace

    GRPO objective (without clipping for simplicity; add PPO-style clipping later):
        L_RL = -mean over all (i, h, t) of:
               advantage_i × (log_pi_current - log_pi_old)

    Note: L_RL updates backbone parameters + z_injector (via the inputs_embeds path).
    It does NOT flow gradients into the VAE encoder (z values are detached).

    ── L_VAE and L_transition (VAE regularisation) ───────────────────────────
    These losses use the repr_h tensors stored in the traces (computed with the
    backbone at trace-generation time, treated as constants here).  This keeps
    L_VAE and L_transition from updating backbone parameters — only the small
    VAE modules receive gradient from these terms.

    Steps:
        repr_list = [torch.stack([t["repr_h"] for t in traces]) for h in 1,2,3]
                    → each tensor: (B*G, hidden_dim), on device
        results   = vae.forward(repr_list)           # differentiable VAE pass
        z_list    = [r[0] for r in results]
        mu_list   = [r[1] for r in results]
        logvar_list = [r[2] for r in results]
        L_VAE     = vae.compute_elbo(repr_list, z_list, mu_list, logvar_list)
        L_trans   = vae.compute_transition_loss(z_list)

    ── Total ─────────────────────────────────────────────────────────────────
        L_total = L_RL + lambda_t × L_trans + L_VAE

    Args:
        model:      backbone in training mode.
        vae:        VAEStateEncoder in training mode.
        z_injector: ZInjector in training mode.
        traces:     output of generate_latent_traces() for this step.
        advantages: aligned list of GRPO advantages (output of compute_grpo_advantages()).
        lambda_t:   current transition loss weight (from lambda_trans_schedule()).
        device:     CUDA device.

    Returns:
        Dict with backward-able scalar tensor "total" and detached scalars
        "l_rl", "l_vae", "l_trans" for logging.
    """
    embed_layer = model.get_input_embeddings()
    model_dtype = embed_layer.weight.dtype
    hidden_dim  = vae.hidden_dim
    B           = len(traces)
    adv         = torch.tensor(advantages, dtype=torch.float32, device=device)

    # ── L_VAE + L_transition (repr_h from traces = detached constants) ─
    repr_list = [
        torch.stack([t["repr_1"] for t in traces]).to(device),
        torch.stack([t["repr_2"] for t in traces]).to(device),
        torch.stack([t["repr_3"] for t in traces]).to(device),
    ]
    results     = vae.forward(repr_list)
    z_list      = [r[0] for r in results]
    mu_list     = [r[1] for r in results]
    logvar_list = [r[2] for r in results]
    l_vae   = vae.compute_elbo(repr_list, z_list, mu_list, logvar_list)
    l_trans = vae.compute_transition_loss(z_list)

    # ── L_RL: three differentiable forward passes, one per chunk ───────
    # z values from traces are detached → only backbone + z_injector get gradient.
    log_ratio_sum = torch.zeros(1, device=device)
    token_count   = 0

    # Chunk 1: token-ID forward pass over [prompt | chunk1]
    chunk1_ids_list = [t["chunk_ids"][0] for t in traces]
    prompt_ids_list = [t["prompt_ids"] for t in traces]
    c1_lens  = [c.shape[0] for c in chunk1_ids_list]
    pl_list  = [p.shape[0] for p in prompt_ids_list]

    full_seqs1 = [
        torch.cat([p.to(device), c.to(device)])
        for p, c in zip(prompt_ids_list, chunk1_ids_list)
    ]
    max_f1 = max(s.shape[0] for s in full_seqs1)
    fi1 = torch.zeros(B, max_f1, dtype=torch.long, device=device)
    fa1 = torch.zeros(B, max_f1, dtype=torch.long, device=device)
    for i, seq in enumerate(full_seqs1):
        L = seq.shape[0]; fi1[i, :L] = seq; fa1[i, :L] = 1
    logits1 = model(fi1, attention_mask=fa1).logits  # (B, max_f1, vocab)
    for i in range(B):
        pl = pl_list[i]; rl = c1_lens[i]
        c1  = chunk1_ids_list[i].to(device)
        sl  = logits1[i, pl-1:pl+rl-1, :]                          # (rl, vocab)
        # log p(token) = logit(token) - logsumexp(logits) — avoids materialising
        # a full (B, seq, vocab) log_softmax tensor in the computation graph.
        curr = sl.gather(1, c1.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        old  = traces[i]["old_log_probs"][0].to(device)[:rl]
        log_ratio_sum = log_ratio_sum + (adv[i] * (curr - old)).sum()
        token_count  += rl
    del fi1, fa1, logits1

    # Chunk 2: inputs_embeds [z_1_detached | chunk1 | chunk2]
    z1_det = torch.stack([t["z_1"].to(device) for t in traces])   # detached from trace
    z_pfx1 = z_injector.get_prefix_embedding(z1_det)               # grad flows into z_injector
    chunk2_ids_list = [t["chunk_ids"][1] for t in traces]
    c2_lens = [c.shape[0] for c in chunk2_ids_list]

    max_f2 = max(1 + c1_lens[i] + c2_lens[i] for i in range(B))
    fe2 = torch.zeros(B, max_f2, hidden_dim, dtype=model_dtype, device=device)
    fa2 = torch.zeros(B, max_f2, dtype=torch.long, device=device)
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]; tot = 1 + L1 + L2
        fe2[i, 0, :]         = z_pfx1[i, 0, :]
        fe2[i, 1:1+L1, :]    = embed_layer(chunk1_ids_list[i].to(device))
        fe2[i, 1+L1:tot, :] = embed_layer(chunk2_ids_list[i].to(device))
        fa2[i, :tot] = 1
    logits2 = model(inputs_embeds=fe2, attention_mask=fa2).logits
    for i in range(B):
        L1 = c1_lens[i]; L2 = c2_lens[i]
        c2  = chunk2_ids_list[i].to(device)
        sl  = logits2[i, L1:L1+L2, :]
        curr = sl.gather(1, c2.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        old  = traces[i]["old_log_probs"][1].to(device)[:L2]
        log_ratio_sum = log_ratio_sum + (adv[i] * (curr - old)).sum()
        token_count  += L2
    del fe2, fa2, logits2

    # Chunk 3: inputs_embeds [z_2_detached | chunk2 | chunk3]
    z2_det = torch.stack([t["z_2"].to(device) for t in traces])
    z_pfx2 = z_injector.get_prefix_embedding(z2_det)
    chunk3_ids_list = [t["chunk_ids"][2] for t in traces]
    c3_lens = [c.shape[0] for c in chunk3_ids_list]

    max_f3 = max(1 + c2_lens[i] + c3_lens[i] for i in range(B))
    fe3 = torch.zeros(B, max_f3, hidden_dim, dtype=model_dtype, device=device)
    fa3 = torch.zeros(B, max_f3, dtype=torch.long, device=device)
    for i in range(B):
        L2 = c2_lens[i]; L3 = c3_lens[i]; tot = 1 + L2 + L3
        fe3[i, 0, :]        = z_pfx2[i, 0, :]
        fe3[i, 1:1+L2, :]   = embed_layer(chunk2_ids_list[i].to(device))
        fe3[i, 1+L2:tot, :] = embed_layer(chunk3_ids_list[i].to(device))
        fa3[i, :tot] = 1
    logits3 = model(inputs_embeds=fe3, attention_mask=fa3).logits
    for i in range(B):
        L2 = c2_lens[i]; L3 = c3_lens[i]
        c3  = chunk3_ids_list[i].to(device)
        sl  = logits3[i, L2:L2+L3, :]
        curr = sl.gather(1, c3.unsqueeze(1)).squeeze(1) - torch.logsumexp(sl, dim=-1)
        old  = traces[i]["old_log_probs"][2].to(device)[:L3]
        log_ratio_sum = log_ratio_sum + (adv[i] * (curr - old)).sum()
        token_count  += L3
    del fe3, fa3, logits3

    l_rl    = -(log_ratio_sum / max(token_count, 1))
    l_total = l_rl + lambda_t * l_trans + l_vae

    return {
        "total":   l_total,
        "l_rl":    l_rl.detach(),
        "l_vae":   l_vae.detach(),
        "l_trans": l_trans.detach(),
    }


# ---------------------------------------------------------------------------
# train_latent — Phase 1 joint RL loop
# ---------------------------------------------------------------------------

def train_latent(config: dict[str, Any], run_dir: Path) -> None:
    """Phase 1: joint GRPO training with live backbone and z injection.

    Loads the Phase 0 VAE checkpoint, unfreezes the backbone, and runs the
    custom latent Markov GRPO loop for max_steps steps:

      L_total = L_RL  +  λ_t × L_transition  +  L_VAE

    L_RL is computed by generate_latent_traces() (rollout collection) and
    latent_training_step() (differentiable re-computation).  L_transition and
    L_VAE are computed from the pre-saved repr_h tensors in the traces.

    Config keys consumed:
      primary.*               — backbone model ID, revision, dtype
      phase0.checkpoint_path  — path to phase0_vae.pt (VAE weights)
      latent_markov.*         — latent_dim, hidden_dim, n_chunks, chunk_tokens
      training.*              — seed, learning_rate, num_generations, batch_size,
                                max_steps, temperature, top_p, gradient_checkpointing,
                                logging_steps, save_steps
      phase1_loss.*           — lambda_trans (starting weight), lambda_vae
      evaluation.path         — MATH-B-I JSONL pool for RL training
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import random as _random

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    primary      = config["primary"]
    training_cfg = config["training"]
    latent_cfg   = config["latent_markov"]
    phase0_cfg   = config["phase0"]
    phase1_cfg   = config.get("phase1_loss", {})

    model_id     = primary["huggingface_repo_id"]
    revision     = primary.get("revision", "main")
    dtype        = getattr(torch, primary.get("dtype", "bfloat16"))
    is_smoke     = (config.get("experiment") or {}).get("profile") == "smoke"

    seed         = training_cfg.get("seed", 42)
    lr_backbone  = float(training_cfg.get("learning_rate", 1e-6))
    lr_vae       = 3e-4             # VAE/ZInjector always trained at Phase 0 rate
    G            = int(training_cfg.get("num_generations", 8))
    batch_size   = int(training_cfg.get("batch_size", 1))   # problems per step
    max_steps    = int(training_cfg.get("max_steps", 200))
    temperature  = float(training_cfg.get("temperature", 1.0))
    top_p        = float(training_cfg.get("top_p", 1.0))
    log_steps    = int(training_cfg.get("logging_steps", 10))
    save_steps   = int(training_cfg.get("save_steps", 50))
    grad_clip    = 1.0

    chunk_tokens = int(latent_cfg.get("chunk_tokens", 341))
    latent_dim   = int(latent_cfg.get("latent_dim", LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim", HIDDEN_DIM))

    vae0_path    = Path(phase0_cfg.get("checkpoint_path", run_dir / "phase0_vae.pt"))
    pool_path    = Path(config["evaluation"]["path"])
    ckpt_path    = run_dir / "phase1"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 1 — device: %s", device)

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
        model_id, revision=revision,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("  backbone ready")

    # ------------------------------------------------------------------
    # VAE — load Phase 0 weights
    # ------------------------------------------------------------------
    vae = VAEStateEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    ckpt = torch.load(vae0_path, weights_only=False, map_location=device)
    vae.load_state_dict(ckpt["vae"])
    vae.train()
    logger.info("VAE loaded from %s (step %d)", vae0_path, ckpt.get("step", 0))

    # ------------------------------------------------------------------
    # ZInjector — fresh for Phase 1 (not part of Phase 0)
    # ------------------------------------------------------------------
    z_injector = ZInjector(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    z_injector.train()

    if training_cfg.get("gradient_checkpointing", False):
        model.config.use_cache = False   # required before gradient_checkpointing_enable
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        logger.info("  gradient checkpointing enabled (use_cache disabled for training)")

    # ------------------------------------------------------------------
    # Optimizer — two learning rates: backbone low, VAE/ZInjector higher
    # ------------------------------------------------------------------
    vae_params = list(vae.parameters()) + list(z_injector.parameters())
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(),  "lr": lr_backbone},
        {"params": vae_params,           "lr": lr_vae},
    ])

    # ------------------------------------------------------------------
    # Training pool (MATH-B-I)
    # ------------------------------------------------------------------
    logger.info("Loading training pool from %s ...", pool_path)
    with open(pool_path, encoding="utf-8") as f:
        problems = [json.loads(l) for l in f if l.strip()]
    logger.info("  %d problems in training pool", len(problems))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))

    log_history: list[dict] = []
    pending: dict[str, float] = {}
    step_bar = tqdm(total=max_steps, desc="phase1", unit="step", dynamic_ncols=True)

    # Maintain a shuffled cycle over the problem pool.
    pool_order: list[int] = []

    for global_step in range(max_steps):
        lambda_t = lambda_trans_schedule(global_step, max_steps)

        # Sample batch_size problems (reshuffle when the pool is exhausted).
        while len(pool_order) < batch_size:
            epoch_order = list(range(len(problems)))
            _random.shuffle(epoch_order)
            pool_order.extend(epoch_order)
        step_problem_idxs = [pool_order.pop(0) for _ in range(batch_size)]
        step_problems = [problems[i] for i in step_problem_idxs]

        # ── Rollout collection (no gradient) ──────────────────────────
        with torch.no_grad():
            model.eval()
            vae.eval()
            z_injector.eval()
            traces = generate_latent_traces(
                model=model,
                tokenizer=tokenizer,
                vae=vae,
                z_injector=z_injector,
                problems=step_problems,
                n_rollouts=G,
                chunk_tokens=chunk_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

        model.train()
        vae.train()
        z_injector.train()

        rewards    = [float(t["reward"]) for t in traces]
        advantages = compute_grpo_advantages(rewards, group_size=G)

        # ── Training step (gradient) ───────────────────────────────────
        optimizer.zero_grad()

        metrics = latent_training_step(
            model=model,
            vae=vae,
            z_injector=z_injector,
            traces=traces,
            advantages=advantages,
            lambda_t=lambda_t,
            device=device,
        )

        loss: torch.Tensor = metrics["total"]
        loss.backward()
        all_params = list(model.parameters()) + vae_params
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────
        for k in ("total", "l_rl", "l_vae", "l_trans"):
            pending[k] = pending.get(k, 0.0) + metrics[k].item()
        pending["reward_rate"] = pending.get("reward_rate", 0.0) + (
            sum(rewards) / len(rewards)
        )

        if (global_step + 1) % log_steps == 0:
            n = log_steps
            entry = {
                "step":        global_step + 1,
                "lambda_t":    round(lambda_t, 4),
                **{k: pending.get(k, 0.0) / n
                   for k in ("total", "l_rl", "l_vae", "l_trans", "reward_rate")},
            }
            log_history.append(entry)
            pending = {}
            logger.info(
                "step %d | λ_t=%.2f | total=%.4f | rl=%.4f | vae=%.4f | "
                "trans=%.4f | reward=%.2f%%",
                entry["step"], entry["lambda_t"], entry["total"],
                entry["l_rl"], entry["l_vae"], entry["l_trans"],
                entry["reward_rate"] * 100,
            )

        if (global_step + 1) % save_steps == 0:
            _save_phase1_checkpoint(
                ckpt_path / f"checkpoint-{global_step + 1}",
                model, vae, z_injector, optimizer, global_step + 1, log_history,
            )

        step_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            rl=f"{metrics['l_rl'].item():.4f}",
            λ_t=f"{lambda_t:.2f}",
        )
        step_bar.update(1)

    step_bar.close()

    # Final save.
    _save_phase1_checkpoint(
        ckpt_path / "final", model, vae, z_injector, optimizer, max_steps, log_history
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
) -> None:
    """Save backbone, VAE, ZInjector, and optimizer state to `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vae":         vae.state_dict(),
            "z_injector":  z_injector.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "step":        step,
        },
        directory / "phase1_latent.pt",
    )
    # Backbone saved separately via HF API (handles QLoRA adapter merging).
    model.save_pretrained(str(directory / "backbone"))
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}, indent=2)
    )
