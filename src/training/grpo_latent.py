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
    N_CHUNKS,
    OutcomeHead,
    VAEStateEncoder,
)

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
# train_latent — Phase 1 joint RL (stub — future session)
# ---------------------------------------------------------------------------

def train_latent(config: dict[str, Any], run_dir: Path) -> None:
    """Phase 1: joint GRPO training with live backbone and z injection.

    Loads the Phase 0 VAE checkpoint, unfreezes the backbone, and runs the
    full latent Markov GRPO loop:

      L_total = L_RL + λ_trans × L_transition + L_VAE

    where L_RL is the GRPO policy gradient (sparse, fires only when ≥1 of G=8
    rollouts is correct), L_transition is the Markov consistency loss (always
    non-zero), and L_VAE is the ELBO (prevents encoder/decoder drift).

    z_h is injected via ZInjector as a soft prefix token at the start of each
    chunk (see src/models/vae_state_encoder.py — ZInjector).

    This function is a stub. Implementation is planned for a future session
    after Phase 0 is validated via the NFR6 gate (check_latent_structure.py).

    See reports/latent_markov_design.md §Phase 1 for full design.
    """
    # ------------------------------------------------------------------ #
    # FUTURE WORK — Phase 1 implementation                                #
    #                                                                     #
    # High-level structure (implement in a future session):               #
    #   1. Load Phase 0 VAE checkpoint (phase0_vae.pt).                  #
    #   2. Load Qwen backbone (unfrozen, QLoRA for smoke / bf16 for A100) #
    #   3. Load ZInjector (64 → 1536 linear).                            #
    #   4. Load hard 40-problem MATH-B-I pool.                           #
    #   5. For each step:                                                 #
    #      a. For each problem, run generate_latent_traces() (G traces):  #
    #         - Chunk 1: no z prefix (no prior state).                    #
    #         - Extract repr_1, compute z_1 via VAE.                     #
    #         - Chunk 2: prepend z_1 via ZInjector.inputs_embeds.        #
    #         - Extract repr_2, compute z_2.                             #
    #         - Chunk 3: prepend z_2.                                     #
    #      b. Grade traces → rewards.                                     #
    #      c. Compute GRPO advantages.                                    #
    #      d. Compute L_RL (policy gradient over all chunks).             #
    #      e. Compute L_transition = ‖f(z_h) − z_{h+1}‖² (Markov).       #
    #      f. Compute L_VAE (ELBO on live repr_h tensors).               #
    #      g. L_total = L_RL + λ_trans * L_transition + L_VAE.           #
    #      h. Backward + optimizer step.                                  #
    # ------------------------------------------------------------------ #
    raise NotImplementedError("train_latent (Phase 1) is not yet implemented.")
