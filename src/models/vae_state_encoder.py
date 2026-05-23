"""Deterministic state encoder, transition model, and policy-conditioning injector.

Architecture (v2 — rung 2 of the research ladder):

  Encoder    1536 → 512 → 128 → split → μ (64)  log_σ² (64)
  Transition  64  → 512 → 64
  OutcomeHead  64 → 64  → 1                    (raw logit; Phase 0 only, discarded before Phase 1)
  ZInjector    64 → 1536                       (Phase 1: prepend z as soft prefix token)

z_h = μ_h  — used deterministically during training for Markov tracking.
σ²_h        — trained explicitly via L_calib to be high on incorrect trajectories
               and low on correct ones. Not a KL side-effect.

reparameterize() is retained for optional noise injection at inference time only.
The decoder and ELBO/KL machinery are removed (rung 2 is a tracker, not a generator).

See reports/latent_markov_design.md §Architecture for full design rationale.
See reports/NEXT_STEPS_V2.md for the rung 2 → rung 3 (diffusion) upgrade path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 1536   # Qwen2.5-1.5B last-layer hidden size
LATENT_DIM = 64     # z dimension
N_CHUNKS   = 3      # fixed: chunks per rollout


# ---------------------------------------------------------------------------
# StateEncoder
# ---------------------------------------------------------------------------

class VAEStateEncoder(nn.Module):
    """Deterministic state encoder with calibrated uncertainty.

    Encodes the backbone's hidden-state summary (repr_h) of one chunk into a
    (μ_h, log_σ²_h) pair.  z_h = μ_h is used throughout the pipeline for
    Markov tracking; log_σ²_h is trained via an explicit calibration loss so
    that σ² is high on incorrect trajectories and low on correct ones.

    A transition model predicts z_{h+1} from z_h alone to enforce the Markov
    property across chunks.

    Usage (single chunk):
        enc = VAEStateEncoder()
        mu, logvar = enc.encode(repr_h)
        z = mu                                  # deterministic during training
        z_next_pred = enc.transition(z)         # predicts z_{h+1}

    Usage (full 3-chunk rollout):
        results = enc.forward([repr_1, repr_2, repr_3])
        # returns list of (z_h, mu_h, logvar_h) per chunk
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: repr_h (hidden_dim) → intermediate → (μ, log_σ²)
        self.enc_fc1     = nn.Linear(hidden_dim, 512)
        self.enc_fc2     = nn.Linear(512, 128)
        self.mu_head     = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

        # Transition: z_h (latent_dim) → z_{h+1}_predicted
        # Pure Markov: z_h alone must predict the next state. repr_h is excluded —
        # including it would let the transition bypass the bottleneck.
        self.trans_fc1 = nn.Linear(latent_dim, 512)
        self.trans_fc2 = nn.Linear(512, latent_dim)

    # ------------------------------------------------------------------
    # Per-chunk forward methods
    # ------------------------------------------------------------------

    def encode(self, repr_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a chunk representation into (μ_h, log_σ²_h).

        Args:
            repr_h: mean-pooled final-layer hidden states for chunk h.
                    Shape: (batch, hidden_dim) or (hidden_dim,) for single sample.

        Returns:
            mu:     state estimate. Shape: (batch, latent_dim).
            logvar: log σ² (uncertainty). Shape: (batch, latent_dim).
                    Both are unconstrained — no output activation.
        """
        x = F.relu(self.enc_fc1(repr_h))
        x = F.relu(self.enc_fc2(x))
        mu     = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Return z = μ during training (deterministic) or μ + ε·σ at eval.

        Training: z = μ  (no sampling noise — z is the Markov state).
        Eval / inference: z = μ + ε·σ for optional diversity.

        Args:
            mu:     posterior mean. Shape: (batch, latent_dim).
            logvar: log σ². Shape: (batch, latent_dim).

        Returns:
            z: latent vector. Shape: (batch, latent_dim).
        """
        if self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def transition(self, z_h: torch.Tensor) -> torch.Tensor:
        """Predict z_{h+1} from the current latent state alone.

        Enforces the Markov property: z_h must be a sufficient summary of the
        reasoning trajectory up to chunk h to predict where chunk h+1 will land.

        Args:
            z_h: current latent state. Shape: (batch, latent_dim).

        Returns:
            z_next_pred: predicted next latent. Shape: (batch, latent_dim).
        """
        x = F.relu(self.trans_fc1(z_h))
        return self.trans_fc2(x)

    # ------------------------------------------------------------------
    # Full-rollout forward
    # ------------------------------------------------------------------

    def forward(
        self,
        repr_list: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode all N_CHUNKS chunk representations in a single rollout.

        Args:
            repr_list: list of N_CHUNKS tensors, each (batch, hidden_dim).

        Returns:
            List of N_CHUNKS tuples (z_h, mu_h, logvar_h).
            z_h = mu_h during training (deterministic).
        """
        assert len(repr_list) == N_CHUNKS, (
            f"Expected {N_CHUNKS} chunk representations, got {len(repr_list)}"
        )
        results = []
        for repr_h in repr_list:
            mu_h, logvar_h = self.encode(repr_h)
            z_h = self.reparameterize(mu_h, logvar_h)
            results.append((z_h, mu_h, logvar_h))
        return results

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_transition_loss(
        self,
        z_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Markov transition consistency loss.

        L_transition = Σ_{h=1}^{N-1} ‖ transition(z_h) − z_{h+1} ‖²

        z_{h+1} is detached — gradient flows through the transition model and
        encoder (via z_h), creating Markov pressure.

        Args:
            z_list: latent states. List of N_CHUNKS tensors (batch, latent_dim).

        Returns:
            Scalar tensor — MSE mean over batch, summed over h.
        """
        loss = torch.zeros(1, device=z_list[0].device)
        for h in range(N_CHUNKS - 1):
            z_next_pred = self.transition(z_list[h])
            target = z_list[h + 1].detach()
            loss = loss + F.mse_loss(z_next_pred, target)
        return loss

    def compute_calibration_loss(
        self,
        logvar_list: list[torch.Tensor],
        rewards: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Calibrate σ² to be high on incorrect trajectories and low on correct ones.

        Framing: treat −mean_logvar as a "correctness logit".
          High σ² (high logvar) → low correctness logit → predicts incorrect.
          Low σ² (low logvar)   → high correctness logit → predicts correct.
        Target = reward (1 = correct minority, 0 = incorrect majority).

        Using BCE_with_logits(−mean_logvar, reward) is mathematically identical to
        the original BCE(sigmoid(mean_logvar), 1−reward); the benefit is that
        pos_weight can upweight the minority correct class (≈18%) to counteract
        the 82% incorrect dominance that caused the calibration loss to trivially
        converge to "everything is uncertain".

        Args:
            logvar_list: log σ² per chunk. List of N_CHUNKS tensors (batch, latent_dim).
            rewards:     binary rewards. Shape: (batch, 1) or (batch,). Float.
            pos_weight:  scalar tensor — weight for the positive (correct) class.
                         Pass (1−pos_rate)/pos_rate clamped to [1, 20] to balance
                         the 82/18 split.  None = no reweighting (unbalanced BCE).

        Returns:
            Scalar tensor — BCE loss.
        """
        stacked = torch.stack(logvar_list, dim=0)   # (N_CHUNKS, batch, latent_dim)
        mean_logvar = stacked.mean(dim=(0, 2))       # (batch,)

        # −mean_logvar: high uncertainty → low logit → predicts incorrect (target=0)
        target = rewards.float().view(-1)            # 1=correct (minority class)

        return F.binary_cross_entropy_with_logits(
            -mean_logvar, target, pos_weight=pos_weight
        )


# ---------------------------------------------------------------------------
# OutcomeHead
# ---------------------------------------------------------------------------

class OutcomeHead(nn.Module):
    """Binary classification head over z_final — used in Phase 0 ONLY.

    Predicts P(trajectory is correct) from the last chunk's latent z_3.
    Provides dense quality-oriented gradient signal to the encoder during
    pretraining, where L_RL is unavailable (sparse reward on hard problems).

    DISCARDED before Phase 1 begins.
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z_final: torch.Tensor) -> torch.Tensor:
        """Return raw logit for P(correct) from the final chunk's latent vector.

        Returns a raw (pre-sigmoid) logit so the caller can use
        F.binary_cross_entropy_with_logits with dynamic pos_weight.

        Args:
            z_final: latent of the last chunk (z_3). Shape: (batch, latent_dim).

        Returns:
            logit: raw correctness logit. Shape: (batch, 1).
        """
        x = F.relu(self.fc1(z_final))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# ZInjector (Phase 1 policy conditioning)
# ---------------------------------------------------------------------------

class ZInjector(nn.Module):
    """Projects z_h into embedding space for use as a soft prefix token.

    During Phase 1 (joint RL training), z_h = μ_h is prepended to chunk h+1's
    input via inputs_embeds — not as a real vocabulary token but as a learned
    linear projection of the 64-dim latent into the 1536-dim embedding space.

    This does NOT consume the 1024-token generation budget (R5.2).
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_dim, bias=False)
        # Near-zero init: default Kaiming-uniform gives std ≈ 0.125, making the
        # prefix embedding O(1) magnitude — same as a real token embedding —
        # which injects random noise and degrades the pretrained model immediately.
        # Starting at std=0.01 keeps the prefix ~12× smaller than a token embedding,
        # making it effectively neutral at init (same principle as LoRA zero-init).
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)

    def get_prefix_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Project z into a prefix embedding vector.

        Args:
            z: latent state (= μ_h). Shape: (batch, latent_dim) or (latent_dim,).

        Returns:
            prefix: embedding to prepend. Shape: (batch, 1, hidden_dim).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.proj(z).unsqueeze(1)
