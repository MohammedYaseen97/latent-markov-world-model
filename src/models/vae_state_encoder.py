"""Deterministic state encoder, transition model, and policy-conditioning injector.

Architecture:

  Encoder     1536 → 512 → 128 → μ (64)
  Transition   64  → 512 → 64
  OutcomeHead  64  → 64  → 1     (raw logit; Phase 0 only, discarded before Phase 1)
  ZInjector    64  → 1536        (Phase 1: prepend z as soft prefix token)

ARM 3 — latent_grpo:
  z_h = μ_h always. Deterministic encoder only — no sampling, no decoder.

ARM 4 — latent_grpo_uncertainty (stub, not yet implemented):
  Will extend this with a logvar head and uncertainty-driven exploration.
  See PROJECT_CONTRACT.md §Phase 3b.

See reports/latent_markov_design.md §Architecture for full design rationale.
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
    """Deterministic state encoder with Markov transition model.

    Encodes the backbone's hidden-state summary (repr_h) into μ_h. z_h = μ_h.
    A transition model predicts z_{h+1} from z_h alone (Markov property).

    Usage (single chunk):
        enc = VAEStateEncoder()
        z = enc.encode(repr_h)
        z_next_pred = enc.transition(z)

    Usage (full 3-chunk rollout):
        z_list = enc.forward([repr_1, repr_2, repr_3])
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder: repr_h (hidden_dim) → μ_h (latent_dim)
        self.enc_fc1 = nn.Linear(hidden_dim, 512)
        self.enc_fc2 = nn.Linear(512, 128)
        self.mu_head = nn.Linear(128, latent_dim)

        # Transition: z_h → z_{h+1}_predicted.
        # Pure Markov: z_h alone must predict the next state. repr_h is excluded —
        # including it would let the transition bypass the bottleneck.
        self.trans_fc1 = nn.Linear(latent_dim, 512)
        self.trans_fc2 = nn.Linear(512, latent_dim)

    # ------------------------------------------------------------------
    # Per-chunk forward methods
    # ------------------------------------------------------------------

    def encode(self, repr_h: torch.Tensor) -> torch.Tensor:
        """Encode a chunk representation into μ_h (= z_h).

        Args:
            repr_h: mean-pooled final-layer hidden states for chunk h.
                    Shape: (batch, hidden_dim) or (hidden_dim,) for single sample.

        Returns:
            mu: latent state. Shape: (batch, latent_dim). Unconstrained.
        """
        x = F.relu(self.enc_fc1(repr_h))
        x = F.relu(self.enc_fc2(x))
        return self.mu_head(x)

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
    ) -> list[torch.Tensor]:
        """Encode all N_CHUNKS chunk representations in a single rollout.

        Args:
            repr_list: list of N_CHUNKS tensors, each (batch, hidden_dim).

        Returns:
            List of N_CHUNKS latent tensors z_h = μ_h. Each (batch, latent_dim).
        """
        assert len(repr_list) == N_CHUNKS, (
            f"Expected {N_CHUNKS} chunk representations, got {len(repr_list)}"
        )
        return [self.encode(repr_h) for repr_h in repr_list]

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
