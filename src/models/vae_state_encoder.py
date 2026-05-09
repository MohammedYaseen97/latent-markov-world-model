"""VAE state encoder, decoder, transition model, and policy-conditioning injector.

Architecture (all dimensions from reports/latent_markov_design.md):

  Encoder   1536 → 512 → 128 → split → μ (64)  log_σ² (64)
  Decoder    64  → 512 → 1536
  Transition  64 → 512 → 64
  OutcomeHead  64 → 64 → 1 → sigmoid         (Phase 0 only, discarded before Phase 1)
  ZInjector    64 → 1536                      (Phase 1: prepend z as soft prefix token)

The encoder is weight-shared across all three chunks of a rollout.
repr_h  — mean-pooled last-layer hidden states of the backbone over chunk h tokens.
          Shape: (batch, HIDDEN_DIM). Static (pre-saved) tensors during Phase 0;
          re-computed live during Phase 1.
z_h     — latent state after chunk h. Shape: (batch, LATENT_DIM).

See reports/latent_markov_design.md §Architecture for full design rationale.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 1536   # Qwen2.5-1.5B last-layer hidden size
LATENT_DIM = 64     # z dimension (design decision Q1)
N_CHUNKS   = 3      # fixed: chunks per rollout


# ---------------------------------------------------------------------------
# VAEStateEncoder
# ---------------------------------------------------------------------------

class VAEStateEncoder(nn.Module):
    """Variational autoencoder over reasoning-trajectory representations.

    Encodes the backbone's hidden-state summary (repr_h) of one chunk into a
    diagonal-Gaussian latent z_h, then decodes z_h back to reconstruct repr_h.
    A transition model predicts z_{h+1} from z_h alone to enforce the Markov
    property across chunks.

    All components are small MLPs with < 10M parameters total (satisfies NFR4).

    Usage (Phase 0 — static repr_h tensors, backbone NOT in graph):
        vae = VAEStateEncoder()
        mu1, logvar1 = vae.encode(repr_1)
        z1 = vae.reparameterize(mu1, logvar1)
        repr_1_hat = vae.decode(z1)
        z2_pred = vae.transition(z1)           # predicts z2 from z1 alone

    Usage (forward over a full 3-chunk rollout):
        results = vae.forward([repr_1, repr_2, repr_3])
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
        self.enc_fc1   = nn.Linear(hidden_dim, 512)
        self.enc_fc2   = nn.Linear(512, 128)
        self.mu_head   = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

        # Decoder: z (latent_dim) → reconstructed repr_h
        self.dec_fc1 = nn.Linear(latent_dim, 512)
        self.dec_fc2 = nn.Linear(512, hidden_dim)

        # Transition: z_h (latent_dim) → z_{h+1}_predicted
        # Pure Markov: z_h alone must predict next state. repr_h is NOT included —
        # the ELBO already handles reconstruction (repr_h → z_h). Adding repr_h here
        # would let the transition bypass the bottleneck and weaken gradient on z_h.
        self.trans_fc1 = nn.Linear(latent_dim, 512)
        self.trans_fc2 = nn.Linear(512, latent_dim)

    # ------------------------------------------------------------------
    # Per-chunk forward methods
    # ------------------------------------------------------------------

    def encode(self, repr_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a chunk representation into the parameters of q(z_h | repr_h).

        Args:
            repr_h: mean-pooled final-layer hidden states for chunk h.
                    Shape: (batch, hidden_dim) or (hidden_dim,) for single sample.

        Returns:
            mu:     mean of the approximate posterior. Shape: (batch, latent_dim).
            logvar: log variance (log σ²) of the posterior. Shape: (batch, latent_dim).
                    Both are unconstrained — no activation applied.
        """
        x = F.relu(self.enc_fc1(repr_h))
        x = F.relu(self.enc_fc2(x))
        mu = self.mu_head(x)
        logvar = self.logvar_head(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ q(z | repr_h) using the reparameterization trick.

        During training: z = μ + ε·exp(0.5·log σ²),  ε ~ N(0, I).
        During eval (inference): z = μ (deterministic — no noise).

        Args:
            mu:     posterior mean. Shape: (batch, latent_dim).
            logvar: log variance (log σ²). Shape: (batch, latent_dim).

        Returns:
            z: sampled latent vector. Shape: (batch, latent_dim).
        """
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct repr_h from latent z.

        Args:
            z: sampled or deterministic latent. Shape: (batch, latent_dim).

        Returns:
            repr_hat: reconstruction of repr_h. Shape: (batch, hidden_dim).
                      Unconstrained — no output activation.
        """
        x = F.relu(self.dec_fc1(z))
        repr_hat = self.dec_fc2(x)
        return repr_hat

    def transition(self, z_h: torch.Tensor) -> torch.Tensor:
        """Predict z_{h+1} from the current latent state alone.

        Enforces the Markov property directly: z_h must be a sufficient summary
        of the reasoning trajectory up to chunk h to predict where chunk h+1 will
        land in latent space. repr_h is deliberately excluded — the ELBO already
        handles repr_h → z_h compression; including repr_h would let the transition
        bypass the bottleneck and reduce gradient pressure on z_h.

        Args:
            z_h: current latent state. Shape: (batch, latent_dim).

        Returns:
            z_next_pred: predicted next latent. Shape: (batch, latent_dim).
        """
        x = F.relu(self.trans_fc1(z_h))
        z_next_pred = self.trans_fc2(x)
        return z_next_pred

    # ------------------------------------------------------------------
    # Full-rollout forward
    # ------------------------------------------------------------------

    def forward(
        self,
        repr_list: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode all N_CHUNKS chunk representations in a single rollout.

        Args:
            repr_list: list of N_CHUNKS tensors, each (batch, hidden_dim),
                       one per chunk in chronological order.

        Returns:
            List of N_CHUNKS tuples (z_h, mu_h, logvar_h), one per chunk.
            z_h:      sampled latent.   Shape: (batch, latent_dim).
            mu_h:     posterior mean.   Shape: (batch, latent_dim).
            logvar_h: posterior log σ². Shape: (batch, latent_dim).
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

    def compute_elbo(
        self,
        repr_list:   list[torch.Tensor],
        z_list:      list[torch.Tensor],
        mu_list:     list[torch.Tensor],
        logvar_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the ELBO loss summed over all N_CHUNKS chunks.

        L_ELBO = Σ_h [ MSE(decode(z_h), repr_h) + KL(N(μ_h, σ_h²) ∥ N(0, I)) ]

        KL for a diagonal Gaussian (per sample, summed over latent dims):
            KL = -0.5 * Σ_j (1 + log σ²_j - μ_j² - σ_j²)

        Args:
            repr_list:   original repr_h tensors. List of N_CHUNKS (batch, hidden_dim).
            z_list:      sampled latents.    List of N_CHUNKS (batch, latent_dim).
            mu_list:     posterior means.    List of N_CHUNKS (batch, latent_dim).
            logvar_list: posterior log σ².   List of N_CHUNKS (batch, latent_dim).

        Returns:
            Scalar tensor — reconstruction + KL, averaged over batch, summed over chunks.
        """
        assert len(repr_list) == len(z_list) == len(mu_list) == len(logvar_list), \
            "All lists must have the same length"
        loss = 0
        for i in range(len(repr_list)):
            rec_loss = F.mse_loss(self.decode(z_list[i]), repr_list[i])
            kl_loss  = -0.5 * (1 + logvar_list[i] - mu_list[i] ** 2
                                - logvar_list[i].exp()).sum(dim=-1).mean()
            loss += rec_loss + kl_loss
        return loss

    def compute_transition_loss(
        self,
        z_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Markov transition consistency loss.

        L_transition = Σ_{h=1}^{N-1} ‖ transition(z_h) − z_{h+1} ‖²

        Sums over h = 1 → 2 and h = 2 → 3 (two transitions for N_CHUNKS=3).
        z_{h+1} is detached — gradient flows through the transition model and
        encoder (via z_h), creating Markov pressure: z_h must be information-dense
        enough to predict z_{h+1} on its own.

        Args:
            z_list: sampled latents. List of N_CHUNKS tensors (batch, latent_dim).

        Returns:
            Scalar tensor — MSE mean over batch, summed over h.
        """
        loss = 0
        for h in range(N_CHUNKS - 1):
            z_next_pred = self.transition(z_list[h])
            target = z_list[h + 1].detach()
            loss += F.mse_loss(z_next_pred, target)
        return loss


# ---------------------------------------------------------------------------
# OutcomeHead
# ---------------------------------------------------------------------------

class OutcomeHead(nn.Module):
    """Binary classification head over z_final — used in Phase 0 ONLY.

    Predicts P(trajectory is correct) from the last chunk's latent z_3.
    Provides dense quality-oriented gradient signal to the encoder during
    pretraining, where L_RL is unavailable (sparse reward on hard problems).

    DISCARDED before Phase 1 begins. It is scaffolding for VAE initialisation,
    not a component of the final latent policy.

    See reports/latent_markov_design.md §Architecture — "Outcome head".
    """

    def __init__(self, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, z_final: torch.Tensor) -> torch.Tensor:
        """Predict P(correct) from the final chunk's latent vector.

        Args:
            z_final: latent of the last chunk (z_3). Shape: (batch, latent_dim).

        Returns:
            prob: predicted probability of correctness. Shape: (batch, 1).
                  Values in (0, 1) — output of sigmoid.
        """
        x = F.relu(self.fc1(z_final))
        return torch.sigmoid(self.fc2(x))


# ---------------------------------------------------------------------------
# ZInjector (Phase 1 policy conditioning)
# ---------------------------------------------------------------------------

class ZInjector(nn.Module):
    """Projects z_h into embedding space for use as a soft prefix token.

    During Phase 1 (joint RL training), z_h is prepended to chunk h+1's input
    via inputs_embeds — not as a real vocabulary token but as a learned linear
    projection of the 64-dim latent into the 1536-dim embedding space.

    This does NOT consume the 1024-token generation budget (satisfies R5.2).
    The prefix is a single embedding vector, not a token ID.
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, hidden_dim, bias=False)

    def get_prefix_embedding(self, z: torch.Tensor) -> torch.Tensor:
        """Project z into a prefix embedding vector.

        Args:
            z: latent state. Shape: (batch, latent_dim) or (latent_dim,).

        Returns:
            prefix: embedding to prepend. Shape: (batch, 1, hidden_dim).
                    Ready to be concatenated with inputs_embeds along dim=1.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return self.proj(z).unsqueeze(1)
