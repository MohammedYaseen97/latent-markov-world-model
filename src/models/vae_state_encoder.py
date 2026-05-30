"""Latent state encoder and z-injector for the latent Markov GRPO arm.

Architecture
────────────
  Encoder    1536 → 512 → 128 → 64   (repr_h → z_h; fp32)
  Injector     64 → 1536              (z_h → soft prefix embedding; bf16-compatible)

repr_h is the last-token hidden state of the backbone's final layer after
processing chunk h under strict Markov context [z_{h-1}_prefix | chunk_h_tokens].
The last token of a causal LM has attended over every preceding token — it is
the model's own attention-weighted summary of the chunk.

z_h = encoder(repr_h) always — deterministic, no sampling.

Design reference: reports/latent_markov_design.md §Architecture
"""
from __future__ import annotations

import torch
import torch.nn as nn

HIDDEN_DIM = 1536   # Qwen2.5-1.5B last-layer hidden size
LATENT_DIM = 64     # z_h dimension
N_CHUNKS   = 3      # chunks per rollout (fixed)


class LatentStateEncoder(nn.Module):
    """Encodes backbone chunk representations into a compact latent state
    and projects that state back into the backbone's embedding space as a
    soft prefix token.

    Both encode() and inject() are shared across all chunks (same weights
    applied at each chunk boundary).

    Usage::

        enc = LatentStateEncoder()
        z   = enc.encode(repr_h)   # [B, HIDDEN_DIM]  → [B, LATENT_DIM]
        pfx = enc.inject(z)        # [B, LATENT_DIM]  → [B, 1, HIDDEN_DIM]
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        z_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

        self.injector = nn.Linear(z_dim, hidden_dim, bias=False)
        # Near-zero init keeps the prefix ~12× smaller than a real token
        # embedding at step 0, making it effectively neutral for the pretrained
        # backbone. Weights grow as Phase 0 distillation proceeds.
        nn.init.normal_(self.injector.weight, std=0.01)

    def encode(self, repr_h: torch.Tensor) -> torch.Tensor:
        """Map a chunk's last-token hidden state to a latent state vector.

        Args:
            repr_h: last-token hidden state of chunk h.
                    Shape: [B, hidden_dim] or [hidden_dim,] for a single sample.
                    Expected dtype: fp32 (backbone hidden states cast before call).

        Returns:
            z_h: latent state. Shape: [B, z_dim]. fp32.
        """
        return self.encoder(repr_h)

    def inject(self, z_h: torch.Tensor) -> torch.Tensor:
        """Project z_h into the backbone's embedding space as a soft prefix token.

        The returned tensor is prepended to chunk h+1's inputs_embeds. It does
        not consume a token-budget slot; it is a virtual prefix that conditions
        the backbone on the current latent state.

        Args:
            z_h: latent state. Shape: [B, z_dim]. fp32 or bf16.

        Returns:
            prefix_embed: Shape: [B, 1, hidden_dim]. Same dtype as z_h input.
        """
        return self.injector(z_h).unsqueeze(1)
