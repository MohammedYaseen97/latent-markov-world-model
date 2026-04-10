"""Latent-state GRPO training scaffold."""


def train_latent(*args, **kwargs):
    """Train latent arm without uncertainty bonus."""
    raise NotImplementedError("TODO: implement latent GRPO loop.")


def train_latent_with_uncertainty(*args, **kwargs):
    """Train latent arm with uncertainty bonus."""
    raise NotImplementedError("TODO: implement latent+uncertainty GRPO loop.")
