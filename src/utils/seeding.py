"""Deterministic seeding for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[misc, assignment]


def set_seed(seed: int) -> None:
    """Set RNG seeds for Python, NumPy, and PyTorch.

    CUDA determinism flags are set when available so that two runs from the
    same seed produce identical outputs.  This incurs a small throughput cost
    but is required for the fair cross-arm comparison in PROJECT_CONTRACT.md.
    """
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
