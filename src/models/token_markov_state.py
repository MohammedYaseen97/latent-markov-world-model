"""Token-Markov state representation — Delethink implementation.

The original design envisioned a separate Yuan-style state predictor module here.
That design was superseded: MATH-Beyond competition problems have no ground-truth
symbolic state (no board configuration, no step-by-step verifier), so a symbolic
state predictor cannot run on this dataset.

The token-Markov arm is implemented instead using Delethink-style RL-learned textual
carryover (Aghajohari et al., ICLR 2026). The state representation is structural:
the last m tokens of each chunk are passed as context to the next chunk, and old
tokens are deleted. There is no separate predictor module — the Markov property is
enforced by construction in the generation loop.

All implementation lives in:
    src/training/grpo_token_markov.py  — Chunk, Trace dataclasses + generation loop
    configs/train_token_markov_grpo.yaml  — hyperparameters (C, m, I, planning_prefix)
    reports/token_markov_design.md  — full design rationale and ASCII diagrams

This file is retained to satisfy the repository layout contract
(src/models/token_markov_state.py) and to document the design decision.
"""

from src.training.grpo_token_markov import Chunk, Trace, generate_delethink_trace

__all__ = ["Chunk", "Trace", "generate_delethink_trace"]
