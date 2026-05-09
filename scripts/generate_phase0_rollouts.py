#!/usr/bin/env python3
"""Generate Phase 0 VAE pretraining rollouts from the easy pretraining pool.

What this script does
─────────────────────
1. Loads the pre-built easy pool (data/math_easy_pool.jsonl by default).
2. Loads the frozen backbone (Qwen2.5-1.5B-Instruct) in inference mode.
3. For each problem, generates G independent 1024-token completions.
4. For each completion, extracts mean-pooled last-layer hidden states for
   three equal-size chunks: repr_1, repr_2, repr_3.
5. Grades the completion (binary: 1 if correct, 0 otherwise).
6. Saves all trajectories to data/phase0_rollouts.pt.

Output format (data/phase0_rollouts.pt)
────────────────────────────────────────
A list of dicts, one per trajectory (n_problems × G entries):
  {
    "problem_id":  str,              # e.g. "math_easy_0000"
    "rollout_idx": int,              # 0 … G-1
    "repr_1":      Tensor([1536]),   # float32, mean-pooled hidden states chunk 1
    "repr_2":      Tensor([1536]),   # float32, mean-pooled hidden states chunk 2
    "repr_3":      Tensor([1536]),   # float32, mean-pooled hidden states chunk 3
    "reward":      int,              # 0 or 1
    "ground_truth": str,             # answer string (for debugging)
  }

Memory estimate: 2974 problems × 8 rollouts × 3 × 1536 × 4 bytes ≈ 440 MB.
Fits comfortably on any machine that can also run the 1.5B backbone.

Usage
─────
  # Default: full easy pool, G=8, output to data/
  python scripts/generate_phase0_rollouts.py

  # Quick smoke test: 4 problems, G=2
  python scripts/generate_phase0_rollouts.py --limit 4 --n-rollouts 2

  # Specify pool and output explicitly
  python scripts/generate_phase0_rollouts.py \\
      --pool-path data/math_easy_pool.jsonl \\
      --output-path data/phase0_rollouts.pt \\
      --n-rollouts 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.grpo_baseline import (
    SYSTEM_PROMPT,
    answers_equivalent,
    extract_answer,
)
from src.models.vae_state_encoder import HIDDEN_DIM, N_CHUNKS

# Primary model (Qwen2.5-1.5B-Instruct) — same as all other arms.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_REVISION  = "main"

TOTAL_TOKENS = 1024   # full generation budget — split evenly into N_CHUNKS chunks


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pool-path",
        type=Path,
        default=REPO_ROOT / "data" / "math_easy_pool.jsonl",
        help="Path to the Phase 0 pool JSONL (default: data/math_easy_pool.jsonl).",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "data" / "phase0_rollouts.pt",
        help="Where to save the rollout .pt file (default: data/phase0_rollouts.pt).",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=8,
        help="Number of independent rollouts per problem (G). Default: 8.",
    )
    p.add_argument(
        "--total-tokens",
        type=int,
        default=TOTAL_TOKENS,
        help=f"Total generation budget per rollout, split into {N_CHUNKS} chunks. Default: {TOTAL_TOKENS}.",
    )
    p.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HuggingFace model ID for the backbone (default: Qwen2.5-1.5B-Instruct).",
    )
    p.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help="HuggingFace model revision (default: main).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of problems to process (default: all). Useful for smoke tests.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for Python RNG (default: 42). Generation uses temperature > 0.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p (nucleus) sampling parameter (default: 1.0).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Problem loading
# ---------------------------------------------------------------------------

def load_pool(path: Path) -> list[dict]:
    """Load problems from a JSONL file. Returns a list of problem dicts."""
    problems = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(problem: dict, tokenizer: AutoTokenizer) -> list[int]:
    """Format a problem into a chat-template prompt and tokenize it.

    Uses the same SYSTEM_PROMPT and chat template as grpo_baseline.py so that
    the distribution of hidden states during Phase 0 matches what the backbone
    will see during Phase 1 RL training.

    Args:
        problem:   dict with at least a "prompt" field (problem text).
        tokenizer: Qwen tokenizer with apply_chat_template support.

    Returns:
        List of token IDs (the prompt, not including the generated response).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem["prompt"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


# ---------------------------------------------------------------------------
# Core rollout generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_chunked_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: list[int],
    total_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """Generate a single full-length completion for one problem.

    Generates `total_tokens` new tokens from the prompt in one continuous call
    (no chunked context reset — the latent arm's state compression happens in
    the VAE, not at the token level). The response is one contiguous sequence.

    Args:
        prompt_ids:   tokenized prompt (list of ints).
        total_tokens: total new tokens to generate (e.g. 1024).
        temperature:  sampling temperature.
        top_p:        nucleus sampling cutoff.
        device:       compute device.

    Returns:
        response_ids: 1-D LongTensor of generated token IDs (length ≤ total_tokens).
        completion:   decoded text of the response (for grading and debugging).
    """
    prompt_t = torch.tensor(prompt_ids, device=device, dtype=torch.long)
    attn_mask = torch.ones(1, len(prompt_ids), dtype=torch.long, device=device)
    out = model.generate(
        prompt_t.unsqueeze(0),
        attention_mask=attn_mask,
        max_new_tokens=total_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = out[0, len(prompt_ids):]
    completion   = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_ids, completion


@torch.no_grad()
def extract_chunk_representations(
    model: AutoModelForCausalLM,
    prompt_ids: list[int],
    response_ids: torch.Tensor,
    n_chunks: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Extract mean-pooled last-layer hidden states for each chunk of a rollout.

    After generation, runs a single forward pass over the full sequence
    (prompt + response) with output_hidden_states=True, then splits the
    response hidden states into n_chunks equal segments and mean-pools each.

    A separate forward pass is used rather than extracting during generation
    because model.generate() returns hidden states in a per-decoding-step format
    that does not align cleanly with the final token sequence. A single forward
    pass over the concatenated sequence gives correctly aligned per-token hidden
    states in one shot.

    Args:
        prompt_ids:   tokenized prompt (list of ints).
        response_ids: 1-D LongTensor of generated tokens, shape (resp_len,).
                      Expected to be on the model's device.
        n_chunks:     number of equal-size segments to split the response into.
        device:       compute device (must match response_ids.device).

    Returns:
        repr_list: list of n_chunks tensors, each shape (HIDDEN_DIM,) float32 on CPU.
                   repr_list[0] covers the first ~341 response tokens, etc.
                   Detached from the computation graph — Phase 0 uses static tensors.
    """
    prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=response_ids.device)
    full_ids  = torch.cat([prompt_t, response_ids])
    out = model(
        full_ids.unsqueeze(0),
        attention_mask=torch.ones(1, full_ids.shape[0], dtype=torch.long,
                                  device=response_ids.device),
        output_hidden_states=True,
    )
    hidden = out.hidden_states[-1][0]          # (seq_len, hidden_dim) — last layer
    resp_hidden = hidden[len(prompt_ids):]     # (resp_len, hidden_dim)
    chunks  = torch.chunk(resp_hidden, n_chunks)
    return [c.mean(dim=0).detach().cpu().float() for c in chunks]


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def grade(completion: str, ground_truth: str) -> int:
    """Grade a completion against the ground truth answer.

    Returns 1 if the extracted answer is mathematically equivalent to
    ground_truth, 0 otherwise. Reuses the same extract_answer and
    answers_equivalent functions as grpo_baseline.py (single source of truth).
    """
    pred = extract_answer(completion)
    if pred is None:
        return 0
    return 1 if answers_equivalent(pred, ground_truth) else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ------------------------------------------------------------------
    # Load pool
    # ------------------------------------------------------------------
    print(f"Loading pool from {args.pool_path} ...", flush=True)
    problems = load_pool(args.pool_path)
    if args.limit is not None:
        problems = problems[: args.limit]
    print(f"  {len(problems)} problems loaded", flush=True)

    # ------------------------------------------------------------------
    # Load frozen backbone (inference only — no gradient)
    # ------------------------------------------------------------------
    print(f"Loading model {args.model_id} @ {args.revision} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print("  model frozen and ready", flush=True)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    all_trajectories: list[dict] = []
    n_problems = len(problems)
    chunk_tokens = args.total_tokens // N_CHUNKS   # tokens per chunk (integer division)

    for prob_idx, problem in enumerate(tqdm(problems, desc="problems", unit="prob")):
        problem_id  = problem.get("problem_id", f"problem_{prob_idx:05d}")
        ground_truth = problem["ground_truth"]
        prompt_ids  = format_prompt(problem, tokenizer)

        for rollout_idx in range(args.n_rollouts):
            # --- generate completion ---
            try:
                response_ids, completion = generate_chunked_completion(
                    model,
                    tokenizer,
                    prompt_ids,
                    total_tokens=args.total_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )
            except NotImplementedError:
                raise   # propagate immediately — don't silently swallow
            except Exception as exc:
                print(
                    f"\n  WARN: generation failed for {problem_id} rollout {rollout_idx}: {exc}",
                    file=sys.stderr,
                )
                continue

            # --- extract hidden states per chunk ---
            try:
                repr_list = extract_chunk_representations(
                    model,
                    prompt_ids,
                    response_ids,
                    n_chunks=N_CHUNKS,
                    device=device,
                )
            except NotImplementedError:
                raise
            except Exception as exc:
                print(
                    f"\n  WARN: hidden state extraction failed for {problem_id} rollout {rollout_idx}: {exc}",
                    file=sys.stderr,
                )
                continue

            # --- grade ---
            reward = grade(completion, ground_truth)

            # --- save trajectory ---
            trajectory = {
                "problem_id":   problem_id,
                "rollout_idx":  rollout_idx,
                "repr_1":       repr_list[0],
                "repr_2":       repr_list[1],
                "repr_3":       repr_list[2],
                "reward":       reward,
                "ground_truth": ground_truth,
            }
            all_trajectories.append(trajectory)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_trajectories, args.output_path)

    n_correct = sum(t["reward"] for t in all_trajectories)
    n_total   = len(all_trajectories)
    print(
        f"\nSaved {n_total} trajectories → {args.output_path}\n"
        f"  Reward rate: {n_correct}/{n_total} = {n_correct/n_total:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
