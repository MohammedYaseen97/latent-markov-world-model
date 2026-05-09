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

Batching strategy
─────────────────
All problems × rollouts in a mini-batch are processed in a single
model.generate() call followed by a single model() forward pass (for
hidden states). On an A100 80 GB with --problems-per-batch 16 this gives
128 sequences per GPU call, reducing total GPU dispatches from 23,792
(serial baseline) to ~186 and cutting wallclock time from ~44 h to ~1-2 h.

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
  python scripts/generate_phase0_rollouts.py --limit 4 --n-rollouts 2 --problems-per-batch 2

  # Specify pool and output explicitly
  python scripts/generate_phase0_rollouts.py \\
      --pool-path data/math_easy_pool.jsonl \\
      --output-path data/phase0_rollouts.pt \\
      --n-rollouts 8 --seed 42

  # A100 80 GB — full run, save checkpoint every 20 batches in case of preemption
  python scripts/generate_phase0_rollouts.py \\
      --n-rollouts 8 --problems-per-batch 16 --save-every 20 --seed 42
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
        "--problems-per-batch",
        type=int,
        default=16,
        help=(
            "Problems processed per GPU call. Effective batch = problems_per_batch × n_rollouts. "
            "Default 16 → 128 sequences/call on A100 80 GB. "
            "Reduce to 4-8 if you hit OOM."
        ),
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=20,
        help=(
            "Save a .tmp.pt checkpoint every N batches (0 = only save at the end). "
            "Use to recover progress after preemption. Default: 20."
        ),
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
# Batched rollout generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    n_rollouts: int,
    total_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    n_chunks: int = N_CHUNKS,
) -> list[dict]:
    """Generate and grade n_rollouts completions for every problem in the batch.

    Packs all (problem × rollout) pairs into a single model.generate() call
    with left-padded inputs, then extracts hidden states with a single batched
    forward pass. This replaces the serial one-sequence-at-a-time loop and
    reduces GPU dispatches by a factor of problems_per_batch × n_rollouts.

    Left-padding is used so that all sequences in the batch share the same
    right-aligned position for the start of generation, as required by
    decoder-only attention patterns.

    Args:
        problems:     list of problem dicts (keys: "prompt", "ground_truth",
                      optionally "problem_id").
        n_rollouts:   G rollouts per problem.
        total_tokens: max new tokens to generate per rollout.
        temperature:  sampling temperature.
        top_p:        nucleus sampling cutoff.
        device:       CUDA device.
        n_chunks:     number of equal segments to split each response into.

    Returns:
        Flat list of trajectory dicts, length = len(problems) × n_rollouts.
        Each dict has the same schema as the full rollout file.
    """
    pad_id = tokenizer.eos_token_id

    # Build flat list: repeat each problem's prompt n_rollouts times so that
    # all rollouts for the same problem are contiguous in the batch.
    all_prompt_ids:  list[list[int]] = []
    all_gt:          list[str]       = []
    all_problem_ids: list[str]       = []
    all_rollout_idxs: list[int]      = []

    for prob in problems:
        pids = format_prompt(prob, tokenizer)
        for r in range(n_rollouts):
            all_prompt_ids.append(pids)
            all_gt.append(prob["ground_truth"])
            all_problem_ids.append(prob.get("problem_id", "unknown"))
            all_rollout_idxs.append(r)

    B = len(all_prompt_ids)
    max_prompt_len = max(len(p) for p in all_prompt_ids)

    # Left-pad: shorter prompts get pad tokens prepended.
    input_ids = torch.full((B, max_prompt_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
    prompt_lengths: list[int] = []

    for i, pids in enumerate(all_prompt_ids):
        offset = max_prompt_len - len(pids)
        input_ids[i, offset:] = torch.tensor(pids, dtype=torch.long, device=device)
        attn_mask[i, offset:] = 1
        prompt_lengths.append(len(pids))

    # ── Generation ────────────────────────────────────────────────────────
    gen_out = model.generate(
        input_ids,
        attention_mask=attn_mask,
        max_new_tokens=total_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # gen_out: (B, max_prompt_len + max_new_tokens)
    # With left-padding the new tokens always start at index max_prompt_len.
    response_ids_list = [gen_out[i, max_prompt_len:] for i in range(B)]
    completions = [
        tokenizer.decode(r, skip_special_tokens=True) for r in response_ids_list
    ]
    del gen_out, input_ids, attn_mask
    torch.cuda.empty_cache()

    # ── Hidden-state forward pass ──────────────────────────────────────────
    # Build full sequences (actual prompt, no left-pad prefix) + response.
    # Right-pad to max_full for the batched forward call.
    full_seqs: list[torch.Tensor] = []
    for i, pids in enumerate(all_prompt_ids):
        prompt_t = torch.tensor(pids, dtype=torch.long, device=device)
        full_seqs.append(torch.cat([prompt_t, response_ids_list[i]]))

    max_full = max(s.shape[0] for s in full_seqs)
    full_ids  = torch.full((B, max_full), pad_id, dtype=torch.long, device=device)
    full_attn = torch.zeros(B, max_full, dtype=torch.long, device=device)

    for i, seq in enumerate(full_seqs):
        L = seq.shape[0]
        full_ids[i, :L]  = seq
        full_attn[i, :L] = 1

    fwd = model(full_ids, attention_mask=full_attn, output_hidden_states=True)
    last_hidden = fwd.hidden_states[-1]  # (B, max_full, hidden_dim)
    del fwd, full_ids, full_attn
    torch.cuda.empty_cache()

    # ── Chunk extraction + grading ─────────────────────────────────────────
    trajectories: list[dict] = []
    for i in range(B):
        pl = prompt_lengths[i]
        rl = response_ids_list[i].shape[0]
        resp_hidden = last_hidden[i, pl : pl + rl, :]   # (resp_len, hidden_dim)
        chunks = torch.chunk(resp_hidden, n_chunks, dim=0)
        reprs  = [c.mean(dim=0).detach().cpu().float() for c in chunks]

        trajectories.append({
            "problem_id":   all_problem_ids[i],
            "rollout_idx":  all_rollout_idxs[i],
            "repr_1":       reprs[0],
            "repr_2":       reprs[1],
            "repr_3":       reprs[2],
            "reward":       grade(completions[i], all_gt[i]),
            "ground_truth": all_gt[i],
        })

    del last_hidden
    torch.cuda.empty_cache()

    return trajectories


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
        args.model_id, revision=args.revision, trust_remote_code=True,
        padding_side="left",   # required for batched generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"   # PyTorch built-in; still faster than eager
    print(f"  attention implementation: {attn_impl}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print("  model frozen and ready", flush=True)

    # ------------------------------------------------------------------
    # Batched generation loop
    # ------------------------------------------------------------------
    all_trajectories: list[dict] = []
    n_batches = (len(problems) + args.problems_per_batch - 1) // args.problems_per_batch
    tmp_path  = args.output_path.with_suffix(".tmp.pt")

    print(
        f"\nGenerating: {len(problems)} problems × {args.n_rollouts} rollouts "
        f"= {len(problems) * args.n_rollouts} trajectories",
        flush=True,
    )
    print(
        f"Batch size: {args.problems_per_batch} problems × {args.n_rollouts} rollouts "
        f"= {args.problems_per_batch * args.n_rollouts} sequences/call | {n_batches} batches total",
        flush=True,
    )

    for batch_idx, batch_start in enumerate(
        tqdm(range(0, len(problems), args.problems_per_batch), desc="batches", unit="batch")
    ):
        batch = problems[batch_start : batch_start + args.problems_per_batch]
        try:
            trajs = run_batch(
                model,
                tokenizer,
                batch,
                n_rollouts=args.n_rollouts,
                total_tokens=args.total_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            all_trajectories.extend(trajs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(
                f"\n  OOM on batch {batch_idx} — skipping. "
                f"Retry with --problems-per-batch {args.problems_per_batch // 2}.",
                file=sys.stderr,
            )
            continue

        # Periodic checkpoint so a preempted run can be inspected / resumed.
        if args.save_every > 0 and (batch_idx + 1) % args.save_every == 0:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(all_trajectories, tmp_path)
            n_correct = sum(t["reward"] for t in all_trajectories)
            n_total   = len(all_trajectories)
            print(
                f"  checkpoint [{batch_idx + 1}/{n_batches}]: "
                f"{n_total} trajectories, reward rate {n_correct/n_total:.1%}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Final save
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

    if tmp_path.exists():
        tmp_path.unlink()   # clean up checkpoint once final file is written


if __name__ == "__main__":
    main()
