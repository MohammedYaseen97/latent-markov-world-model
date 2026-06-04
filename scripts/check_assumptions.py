"""Assumption-checking script for the latent Markov arm.

Run incrementally — each section is gated behind --check <name>.

Checks (added progressively):
  traces      — A0/A1/A9/A10: print 20 raw teacher traces to inspect
                chunk quality, boundary alignment, and problem coverage.

Usage (local, RTX 4060 8 GB):
  python scripts/check_assumptions.py --check traces
  python scripts/check_assumptions.py --check traces --n 5 --difficulties 1 2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Repo root on sys.path ──────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.grpo_latent import (
    _LATENT_SYSTEM_PROMPT,
    _find_boxed,
    _has_boxed,
    format_prompt,
)
from src.training.grpo_baseline import answers_equivalent, extract_answer
from src.utils.config_loader import load_yaml_with_extends

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_CONFIG  = REPO_ROOT / "configs" / "train_latent_grpo.yaml"
DEFAULT_POOL    = REPO_ROOT / "data" / "math_easy_pool.jsonl"
CHUNK_TOKENS    = 341   # matches train config
SEP             = "─" * 72


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model_local(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-1.5B in bfloat16 on CUDA (or CPU fallback).

    Fits inside 8 GB VRAM at batch_size=1 with no KV-cache accumulation.
    torch.compile and vLLM are off — this is a diagnostic script.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} (revision={revision[:8]}…) on {device} …", flush=True)

    tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # lets HF place layers given available VRAM
        attn_implementation="sdpa",
    )
    model.eval()
    print("  model loaded.\n", flush=True)
    return model, tok


def load_pool(
    pool_path: Path,
    difficulties: list[int] | None = None,
    n: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Sample n problems from the JSONL pool, optionally filtered by difficulty."""
    problems = []
    with open(pool_path) as f:
        for line in f:
            p = json.loads(line)
            if difficulties is None or p.get("difficulty") in difficulties:
                problems.append(p)
    rng = random.Random(seed)
    return rng.sample(problems, min(n, len(problems)))


@torch.no_grad()
def generate_teacher_steps_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem: dict,
    step_tokens: int,
    max_chunks: int,
    device: str,
) -> list[str]:
    """Generate reasoning one step at a time until ``\\boxed{}`` appears.

    Each generation call produces exactly one reasoning step.
    Returns a list of decoded step texts (length 1..max_chunks).
    """
    pad_id = tokenizer.eos_token_id
    prompt_ids = torch.tensor(
        [format_prompt(problem, tokenizer)], dtype=torch.long
    )
    # Accumulated token IDs from prior steps (just the generated tokens, not prompt)
    accumulated: list[int] = []
    steps: list[str] = []

    for _ in range(max_chunks):
        total_ids = prompt_ids[0].tolist() + accumulated
        input_ids = torch.tensor([total_ids], dtype=torch.long)
        attn = torch.ones_like(input_ids)
        out = model.generate(
            input_ids.to(device), attention_mask=attn.to(device),
            max_new_tokens=step_tokens,
            do_sample=True, temperature=1.0, top_p=1.0,
            pad_token_id=pad_id,
        )
        new_ids = out[0, input_ids.shape[1]:].cpu()
        step_text = tokenizer.decode(new_ids.tolist(), skip_special_tokens=True)
        steps.append(step_text)
        accumulated.extend(new_ids.tolist())

        if _find_boxed(step_text) is not None:
            break

    return steps


def print_trace(
    idx: int,
    problem: dict,
    steps: list[str],
    ground_truth: str,
) -> None:
    full = "".join(steps)
    pred = extract_answer(full)
    reward = int(pred is not None and answers_equivalent(pred, ground_truth))
    status = "✅ CORRECT" if reward else "❌ WRONG"
    pred_s = repr(pred) if pred else "None"

    print(SEP)
    print(f"[{idx+1}] {status}  |  gt={ground_truth!r}  pred={pred_s}")
    print(f"     diff={problem.get('difficulty','?')}  topic={problem.get('topic','?')}")
    print(f"     problem: {problem['prompt'][:120]}")
    print(f"     steps: {len(steps)}")
    print()

    for k, text in enumerate(steps):
        label = f"STEP {k+1}" if k < len(steps) - 1 or _find_boxed(text) is None else f"STEP {k+1} (final)"
        print(f"  ── {label} ({len(text.split())} words) ──")
        if len(text) > 450:
            head = text[:300].replace("\n", " ")
            tail = text[-120:].replace("\n", " ")
            print("  " + textwrap.fill(head, 80, subsequent_indent="  "))
            print("  …")
            print("  " + textwrap.fill(tail, 80, subsequent_indent="  "))
        else:
            print("  " + textwrap.fill(text.replace("\n", " "), 80, subsequent_indent="  "))
        print()
    print()


# ── Check: traces ──────────────────────────────────────────────────────────────

def check_traces(args: argparse.Namespace) -> None:
    """A0/A1/A9/A10: inspect teacher traces with one-step-at-a-time generation.

    What to look for:
      A0 — Does each step end cleanly (model self-stops)?
      A1 — Are steps complete reasoning units, or cut mid-sentence?
      A9 — Is the teacher's reasoning coherent and correct?
      A10 — Could a reader resume from the start of step 2/3/…?
    """
    pool_path   = Path(args.pool) if args.pool else DEFAULT_POOL
    difficulties = args.difficulties or None
    n           = args.n
    seed        = args.seed
    step_tokens = args.step_tokens
    max_chunks  = args.max_chunks

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_local()

    problems = load_pool(pool_path, difficulties=difficulties, n=n, seed=seed)
    print(f"Sampled {len(problems)} problems from {pool_path.name}")
    print(f"step_tokens={step_tokens}  max_chunks={max_chunks}\n")

    correct = 0
    total_steps = 0
    for i, prob in enumerate(problems):
        print(f"\nGenerating step-by-step trace {i+1}/{len(problems)} …", end=" ", flush=True)
        steps = generate_teacher_steps_single(
            model, tokenizer, prob, step_tokens, max_chunks, device,
        )
        total_steps += len(steps)
        full = "".join(steps)
        pred = extract_answer(full)
        reward = int(pred is not None and answers_equivalent(pred, prob["ground_truth"]))
        correct += reward
        print(f"done ({len(steps)} steps)", flush=True)

        print_trace(i, prob, steps, prob["ground_truth"])

    avg_steps = total_steps / len(problems)
    print(SEP)
    print(f"\nTeacher reward rate: {correct}/{len(problems)} = {correct/len(problems):.1%}")
    print(f"Average steps per problem: {avg_steps:.1f}")
    print("\nThings to note per trace:")
    print("  A0  — does each step end cleanly (model self-stops at step boundary)?")
    print("  A1  — is each step a complete reasoning unit?")
    print("  A9  — is teacher reasoning coherent and step-by-step?")
    print("  A10 — could a reader resume from the start of step 2?")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Incrementally test latent Markov design assumptions (local, 8 GB)."
    )
    p.add_argument(
        "--check", required=True,
        choices=["traces"],
        help="Which assumption check to run.",
    )
    p.add_argument(
        "--n", type=int, default=20,
        help="Number of problems to sample (default: 20).",
    )
    p.add_argument(
        "--difficulties", type=int, nargs="+", default=None,
        help="Filter pool by difficulty levels e.g. --difficulties 1 2 3 4",
    )
    p.add_argument(
        "--pool", type=str, default=None,
        help="Path to JSONL pool (default: data/math_easy_pool.jsonl).",
    )
    p.add_argument(
        "--step_tokens", type=int, default=341,
        help="Max tokens per generation step (default: 341).",
    )
    p.add_argument(
        "--max_chunks", type=int, default=10,
        help="Max reasoning steps before stopping (default: 10).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.check == "traces":
        check_traces(args)
    else:
        print(f"Unknown check: {args.check}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
