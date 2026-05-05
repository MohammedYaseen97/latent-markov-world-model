#!/usr/bin/env python3
"""Calibration check: pass@k on a candidate Phase 0 pretraining pool.

Runs the model on each problem with n=k samples and reports whether the
pool provides enough positive training examples for L_outcome.

Rule of thumb: if ≥ 20% of problems are solved at pass@8, the pool is usable.
If < 5%, switch to an easier pool or increase G during pretraining.

Default k:    8  (mirrors G=8 rollouts used during Phase 0 pretraining)
Default seed: 42 (fully reproducible generations)

Usage:
    python scripts/calibrate_pool.py --pool-path data/math_easy_pool.jsonl
    python scripts/calibrate_pool.py --pool-path data/math_easy_pool.jsonl --n-samples 16
    python scripts/calibrate_pool.py --pool-path data/math_easy_pool.jsonl --limit 30
    python scripts/calibrate_pool.py --pool-path data/math_easy_pool.jsonl --output results/calib_easy.json

See reports/latent_markov_design.md (Implementation Deliverables, step 1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))
from src.training.grpo_baseline import extract_answer, answers_equivalent

SYSTEM_PROMPT = (
    "Please reason step by step. You may think as long as you need to. "
    "Put your final answer inside \\boxed{answer}."
)


def load_pool(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def sample_completions(
    model,
    tokenizer,
    prompt: str,
    n: int,
    max_new_tokens: int,
    device: str,
) -> list[str]:
    """Generate n completions for a single prompt in one model.generate() call.

    Using num_return_sequences=n lets HF compute the prompt KV-cache once and
    branch n independent samples from it, which is significantly faster than
    calling generate() n times or manually expanding the batch.
    """
    enc = tokenizer(prompt, return_tensors="pt", padding=False)
    input_ids = enc.input_ids.to(device)       # [1, seq_len]
    attn_mask = enc.attention_mask.to(device)  # [1, seq_len], all-ones

    out = model.generate(
        input_ids,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )  # [n, prompt_len + gen_len]

    prompt_len = input_ids.shape[1]
    return [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        for seq in out
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool-path", type=Path, required=True,
                   help="Path to the JSONL file to calibrate (required)")
    p.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--revision", type=str, default="989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
    p.add_argument("--n-samples", type=int, default=8, help="Samples per problem / k in pass@k (default 8)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens to generate per sample (default 512; calibration doesn't need 1024)")
    p.add_argument("--seed", type=int, default=42,
                   help="Global RNG seed — pins torch, numpy, random, and HF sampling (default 42)")
    p.add_argument("--output", type=Path, default=None, help="Optional JSON output path for per-problem results")
    p.add_argument("--limit", type=int, default=None, help="Only run first N problems (sequential, preserves pool order)")
    p.add_argument("--sample", type=int, default=None,
                   help="Randomly sample N problems from the pool (uses --seed). "
                        "Preferred over --limit when the pool is sorted by difficulty.")
    args = p.parse_args()

    # --- reproducibility -------------------------------------------------------
    set_seed(args.seed)  # sets torch / numpy / random seeds in one call
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:  {device}", flush=True)
    print(f"Seed:    {args.seed}", flush=True)
    print(f"Loading  {args.model_id} @ {args.revision[:8]}...", flush=True)

    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    # Use a distinct pad token so HF can set a proper attention_mask.
    # We never actually pad — prompts are processed one at a time — so the
    # token choice doesn't affect results.
    if tokenizer.pad_token_id is None:
        vocab = tokenizer.get_vocab()
        if "<|fim_pad|>" in vocab:
            tokenizer.pad_token = "<|fim_pad|>"
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=torch.bfloat16,   # use `dtype` not deprecated `torch_dtype`
        device_map="auto",
    )
    model.eval()

    load_time = time.perf_counter() - t0
    print(f"Loaded   in {load_time:.1f}s", flush=True)

    import random
    problems = load_pool(args.pool_path)
    if args.sample:
        problems = random.Random(args.seed).sample(problems, min(args.sample, len(problems)))
        problems.sort(key=lambda r: r["source_index"])  # stable display order
    elif args.limit:
        problems = problems[:args.limit]

    print(f"Pool:    {args.pool_path.name}  ({len(problems)} problems)", flush=True)
    print(f"n={args.n_samples} samples/problem  max_new_tokens={args.max_new_tokens}", flush=True)
    print("-" * 70, flush=True)

    results = []
    n_solved = 0
    gen_start = time.perf_counter()

    for i, row in enumerate(problems):
        t_prob = time.perf_counter()
        prompt = make_prompt(row["prompt"], tokenizer)
        completions = sample_completions(
            model, tokenizer, prompt,
            n=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )

        correct = [
            answers_equivalent(extract_answer(c) or "", row["ground_truth"])
            for c in completions
        ]
        n_correct = sum(correct)
        solved = n_correct > 0
        if solved:
            n_solved += 1

        elapsed = time.perf_counter() - t_prob
        status = f"✓ {n_correct}/{args.n_samples}" if solved else f"✗ 0/{args.n_samples}"
        print(f"[{i+1:3d}/{len(problems)}] {row['problem_id']}  {status}  ({elapsed:.1f}s)", flush=True)

        results.append({
            "problem_id": row["problem_id"],
            "source_index": row["source_index"],
            "n_correct": n_correct,
            "n_samples": args.n_samples,
            "solved": solved,
            "ground_truth": row["ground_truth"],
        })

    total_gen_time = time.perf_counter() - gen_start
    print("-" * 70)

    pct_problems_solved = 100 * n_solved / len(problems)
    total_correct = sum(r["n_correct"] for r in results)
    total_samples = len(problems) * args.n_samples
    per_sample_rate = 100 * total_correct / total_samples

    print(f"\nSummary  ({len(problems)} problems × n={args.n_samples}  |  seed={args.seed}  |  {total_gen_time:.0f}s total)")
    print(f"  Problems with ≥1 correct:  {n_solved}/{len(problems)}  ({pct_problems_solved:.1f}%)")
    print(f"  Per-sample success rate:   {total_correct}/{total_samples}  ({per_sample_rate:.3f}%)")
    print(f"  Avg time per problem:      {total_gen_time/len(problems):.1f}s")
    print()
    print("Interpretation for Phase 0:")
    if pct_problems_solved >= 20:
        print(f"  GOOD — {pct_problems_solved:.0f}% of problems solved at n={args.n_samples}.")
        print("  L_outcome will see positive training examples. Phase 0 design is sound.")
    elif pct_problems_solved >= 5:
        print(f"  MARGINAL — {pct_problems_solved:.0f}% of problems solved at n={args.n_samples}.")
        print("  Some positive examples exist. Consider increasing G during pretraining.")
    else:
        print(f"  LOW — only {pct_problems_solved:.0f}% of problems solved at n={args.n_samples}.")
        print("  L_outcome will be starved of positive examples. Switch to an easier pool.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {
            "pool_path": str(args.pool_path),
            "model_id": args.model_id,
            "revision": args.revision,
            "seed": args.seed,
            "n_samples": args.n_samples,
            "max_new_tokens": args.max_new_tokens,
            "n_problems": len(problems),
            "n_problems_solved": n_solved,
            "pct_problems_solved": pct_problems_solved,
            "per_sample_success_rate_pct": per_sample_rate,
            "total_gen_time_s": round(total_gen_time, 1),
            "per_problem": results,
        }
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
