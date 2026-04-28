#!/usr/bin/env python3
"""Evaluate pass@k (k ∈ {1, 16, 1024}) on a MATH-B JSONL pool.

Loads eval + model config, reads problems, runs generation and grading, writes a machine-readable
metrics file. Core sampling and grading logic lives in ``src/eval/metrics.py``.

Two generation modes are supported (--generation-mode):

  baseline      Single flat sequence per sample, up to max_new_tokens. Supports vLLM
                (recommended for pass@1024) and HF generate (smoke / fallback).

  token_markov  Delethink chunked generation — matches the regime the arm was trained
                in. Context is reset at each chunk boundary; only the carryover window
                carries forward. Requires --train-config to read chunk parameters.
                Supports vLLM (production, use_vllm: true in eval config) and HF
                sequential (smoke fallback). vLLM path uses multi-round batched
                generation; wall-clock ≈ baseline (round 1 identical; rounds 2+ are
                fast). Full pass@1024 on A100 in ~30-45 min.

See ``configs/eval_math_beyond.yaml`` and ``PROJECT_CONTRACT.md`` (Phase 2).
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_yaml_with_extends
from src.training.grpo_baseline import SYSTEM_PROMPT, answers_equivalent, extract_answer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--eval-config",
        type=Path,
        default=REPO_ROOT / "configs" / "eval_math_beyond.yaml",
        help="YAML with primary_claim_path and metrics list.",
    )
    p.add_argument(
        "--base-model-config",
        type=Path,
        default=REPO_ROOT / "configs" / "base_model.yaml",
        help="Checkpoint ids for generation (no extends).",
    )
    p.add_argument(
        "--pool",
        choices=("primary", "secondary_strict", "full"),
        default="primary",
        help="Which pool from eval config to use.",
    )
    p.add_argument(
        "--pool-path",
        type=Path,
        default=None,
        help="Override JSONL path (otherwise taken from eval config).",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="HF repo id or local path to weights for generation (default: primary from base_model.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write metrics JSON here (default: stdout).",
    )
    p.add_argument(
        "--arm-name",
        type=str,
        default="eval",
        help="Label for downstream ablation table (e.g. baseline_grpo).",
    )
    p.add_argument(
        "--generation-mode",
        choices=("baseline", "token_markov"),
        default="baseline",
        help=(
            "baseline: flat single-sequence generation (default). "
            "token_markov: Delethink chunked generation — requires --train-config."
        ),
    )
    p.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help=(
            "Training config YAML for the arm being evaluated. Required when "
            "--generation-mode token_markov (reads chunk_size_tokens, "
            "max_carryover_tokens, iteration_cap, planning_prefix_tokens)."
        ),
    )
    return p.parse_args()


def _load_eval_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "evaluation" not in data:
        raise ValueError(f"Expected top-level 'evaluation' key in {path}")
    return data["evaluation"]


def _pool_path(eval_cfg: dict[str, Any], pool: str, override: Path | None) -> Path:
    if override is not None:
        return (REPO_ROOT / override).resolve() if not override.is_absolute() else override
    key = {
        "primary": "primary_claim_path",
        "secondary_strict": "secondary_strict_path",
        "full": "optional_full_set_path",
    }[pool]
    rel = eval_cfg.get(key)
    if not rel:
        raise ValueError(f"eval config missing {key}")
    return (REPO_ROOT / rel).resolve()


def _load_jsonl_problems(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pool JSONL not found: {path}. Run scripts/prepare_data.py first.")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021, HumanEval).

    Given n total samples for a problem, c of which are correct:
        pass@k = 1 - C(n-c, k) / C(n, k)

    This avoids the variance of randomly choosing k from n; averaging over problems
    gives an unbiased estimate of the true pass@k.
    """
    if n < k:
        # Fewer samples than k — not enough to compute; treat as solved if any correct.
        return 1.0 if c > 0 else 0.0
    if n - c < k:
        # Not enough incorrect samples to fill k slots → guaranteed at least one correct.
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _build_prompts(
    problems: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[str]:
    """Format all problems into chat-template strings ready for generation."""
    prompts = []
    for problem in problems:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem["prompt"]},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        ))
    return prompts


def _grade(completions: list[str], ground_truth: str) -> int:
    return sum(
        1 for comp in completions
        if (pred := extract_answer(comp)) is not None
        and answers_equivalent(pred, ground_truth)
    )


def _estimate_pass_at_k_metrics(
    *,
    problems: list[dict[str, Any]],
    checkpoint: str,
    ks: list[int],
    eval_cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute pass@k for each k using the unbiased estimator (Chen et al. 2021).

    Two generation backends are supported, selected by ``eval_cfg["use_vllm"]``:

    - **vLLM** (production, A100): generates all problems × n_samples in one call.
      Orders of magnitude faster for pass@1024; recommended for any run where max(ks) > 16.

    - **HF generate** (smoke / fallback): loops per problem, generating completions in
      chunks of ``gen_batch_size``. Works on any GPU without vLLM installed.
    """
    n_samples = max(ks)
    temperature = eval_cfg.get("temperature", 1.0)
    top_p = eval_cfg.get("top_p", 1.0)
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 1024))
    use_vllm = eval_cfg.get("use_vllm", False)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = _build_prompts(problems, tokenizer)
    per_problem: list[tuple[int, int]] = []  # (n_sampled, n_correct) per problem

    if use_vllm:
        from vllm import LLM, SamplingParams  # deferred — not required for smoke

        # Fail fast with a clear message if another process is holding GPU memory.
        # vLLM's own error ("Free memory … less than desired utilization") surfaces
        # only after a slow EngineCore subprocess spawn; this check catches it upfront.
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            utilization = eval_cfg.get("vllm_gpu_memory_utilization", 0.85)
            required_bytes = total_bytes * utilization
            if free_bytes < required_bytes:
                free_gib = free_bytes / 2**30
                total_gib = total_bytes / 2**30
                required_gib = required_bytes / 2**30
                raise RuntimeError(
                    f"Not enough free GPU memory to start vLLM "
                    f"({free_gib:.1f} GiB free / {total_gib:.1f} GiB total, "
                    f"need {required_gib:.1f} GiB for gpu_memory_utilization={utilization}). "
                    f"Kill any orphaned GPU processes (check: nvidia-smi) and retry."
                )

        llm = LLM(
            model=checkpoint,
            dtype="bfloat16",
            gpu_memory_utilization=eval_cfg.get("vllm_gpu_memory_utilization", 0.85),
        )
        # Generate in chunks rather than a single llm.generate(n=1024) call.
        # With max_tokens=1024 the EngineCore must detokenize ~40k sequences
        # (~20M tokens) before it can return anything, which stalls for 20+ min.
        # Smaller chunks keep each round-trip fast.
        chunk_size: int = int(eval_cfg.get("vllm_chunk_size", 64))
        n_chunks = -(-n_samples // chunk_size)  # ceil division
        accumulated: list[list[str]] = [[] for _ in range(len(prompts))]

        print(
            f"Generating {n_samples} completions × {len(problems)} problems via vLLM "
            f"(chunks of {chunk_size}) …",
            file=sys.stderr,
        )
        remaining = n_samples
        for chunk_idx in range(1, n_chunks + 1):
            this_n = min(chunk_size, remaining)
            chunk_params = SamplingParams(
                n=this_n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            print(f"  chunk {chunk_idx}/{n_chunks} (n={this_n}) …", file=sys.stderr)
            chunk_outputs = llm.generate(prompts, chunk_params)
            for i, out in enumerate(chunk_outputs):
                accumulated[i].extend(o.text for o in out.outputs)
            remaining -= this_n

        for completions, problem in zip(accumulated, problems):
            per_problem.append((len(completions), _grade(completions, problem["ground_truth"])))

    else:
        gen_batch_size = int(eval_cfg.get("gen_batch_size", 64))

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        for prompt_text, problem in tqdm(
            zip(prompts, problems), total=len(problems), desc="Evaluating", unit="problem"
        ):
            enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)
            attention_mask = enc.attention_mask.to(model.device)
            prompt_len = input_ids.shape[-1]

            completions: list[str] = []
            remaining = n_samples
            while remaining > 0:
                this_batch = min(gen_batch_size, remaining)
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        num_return_sequences=this_batch,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                for seq in output_ids:
                    completions.append(
                        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    )
                remaining -= this_batch

            per_problem.append((len(completions), _grade(completions, problem["ground_truth"])))

    return {
        f"pass@{k}": sum(_unbiased_pass_at_k(n, c, k) for n, c in per_problem) / len(per_problem)
        for k in ks
    }


def _estimate_pass_at_k_metrics_token_markov(
    *,
    problems: list[dict[str, Any]],
    checkpoint: str,
    ks: list[int],
    eval_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute pass@k for the token-Markov arm using Delethink chunked generation.

    Two backends, selected by eval_cfg["use_vllm"]:

    vLLM (production, use_vllm=true):
      Multi-round batched generation — same continuous-batching throughput as
      the baseline vLLM path.

      Round 1  — identical to baseline: all problems × n_samples via vLLM,
                 chunk_size completions at a time (max_tokens=C).
      Round 2+ — 40k unique prompts (query + carryover_i) in chunks of
                 chunk_size, n=1 each (max_tokens=C-m, much shorter).

      Wall-clock: round 1 ≈ baseline time (~30 min); rounds 2+ are short
      because max_tokens is 2-4× smaller and n=1. Total ≈ baseline ± 25%.

    HF generate (smoke / no-vLLM fallback):
      Sequential, one trace at a time. Correct but slow; use for smoke only.
    """
    n_samples = max(ks)
    temperature = eval_cfg.get("temperature", 1.0)
    top_p = eval_cfg.get("top_p", 1.0)
    use_vllm = eval_cfg.get("use_vllm", False)
    tm_cfg = train_cfg.get("token_markov", {})

    if not tm_cfg:
        raise ValueError(
            "train_cfg has no 'token_markov' block — "
            "pass the token-Markov training config via --train-config."
        )

    C = int(tm_cfg["chunk_size_tokens"])
    m = int(tm_cfg["max_carryover_tokens"])
    I = int(tm_cfg["iteration_cap"])
    P = int(tm_cfg["planning_prefix_tokens"])

    from src.training.grpo_token_markov import make_tm_system_prompt
    tm_system_prompt = make_tm_system_prompt(m)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_prompts = _build_prompts(problems, tokenizer, system_prompt=tm_system_prompt)

    if use_vllm:
        from vllm import LLM, SamplingParams

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            utilization = eval_cfg.get("vllm_gpu_memory_utilization", 0.85)
            required_bytes = total_bytes * utilization
            if free_bytes < required_bytes:
                free_gib = free_bytes / 2**30
                total_gib = total_bytes / 2**30
                required_gib = required_bytes / 2**30
                raise RuntimeError(
                    f"Not enough free GPU memory to start vLLM "
                    f"({free_gib:.1f} GiB free / {total_gib:.1f} GiB total, "
                    f"need {required_gib:.1f} GiB). Kill orphaned GPU processes and retry."
                )

        llm = LLM(
            model=checkpoint,
            dtype="bfloat16",
            gpu_memory_utilization=eval_cfg.get("vllm_gpu_memory_utilization", 0.85),
        )
        chunk_size: int = int(eval_cfg.get("vllm_chunk_size", 64))
        n_problems = len(problems)

        # chunk_texts[prob_idx][samp_idx] = list of per-chunk text strings
        chunk_texts: list[list[list[str]]] = [
            [[] for _ in range(n_samples)] for _ in range(n_problems)
        ]

        # ── Round 1: identical to baseline vLLM path ─────────────────────────
        # Same prompt for all n_samples traces of each problem → use n=this_n.
        # Accumulate into a flat list per problem, then assign to chunk_texts.
        print(
            f"[token-Markov eval] Round 1/{I}: {n_samples} completions × "
            f"{n_problems} problems (max_tokens={C}) …",
            file=sys.stderr,
        )
        accumulated_r1: list[list[str]] = [[] for _ in range(n_problems)]
        remaining = n_samples
        n_chunks_r1 = -(-n_samples // chunk_size)
        for chunk_idx in range(1, n_chunks_r1 + 1):
            this_n = min(chunk_size, remaining)
            params = SamplingParams(n=this_n, temperature=temperature, top_p=top_p, max_tokens=C)
            print(f"  chunk {chunk_idx}/{n_chunks_r1} (n={this_n}) …", file=sys.stderr)
            outputs = llm.generate(base_prompts, params)
            for prob_idx, out in enumerate(outputs):
                accumulated_r1[prob_idx].extend(o.text for o in out.outputs)
            remaining -= this_n

        # Assign chunk-1 texts into chunk_texts[prob][samp][chunk=0].
        for prob_idx in range(n_problems):
            for samp_idx, text in enumerate(accumulated_r1[prob_idx]):
                chunk_texts[prob_idx][samp_idx].append(text)

        # ── Compute modified queries (base prompt + planning prefix) ──────────
        # planning prefix = first P tokens of chunk-1 output, appended permanently.
        modified_queries: list[list[str]] = []
        for prob_idx in range(n_problems):
            queries: list[str] = []
            for samp_idx in range(n_samples):
                c1_text = chunk_texts[prob_idx][samp_idx][0]
                ids = tokenizer.encode(c1_text, add_special_tokens=False)
                planning_text = tokenizer.decode(ids[:P], skip_special_tokens=True)
                queries.append(base_prompts[prob_idx] + planning_text)
            modified_queries.append(queries)

        # ── Rounds 2 .. I ─────────────────────────────────────────────────────
        # Each trace has a unique prompt = modified_query + carryover from prev chunk.
        # Flatten all (prob, samp) pairs into one list, batch with chunk_size, n=1.
        for round_idx in range(1, I):
            max_new = C - m
            flat_prompts: list[str] = []
            for prob_idx in range(n_problems):
                for samp_idx in range(n_samples):
                    prev_text = chunk_texts[prob_idx][samp_idx][-1]
                    ids = tokenizer.encode(prev_text, add_special_tokens=False)
                    carryover = tokenizer.decode(ids[-m:], skip_special_tokens=True)
                    flat_prompts.append(modified_queries[prob_idx][samp_idx] + carryover)

            print(
                f"[token-Markov eval] Round {round_idx + 1}/{I}: "
                f"{len(flat_prompts)} unique prompts (max_tokens={max_new}) …",
                file=sys.stderr,
            )
            flat_outputs: list[str] = [""] * len(flat_prompts)
            params = SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_new)
            for start in range(0, len(flat_prompts), chunk_size):
                batch = flat_prompts[start : start + chunk_size]
                outs = llm.generate(batch, params)
                for j, out in enumerate(outs):
                    flat_outputs[start + j] = out.outputs[0].text

            for prob_idx in range(n_problems):
                for samp_idx in range(n_samples):
                    idx = prob_idx * n_samples + samp_idx
                    chunk_texts[prob_idx][samp_idx].append(flat_outputs[idx])

        # ── Grade ─────────────────────────────────────────────────────────────
        per_problem: list[tuple[int, int]] = []
        for prob_idx, problem in enumerate(problems):
            completions = [
                "".join(chunk_texts[prob_idx][samp_idx]) for samp_idx in range(n_samples)
            ]
            per_problem.append((len(completions), _grade(completions, problem["ground_truth"])))

    else:
        # ── HF sequential fallback (smoke / no-vLLM) ─────────────────────────
        from src.training.grpo_token_markov import generate_delethink_trace

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        per_problem = []
        for problem in tqdm(problems, desc="Evaluating (token-Markov HF)", unit="problem"):
            messages = [
                {"role": "system", "content": tm_system_prompt},
                {"role": "user", "content": problem["prompt"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            query_ids: list[int] = tokenizer(prompt_text, add_special_tokens=False).input_ids

            completions: list[str] = []
            for _ in range(n_samples):
                with torch.no_grad():
                    trace = generate_delethink_trace(
                        model=model,
                        tokenizer=tokenizer,
                        query_ids=query_ids,
                        cfg=tm_cfg,
                        temperature=temperature,
                        top_p=top_p,
                    )
                if trace.chunks:
                    all_resp_ids = torch.cat([c.response_ids for c in trace.chunks])
                    full_text = tokenizer.decode(all_resp_ids, skip_special_tokens=True)
                else:
                    full_text = ""
                completions.append(full_text)

            per_problem.append((len(completions), _grade(completions, problem["ground_truth"])))

    return {
        f"pass@{k}": sum(_unbiased_pass_at_k(n, c, k) for n, c in per_problem) / len(per_problem)
        for k in ks
    }


def main() -> None:
    args = parse_args()
    eval_cfg = _load_eval_yaml(args.eval_config.resolve())
    base_cfg = load_yaml_with_extends(args.base_model_config.resolve(), root=REPO_ROOT)
    pool_file = _pool_path(eval_cfg, args.pool, args.pool_path)
    problems = _load_jsonl_problems(pool_file)

    checkpoint = args.checkpoint or (base_cfg.get("primary") or {}).get("huggingface_repo_id")
    if not checkpoint:
        raise ValueError("No --checkpoint and no primary.huggingface_repo_id in base_model config.")

    metrics_keys = eval_cfg.get("metrics") or ["pass@1", "pass@16", "pass@1024"]
    ks = []
    for m in metrics_keys:
        if m.startswith("pass@"):
            ks.append(int(m.split("@", 1)[1]))

    if args.generation_mode == "token_markov":
        if args.train_config is None:
            raise ValueError(
                "--train-config is required for --generation-mode token_markov. "
                "Pass e.g. configs/train_token_markov_grpo.yaml (or the 2-chunk variant)."
            )
        train_cfg = load_yaml_with_extends(args.train_config.resolve(), root=REPO_ROOT)
        metrics = _estimate_pass_at_k_metrics_token_markov(
            problems=problems,
            checkpoint=checkpoint,
            ks=ks,
            eval_cfg=eval_cfg,
            train_cfg=train_cfg,
        )
    else:
        metrics = _estimate_pass_at_k_metrics(
            problems=problems,
            checkpoint=checkpoint,
            ks=ks,
            eval_cfg=eval_cfg,
        )

    out: dict[str, Any] = {
        "arm": args.arm_name,
        "checkpoint": checkpoint,
        "pool_path": str(pool_file.relative_to(REPO_ROOT)),
        "n_problems": len(problems),
        "metrics": metrics,
        "claim_metric": eval_cfg.get("claim_metric"),
    }
    text = json.dumps(out, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
