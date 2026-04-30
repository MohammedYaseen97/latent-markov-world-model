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
    count = 0
    for comp in completions:
        pred = extract_answer(comp)
        if pred is None:
            continue
        try:
            if answers_equivalent(pred, ground_truth):
                count += 1
        except Exception:
            pass
    return count


def _make_vllm(checkpoint: str, eval_cfg: dict[str, Any]) -> "LLM":  # type: ignore[name-defined]
    """Construct a vLLM LLM instance with a pre-flight GPU memory check.

    Raises RuntimeError with a clear message if the GPU doesn't have enough
    free memory to satisfy gpu_memory_utilization before the slow EngineCore
    subprocess spawn. Accepts an optional vllm_max_model_len to cap KV-cache
    block allocation (important: without this vLLM pre-allocates for the
    model's full max_position_embeddings, which is 32 768 for Qwen2.5 and
    wastes gigabytes of RAM for sequences that are never that long).
    """
    from vllm import LLM

    utilization = eval_cfg.get("vllm_gpu_memory_utilization", 0.85)
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        required_bytes = total_bytes * utilization
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"Not enough free GPU memory to start vLLM "
                f"({free_bytes / 2**30:.1f} GiB free / {total_bytes / 2**30:.1f} GiB total, "
                f"need {required_bytes / 2**30:.1f} GiB for "
                f"gpu_memory_utilization={utilization}). "
                f"Kill any orphaned GPU processes (nvidia-smi) and retry."
            )

    kwargs: dict[str, Any] = dict(
        model=checkpoint,
        dtype="bfloat16",
        gpu_memory_utilization=utilization,
        seed=int(eval_cfg.get("vllm_seed", 42)),
    )
    max_model_len = eval_cfg.get("vllm_max_model_len", None)
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)

    return LLM(**kwargs)


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
        from vllm import SamplingParams  # deferred — not required for smoke

        llm = _make_vllm(checkpoint, eval_cfg)

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
      Problems are processed in groups of ``tm_vllm_problem_group_size`` (default 4).
      Processing one group at a time keeps peak system RAM bounded to roughly:
          group_sz × n_samples × I × ~4 KB ≈ 50–100 MB of Python string data
      regardless of total problem count. Without grouping, all I rounds of text
      for all 40 × 1024 samples live in RAM simultaneously, which triggers the
      Linux OOM killer on machines with limited system RAM.

      Per group:
        Round 1  — same prompt for all n_samples of each problem in the group →
                   batched with SamplingParams(n=chunk_size). Exactly like the
                   baseline path but limited to the current group.
        Planning — first P tokens of each chunk-1 output folded into the query
                   permanently (same as training). Computed once per group.
        Rounds 2+ — only unfinished traces (EOS not yet hit) need new chunks.
                    Their unique prompts are flattened and batched in chunk_size.
        Grading  — after all rounds, grade problem-by-problem and free strings
                   before moving to the next group.

    HF generate (smoke / no-vLLM fallback):
      Sequential, one trace at a time (already memory-efficient).
    """
    import gc
    import math

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
    n_problems = len(problems)
    per_problem: list[tuple[int, int]] = []

    if use_vllm:
        from vllm import SamplingParams

        llm = _make_vllm(checkpoint, eval_cfg)
        chunk_size: int = int(eval_cfg.get("vllm_chunk_size", 32))

        # Number of problems to process together. Larger = better vLLM batch
        # utilisation; smaller = lower peak RAM. Default 4 keeps peak string
        # data below ~100 MB while still presenting a reasonable batch to vLLM.
        group_sz: int = int(eval_cfg.get("tm_vllm_problem_group_size", 4))
        n_r1_iters = math.ceil(n_samples / chunk_size)
        n_groups = math.ceil(n_problems / group_sz)

        for g_idx, g_start in enumerate(range(0, n_problems, group_sz)):
            g_end = min(g_start + group_sz, n_problems)
            g_probs = problems[g_start:g_end]
            g_prompts = base_prompts[g_start:g_end]
            G = len(g_probs)

            # chunk_texts[li][s] grows from [] to [c1_text, c2_text, …] over rounds.
            chunk_texts: list[list[list[str]]] = [
                [[] for _ in range(n_samples)] for _ in range(G)
            ]
            # finished[li][s] = True once EOS is seen; such samples skip further rounds.
            finished: list[list[bool]] = [[False] * n_samples for _ in range(G)]

            # ── Round 1 ──────────────────────────────────────────────────────
            print(
                f"\n[TM eval] group {g_idx + 1}/{n_groups} — round 1/{I}: "
                f"{n_samples} × {G} problems (max_tokens={C}) …",
                file=sys.stderr,
            )
            # Temporary per-problem accumulators — freed immediately after use.
            r1_acc: list[list[str]] = [[] for _ in range(G)]
            r1_fin: list[list[bool]] = [[] for _ in range(G)]
            remaining = n_samples
            for iter_i in range(n_r1_iters):
                this_n = min(chunk_size, remaining)
                outputs = llm.generate(
                    g_prompts,
                    SamplingParams(n=this_n, temperature=temperature, top_p=top_p, max_tokens=C),
                )
                for li, out in enumerate(outputs):
                    for o in out.outputs:
                        r1_acc[li].append(o.text)
                        r1_fin[li].append(o.finish_reason == "stop")
                remaining -= this_n
                print(f"  r1 iter {iter_i + 1}/{n_r1_iters}", file=sys.stderr, end="\r")

            for li in range(G):
                for s in range(n_samples):
                    chunk_texts[li][s].append(r1_acc[li][s])
                    finished[li][s] = r1_fin[li][s]
            del r1_acc, r1_fin
            gc.collect()

            # ── Planning prefix (compute once from chunk-1, reuse for all rounds) ──
            # g_mq[li][s] = base_prompt[li] + first-P-tokens of chunk-1 output for sample s.
            g_mq: list[list[str]] = []
            for li in range(G):
                mqs: list[str] = []
                for s in range(n_samples):
                    ids = tokenizer.encode(chunk_texts[li][s][0], add_special_tokens=False)
                    planning_text = tokenizer.decode(ids[:P], skip_special_tokens=True)
                    mqs.append(g_prompts[li] + planning_text)
                g_mq.append(mqs)

            # ── Rounds 2 .. I ────────────────────────────────────────────────
            for round_idx in range(1, I):
                max_new = C - m

                # Only generate for samples whose traces are still live (no EOS yet).
                active: list[tuple[int, int]] = [
                    (li, s)
                    for li in range(G)
                    for s in range(n_samples)
                    if not finished[li][s]
                ]

                print(
                    f"\n[TM eval] group {g_idx + 1}/{n_groups} — round {round_idx + 1}/{I}: "
                    f"{len(active)} active traces (max_tokens={max_new}) …",
                    file=sys.stderr,
                )

                if active:
                    # Build flat prompt list: modified_query + last-m-token carryover.
                    flat_prompts: list[str] = []
                    for li, s in active:
                        prev_ids = tokenizer.encode(
                            chunk_texts[li][s][-1], add_special_tokens=False
                        )
                        carryover = tokenizer.decode(prev_ids[-m:], skip_special_tokens=True)
                        flat_prompts.append(g_mq[li][s] + carryover)

                    flat_texts: list[str] = [""] * len(flat_prompts)
                    flat_fin: list[bool] = [False] * len(flat_prompts)
                    params = SamplingParams(
                        n=1, temperature=temperature, top_p=top_p, max_tokens=max_new
                    )
                    n_batches = math.ceil(len(flat_prompts) / chunk_size)
                    for b_idx, start in enumerate(range(0, len(flat_prompts), chunk_size)):
                        outs = llm.generate(flat_prompts[start : start + chunk_size], params)
                        for j, out in enumerate(outs):
                            flat_texts[start + j] = out.outputs[0].text
                            flat_fin[start + j] = out.outputs[0].finish_reason == "stop"
                        print(
                            f"  r{round_idx + 1} batch {b_idx + 1}/{n_batches}",
                            file=sys.stderr,
                            end="\r",
                        )

                    del flat_prompts
                    gc.collect()

                    for fi, (li, s) in enumerate(active):
                        chunk_texts[li][s].append(flat_texts[fi])
                        finished[li][s] = flat_fin[fi]

                    del flat_texts, flat_fin
                    gc.collect()

                # Samples that were already finished get an empty placeholder so
                # every chunk_texts[li][s] list has length == round_idx + 1.
                for li in range(G):
                    for s in range(n_samples):
                        if len(chunk_texts[li][s]) <= round_idx:
                            chunk_texts[li][s].append("")

            # ── Grade this group (problem-by-problem to minimise peak RAM) ───
            print(
                f"\n[TM eval] group {g_idx + 1}/{n_groups} — grading "
                f"{G * n_samples} completions (math-verify) …",
                file=sys.stderr,
            )
            for li, problem in enumerate(g_probs):
                n_correct = 0
                for s in range(n_samples):
                    full_text = "".join(chunk_texts[li][s])
                    pred = extract_answer(full_text)
                    if pred is not None:
                        try:
                            if answers_equivalent(pred, problem["ground_truth"]):
                                n_correct += 1
                        except Exception:
                            pass
                per_problem.append((n_samples, n_correct))
                print(
                    f"  problem {g_start + li + 1}/{n_problems}: "
                    f"{n_correct}/{n_samples} correct",
                    file=sys.stderr,
                )

            del chunk_texts, g_mq, finished
            gc.collect()

    else:
        # ── HF sequential fallback (smoke / no-vLLM) ─────────────────────────
        # One trace at a time — already bounded RAM, no grouping needed.
        from src.training.grpo_token_markov import generate_delethink_trace

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

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
                "Pass e.g. configs/train_token_markov_grpo.yaml."
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
