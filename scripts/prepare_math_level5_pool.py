#!/usr/bin/env python3
"""Build the Level 5 hard pool for training and eval.

Loads all Level 5 problems from EleutherAI/hendrycks_math, runs the pretrained
model at pass@128, and keeps only the problems where pass@128 = 0.

Rationale: if the pretrained model can already solve a problem at 128 samples,
it is not hard enough to test the capability-ceiling hypothesis. This filter is
model-specific but reproducible given a fixed checkpoint and seed.

Mutual exclusivity guarantee: Level 5 problems are never in the easy pool (which
uses Level 1–4 only). Verified by source_level field in both manifest files.

Output: ``data/math_level5_hard_pool.jsonl``

Schema (one JSON object per line):
  problem_id   : "math_l5_{i:04d}"
  source_index : i (sequential, 0-based within filtered set)
  prompt       : problem text
  ground_truth : answer string extracted from the LaTeX \\boxed{} in the solution
  data_source  : "hendrycks_math"
  topic        : subject category (Algebra, Geometry, etc.)
  difficulty   : 5 (all rows)
  source_level : 5 (for manifest mutual exclusivity check)

Usage:
    # Model ID and revision are read from configs/base_model.yaml automatically.
    python scripts/prepare_math_level5_pool.py

    # With explicit overrides:
    python scripts/prepare_math_level5_pool.py \\
        --model-id Qwen/Qwen2.5-1.5B-Instruct \\
        --model-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \\
        --n-samples 128 \\
        --batch-size 16 \\
        --output data/math_level5_hard_pool.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Read pinned model ID + revision from base_model.yaml — single source of truth.
_base_cfg = yaml.safe_load((REPO_ROOT / "configs" / "base_model.yaml").read_text())
_PRIMARY   = _base_cfg["primary"]
_DEFAULT_MODEL_ID       = _PRIMARY["huggingface_repo_id"]
_DEFAULT_MODEL_REVISION = _PRIMARY["revision"]
_DEFAULT_ATTN_IMPL      = _PRIMARY.get("attn_implementation", "sdpa")

from src.training.grpo_baseline import (
    SYSTEM_PROMPT,
    _extract_boxed as extract_boxed,
    answers_equivalent,
    extract_answer,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DATASET    = "EleutherAI/hendrycks_math"
HF_REVISION   = "21a5633873b6a120296cce3e2df9d5550074f4a3"  # pinned
CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
MANIFEST_PATH = REPO_ROOT / "data" / "level5_pool_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_library_versions() -> dict[str, str]:
    out = {}
    for pkg in ("datasets", "huggingface_hub", "transformers", "torch"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_prompt_str(problem_text: str, tokenizer: AutoTokenizer) -> str:
    """Return the chat-formatted prompt string (for vLLM or HF tokenisation)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": problem_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_prompt(problem_text: str, tokenizer: AutoTokenizer) -> list[int]:
    text = build_prompt_str(problem_text, tokenizer)
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def estimate_pass_at_k(n_correct: int, n_total: int) -> float:
    """Unbiased pass@k estimator (Chen et al. 2021).
    For k == n_total: pass@n = n_correct / n_total.
    Here we use pass@n_total as a proxy for pass@128.
    """
    if n_total == 0:
        return 0.0
    return n_correct / n_total


# ---------------------------------------------------------------------------
# pass@n scoring for a single problem
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_problem(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problem_text: str,
    ground_truth: str,
    n_samples: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> int:
    """Return number of correct completions out of n_samples."""
    prompt_ids = build_prompt(problem_text, tokenizer)
    input_ids  = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    prompt_len = input_ids.shape[1]
    pad_id     = tokenizer.eos_token_id

    n_correct = 0
    remaining = n_samples

    while remaining > 0:
        this_batch = min(batch_size, remaining)
        inp = input_ids.expand(this_batch, -1)
        attn = torch.ones(this_batch, prompt_len, dtype=torch.long, device=device)

        out = model.generate(
            inp,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for i in range(this_batch):
            completion = tokenizer.decode(
                out[i, prompt_len:], skip_special_tokens=True
            )
            pred = extract_answer(completion)
            if pred is not None and answers_equivalent(pred, ground_truth):
                n_correct += 1

        remaining -= this_batch

    return n_correct


def score_all_problems_vllm(
    problems: list[dict],
    tokenizer: AutoTokenizer,
    model_id: str,
    model_revision: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
    max_model_len: int,
    seed: int,
) -> list[int]:
    """Score all problems in one vLLM call — uses continuous batching to saturate GPU.

    Returns a list of n_correct (int) per problem, same order as `problems`.
    Each prompt gets n_samples completions; vLLM handles scheduling internally.
    """
    from vllm import LLM, SamplingParams  # imported lazily — not required for HF path

    logger.info(
        "vLLM: loading %s @ %s  gpu_mem=%.2f  max_model_len=%d",
        model_id, model_revision, gpu_memory_utilization, max_model_len,
    )
    llm = LLM(
        model=model_id,
        revision=model_revision,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )

    prompts = [build_prompt_str(p["prompt"], tokenizer) for p in problems]
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    logger.info(
        "vLLM: generating %d problems × %d samples = %d completions ...",
        len(prompts), n_samples, len(prompts) * n_samples,
    )
    outputs = llm.generate(prompts, sampling_params)

    results: list[int] = []
    for prob, output in zip(problems, outputs):
        n_correct = 0
        for completion in output.outputs:
            pred = extract_answer(completion.text)
            if pred is not None and answers_equivalent(pred, prob["ground_truth"]):
                n_correct += 1
        results.append(n_correct)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model-id", type=str, default=_DEFAULT_MODEL_ID,
        help="HF model ID for pass@128 filtering (default: from configs/base_model.yaml).",
    )
    p.add_argument(
        "--model-revision", type=str, default=_DEFAULT_MODEL_REVISION,
        help="HF model revision SHA (default: pinned commit from configs/base_model.yaml).",
    )
    p.add_argument(
        "--n-samples", type=int, default=128,
        help="Samples per problem for filtering (default: 128).",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Generation batch size per problem (default: 16).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Max tokens per completion (default: 1024).",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    p.add_argument(
        "--hf-revision", type=str, default=HF_REVISION,
        help="Pinned HF dataset git SHA.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--output", type=Path, default=REPO_ROOT / "data" / "math_level5_hard_pool.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="HF splits to pull from (default: train test).",
    )

    # vLLM settings (production — saturates GPU via continuous batching)
    p.add_argument(
        "--use-vllm", action=argparse.BooleanOptionalAction, default=True,
        help="Use vLLM for scoring (default: true). Pass --no-use-vllm for HF fallback.",
    )
    p.add_argument(
        "--vllm-gpu-memory-utilization", type=float, default=0.90,
        help="Fraction of GPU memory given to vLLM KV cache (default: 0.90).",
    )
    p.add_argument(
        "--vllm-max-model-len", type=int, default=2048,
        help="vLLM max sequence length (prompt + completion, default: 2048).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s  use_vllm=%s", device, args.use_vllm)

    # ------------------------------------------------------------------
    # Tokenizer — always needed (for prompt formatting and HF-path scoring)
    # Model     — only loaded for HF path; vLLM loads its own copy internally
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer %s @ %s ...", args.model_id, args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    if not args.use_vllm:
        logger.info("Loading HF model %s @ %s  attn=%s ...", args.model_id, args.model_revision, _DEFAULT_ATTN_IMPL)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.model_revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=_DEFAULT_ATTN_IMPL,
        )
        model.eval()
        logger.info("  HF model loaded")

    # ------------------------------------------------------------------
    # Load Level 5 problems from all configs
    # ------------------------------------------------------------------
    logger.info("Loading MATH Level 5 problems from %s ...", HF_DATASET)
    raw_problems: list[dict] = []
    skipped = 0

    for config in CONFIGS:
        for split in args.splits:
            try:
                ds = load_dataset(
                    HF_DATASET, config, split=split, revision=args.hf_revision
                )
            except Exception as e:
                logger.warning("SKIP %s/%s: %s", config, split, e)
                continue
            for row in ds:
                if row["level"] != "Level 5":
                    continue
                answer = extract_boxed(row["solution"])
                if answer is None or r"\text{" in answer:
                    skipped += 1
                    continue
                raw_problems.append({
                    "prompt":       row["problem"],
                    "ground_truth": answer,
                    "topic":        row["type"],
                    "_config":      config,
                    "_split":       split,
                })

    logger.info("  Level 5 problems loaded: %d  (skipped %d no-answer)", len(raw_problems), skipped)

    # Deduplicate by prompt text
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in raw_problems:
        if p["prompt"] not in seen:
            seen.add(p["prompt"])
            deduped.append(p)
    logger.info("  After dedup: %d problems", len(deduped))

    # ------------------------------------------------------------------
    # Filter: keep only problems where pass@n_samples == 0
    # ------------------------------------------------------------------
    logger.info(
        "Filtering: keeping problems where pass@%d = 0 (pretrained model) ...",
        args.n_samples,
    )

    if args.use_vllm:
        n_correct_list = score_all_problems_vllm(
            problems=deduped,
            tokenizer=tokenizer,
            model_id=args.model_id,
            model_revision=args.model_revision,
            n_samples=args.n_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            seed=args.seed,
        )
    else:
        n_correct_list = []
        for prob in tqdm(deduped, desc="scoring", unit="prob"):
            n_correct_list.append(score_problem(
                model=model,
                tokenizer=tokenizer,
                problem_text=prob["prompt"],
                ground_truth=prob["ground_truth"],
                n_samples=args.n_samples,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            ))

    hard_problems: list[dict] = []
    n_solved = 0
    for prob, n_correct in zip(deduped, n_correct_list):
        if n_correct == 0:
            hard_problems.append(prob)
        else:
            n_solved += 1
            logger.debug(
                "  SOLVED (%d/%d): %s...",
                n_correct, args.n_samples, prob["prompt"][:60],
            )

    logger.info(
        "Filter complete: %d hard (pass@%d=0)  /  %d total  (%d solvable excluded)",
        len(hard_problems), args.n_samples, len(deduped), n_solved,
    )

    # ------------------------------------------------------------------
    # Sort, assign IDs, write
    # ------------------------------------------------------------------
    hard_problems.sort(key=lambda r: (r["topic"], r["prompt"]))

    final_records = []
    for i, prob in enumerate(hard_problems):
        final_records.append({
            "problem_id":   f"math_l5_{i:04d}",
            "source_index": i,
            "prompt":       prob["prompt"],
            "ground_truth": prob["ground_truth"],
            "data_source":  "hendrycks_math",
            "topic":        prob["topic"],
            "difficulty":   5,
            "source_level": 5,
        })

    write_jsonl(args.output, final_records)
    sha = sha256_file(args.output)

    logger.info("Wrote %d problems → %s", len(final_records), args.output)
    logger.info("SHA-256: %s", sha)

    manifest = {
        "hf_dataset":               HF_DATASET,
        "hf_revision":              args.hf_revision,
        "splits_used":              args.splits,
        "source_level":             5,
        "filter_model":             args.model_id,
        "filter_model_revision":    args.model_revision,
        "filter_n_samples":         args.n_samples,
        "filter_criterion":         "pass@n_samples == 0",
        "filter_backend":           "vllm" if args.use_vllm else "hf_generate",
        "seed":                     args.seed,
        "raw_level5_count":         len(deduped),
        "solvable_excluded":        n_solved,
        "row_count":                len(final_records),
        "output_path":              str(args.output.relative_to(REPO_ROOT)),
        "sha256":                   sha,
        "library_versions_at_build": pinned_library_versions(),
        "mutual_exclusivity_note":  "Level 5 only; easy pool uses Level 1-4 only. Intersection = empty.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest → %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
