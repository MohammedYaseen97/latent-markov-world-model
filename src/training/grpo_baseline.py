"""Baseline GRPO training scaffold."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from math_verify import parse as mv_parse, verify as mv_verify


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in text, handling nested braces."""
    positions = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not positions:
        return None
    start = positions[-1] + len(r"\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def _extract_last_number(text: str) -> str | None:
    """Return the last integer, decimal, or simple fraction (a/b) in text."""
    hits = re.findall(r"-?\d+(?:[./]\d+)?", text)
    return hits[-1] if hits else None


def extract_answer(text: str) -> str | None:
    """Extract the final answer from a model completion.

    Priority:
    1. Last ``\\boxed{...}`` — standard competition-math format; Qwen Instruct follows this.
    2. Last number / fraction in the text — fallback for bare numeric answers.
    """
    return _extract_boxed(text) or _extract_last_number(text)


# ---------------------------------------------------------------------------
# Symbolic equivalence
# ---------------------------------------------------------------------------

def answers_equivalent(pred: str, gold: str) -> bool:
    """True when pred and gold represent the same mathematical value.

    Uses math-verify (HuggingFace), built specifically for LLM math evaluation.
    Handles LaTeX, fractions, radicals, symbolic and numeric equivalence.
    math-verify's own signal.alarm() timeout prevents hangs on pathological
    inputs — it must run on the main thread (signal.alarm requires it).
    Falls back to normalised string equality on any failure.
    """
    try:
        return bool(mv_verify(mv_parse(pred), mv_parse(gold)))
    except Exception:
        return pred.strip().lower() == gold.strip().lower()


# ---------------------------------------------------------------------------
# Reward function (TRL-compatible)
# ---------------------------------------------------------------------------

def _completion_to_str(completion) -> str:
    """Normalise a completion to a plain string.

    Newer TRL versions pass completions as a list of chat-message dicts
    (``[{"role": "assistant", "content": "..."}]``).  Older versions pass
    plain strings.  This helper handles both.
    """
    if isinstance(completion, str):
        return completion
    # List[dict] – concatenate all message content in order.
    return "".join(msg.get("content", "") for msg in completion if isinstance(msg, dict))


def math_reward(completions, answer: list[str], **kwargs) -> list[float]:
    """Binary reward: 1.0 if the extracted final answer is mathematically equivalent to
    ``ground_truth``, 0.0 otherwise.  Same function used across all four arms (contract).

    Args:
        completions: Raw model completions for a batch (str or list[dict]).
        answer:      Corresponding ``ground_truth`` strings from the JSONL pool.
    """
    rewards = []
    for completion, ground_truth in zip(completions, answer):
        pred = extract_answer(_completion_to_str(completion))
        rewards.append(1.0 if pred is not None and answers_equivalent(pred, ground_truth) else 0.0)
    return rewards


def make_logged_reward(run_dir: Path):
    """Wrap math_reward to SHA-256-hash each step's completions into completions_hashes.jsonl.

    The hash covers the full text of every completion in the batch, in order, so any
    difference in generated tokens — even a single token — will flip the hash.
    """
    log_path = run_dir / "completions_hashes.jsonl"
    step: list[int] = [0]

    def _reward(completions, answer: list[str], **kwargs) -> list[float]:
        step[0] += 1
        strs = [_completion_to_str(c) for c in completions]
        # Null-byte separator prevents cross-boundary collisions between completions.
        digest = hashlib.sha256("\n\x00\n".join(strs).encode()).hexdigest()
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step[0], "completions_hash": digest}) + "\n")
        return math_reward(completions, answer, **kwargs)

    return _reward


SYSTEM_PROMPT = (
    "Please reason step by step. You may think as long as you need to. "
    "Put your final answer inside \\boxed{answer}."
)


def map_keys(ex):
    return {
        # Chat-message format — TRL applies the model's chat template automatically.
        # The system prompt instructs the model to reason freely and end with \boxed{}.
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ex["prompt"]},
        ],
        "answer": ex["ground_truth"],  # passed as kwarg to reward function
    }


def train_baseline(config: dict[str, Any], run_dir: Path) -> None:
    """Run the history-as-state GRPO baseline (Arm 1 of the four-arm ablation).

    The model receives the full conversation history as its only state representation —
    no latent bottleneck, no token-Markov head. This is the vanilla GRPO baseline against
    which the other arms are compared.

    Dispatch:
    - ``smoke`` profile: 4-bit QLoRA on 8 GB GPU, 4 steps, 64-token completions.
    - ``full`` profile: bfloat16 on A100 80 GB, ``num_epochs`` passes over the pool.

    Checkpoints and per-step metrics are written under ``run_dir``.
    See ``configs/train_baseline_grpo.yaml`` and ``configs/final_parity/base_parity.yaml``
    for the frozen shared hyperparameters used across all arms.
    """

    is_smoke = (config.get("experiment") or {}).get("profile") == "smoke"

    primary = config["primary"]
    training = config["training"]
    model_id = primary["huggingface_repo_id"]
    tokenizer_id = primary["tokenizer_repo_id"]
    revision = primary.get("revision")
    dtype = getattr(torch, primary["dtype"])

    # Smoke: QLoRA (4-bit base + LoRA adapters) to fit on 8 GB.
    # Full: native dtype, no quantization.
    if is_smoke:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            quantization_config=bnb_config,
            device_map="auto",
        )
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            target_modules="all-linear",
            lora_dropout=0.05,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype,
            device_map="auto",
        )
        peft_config = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Ensure checkpoints are saved with Qwen2TokenizerFast so vLLM can load them
    # without hitting "Qwen2Tokenizer has no attribute all_special_tokens_extended".
    if getattr(tokenizer, "init_kwargs", {}).get("tokenizer_class") == "Qwen2Tokenizer":
        tokenizer.init_kwargs["tokenizer_class"] = "Qwen2TokenizerFast"

    raw_dataset = load_dataset(
        "json",
        data_files=config["evaluation"]["path"],
        split="train",
    )
    # map_keys adds 'answer' and updates 'prompt'; remove everything else
    dataset = raw_dataset.map(
        map_keys,
        remove_columns=[c for c in raw_dataset.column_names if c != "prompt"],
    )

    grpo_kwargs: dict[str, Any] = {}
    if training["use_vllm"]:
        grpo_kwargs["vllm_gpu_memory_utilization"] = 0.3

    args = GRPOConfig(
        output_dir=str(run_dir),
        seed=training["seed"],
        learning_rate=training["learning_rate"],
        bf16=not is_smoke and primary["dtype"] == "bfloat16",
        per_device_train_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["grad_accum_steps"],
        gradient_checkpointing=training["gradient_checkpointing"],
        num_generations=training["num_generations"],
        max_completion_length=training["max_completion_length"],
        temperature=training.get("temperature", 1.0),
        top_p=training.get("top_p", 1.0),
        logging_steps=training.get("logging_steps", 1),
        save_steps=training.get("save_steps", 5),
        max_steps=training.get("max_steps") or -1,
        num_train_epochs=training.get("num_epochs") or 1,
        use_vllm=training["use_vllm"],
        **grpo_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_logged_reward(run_dir),
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
