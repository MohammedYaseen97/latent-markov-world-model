"""Phase 0 sanity check — run after pretrain_distill before launching Phase 1.

Three checks:
  1. CE distillation loss on held-out L1–L4 problems is below threshold.
  2. z latent vectors have meaningful variance (encoder isn't collapsed).
  3. Print a sample completion in strict Markov mode to visually verify the
     student can produce coherent text with z-prefix injection.

Pass criteria (from configs/train_latent_grpo.yaml §phase0_sanity):
  - mean_ce_loss      < ce_loss_threshold   (default: 2.0 nats)
  - mean_z_std        > z_variance_threshold (default: 0.1)

Usage:
    python scripts/run_phase0_sanity.py \
        --config configs/train_latent_grpo.yaml

    # Quick smoke run:
    python scripts/run_phase0_sanity.py \
        --config configs/train_latent_grpo_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_yaml_with_extends
from src.models.vae_state_encoder import HIDDEN_DIM, LATENT_DIM, LatentStateEncoder
from src.training.grpo_latent import (
    _distill_loss,
    _fwd,
    _generate_teacher_chunks,
    _last_token_repr,
    _setup_compile,
    format_prompt,
    generate_latent_traces,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 0 sanity check.")
    p.add_argument(
        "--config", type=Path,
        default=REPO_ROOT / "configs" / "train_latent_grpo.yaml",
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the Phase 0 artifact directory (artifacts/<arm>/<run_id>/phase0/).",
    )
    return p.parse_args()


@torch.no_grad()
def _run_distill_loss_check(
    student: AutoModelForCausalLM,
    teacher: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    batch_size: int,
    chunk_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    threshold: float,
) -> float:
    """Evaluate mean CE distillation loss on held-out problems."""
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]
        teacher_chunks = _generate_teacher_chunks(
            teacher, tokenizer, batch,
            chunk_tokens=chunk_tokens,
            temperature=temperature, top_p=top_p,
            device=device,
        )
        loss = _distill_loss(student, encoder, tokenizer, batch, teacher_chunks, device)
        total_loss += loss.item()
        n_batches  += 1

    mean_loss = total_loss / max(n_batches, 1)
    status    = "PASS" if mean_loss < threshold else "FAIL"
    logger.info(
        "[Check 1] distill CE loss: %.4f nats (threshold < %.4f) → %s",
        mean_loss, threshold, status,
    )
    return mean_loss


@torch.no_grad()
def _run_z_variance_check(
    student: AutoModelForCausalLM,
    teacher: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    batch_size: int,
    chunk_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    threshold: float,
) -> float:
    """Check that z_1 vectors have meaningful variance across problems.

    A collapsed encoder produces the same z regardless of input — variance = 0.
    Any mean per-dim std > threshold indicates the encoder is discriminating.
    """
    all_z: list[torch.Tensor] = []

    for i in range(0, len(problems), batch_size):
        batch   = problems[i : i + batch_size]
        B       = len(batch)
        dtype   = next(student.parameters()).dtype

        teacher_chunks = _generate_teacher_chunks(
            teacher, tokenizer, batch,
            chunk_tokens=chunk_tokens,
            temperature=temperature, top_p=top_p,
            device=device,
        )

        prompt_ids_list = [format_prompt(p, tokenizer) for p in batch]
        c1_ids_list     = [teacher_chunks[j][0] for j in range(B)]
        prompt_lens     = [len(p) for p in prompt_ids_list]
        c1_lens         = [c.shape[0] for c in c1_ids_list]
        full1_lens      = [prompt_lens[j] + c1_lens[j] for j in range(B)]
        max_full1       = max(full1_lens)

        fi1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
        fa1 = torch.zeros(B, max_full1, dtype=torch.long, device=device)
        for j in range(B):
            L   = full1_lens[j]
            seq = torch.cat([
                torch.tensor(prompt_ids_list[j], dtype=torch.long),
                c1_ids_list[j],
            ]).to(device)
            fi1[j, :L] = seq
            fa1[j, :L] = 1

        _, hidden1 = _fwd(student, input_ids=fi1, attention_mask=fa1, bypass_lm_head=True)
        last1  = torch.tensor([full1_lens[j] - 1 for j in range(B)], device=device)
        repr_1 = _last_token_repr(hidden1, last1)
        z_1    = encoder.encode(repr_1)
        all_z.append(z_1.cpu())

    z_cat     = torch.cat(all_z, dim=0)              # [N, z_dim]
    z_std     = z_cat.float().std(dim=0).mean().item()
    status    = "PASS" if z_std > threshold else "FAIL"
    logger.info(
        "[Check 2] z mean per-dim std: %.4f (threshold > %.4f) → %s",
        z_std, threshold, status,
    )
    return z_std


@torch.no_grad()
def _run_qualitative_sample(
    student: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    chunk_tokens: int,
    device: torch.device,
) -> None:
    """Generate one sample in strict Markov mode and print first chunk."""
    logger.info("[Check 3] Qualitative sample (strict Markov generation):")
    problem = problems[0]
    traces  = generate_latent_traces(
        model=student, tokenizer=tokenizer, encoder=encoder,
        problems=[problem], n_rollouts=1,
        chunk_tokens=chunk_tokens,
        temperature=1.0, top_p=1.0, device=device,
    )
    if traces:
        chunk1 = tokenizer.decode(traces[0]["chunk_ids"][0], skip_special_tokens=True)
        logger.info("  Problem: %s", problem["prompt"][:120])
        logger.info("  Chunk 1: %s", chunk1[:200])
        logger.info("  Reward:  %d", traces[0]["reward"])


def main() -> None:
    args   = parse_args()
    config = load_yaml_with_extends(args.config.resolve(), root=REPO_ROOT)

    primary      = config["primary"]
    phase0_cfg   = config["phase0"]
    latent_cfg   = config["latent_markov"]
    sanity_cfg   = config.get("phase0_sanity", {})
    training_cfg = config.get("training", {})

    model_id  = primary["huggingface_repo_id"]
    revision  = primary.get("revision", "main")
    dtype     = getattr(torch, primary.get("dtype", "bfloat16"))
    attn_impl = primary.get("attn_implementation", "sdpa")

    n_val   = int(sanity_cfg.get("n_val_problems",         50))
    n_z     = int(sanity_cfg.get("n_z_problems",          100))
    ce_thr  = float(sanity_cfg.get("ce_loss_threshold",    2.0))
    z_thr   = float(sanity_cfg.get("z_variance_threshold", 0.1))

    chunk_tokens = int(latent_cfg.get("chunk_tokens", 341))
    latent_dim   = int(latent_cfg.get("latent_dim",   LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim",   HIDDEN_DIM))
    batch_size   = int(phase0_cfg.get("batch_size",   64))
    temperature  = float(phase0_cfg.get("temperature", 1.0))
    top_p        = float(phase0_cfg.get("top_p",       1.0))

    phase0_dir   = args.checkpoint.resolve()
    enc_ckpt     = phase0_dir / "phase0_encoder.pt"
    backbone_dir = phase0_dir / "backbone"
    pool_path    = Path(phase0_cfg.get("pool_path", "data/math_easy_pool.jsonl"))
    out_path     = phase0_dir / "phase0_sanity.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Phase 0 sanity — device: %s", device)
    logger.info("Checkpoint dir: %s", phase0_dir)

    # ── Load student (Phase 0 trained backbone) ────────────────────────────────
    student_src = str(backbone_dir) if backbone_dir.is_dir() else model_id
    logger.info("Loading student from %s ...", student_src)
    student = AutoModelForCausalLM.from_pretrained(
        student_src, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
    )
    student.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        student_src, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load teacher (original frozen backbone) ────────────────────────────────
    logger.info("Loading teacher (original backbone %s) ...", model_id)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Load encoder ───────────────────────────────────────────────────────────
    encoder = LatentStateEncoder(hidden_dim=hidden_dim, z_dim=latent_dim).to(device)
    if enc_ckpt.is_file():
        ckpt = torch.load(enc_ckpt, weights_only=False, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        logger.info("Encoder loaded from %s (step %d)", enc_ckpt, ckpt.get("step", 0))
    else:
        logger.error("Encoder checkpoint not found: %s", enc_ckpt)
        sys.exit(1)
    encoder.eval()
    _setup_compile(student, encoder)

    # ── Load problem pool ──────────────────────────────────────────────────────
    with open(pool_path, encoding="utf-8") as f:
        all_problems = [json.loads(line) for line in f if line.strip()]
    random.shuffle(all_problems)

    val_problems = all_problems[:n_val]
    z_problems   = all_problems[:n_z]

    # ── Run checks ────────────────────────────────────────────────────────────
    mean_ce  = _run_distill_loss_check(
        student, teacher, encoder, tokenizer,
        val_problems, batch_size=min(batch_size, len(val_problems)),
        chunk_tokens=chunk_tokens, temperature=temperature, top_p=top_p,
        device=device, threshold=ce_thr,
    )

    mean_z_std = _run_z_variance_check(
        student, teacher, encoder, tokenizer,
        z_problems, batch_size=min(batch_size, len(z_problems)),
        chunk_tokens=chunk_tokens, temperature=temperature, top_p=top_p,
        device=device, threshold=z_thr,
    )

    _run_qualitative_sample(student, encoder, tokenizer, val_problems[:1],
                            chunk_tokens, device)

    # ── Report ────────────────────────────────────────────────────────────────
    ce_pass  = mean_ce < ce_thr
    z_pass   = mean_z_std > z_thr
    all_pass = ce_pass and z_pass

    results = {
        "mean_ce_loss":     mean_ce,
        "mean_z_std":       mean_z_std,
        "ce_threshold":     ce_thr,
        "z_threshold":      z_thr,
        "ce_pass":          ce_pass,
        "z_pass":           z_pass,
        "overall_pass":     all_pass,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved → %s", out_path)

    if all_pass:
        logger.info("✓ Phase 0 sanity check PASSED. Safe to proceed to Phase 1.")
        sys.exit(0)
    else:
        failures = [k for k, v in [("CE loss", ce_pass), ("z variance", z_pass)] if not v]
        logger.error("✗ Phase 0 sanity check FAILED: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
