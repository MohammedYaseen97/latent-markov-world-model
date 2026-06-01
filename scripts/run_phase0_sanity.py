"""Phase 0 sanity check — run after pretrain_distill before launching Phase 1.

Objective
---------
Phase 0 trains the student (LoRA backbone + encoder + z-injector) to generate
full 3-chunk reasoning sequences on L1-L4 problems by conditioning only on the
z-prefix, without ever seeing the preceding chunks directly.  The canonical
success criterion is:

    student(z_prefix) reward rate  ≥  teacher(full context) × coverage_threshold

If the student can recover a meaningful fraction of the teacher's reward rate
using *only* the compressed latent state z, the z-prefix mechanism works and
Phase 1 GRPO on L5 problems has a viable gradient signal.

Check structure
---------------
  1. [GATE]  CE distillation loss on held-out L1-L4 problems is below
             threshold.  Fast training-health proxy.  Low CE is necessary
             but not sufficient: teacher-forced loss is blind to exposure
             bias — the model can ace CE while generating incoherently.

  2. [GATE]  z latent vectors have meaningful per-dim variance (encoder isn't
             collapsed).  Zero variance = encoder is dead; catches catastrophic
             failure cheaply.

  3. [INFO]  z-transition diagnostic: temporal_delta (how much z changes
             between chunks of the same problem) vs. cross_variation (how
             much z_1 varies across problems).  temporal_delta > cross_variation
             means z tracks solution-state transitions.  Informational only —
             not a gate because structured z can coexist with broken generation
             and vice-versa.

  4. [GATE]  Teacher reward baseline: run teacher (standard generation, no z)
             on the same easy problems × K rollouts to establish what the
             upper-bound reward rate looks like.  This anchors check 5.

  5. [GATE]  Student vs. teacher reward rate: run student in strict Markov mode
             (z-prefix only, no chunk context) on the same problems.  Gate:
               student_rate ≥ teacher_rate × coverage_threshold   (default 0.30)
               student_rate ≥ min_student_reward                  (default 0.10)
             Also reports chunk-1 crutch rate: what fraction of student wins
             are carried by chunk 1 alone (answer already boxed before chunk 2).
             If > 50% the backbone is ignoring z for chunks 2+3 — still broken.

  6. [INFO]  Qualitative samples: full decoded 3-chunk output per easy problem.
             Indispensable for visually confirming chunks 2+3 are coherent math.

Pass criteria (from configs/train_latent_grpo.yaml §phase0_sanity):
  - mean_ce_loss                    < ce_loss_threshold          (default: 2.0 nats)
  - mean_z_std                      > z_variance_threshold       (default: 0.1)
  - student_rate                   ≥ teacher_rate × coverage_threshold  (default: 0.30)
  - student_rate                   ≥ min_student_reward          (default: 0.10)

Usage:
    python scripts/run_phase0_sanity.py \
        --config configs/train_latent_grpo.yaml \
        --checkpoint artifacts/latent_grpo/<run_id>/phase0

    # Quick smoke run:
    python scripts/run_phase0_sanity.py \
        --config configs/train_latent_grpo_smoke.yaml \
        --checkpoint artifacts/latent_grpo/<run_id>/phase0
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
from src.training.grpo_baseline import answers_equivalent, extract_answer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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
def _run_z_transition_check(
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
    margin: float,
) -> tuple[float, float, float]:
    """Check that z tracks solution-state transitions within a problem.

    For each problem we compute z_1 (after chunk 1) and z_2 (after chunk 2).

    temporal_delta   = 1 − mean cos_sim(z_1_i, z_2_i)
        How much z changes between consecutive chunks of the same problem.

    cross_variation  = 1 − mean cos_sim(z_1_i, z_1_j)  for i ≠ j
        How much z_1 varies across problems at the same stage.

    If z is tracking solution state, temporal_delta > cross_variation:
    the state transition signal dominates over between-problem noise.
    Gate: temporal_delta − cross_variation > margin.
    """
    z1_list: list[torch.Tensor] = []
    z2_list: list[torch.Tensor] = []

    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]
        B     = len(batch)

        teacher_chunks = _generate_teacher_chunks(
            teacher, tokenizer, batch,
            chunk_tokens=chunk_tokens,
            temperature=temperature, top_p=top_p,
            device=device,
        )

        prompt_ids_list = [format_prompt(p, tokenizer) for p in batch]

        # ── z_1: after prompt + chunk 1 ───────────────────────────────────────
        c1_ids_list = [teacher_chunks[j][0] for j in range(B)]
        full1_lens  = [len(prompt_ids_list[j]) + c1_ids_list[j].shape[0] for j in range(B)]
        max_f1      = max(full1_lens)
        fi1 = torch.zeros(B, max_f1, dtype=torch.long, device=device)
        fa1 = torch.zeros(B, max_f1, dtype=torch.long, device=device)
        for j in range(B):
            seq = torch.cat([
                torch.tensor(prompt_ids_list[j], dtype=torch.long),
                c1_ids_list[j],
            ]).to(device)
            fi1[j, :full1_lens[j]] = seq
            fa1[j, :full1_lens[j]] = 1
        _, h1 = _fwd(student, input_ids=fi1, attention_mask=fa1, bypass_lm_head=True)
        last1 = torch.tensor([l - 1 for l in full1_lens], device=device)
        z1_list.append(encoder.encode(_last_token_repr(h1, last1)).cpu())

        # ── z_2: after prompt + chunk 1 + chunk 2 ────────────────────────────
        c2_ids_list = [teacher_chunks[j][1] if len(teacher_chunks[j]) > 1
                       else teacher_chunks[j][0] for j in range(B)]
        full2_lens  = [full1_lens[j] + c2_ids_list[j].shape[0] for j in range(B)]
        max_f2      = max(full2_lens)
        fi2 = torch.zeros(B, max_f2, dtype=torch.long, device=device)
        fa2 = torch.zeros(B, max_f2, dtype=torch.long, device=device)
        for j in range(B):
            seq = torch.cat([fi1[j, :full1_lens[j]], c2_ids_list[j].to(device)])
            fi2[j, :full2_lens[j]] = seq
            fa2[j, :full2_lens[j]] = 1
        _, h2 = _fwd(student, input_ids=fi2, attention_mask=fa2, bypass_lm_head=True)
        last2 = torch.tensor([l - 1 for l in full2_lens], device=device)
        z2_list.append(encoder.encode(_last_token_repr(h2, last2)).cpu())

    z1 = torch.cat(z1_list, dim=0).float()   # [N, z_dim]
    z2 = torch.cat(z2_list, dim=0).float()   # [N, z_dim]

    z1n = torch.nn.functional.normalize(z1, dim=-1)
    z2n = torch.nn.functional.normalize(z2, dim=-1)

    within_sim     = (z1n * z2n).sum(dim=-1).mean().item()
    temporal_delta = 1.0 - within_sim

    sim_mat        = z1n @ z1n.T
    N              = z1.shape[0]
    off_diag       = ~torch.eye(N, dtype=torch.bool)
    cross_sim      = sim_mat[off_diag].mean().item()
    cross_variation = 1.0 - cross_sim

    gap    = temporal_delta - cross_variation
    status = "PASS" if gap > margin else "FAIL"
    logger.info(
        "[Check 3] z-transition: temporal_delta=%.4f  cross_variation=%.4f  gap=%.4f"
        " (threshold > %.4f) → %s",
        temporal_delta, cross_variation, gap, margin, status,
    )
    return temporal_delta, cross_variation, gap


@torch.no_grad()
def _run_teacher_reward_check(
    teacher: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    max_new_tokens: int,
    n_rollouts: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> float:
    """Establish teacher reward baseline on easy problems.

    The teacher generates in standard autoregressive mode (no chunking, full
    context) to produce the upper-bound reward rate that the student should
    approach.  This anchors the student coverage gate in check 5.

    Returns teacher_rate: fraction of (problem × rollout) pairs graded correct.
    """
    correct = 0
    total   = 0

    for problem in problems:
        prompt_ids = format_prompt(problem, tokenizer)
        input_ids  = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        for _ in range(n_rollouts):
            out = teacher.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = out[0, input_ids.shape[1]:]
            text      = tokenizer.decode(generated, skip_special_tokens=True)
            pred      = extract_answer(text)
            total    += 1
            if pred is not None and answers_equivalent(pred, problem["ground_truth"]):
                correct += 1

    teacher_rate = correct / max(total, 1)
    logger.info(
        "[Check 4] teacher reward rate (standard generation): %.4f  (%d/%d correct)",
        teacher_rate, correct, total,
    )
    return teacher_rate


@torch.no_grad()
def _run_student_vs_teacher_check(
    student: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    chunk_tokens: int,
    n_rollouts: int,
    device: torch.device,
    teacher_rate: float,
    coverage_threshold: float,
    min_student_reward: float,
) -> tuple[float, float]:
    """Check student reward rate against teacher baseline in strict Markov mode.

    The student generates all three chunks using only the z-prefix — no chunk
    context.  We grade the full concatenated 3-chunk output and also track the
    chunk-1 crutch rate: the fraction of wins where the answer already appears
    in chunk 1 alone.  A high crutch rate means chunks 2+3 are still ignored,
    which means z_prefix conditioning is not working for the later chunks.

    Gate:
      student_rate ≥ teacher_rate × coverage_threshold   (default 0.30)
      student_rate ≥ min_student_reward                  (default 0.10)

    Returns (student_rate, chunk1_crutch_rate).
    """
    correct        = 0
    chunk1_correct = 0
    total          = 0

    for problem in problems:
        traces = generate_latent_traces(
            model=student, tokenizer=tokenizer, encoder=encoder,
            problems=[problem], n_rollouts=n_rollouts,
            chunk_tokens=chunk_tokens,
            temperature=1.0, top_p=1.0, device=device,
        )
        for trace in traces:
            total += 1
            chunk_ids = trace["chunk_ids"]

            full_text = tokenizer.decode(
                [tok for chunk in chunk_ids for tok in chunk.tolist()],
                skip_special_tokens=True,
            )
            pred = extract_answer(full_text)
            if pred is not None and answers_equivalent(pred, problem["ground_truth"]):
                correct += 1
                # Check whether chunk 1 alone already contains the answer
                c1_text   = tokenizer.decode(chunk_ids[0].tolist(), skip_special_tokens=True)
                c1_pred   = extract_answer(c1_text)
                if c1_pred is not None and answers_equivalent(c1_pred, problem["ground_truth"]):
                    chunk1_correct += 1

    student_rate      = correct / max(total, 1)
    chunk1_crutch_rate = chunk1_correct / max(correct, 1)

    coverage_target = teacher_rate * coverage_threshold
    rate_pass       = (
        student_rate >= coverage_target and
        student_rate >= min_student_reward
    )
    status = "PASS" if rate_pass else "FAIL"

    logger.info(
        "[Check 5] student reward rate (Markov): %.4f  (%d/%d correct) → %s",
        student_rate, correct, total, status,
    )
    logger.info(
        "         teacher rate: %.4f  coverage needed: %.4f (%.0f%% of teacher)  min floor: %.4f",
        teacher_rate, coverage_target, coverage_threshold * 100, min_student_reward,
    )
    crutch_warning = "  ⚠ chunk-1 crutch" if chunk1_crutch_rate > 0.5 else ""
    logger.info(
        "         chunk-1 crutch rate: %.4f (wins from chunk-1 alone / total wins)%s",
        chunk1_crutch_rate, crutch_warning,
    )
    return student_rate, chunk1_crutch_rate


@torch.no_grad()
def _run_qualitative_samples(
    student: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    tokenizer: AutoTokenizer,
    problems: list[dict],
    chunk_tokens: int,
    device: torch.device,
    n_samples: int = 4,
) -> list[dict]:
    """Generate n_samples full 3-chunk traces in strict Markov mode.

    Returns a list of dicts suitable for JSON serialisation, each containing
    the problem prompt, all three decoded chunks, and the reward.
    """
    logger.info("[Qualitative] Generating %d full 3-chunk samples …", n_samples)
    records: list[dict] = []

    for i, problem in enumerate(problems[:n_samples]):
        # Generate each problem independently — avoids batch index aliasing
        traces = generate_latent_traces(
            model=student, tokenizer=tokenizer, encoder=encoder,
            problems=[problem], n_rollouts=1,
            chunk_tokens=chunk_tokens,
            temperature=1.0, top_p=1.0, device=device,
        )
        if not traces:
            continue
        trace = traces[0]
        chunks_decoded = [
            tokenizer.decode(chunk_ids, skip_special_tokens=True)
            for chunk_ids in trace["chunk_ids"]
        ]
        record = {
            "problem":  problem["prompt"],
            "chunks":   chunks_decoded,
            "reward":   int(trace["reward"]),
        }
        records.append(record)
        logger.info("  [Sample %d] problem: %s", i + 1, problem["prompt"][:100])
        for j, chunk_text in enumerate(chunks_decoded):
            logger.info("    chunk %d (%d tok): %s", j + 1, len(trace["chunk_ids"][j]), chunk_text[:180])
        logger.info("    reward: %d", record["reward"])

    return records


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

    n_val               = int(sanity_cfg.get("n_val_problems",           50))
    n_z                 = int(sanity_cfg.get("n_z_problems",            100))
    ce_thr              = float(sanity_cfg.get("ce_loss_threshold",      2.0))
    z_thr               = float(sanity_cfg.get("z_variance_threshold",   0.1))
    z_trans_thr         = float(sanity_cfg.get("z_transition_margin",   0.02))
    n_reward_problems   = int(sanity_cfg.get("n_reward_problems",        20))
    teacher_rollouts    = int(sanity_cfg.get("teacher_reward_rollouts",   4))
    student_rollouts    = int(sanity_cfg.get("student_reward_rollouts",   4))
    coverage_threshold  = float(sanity_cfg.get("coverage_threshold",    0.30))
    min_student_reward  = float(sanity_cfg.get("min_student_reward",    0.10))

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

    # ── Load student (Phase 0 trained backbone, merge LoRA if present) ────────
    adapter_cfg = backbone_dir / "adapter_config.json"
    if backbone_dir.is_dir() and adapter_cfg.is_file():
        logger.info("Phase 0 LoRA adapter found — loading base + merging: %s", backbone_dir)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision,
            torch_dtype=dtype, device_map="auto",
            attn_implementation=attn_impl,
        )
        student = PeftModel.from_pretrained(base, str(backbone_dir))
        student = student.merge_and_unload()
        logger.info("  LoRA merged into student backbone")
    else:
        student_src = str(backbone_dir) if backbone_dir.is_dir() else model_id
        logger.info("Loading student from %s ...", student_src)
        student = AutoModelForCausalLM.from_pretrained(
            student_src, torch_dtype=dtype, device_map="auto",
            attn_implementation=attn_impl,
        )
    student.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
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
    encoder = LatentStateEncoder(hidden_dim=hidden_dim, z_dim=latent_dim).to(device=device, dtype=dtype)
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

    val_problems    = all_problems[:n_val]
    z_problems      = all_problems[:n_z]
    reward_problems = all_problems[:n_reward_problems]

    # ── [Check 1] CE distillation loss ────────────────────────────────────────
    mean_ce = _run_distill_loss_check(
        student, teacher, encoder, tokenizer,
        val_problems, batch_size=min(batch_size, len(val_problems)),
        chunk_tokens=chunk_tokens, temperature=temperature, top_p=top_p,
        device=device, threshold=ce_thr,
    )

    # ── [Check 2] z encoder variance ──────────────────────────────────────────
    mean_z_std = _run_z_variance_check(
        student, teacher, encoder, tokenizer,
        z_problems, batch_size=min(batch_size, len(z_problems)),
        chunk_tokens=chunk_tokens, temperature=temperature, top_p=top_p,
        device=device, threshold=z_thr,
    )

    # ── [Diagnostic] z-transition (informational, not a gate) ─────────────────
    z_tdelta, z_cvar, z_gap = _run_z_transition_check(
        student, teacher, encoder, tokenizer,
        z_problems, batch_size=min(batch_size, len(z_problems)),
        chunk_tokens=chunk_tokens, temperature=temperature, top_p=top_p,
        device=device, margin=z_trans_thr,
    )

    # ── [Check 4] Teacher reward baseline ─────────────────────────────────────
    teacher_rate = _run_teacher_reward_check(
        teacher, tokenizer,
        reward_problems,
        max_new_tokens=chunk_tokens * 3,
        n_rollouts=teacher_rollouts,
        temperature=temperature, top_p=top_p,
        device=device,
    )

    # ── [Check 5] Student vs teacher reward rate ──────────────────────────────
    student_rate, chunk1_crutch_rate = _run_student_vs_teacher_check(
        student, encoder, tokenizer,
        reward_problems,
        chunk_tokens=chunk_tokens,
        n_rollouts=student_rollouts,
        device=device,
        teacher_rate=teacher_rate,
        coverage_threshold=coverage_threshold,
        min_student_reward=min_student_reward,
    )

    # ── [Qualitative] sample 3-chunk traces ───────────────────────────────────
    qualitative = _run_qualitative_samples(student, encoder, tokenizer, val_problems,
                                           chunk_tokens, device, n_samples=4)

    # ── Report ────────────────────────────────────────────────────────────────
    ce_pass      = mean_ce < ce_thr
    z_pass       = mean_z_std > z_thr
    student_pass = (
        student_rate >= teacher_rate * coverage_threshold and
        student_rate >= min_student_reward
    )
    # z-transition is informational: included in output but NOT in gate
    all_pass = ce_pass and z_pass and student_pass

    results = {
        # ── gates ──
        "mean_ce_loss":            mean_ce,
        "mean_z_std":              mean_z_std,
        "teacher_reward_rate":     teacher_rate,
        "student_reward_rate":     student_rate,
        "coverage_ratio":          student_rate / max(teacher_rate, 1e-6),
        "chunk1_crutch_rate":      chunk1_crutch_rate,
        # ── diagnostic (not a gate) ──
        "z_temporal_delta":        z_tdelta,
        "z_cross_variation":       z_cvar,
        "z_transition_gap":        z_gap,
        # ── thresholds ──
        "ce_threshold":            ce_thr,
        "z_threshold":             z_thr,
        "coverage_threshold":      coverage_threshold,
        "min_student_reward":      min_student_reward,
        "z_transition_margin":     z_trans_thr,
        # ── pass/fail ──
        "ce_pass":                 ce_pass,
        "z_pass":                  z_pass,
        "student_vs_teacher_pass": student_pass,
        "z_transition_info":       z_gap > z_trans_thr,
        "overall_pass":            all_pass,
        # ── qualitative ──
        "qualitative_samples":     qualitative,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved → %s", out_path)

    logger.info(
        "──── Summary ────────────────────────────────────────────────────────\n"
        "  [1] CE loss:           %.4f nats    (< %.4f)  → %s\n"
        "  [2] z variance:        %.4f         (> %.4f)  → %s\n"
        "  [D] z-transition gap:  %.4f         (> %.4f)  → %s (info only)\n"
        "  [4] teacher rate:      %.4f\n"
        "  [5] student rate:      %.4f  coverage: %.1f%%  crutch: %.1f%% → %s\n"
        "  Overall: %s",
        mean_ce,      ce_thr,      "PASS" if ce_pass      else "FAIL",
        mean_z_std,   z_thr,       "PASS" if z_pass       else "FAIL",
        z_gap,        z_trans_thr, "PASS" if z_gap > z_trans_thr else "FAIL",
        teacher_rate,
        student_rate, results["coverage_ratio"] * 100, chunk1_crutch_rate * 100,
                                   "PASS" if student_pass else "FAIL",
        "PASS ✓" if all_pass else "FAIL ✗",
    )

    if all_pass:
        logger.info("Phase 0 sanity PASSED — safe to proceed to Phase 1.")
        sys.exit(0)
    else:
        failures = [k for k, v in [
            ("CE loss",              ce_pass),
            ("z variance",          z_pass),
            ("student vs teacher",  student_pass),
        ] if not v]
        logger.error("Phase 0 sanity FAILED: %s", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
