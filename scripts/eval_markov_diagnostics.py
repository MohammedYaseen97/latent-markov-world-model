#!/usr/bin/env python3
"""Post-Phase-1 Markov property diagnostics (E1 — z consistency).

Measures whether the learned z vectors are problem-specific rather than
noisy. For each problem, G rollouts produce G latent trajectories. If z
carries meaningful state, z_final vectors from the same problem should be
more similar to each other than to those from different problems.

E1 — z consistency
    Within-problem mean cosine similarity of z_3 vectors vs cross-problem
    mean cosine similarity.
    A large within/cross gap indicates z is discriminating between problems
    (necessary condition for the Markov state to be meaningful).
    Output: e1_result.json

Usage
─────
  # Standard post-Phase-1 run (use Phase 1 final checkpoint)
  PHASE1_CKPT=$(ls -td artifacts/latent_grpo/*/phase1/final | head -1)
  python scripts/eval_markov_diagnostics.py \\
      --checkpoint "$PHASE1_CKPT" \\
      --config configs/train_latent_grpo.yaml \\
      --output-dir runs/latent_grpo/diagnostics

  # Quick check on Phase 0 encoder (before Phase 1)
  python scripts/eval_markov_diagnostics.py \\
      --checkpoint runs/latent_grpo \\
      --phase0-only \\
      --config configs/train_latent_grpo.yaml \\
      --n-problems 50 \\
      --output-dir runs/latent_grpo/diagnostics_phase0
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.vae_state_encoder import HIDDEN_DIM, LATENT_DIM, LatentStateEncoder
from src.training.grpo_latent import generate_latent_traces, _pipeline_with_grad, _setup_compile
from src.utils.config_loader import load_yaml_with_extends
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help=(
            "Path to a Phase 1 final checkpoint directory (contains backbone/ and "
            "phase1_latent.pt). Use --phase0-only for pre-Phase-1 diagnostics."
        ),
    )
    p.add_argument(
        "--phase0-only", action="store_true",
        help="Load Phase 0 checkpoint (phase0_encoder.pt) from --checkpoint dir.",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--pool-path", type=Path, default=None,
        help="JSONL pool for generating held-out rollouts.",
    )
    p.add_argument("--n-problems", type=int, default=200)
    p.add_argument(
        "--n-rollouts", type=int, default=4,
        help="Rollouts per problem (default: 4; more → better E1 estimate).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "runs" / "latent_grpo" / "diagnostics",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Problems per generate call (default: 4).",
    )
    return p.parse_args()


@torch.no_grad()
def collect_z_vectors(
    model: AutoModelForCausalLM,
    encoder: LatentStateEncoder,
    traces: list[dict],
    device: torch.device,
) -> list[dict]:
    """Re-run the strict Markov pipeline to extract z_1, z_2, z_3 per trace.

    Returns list of dicts with keys: reward, z_1, z_2, z_3 (CPU tensors).
    """
    pipe = _pipeline_with_grad(model, encoder, traces, device)

    result: list[dict] = []
    for i, trace in enumerate(traces):
        result.append({
            "reward": trace["reward"],
            "z_1":    pipe["z_list"][0][i].cpu(),
            "z_2":    pipe["z_list"][1][i].cpu(),
            "z_3":    pipe["z_list"][2][i].cpu(),
        })
    return result


def eval_e1_z_consistency(
    traj_data: list[dict],
    n_rollouts: int,
) -> dict:
    """E1: measure within-problem vs cross-problem z_3 cosine similarity.

    Groups trajectories by problem (every n_rollouts consecutive traces
    belong to the same problem, matching generate_latent_traces ordering).

    Returns dict with:
        within_sim:  mean cosine sim of z_3 pairs from the same problem
        cross_sim:   mean cosine sim of z_3 pairs from different problems
        delta:       within_sim - cross_sim  (higher = more discriminating)
    """
    N = len(traj_data)
    n_probs = N // n_rollouts

    # Group z_3 by problem
    problem_z3: list[torch.Tensor] = []
    for p in range(n_probs):
        vecs = torch.stack([
            traj_data[p * n_rollouts + r]["z_3"]
            for r in range(n_rollouts)
        ])  # [G, z_dim]
        problem_z3.append(F.normalize(vecs.float(), dim=-1))

    # Within-problem: all pairs within each problem
    within_sims: list[float] = []
    for vecs in problem_z3:
        if vecs.shape[0] < 2:
            continue
        sim_mat = vecs @ vecs.T                     # [G, G]
        mask    = ~torch.eye(vecs.shape[0], dtype=torch.bool)
        within_sims.extend(sim_mat[mask].tolist())

    # Cross-problem: sample pairs from different problems
    cross_sims: list[float] = []
    if len(problem_z3) > 1:
        all_z3 = torch.cat(problem_z3, dim=0)       # [N, z_dim]
        labels  = torch.cat([
            torch.full((vecs.shape[0],), p, dtype=torch.long)
            for p, vecs in enumerate(problem_z3)
        ])                                           # [N]
        # Random sample of cross pairs
        rng = random.Random(0)
        indices = list(range(len(all_z3)))
        for _ in range(min(5000, len(all_z3) ** 2)):
            a, b = rng.sample(indices, 2)
            if labels[a] != labels[b]:
                cross_sims.append((all_z3[a] @ all_z3[b]).item())

    within = sum(within_sims) / max(len(within_sims), 1)
    cross  = sum(cross_sims)  / max(len(cross_sims),  1)
    delta  = within - cross

    return {
        "within_problem_z3_sim": round(within, 4),
        "cross_problem_z3_sim":  round(cross,  4),
        "delta":                 round(delta,  4),
        "n_within_pairs":        len(within_sims),
        "n_cross_pairs":         len(cross_sims),
        "interpretation": (
            "PASS (z is discriminating)"  if delta > 0.05 else
            "BORDERLINE"                  if delta > 0.0  else
            "FAIL (z collapses to noise)"
        ),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    config       = load_yaml_with_extends(args.config.resolve(), root=REPO_ROOT)
    primary_cfg  = config["primary"]
    latent_cfg   = config["latent_markov"]
    phase0_cfg   = config["phase0"]
    training_cfg = config["training"]

    model_id     = primary_cfg["huggingface_repo_id"]
    revision     = primary_cfg.get("revision", "main")
    dtype        = getattr(torch, primary_cfg.get("dtype", "bfloat16"))
    attn_impl    = primary_cfg.get("attn_implementation", "sdpa")
    chunk_tokens = int(latent_cfg.get("chunk_tokens", 341))
    latent_dim   = int(latent_cfg.get("latent_dim",   LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim",   HIDDEN_DIM))
    temperature  = float(training_cfg.get("temperature", 1.0))
    top_p        = float(training_cfg.get("top_p",        1.0))

    pool_path = args.pool_path or Path(
        phase0_cfg.get("pool_path", "data/math_easy_pool.jsonl")
    )

    # ── Load backbone ──────────────────────────────────────────────────────────
    if args.phase0_only:
        backbone_src = model_id
        print(f"Loading pretrained backbone {model_id} ...", flush=True)
    else:
        backbone_src = str(args.checkpoint / "backbone")
        print(f"Loading Phase 1 backbone from {backbone_src} ...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        backbone_src, torch_dtype=dtype, device_map="auto",
        attn_implementation=attn_impl,
    )
    model.eval()

    tok_src   = backbone_src if not args.phase0_only else model_id
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, revision=None if not args.phase0_only else revision,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  backbone ready", flush=True)

    # ── Load encoder ───────────────────────────────────────────────────────────
    enc_ckpt = args.checkpoint / (
        "phase0_encoder.pt" if args.phase0_only else "phase1_latent.pt"
    )
    print(f"Loading encoder from {enc_ckpt} ...", flush=True)
    ckpt    = torch.load(str(enc_ckpt), weights_only=False, map_location=device)
    encoder = LatentStateEncoder(hidden_dim=hidden_dim, z_dim=latent_dim).to(device=device, dtype=dtype)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    _setup_compile(model, encoder)
    print("  encoder loaded", flush=True)

    # ── Load pool ─────────────────────────────────────────────────────────────
    with open(pool_path, encoding="utf-8") as f:
        all_problems = [json.loads(line) for line in f if line.strip()]
    print(f"Pool: {len(all_problems)} problems from {pool_path}", flush=True)

    random.seed(99)
    sampled = random.sample(all_problems, min(args.n_problems, len(all_problems)))
    print(f"Sampled {len(sampled)} problems", flush=True)

    # ── Generate traces and collect z vectors ─────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_traj_data: list[dict] = []
    n_batches = -(-len(sampled) // args.batch_size)   # ceil

    print(
        f"\nGenerating {args.n_rollouts} rollouts × {len(sampled)} problems "
        f"in {n_batches} batches ...",
        flush=True,
    )

    for bi in range(n_batches):
        batch_probs = sampled[bi * args.batch_size:(bi + 1) * args.batch_size]
        with torch.no_grad():
            traces = generate_latent_traces(
                model=model, tokenizer=tokenizer, encoder=encoder,
                problems=batch_probs, n_rollouts=args.n_rollouts,
                chunk_tokens=chunk_tokens,
                temperature=temperature, top_p=top_p, device=device,
            )
            traj_data = collect_z_vectors(model, encoder, traces, device)
        all_traj_data.extend(traj_data)
        n_correct = sum(t["reward"] for t in all_traj_data)
        print(
            f"  batch {bi + 1}/{n_batches}  "
            f"({len(all_traj_data)} trajectories, "
            f"reward {n_correct / len(all_traj_data):.1%})",
            flush=True,
        )

    print(f"\nTotal: {len(all_traj_data)} trajectories", flush=True)

    # ── E1 — z consistency ────────────────────────────────────────────────────
    print("\nE1 — z consistency ...", flush=True)
    e1 = eval_e1_z_consistency(all_traj_data, n_rollouts=args.n_rollouts)
    print(
        f"  within_sim={e1['within_problem_z3_sim']:.4f}  "
        f"cross_sim={e1['cross_problem_z3_sim']:.4f}  "
        f"delta={e1['delta']:.4f}  → {e1['interpretation']}",
        flush=True,
    )
    (args.output_dir / "e1_result.json").write_text(json.dumps(e1, indent=2))

    summary = {**e1, "n_problems": len(sampled), "n_rollouts": args.n_rollouts}
    (args.output_dir / "markov_diagnostics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nDone. → {args.output_dir}/markov_diagnostics.json", flush=True)


if __name__ == "__main__":
    main()
