#!/usr/bin/env python3
"""NFR6 gate (v3): latent-space structure check after Phase 0 online pretraining.

Runs the FULL trained Phase 0 pipeline — backbone + trained ZInjector + trained
VAE — to collect z_3 (z_final) for each trajectory.  This is the correct
distribution: the same backbone + injected-prefix path that was trained during
Phase 0.  Using the base backbone without injection (generate_phase0_rollouts.py)
gives repr_h from a different distribution and should NOT be used for this gate.

Pipeline per trajectory:
    Chunk 1: generate(prompt)  → chunk1_ids
             forward([prompt|chunk1]) → repr_1 → z_1 → prefix_1
    Chunk 2: generate([z_pfx1|chunk1]) → chunk2_ids
             forward([z_pfx1|chunk1|chunk2]) → repr_2 → z_2 → prefix_2
    Chunk 3: generate([z_pfx2|chunk2]) → chunk3_ids
             grade → reward
             forward([z_pfx2|chunk2|chunk3]) → repr_3 → z_3   ← collected here

Pass criterion (NFR6): UMAP of z_3 coloured by outcome (green = correct,
red = incorrect) should show visually separable clusters.  Fully mixed = Phase 0
failed to orient z toward outcome quality; do NOT proceed to Phase 1.

Usage
─────
  # Quickest (~10 min on A100): 200 problems × 2 rollouts = 400 trajectories
  python scripts/run_nfr6_gate.py \\
      --config configs/train_latent_grpo.yaml \\
      --n-problems 200 --n-rollouts 2

  # Full diagnostic (richer plot, ~25 min): all 4974 problems × 1 rollout
  python scripts/run_nfr6_gate.py \\
      --config configs/train_latent_grpo.yaml \\
      --n-problems 0                # 0 = all problems in the pool
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.vae_state_encoder import VAEStateEncoder, ZInjector
from src.training.grpo_latent import (
    generate_latent_traces,
    _run_pipeline_with_grad,
    OutcomeHead,
    N_CHUNKS,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config YAML (configs/train_latent_grpo.yaml).",
    )
    p.add_argument(
        "--n-problems",
        type=int,
        default=200,
        help="Number of problems to sample from the easy pool (0 = all). Default: 200.",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=2,
        help="Rollouts per problem (default: 2). Total trajectories = n-problems × n-rollouts.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Problems per inference batch (default: 16). Reduce if OOM.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output plots (default: nfr6.output_dir from config).",
    )
    p.add_argument(
        "--method",
        choices=["tsne", "umap"],
        default=None,
        help="Dim-reduction method (default: nfr6.method from config, or 'umap').",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for problem sampling and dim-reduction (default: 42).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading (mirrors train_latent.py logic — handles extends)
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict[str, Any]:
    import yaml

    def _load_yaml(p: Path) -> dict:
        with p.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    cfg = _load_yaml(path)
    if "extends" in cfg:
        parent_path = path.parent / cfg.pop("extends")
        parent = _load_yaml(parent_path)
        cfg = _deep_merge(parent, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Backbone + Phase 0 checkpoint loading
# ---------------------------------------------------------------------------

def load_phase0_pipeline(
    config: dict[str, Any],
    device: torch.device,
):
    """Load backbone, tokenizer, VAE, ZInjector, OutcomeHead from Phase 0 ckpt."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    primary_cfg = config["primary"]
    latent_cfg  = config["latent_markov"]
    phase0_cfg  = config["phase0"]

    model_id    = primary_cfg["huggingface_repo_id"]
    revision    = primary_cfg.get("revision", "main")
    latent_dim  = int(latent_cfg.get("latent_dim", 64))
    hidden_dim  = int(latent_cfg.get("hidden_dim", 1536))
    ckpt_path   = Path(phase0_cfg.get("checkpoint_path", "runs/latent_grpo/phase0_vae.pt"))

    print(f"Loading backbone {model_id} @ {revision} …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Phase 0 checkpoint from {ckpt_path} …", flush=True)
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=device)

    vae = VAEStateEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    z_injector = ZInjector(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    z_injector.load_state_dict(ckpt["z_injector"])
    z_injector.eval()

    outcome_head = OutcomeHead(latent_dim=latent_dim).to(device)
    outcome_head.load_state_dict(ckpt["outcome_head"])
    outcome_head.eval()

    print(f"  Phase 0 checkpoint loaded (step {ckpt.get('step', '?')})", flush=True)
    return model, tokenizer, vae, z_injector, outcome_head


# ---------------------------------------------------------------------------
# z_3 collection — full trained pipeline, inference only
# ---------------------------------------------------------------------------

def collect_z_finals(
    model,
    tokenizer,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    problems: list[dict],
    config: dict[str, Any],
    device: torch.device,
    n_rollouts: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the full Phase 0 trained pipeline and collect (z_3, reward) pairs.

    Processes `problems` in batches of `batch_size` to avoid OOM.
    Each batch calls generate_latent_traces (no_grad inference) then
    _run_pipeline_with_grad under torch.no_grad() to extract z_3.

    Returns:
        z_finals: float32 array of shape (N, latent_dim).
        labels:   int32 array of shape (N,), values 0 or 1.
    """
    latent_cfg   = config["latent_markov"]
    chunk_tokens = int(latent_cfg.get("chunk_tokens", 341))
    temperature  = float(config.get("evaluation", {}).get("temperature", 1.0))
    top_p        = float(config.get("evaluation", {}).get("top_p", 1.0))

    z_finals_list: list[torch.Tensor] = []
    labels_list:   list[int]          = []

    n_problems  = len(problems)
    n_batches   = (n_problems + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_probs = problems[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # ── Step 1: inference rollout (no_grad internally) ────────────────
        traces = generate_latent_traces(
            model=model,
            tokenizer=tokenizer,
            vae=vae,
            z_injector=z_injector,
            problems=batch_probs,
            n_rollouts=n_rollouts,
            chunk_tokens=chunk_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )

        # ── Step 2: re-run with grad (under no_grad) to get z_3 ───────────
        # _run_pipeline_with_grad re-runs the full pipeline from stored chunk_ids.
        # Under torch.no_grad() this is pure inference — same z_h values as
        # the rollout step since model/vae/z_injector are in eval mode.
        with torch.no_grad():
            pipe = _run_pipeline_with_grad(
                model=model,
                vae=vae,
                z_injector=z_injector,
                traces=traces,
                device=device,
            )

        z_3 = pipe["z_list"][2].detach().cpu()  # (B, latent_dim)

        for i, traj in enumerate(traces):
            z_finals_list.append(z_3[i])
            labels_list.append(int(traj["reward"]))

        n_done    = min((batch_idx + 1) * batch_size, n_problems)
        n_correct = sum(labels_list)
        print(
            f"  batch {batch_idx + 1}/{n_batches}  "
            f"({n_done * n_rollouts} trajectories, "
            f"reward rate {n_correct / len(labels_list):.1%})",
            flush=True,
        )
        torch.cuda.empty_cache()

    z_finals = torch.stack(z_finals_list).numpy().astype("float32")
    labels   = np.array(labels_list, dtype=np.int32)
    return z_finals, labels


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    z_finals: np.ndarray,
    method: str,
    seed: int,
) -> np.ndarray:
    """Reduce z_finals to 2-D via UMAP or t-SNE."""
    np.random.seed(seed)
    if method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=seed)
        coords  = reducer.fit_transform(z_finals)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        tsne   = TSNE(n_components=2, perplexity=30, random_state=seed, max_iter=1000)
        coords = tsne.fit_transform(z_finals)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'umap' or 'tsne'.")
    return coords.astype("float32")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_and_save(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    output_dir: Path,
) -> None:
    """Save scatter plot coloured by outcome and a JSON summary."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        coords[labels == 0, 0], coords[labels == 0, 1],
        c="tab:red",   alpha=0.5, s=12,
        label=f"Incorrect (n={int((labels == 0).sum())})",
    )
    ax.scatter(
        coords[labels == 1, 0], coords[labels == 1, 1],
        c="tab:green", alpha=0.5, s=12,
        label=f"Correct (n={int((labels == 1).sum())})",
    )
    ax.set_title(
        f"Phase 0 Latent Space — {method.upper()} of z_final\n"
        f"NFR6 gate: clusters should be visually separable",
        fontsize=11,
    )
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    out_path = output_dir / f"latent_structure_{method}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {out_path}", flush=True)

    summary = {
        "n_correct":   int((labels == 1).sum()),
        "n_incorrect": int((labels == 0).sum()),
        "reward_rate": float((labels == 1).mean()),
        "method":      method,
        "plot_path":   str(out_path),
    }
    summary_path = output_dir / "nfr6_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary → {summary_path}", flush=True)
    print(
        f"\nNFR6 result: {summary['n_correct']} correct / "
        f"{summary['n_correct'] + summary['n_incorrect']} total "
        f"({summary['reward_rate']:.1%} reward rate)",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = _load_config(args.config)

    nfr6_cfg   = config.get("nfr6", {})
    output_dir = args.output_dir or Path(nfr6_cfg.get("output_dir", "runs/latent_grpo/plots"))
    method     = args.method    or nfr6_cfg.get("method", "umap")

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"NFR6 gate — device: {device}", flush=True)

    # ── Load pipeline ──────────────────────────────────────────────────────
    model, tokenizer, vae, z_injector, outcome_head = load_phase0_pipeline(
        config, device,
    )

    # ── Load easy pool ────────────────────────────────────────────────────
    pool_path = Path(config["phase0"]["pool_path"])
    print(f"Loading pool from {pool_path} …", flush=True)
    problems: list[dict] = []
    with pool_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    print(f"  {len(problems)} problems loaded", flush=True)

    # Sample n_problems (0 = all).
    if args.n_problems > 0 and args.n_problems < len(problems):
        problems = random.sample(problems, args.n_problems)
        print(f"  Sampled {len(problems)} problems (seed={args.seed})", flush=True)

    n_total = len(problems) * args.n_rollouts
    print(
        f"\nCollecting z_3 for {len(problems)} problems × {args.n_rollouts} rollouts "
        f"= {n_total} trajectories …",
        flush=True,
    )

    # ── Collect z_finals ─────────────────────────────────────────────────
    z_finals, labels = collect_z_finals(
        model=model,
        tokenizer=tokenizer,
        vae=vae,
        z_injector=z_injector,
        problems=problems,
        config=config,
        device=device,
        n_rollouts=args.n_rollouts,
        batch_size=args.batch_size,
    )

    # ── Dim reduction ────────────────────────────────────────────────────
    print(
        f"\nRunning {method.upper()} on {len(z_finals)} latent vectors "
        f"(dim={z_finals.shape[1]}) …",
        flush=True,
    )
    coords = reduce_dimensions(z_finals, method=method, seed=args.seed)

    # ── Plot ─────────────────────────────────────────────────────────────
    plot_and_save(coords, labels, method, output_dir)


if __name__ == "__main__":
    main()
