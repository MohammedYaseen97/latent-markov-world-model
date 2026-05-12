#!/usr/bin/env python3
"""NFR6 gate: visualise latent space structure after Phase 0 VAE pretraining.

Runs the trained VAE encoder on all saved Phase 0 rollout trajectories,
collects z_final (z_3 from the last chunk), and produces a 2-D scatter plot
coloured by trajectory outcome (correct = green, incorrect = red).

Pass criteria (NFR6): correct and incorrect trajectories must form visually
separable clusters. If they are fully mixed, Phase 0 failed to orient the
encoder toward outcome quality. See reports/latent_markov_design.md §NFR6.

Failure diagnosis:
  - Increase lambda_out in configs/train_latent_grpo_smoke.yaml
  - Train for more epochs (num_epochs)
  - Increase latent_dim
  - Check that reward_rate in rollouts is not 0% or 100% (no variance = no signal)

Usage
─────
  # After Phase 0 training:
  python scripts/check_latent_structure.py \\
      --rollout-path data/phase0_rollouts.pt \\
      --checkpoint-path runs/latent_grpo/phase0_vae.pt \\
      --output-dir runs/latent_grpo/plots

  # With UMAP instead of t-SNE (requires: pip install umap-learn):
  python scripts/check_latent_structure.py \\
      --rollout-path data/phase0_rollouts.pt \\
      --checkpoint-path runs/latent_grpo/phase0_vae.pt \\
      --output-dir runs/latent_grpo/plots \\
      --method umap

  # Smoke: limit to first 100 trajectories for quick inspection
  python scripts/check_latent_structure.py \\
      --rollout-path data/phase0_rollouts_smoke.pt \\
      --checkpoint-path runs/latent_grpo_smoke/phase0_vae.pt \\
      --output-dir runs/latent_grpo_smoke/plots \\
      --limit 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.vae_state_encoder import VAEStateEncoder

from sklearn.manifold import TSNE
import umap


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rollout-path",
        type=Path,
        required=True,
        help="Path to phase0_rollouts.pt produced by generate_phase0_rollouts.py.",
    )
    p.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to phase0_vae.pt produced by pretrain_vae_online().",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "latent_grpo" / "plots",
        help="Directory for saved figures (default: runs/latent_grpo/plots).",
    )
    p.add_argument(
        "--method",
        choices=["tsne", "umap"],
        default="tsne",
        help="Dimensionality reduction method (default: tsne).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on number of trajectories to include (default: all).",
    )
    p.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity (ignored for UMAP). Default: 30.",
    )
    p.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors (ignored for t-SNE). Default: 15.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the dimensionality reduction (default: 42).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load rollouts and run encoder
# ---------------------------------------------------------------------------

def collect_z_finals(
    rollout_path: Path,
    checkpoint_path: Path,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load VAE checkpoint and encode all z_final vectors from saved rollouts.

    Args:
        rollout_path:     path to phase0_rollouts.pt.
        checkpoint_path:  path to phase0_vae.pt.
        limit:            if given, only process the first `limit` trajectories.

    Returns:
        z_finals: float32 array of shape (N, LATENT_DIM).
        labels:   int array of shape (N,), values 0 or 1 (incorrect / correct).
    """
    # Load rollouts.
    print(f"Loading rollouts from {rollout_path} ...", flush=True)
    rollouts: list[dict] = torch.load(rollout_path, weights_only=False)
    if limit is not None:
        rollouts = rollouts[:limit]
    print(f"  {len(rollouts)} trajectories loaded", flush=True)

    # Load VAE.
    print(f"Loading VAE from {checkpoint_path} ...", flush=True)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    vae = VAEStateEncoder()
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    print("  VAE loaded", flush=True)

    # Run encoder — deterministic (eval mode → reparameterize returns μ).
    z_finals_list: list[torch.Tensor] = []
    labels_list:   list[int]          = []

    with torch.no_grad():
        for traj in rollouts:
            repr_3 = traj["repr_3"].float().unsqueeze(0)   # (1, HIDDEN_DIM)
            mu, logvar = vae.encode(repr_3)
            # eval mode: z = μ (deterministic)
            z = vae.reparameterize(mu, logvar)             # (1, LATENT_DIM)
            z_finals_list.append(z.squeeze(0))
            labels_list.append(int(traj["reward"]))

    z_finals = torch.stack(z_finals_list).numpy().astype("float32")
    labels   = np.array(labels_list, dtype=np.int32)

    n_correct   = labels.sum()
    n_incorrect = len(labels) - n_correct
    print(
        f"  {n_correct} correct, {n_incorrect} incorrect "
        f"({n_correct/len(labels):.1%} reward rate)",
        flush=True,
    )
    return z_finals, labels


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    z_finals: np.ndarray,
    method: str,
    perplexity: float,
    n_neighbors: int,
    seed: int,
) -> np.ndarray:
    """Reduce z_finals from LATENT_DIM to 2-D for scatter plotting.

    t-SNE and UMAP are used because the Markov property criterion is a clustering
    test — correct and incorrect trajectories should occupy distinct regions in
    latent space. Both methods are non-linear and excel at revealing cluster
    structure that linear projections like PCA would flatten out. t-SNE optimises
    for local neighbourhood preservation (best for cluster shape); UMAP additionally
    preserves global structure and is faster for large datasets.

    Args:
        z_finals:    (N, LATENT_DIM) float32 array of encoder outputs.
        method:      "tsne" or "umap".
        perplexity:  t-SNE perplexity (controls local neighbourhood size). Ignored for UMAP.
        n_neighbors: UMAP n_neighbors (controls local neighbourhood size). Ignored for t-SNE.
        seed:        random state for reproducibility.

    Returns:
        coords: (N, 2) float32 array — 2-D projection of z_finals.
    """
    np.random.seed(seed)
    if method == "tsne":
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    random_state=seed, max_iter=1000)
        coords = tsne.fit_transform(z_finals)
    elif method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed)
        coords = reducer.fit_transform(z_finals)
    else:
        raise ValueError(f"Unknown reduction method '{method}'. Choose 'tsne' or 'umap'.")
    return coords.astype("float32")


# ---------------------------------------------------------------------------
# Plotting (scaffolding complete — runs after you fill in reduce_dimensions)
# ---------------------------------------------------------------------------

def plot_latent_structure(
    coords: np.ndarray,
    labels: np.ndarray,
    method: str,
    output_dir: Path,
) -> None:
    """Scatter-plot the 2-D projection of z_final coloured by outcome.

    Args:
        coords:     (N, 2) projection from reduce_dimensions.
        labels:     (N,) int array — 0 = incorrect, 1 = correct.
        method:     "tsne" or "umap" (used in plot title and filename).
        output_dir: directory to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — run: pip install matplotlib", file=sys.stderr)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    colors = np.where(labels == 1, "tab:green", "tab:red")
    alphas = 0.5

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter_incorrect = ax.scatter(
        coords[labels == 0, 0], coords[labels == 0, 1],
        c="tab:red",   alpha=alphas, s=12, label=f"Incorrect (n={int((labels==0).sum())})",
    )
    scatter_correct = ax.scatter(
        coords[labels == 1, 0], coords[labels == 1, 1],
        c="tab:green", alpha=alphas, s=12, label=f"Correct (n={int((labels==1).sum())})",
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
    print(f"Plot saved → {out_path}", flush=True)

    # Also save a JSON summary for programmatic inspection.
    import json
    summary = {
        "n_correct":   int((labels == 1).sum()),
        "n_incorrect": int((labels == 0).sum()),
        "reward_rate": float((labels == 1).mean()),
        "method":      method,
        "plot_path":   str(out_path),
    }
    (output_dir / "nfr6_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary → {output_dir / 'nfr6_summary.json'}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    z_finals, labels = collect_z_finals(
        args.rollout_path,
        args.checkpoint_path,
        args.limit,
    )

    print(
        f"Running {args.method.upper()} on {len(z_finals)} latent vectors "
        f"(dim={z_finals.shape[1]}) ...",
        flush=True,
    )
    coords = reduce_dimensions(
        z_finals,
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        seed=args.seed,
    )

    plot_latent_structure(coords, labels, args.method, args.output_dir)


if __name__ == "__main__":
    main()
