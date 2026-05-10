#!/usr/bin/env python3
"""Post-Phase-1 Markov property diagnostics (E1 and E3 from the requirements doc).

What this script produces
─────────────────────────
E1 — Transition sufficiency
    Evaluates the trained transition model f(z_h) → z_{h+1} on a held-out split
    of the Phase 0 rollouts (last 20% by default).  A low MSE transition loss on
    data the VAE never saw during training is empirical evidence that z_h alone is
    a sufficient state summary — i.e. the Markov property holds.

    Output: JSON entry {"e1_transition_loss_held_out": float}

E3 — Uncertainty calibration
    Checks whether the encoder's per-chunk variance σ_h² (= exp(logvar_h)) correlates
    with trajectory outcome and problem difficulty:
      - Correct trajectories should have lower σ_h² than incorrect ones (the model
        is more certain when it is on the right track).
      - Harder problems (lower reward rate) should show higher average σ_h².

    Output: JSON summary + box-plot PNG of σ_h² by outcome per chunk.

E2 note:
    E2 (policy sufficiency / last-state-only ablation) is covered by the ablation
    table — compare latent_grpo pass@1024 vs baseline_grpo pass@1024 using
    eval_passk.py.  No separate script needed.

Usage
─────
  # Standard post-Phase-1 run
  python scripts/eval_markov_diagnostics.py \\
      --vae-checkpoint runs/latent_grpo/final/phase1_latent.pt \\
      --rollout-path data/phase0_rollouts.pt \\
      --output-dir runs/latent_grpo/diagnostics

  # Smoke: small subset, CPU-friendly
  python scripts/eval_markov_diagnostics.py \\
      --vae-checkpoint runs/latent_grpo/phase0_vae.pt \\
      --rollout-path data/phase0_rollouts_smoke.pt \\
      --output-dir runs/latent_grpo/diagnostics_smoke \\
      --held-out-frac 0.5

Required files:
  - phase1_latent.pt (or phase0_vae.pt for smoke): contains key "vae"
  - phase0_rollouts.pt: list of trajectory dicts produced by generate_phase0_rollouts.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.vae_state_encoder import HIDDEN_DIM, LATENT_DIM, VAEStateEncoder


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--vae-checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a .pt file with key 'vae' containing VAE state dict.  "
            "Either phase1_latent.pt (post-Phase-1) or phase0_vae.pt (smoke)."
        ),
    )
    p.add_argument(
        "--rollout-path",
        type=Path,
        default=REPO_ROOT / "data" / "phase0_rollouts.pt",
        help="Path to the rollout .pt file from generate_phase0_rollouts.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "latent_grpo" / "diagnostics",
        help="Directory to write JSON summary and PNG plots.",
    )
    p.add_argument(
        "--held-out-frac",
        type=float,
        default=0.2,
        help="Fraction of rollouts to hold out for E1 evaluation (default: 0.2).",
    )
    p.add_argument(
        "--latent-dim",
        type=int,
        default=LATENT_DIM,
        help=f"VAE latent dimension (default: {LATENT_DIM}).",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=HIDDEN_DIM,
        help=f"Backbone hidden dimension (default: {HIDDEN_DIM}).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# E1 — Transition sufficiency
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_e1_transition(
    vae: VAEStateEncoder,
    held_out: list[dict],
    device: torch.device,
) -> float:
    """Compute mean transition loss on held-out trajectories.

    For each trajectory:
        z_h = vae.reparameterize(*vae.encode(repr_h))   for h in {1, 2, 3}
        L_transition = ||f(z_1) − z_2||² + ||f(z_2) − z_3||²

    The held-out split ensures the reported loss is not on training data.
    A low value is empirical evidence for Markov sufficiency (E1).

    Args:
        vae:      trained VAEStateEncoder in eval mode.
        held_out: list of trajectory dicts (keys: repr_1, repr_2, repr_3).
        device:   compute device.

    Returns:
        Mean L_transition across held-out trajectories.
    """
    total_loss = 0.0
    n = len(held_out)

    for traj in held_out:
        repr_list = [
            traj["repr_1"].unsqueeze(0).to(device),
            traj["repr_2"].unsqueeze(0).to(device),
            traj["repr_3"].unsqueeze(0).to(device),
        ]
        results = vae.forward(repr_list)
        z_list  = [r[0] for r in results]
        loss    = vae.compute_transition_loss(z_list)
        total_loss += loss.item()

    return total_loss / n


# ---------------------------------------------------------------------------
# E3 — Uncertainty calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_e3_uncertainty(
    vae: VAEStateEncoder,
    trajectories: list[dict],
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Compute per-chunk σ_h² statistics grouped by trajectory outcome.

    For each trajectory, encodes repr_1, repr_2, repr_3 through the VAE to obtain
    log-variance estimates logvar_h.  Converts to per-chunk mean variance:
        sigma_sq_h = exp(logvar_h).mean(dim=-1)   (mean over latent dims)

    Then groups by reward (0 = incorrect, 1 = correct) and computes:
      - mean σ_h² per group per chunk
      - optionally, Pearson correlation between per-trajectory avg σ² and reward

    Produces:
      - JSON summary (mean σ_h² by outcome, correlation coefficient)
      - Box-plot PNG: σ_h² distributions by outcome for each chunk

    Args:
        vae:           trained VAEStateEncoder in eval mode.
        trajectories:  full rollout list (train + held-out — use everything for E3).
        device:        compute device.
        output_dir:    directory to write e3_summary.json and e3_sigma_boxplot.png.

    Returns:
        Dict with the E3 summary metrics (also written to output_dir/e3_summary.json).

    Hints
    ─────
    • vae.encode(repr_h) returns (mu_h, logvar_h).  You want logvar_h (do NOT call
      reparameterize — you want the distribution parameters, not a sample).
    • Per-trajectory mean variance: sigma_sq = exp(logvar_h).mean(dim=-1).item()
      This collapses the latent_dim axis so you get one scalar per chunk per trajectory.
    • Group by reward: build two lists (sigma_correct, sigma_incorrect) per chunk.
    • Correlation: use scipy.stats.pearsonr(sigma_avg_per_traj, rewards_per_traj).
      Report (r, p_value) — a negative r means higher variance → lower reward,
      consistent with the hypothesis.
    • Box-plot: matplotlib.pyplot.boxplot([sigma_correct_h, sigma_incorrect_h])
      one subplot per chunk h.  Label axes and save to output_dir/e3_sigma_boxplot.png.
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for server runs
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    sigmas: dict[int, dict[str, list[float]]] = {
        h: {"correct": [], "incorrect": []} for h in (1, 2, 3)
    }
    all_sigma_avg: list[float] = []
    all_rewards:   list[float] = []

    for traj in trajectories:
        reward = int(traj["reward"])
        label  = "correct" if reward == 1 else "incorrect"
        chunk_sigmas: list[float] = []
        for h, key in enumerate(("repr_1", "repr_2", "repr_3"), start=1):
            repr_h = traj[key].unsqueeze(0).to(device)           # (1, hidden_dim)
            with torch.no_grad():
                _, logvar_h = vae.encode(repr_h)                  # (1, latent_dim)
            sigma_sq = logvar_h.exp().mean().item()               # mean over latent dims
            sigmas[h][label].append(sigma_sq)
            chunk_sigmas.append(sigma_sq)
        all_sigma_avg.append(sum(chunk_sigmas) / len(chunk_sigmas))
        all_rewards.append(float(reward))

    # Pearson correlation between per-trajectory mean σ² and binary reward.
    # Expect r < 0: higher uncertainty → wrong answer.
    r_val, p_val = pearsonr(all_sigma_avg, all_rewards)

    # Means for logging
    def _safe_mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    mean_correct   = _safe_mean([s for h in (1,2,3) for s in sigmas[h]["correct"]])
    mean_incorrect = _safe_mean([s for h in (1,2,3) for s in sigmas[h]["incorrect"]])

    # Box-plot: σ² by outcome, one subplot per chunk
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for h in (1, 2, 3):
        ax = axes[h - 1]
        data = [sigmas[h]["correct"], sigmas[h]["incorrect"]]
        bp   = ax.boxplot(data, patch_artist=True,
                          medianprops={"color": "black", "linewidth": 2})
        bp["boxes"][0].set_facecolor("#4caf50")   # green = correct
        bp["boxes"][1].set_facecolor("#f44336")   # red   = incorrect
        ax.set_title(f"Chunk {h}")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Correct", "Incorrect"])
        ax.set_ylabel("σ² (mean over latent dims)" if h == 1 else "")
    fig.suptitle(
        f"E3 — Encoder variance by outcome\n"
        f"Pearson r={r_val:.3f} (p={p_val:.3e})  |  "
        f"mean σ² correct={mean_correct:.4f}  incorrect={mean_incorrect:.4f}"
    )
    fig.tight_layout()
    plot_path = output_dir / "e3_sigma_boxplot.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  E3 box-plot saved → {plot_path}", flush=True)

    summary = {
        "e3_mean_sigma_correct":   mean_correct,
        "e3_mean_sigma_incorrect": mean_incorrect,
        "e3_pearson_r":            r_val,
        "e3_pearson_p":            p_val,
        "n_correct":               sum(len(sigmas[h]["correct"])   for h in (1,2,3)) // 3,
        "n_incorrect":             sum(len(sigmas[h]["incorrect"]) for h in (1,2,3)) // 3,
    }
    (output_dir / "e3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ------------------------------------------------------------------
    # Load VAE checkpoint
    # ------------------------------------------------------------------
    print(f"Loading VAE from {args.vae_checkpoint} ...", flush=True)
    ckpt = torch.load(args.vae_checkpoint, weights_only=False, map_location=device)
    vae = VAEStateEncoder(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()
    print("  VAE loaded", flush=True)

    # ------------------------------------------------------------------
    # Load rollouts
    # ------------------------------------------------------------------
    print(f"Loading rollouts from {args.rollout_path} ...", flush=True)
    all_rollouts: list[dict] = torch.load(
        args.rollout_path, weights_only=False, map_location="cpu"
    )
    print(f"  {len(all_rollouts)} trajectories", flush=True)

    n_total   = len(all_rollouts)
    n_held    = max(1, int(n_total * args.held_out_frac))
    n_train   = n_total - n_held
    train_set = all_rollouts[:n_train]   # noqa: F841 (unused — E1 only uses held-out)
    held_set  = all_rollouts[n_train:]

    print(
        f"  split: {n_train} train / {n_held} held-out "
        f"(held-out-frac={args.held_out_frac})",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # E1 — Transition sufficiency
    # ------------------------------------------------------------------
    print("\nE1 — Transition sufficiency ...", flush=True)
    e1_loss = eval_e1_transition(vae, held_set, device)
    print(f"  held-out L_transition = {e1_loss:.6f}", flush=True)
    e1_result = {"e1_transition_loss_held_out": e1_loss, "n_held_out": n_held}
    (args.output_dir / "e1_result.json").write_text(
        json.dumps(e1_result, indent=2)
    )

    # ------------------------------------------------------------------
    # E3 — Uncertainty calibration
    # ------------------------------------------------------------------
    print("\nE3 — Uncertainty calibration ...", flush=True)
    e3_result = eval_e3_uncertainty(vae, all_rollouts, device, args.output_dir)

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    summary = {**e1_result, **e3_result}
    (args.output_dir / "markov_diagnostics.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        f"\nDone. Results written to {args.output_dir}/markov_diagnostics.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
