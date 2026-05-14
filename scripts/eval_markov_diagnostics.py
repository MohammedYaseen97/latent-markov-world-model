#!/usr/bin/env python3
"""Post-Phase-1 Markov property diagnostics (E1 and E3).

v3 design: repr_h values are never saved to disk — they are computed live during
training.  This script reproduces the same pipeline in inference mode:

  1. Load the Phase 1 backbone + VAE + ZInjector from the Phase 1 checkpoint.
  2. Sample problems from a held-out pool.
  3. Run generate_latent_traces (no_grad) → chunk_ids + reward per trace.
  4. Re-run the 3-chunk forward pass (no_grad) to collect repr_h + logvar_h.
  5. Feed repr_h into E1 and E3.

E1 — Transition sufficiency
    Evaluates the trained transition model f(z_h) → z_{h+1} on held-out rollouts.
    Low MSE is empirical evidence that z_h alone is a sufficient state summary
    (Markov property holds).
    Output: e1_result.json  {"e1_transition_loss_held_out": float}

E3 — Uncertainty calibration
    Checks whether the encoder's per-chunk variance σ_h² = exp(logvar_h) correlates
    with trajectory outcome:
      - Correct trajectories → lower σ_h²  (model is certain when on the right track)
      - Incorrect trajectories → higher σ_h²
    Pearson r between per-trajectory mean σ² and binary reward should be negative.
    Output: e3_summary.json + e3_sigma_boxplot.png

E2 note:
    E2 (policy sufficiency) is covered by the ablation table — compare latent_grpo
    pass@1024 vs baseline_grpo pass@1024 in reports/ablation_core.md.  No script needed.

Usage
─────
  # Standard post-Phase-1 run (use Phase 1 final checkpoint)
  PHASE1_CKPT=$(ls -td artifacts/latent_grpo/*/phase1/final | head -1)
  python scripts/eval_markov_diagnostics.py \\
      --checkpoint "$PHASE1_CKPT" \\
      --config configs/train_latent_grpo.yaml \\
      --output-dir runs/latent_grpo/diagnostics

  # Quick check on Phase 0 VAE (before Phase 1) — pass phase0 ckpt dir
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
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.models.vae_state_encoder import HIDDEN_DIM, LATENT_DIM, VAEStateEncoder, ZInjector
from src.training.grpo_latent import generate_latent_traces, _run_pipeline_with_grad
from src.utils.config_loader import load_yaml_with_extends


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to a Phase 1 final checkpoint directory (contains backbone/ and "
            "phase1_latent.pt).  Pass the Phase 0 checkpoint dir with --phase0-only "
            "for pre-Phase-1 diagnostics."
        ),
    )
    p.add_argument(
        "--phase0-only",
        action="store_true",
        help=(
            "Load Phase 0 checkpoint (phase0_vae.pt) from --checkpoint dir instead "
            "of a Phase 1 checkpoint.  The pretrained backbone is loaded from HF."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training config YAML (e.g. configs/train_latent_grpo.yaml).",
    )
    p.add_argument(
        "--pool-path",
        type=Path,
        default=None,
        help=(
            "Path to the problem pool JSONL for generating held-out rollouts.  "
            "Defaults to phase0.pool_path from the config (math_easy_pool.jsonl)."
        ),
    )
    p.add_argument(
        "--n-problems",
        type=int,
        default=200,
        help="Number of problems to sample from the pool for diagnostics (default: 200).",
    )
    p.add_argument(
        "--n-rollouts",
        type=int,
        default=4,
        help="Rollouts per problem for generating trajectories (default: 4).",
    )
    p.add_argument(
        "--held-out-frac",
        type=float,
        default=0.2,
        help="Fraction of rollouts to hold out for E1 (default: 0.2).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "latent_grpo" / "diagnostics",
        help="Directory to write JSON summaries and PNG plots.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Problems per generate call (controls CUDA memory; total sequences = "
            "batch_size × n_rollouts).  Default: 4."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Collect repr_h from a batch of traces
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_repr(
    model,
    vae: VAEStateEncoder,
    z_injector: ZInjector,
    traces: list[dict],
    device: torch.device,
) -> list[dict]:
    """Re-run the 3-chunk pipeline in inference mode to extract repr_h and logvar_h.

    Returns a list of dicts, one per trace, each containing:
        "reward":   int (0 or 1)
        "repr_1/2/3": Tensor (hidden_dim,) on CPU
        "logvar_1/2/3": Tensor (latent_dim,) on CPU
    """
    # _run_pipeline_with_grad works under no_grad; it just returns live tensors
    # (which are detached by context).
    pipe = _run_pipeline_with_grad(model, vae, z_injector, traces, device)

    # Encode repr_h to get logvar_h (the distribution parameter we need for E3).
    # Note: vae.eval() → reparameterize returns mu (no sampling), but we want
    # logvar directly from encode(), which we call here separately.
    logvar_list = []
    for repr_h in pipe["repr_list"]:
        _, lv = vae.encode(repr_h.to(device))    # (B, latent_dim)
        logvar_list.append(lv.cpu())

    result: list[dict] = []
    for i, trace in enumerate(traces):
        result.append({
            "reward":   trace["reward"],
            "repr_1":   pipe["repr_list"][0][i].cpu(),
            "repr_2":   pipe["repr_list"][1][i].cpu(),
            "repr_3":   pipe["repr_list"][2][i].cpu(),
            "logvar_1": logvar_list[0][i],
            "logvar_2": logvar_list[1][i],
            "logvar_3": logvar_list[2][i],
        })
    return result


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

    Low held-out loss = Markov property holds (z_h alone predicts z_{h+1}).

    Returns:
        Mean L_transition across held-out trajectories.
    """
    total_loss = 0.0
    for traj in held_out:
        repr_list = [
            traj["repr_1"].unsqueeze(0).to(device),
            traj["repr_2"].unsqueeze(0).to(device),
            traj["repr_3"].unsqueeze(0).to(device),
        ]
        results = vae.forward(repr_list)
        z_list  = [r[0] for r in results]
        total_loss += vae.compute_transition_loss(z_list).item()
    return total_loss / max(len(held_out), 1)


# ---------------------------------------------------------------------------
# E3 — Uncertainty calibration
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_e3_uncertainty(
    trajectories: list[dict],
    output_dir: Path,
) -> dict:
    """Compute per-chunk σ_h² statistics grouped by trajectory outcome.

    logvar_h is taken directly from the pre-collected traj dicts (no VAE
    re-forward needed).  σ² = exp(logvar).mean(dim=-1) per chunk per traj.

    Pearson r between per-trajectory mean σ² and binary reward:
        Expected sign: r < 0  (higher uncertainty → more likely wrong).

    Produces:
        e3_summary.json — mean σ_h² by outcome, Pearson r/p
        e3_sigma_boxplot.png — box-plots of σ_h² by outcome per chunk
    """
    import matplotlib
    matplotlib.use("Agg")
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
        for h, key in enumerate(("logvar_1", "logvar_2", "logvar_3"), start=1):
            sigma_sq = traj[key].exp().mean().item()
            sigmas[h][label].append(sigma_sq)
            chunk_sigmas.append(sigma_sq)
        all_sigma_avg.append(sum(chunk_sigmas) / len(chunk_sigmas))
        all_rewards.append(float(reward))

    r_val, p_val = pearsonr(all_sigma_avg, all_rewards)

    def _safe_mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    mean_correct   = _safe_mean([s for h in (1,2,3) for s in sigmas[h]["correct"]])
    mean_incorrect = _safe_mean([s for h in (1,2,3) for s in sigmas[h]["incorrect"]])
    n_correct   = len(sigmas[1]["correct"])
    n_incorrect = len(sigmas[1]["incorrect"])

    # Box-plot
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
        f"E3 — Encoder variance by outcome  "
        f"(n_correct={n_correct}, n_incorrect={n_incorrect})\n"
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
        "n_correct":               n_correct,
        "n_incorrect":             n_incorrect,
        "interpretation": (
            "PASS" if r_val < -0.05 else "WEAK"
            if r_val < 0 else "FAIL"
        ),
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
    # Load config
    # ------------------------------------------------------------------
    config = load_yaml_with_extends(args.config.resolve(), root=REPO_ROOT)
    primary_cfg  = config["primary"]
    latent_cfg   = config["latent_markov"]
    phase0_cfg   = config["phase0"]
    training_cfg = config["training"]

    model_id    = primary_cfg["huggingface_repo_id"]
    revision    = primary_cfg.get("revision", "main")
    dtype       = getattr(torch, primary_cfg.get("dtype", "bfloat16"))
    chunk_tokens = int(latent_cfg.get("chunk_tokens", 341))
    latent_dim   = int(latent_cfg.get("latent_dim",  LATENT_DIM))
    hidden_dim   = int(latent_cfg.get("hidden_dim",  HIDDEN_DIM))
    temperature  = float(training_cfg.get("temperature", 1.0))
    top_p        = float(training_cfg.get("top_p",        1.0))

    pool_path = args.pool_path or Path(
        phase0_cfg.get("pool_path", "data/math_easy_pool.jsonl")
    )

    # ------------------------------------------------------------------
    # Load backbone
    # ------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.phase0_only:
        print(f"Loading pretrained backbone {model_id} @ {revision} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype, device_map="auto",
        )
    else:
        backbone_dir = args.checkpoint / "backbone"
        print(f"Loading Phase 1 backbone from {backbone_dir} ...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(backbone_dir), torch_dtype=dtype, device_map="auto",
        )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("  backbone ready", flush=True)

    # ------------------------------------------------------------------
    # Load VAE + ZInjector
    # ------------------------------------------------------------------
    if args.phase0_only:
        vae_pt = args.checkpoint / "phase0_vae.pt"
    else:
        vae_pt = args.checkpoint / "phase1_latent.pt"

    print(f"Loading VAE from {vae_pt} ...", flush=True)
    ckpt = torch.load(str(vae_pt), weights_only=False, map_location=device)
    vae = VAEStateEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    vae.load_state_dict(ckpt["vae"])
    vae.eval()

    z_injector = ZInjector(latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    if "z_injector" in ckpt:
        z_injector.load_state_dict(ckpt["z_injector"])
    z_injector.eval()
    print("  VAE + ZInjector loaded", flush=True)

    # ------------------------------------------------------------------
    # Load problem pool and sample
    # ------------------------------------------------------------------
    import json as _json
    import random
    with open(pool_path, encoding="utf-8") as f:
        all_problems = [_json.loads(line) for line in f if line.strip()]
    print(f"Pool: {len(all_problems)} problems from {pool_path}", flush=True)

    random.seed(99)   # fixed seed distinct from training seeds (42 / 11)
    sampled = random.sample(all_problems, min(args.n_problems, len(all_problems)))
    print(f"Sampled {len(sampled)} problems for diagnostics", flush=True)

    # ------------------------------------------------------------------
    # Generate traces and collect repr_h
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_traj_data: list[dict] = []
    problems_per_call = max(1, args.batch_size)
    total_calls = (len(sampled) + problems_per_call - 1) // problems_per_call

    print(
        f"\nGenerating {args.n_rollouts} rollouts × {len(sampled)} problems "
        f"in {total_calls} batches ...",
        flush=True,
    )

    for call_idx in range(total_calls):
        batch_probs = sampled[call_idx * problems_per_call:(call_idx + 1) * problems_per_call]
        with torch.no_grad():
            traces = generate_latent_traces(
                model=model, tokenizer=tokenizer,
                vae=vae, z_injector=z_injector,
                problems=batch_probs, n_rollouts=args.n_rollouts,
                chunk_tokens=chunk_tokens,
                temperature=temperature, top_p=top_p,
                device=device,
            )
            traj_data = collect_repr(model, vae, z_injector, traces, device)
        all_traj_data.extend(traj_data)
        n_correct = sum(t["reward"] for t in all_traj_data)
        print(
            f"  batch {call_idx + 1}/{total_calls}  "
            f"({len(all_traj_data)} trajectories, "
            f"reward rate {n_correct / len(all_traj_data):.1%})",
            flush=True,
        )

    print(f"\nTotal: {len(all_traj_data)} trajectories collected", flush=True)

    # ------------------------------------------------------------------
    # E1 — Transition sufficiency (held-out split)
    # ------------------------------------------------------------------
    n_held = max(1, int(len(all_traj_data) * args.held_out_frac))
    held_set = all_traj_data[-n_held:]   # last 20%: unseen during ordering

    print(f"\nE1 — Transition sufficiency (n_held={n_held}) ...", flush=True)
    e1_loss = eval_e1_transition(vae, held_set, device)
    print(f"  held-out L_transition = {e1_loss:.6f}", flush=True)
    e1_result = {
        "e1_transition_loss_held_out": e1_loss,
        "n_held_out": n_held,
        "interpretation": (
            "PASS (Markov property holds)" if e1_loss < 0.5
            else "BORDERLINE" if e1_loss < 1.0
            else "FAIL (high transition error)"
        ),
    }
    (args.output_dir / "e1_result.json").write_text(_json.dumps(e1_result, indent=2))

    # ------------------------------------------------------------------
    # E3 — Uncertainty calibration (all trajectories)
    # ------------------------------------------------------------------
    print(f"\nE3 — Uncertainty calibration (all {len(all_traj_data)} trajectories) ...",
          flush=True)
    e3_result = eval_e3_uncertainty(all_traj_data, args.output_dir)
    print(
        f"  Pearson r={e3_result['e3_pearson_r']:.3f}  "
        f"p={e3_result['e3_pearson_p']:.3e}  "
        f"→ {e3_result['interpretation']}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------
    summary = {**e1_result, **e3_result}
    (args.output_dir / "markov_diagnostics.json").write_text(_json.dumps(summary, indent=2))
    print(
        f"\nDone. Full summary → {args.output_dir}/markov_diagnostics.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
