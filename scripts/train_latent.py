"""Train the latent Markov GRPO arm.

Dispatches to Phase 0 (teacher-forced distillation) or Phase 1 (joint RL)
based on the --phase argument. All hyperparameters are read from the config.

Usage:
    # Phase 0 — distillation pretraining (smoke, ~1 min):
    python scripts/train_latent.py \
        --config configs/train_latent_grpo_smoke.yaml --phase 0

    # Phase 0 — full run (RTX Pro 6000, ~2-3h):
    python scripts/train_latent.py \
        --config configs/train_latent_grpo.yaml --phase 0

    # Phase 1 — GRPO on L5 hard pool (~2-2.5h):
    python scripts/train_latent.py \
        --config configs/train_latent_grpo.yaml --phase 1
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_yaml_with_extends
from src.utils.seeding import set_seed
from src.training.grpo_latent import pretrain_distill, train_latent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train latent Markov GRPO arm.")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_latent_grpo_smoke.yaml",
        help="Path to YAML config (default: smoke config).",
    )
    p.add_argument(
        "--phase",
        type=int,
        choices=[0, 1],
        default=0,
        help="Training phase: 0 = distillation pretraining, 1 = GRPO (default: 0).",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Reuse an existing run-id so Phase 0 and Phase 1 share the same "
            "artifact directory.  Auto-generated (UTC timestamp) if omitted."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_yaml_with_extends(args.config.resolve(), root=REPO_ROOT)

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    run_id  = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    arm     = config.get("experiment", {}).get("name", "latent_grpo")
    run_dir = Path("artifacts") / arm / run_id

    # Phase 0 writes to  run_dir/phase0/
    # Phase 1 reads from run_dir/phase0/  and writes to  run_dir/phase1/
    phase_dir = run_dir / ("phase0" if args.phase == 0 else "phase1")

    logger.info("arm=%s  phase=%d  run_id=%s  seed=%d", arm, args.phase, run_id, seed)
    logger.info("artifacts → %s", phase_dir)
    print(f"\n  RUN_ID={run_id}\n", flush=True)

    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / "run_manifest.json").write_text(json.dumps({
        "arm":         arm,
        "phase":       args.phase,
        "run_id":      run_id,
        "seed":        seed,
        "config_path": str(args.config),
    }, indent=2))

    if args.phase == 0:
        pretrain_distill(config, phase_dir)
    else:
        train_latent(config, run_dir)


if __name__ == "__main__":
    main()
