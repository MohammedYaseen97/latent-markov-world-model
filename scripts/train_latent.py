"""Train the latent Markov GRPO arm.

Dispatches to Phase 0 (VAE pretraining) or Phase 1 (joint RL) based on the
config's `phase` key. Both phases read all hyperparameters from the config file.

Usage:
    # Phase 0 — VAE pretraining (RTX 4060 smoke):
    python scripts/train_latent.py \
        --config configs/train_latent_grpo_smoke.yaml \
        --phase 0

    # Phase 0 — full run (A100):
    python scripts/train_latent.py \
        --config configs/train_latent_grpo.yaml \
        --phase 0

    # Phase 1 — joint RL:
    python scripts/train_latent.py \
        --config configs/train_latent_grpo.yaml \
        --phase 1
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
from src.training.grpo_latent import pretrain_vae_online, train_latent

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
        help="Training phase: 0 = VAE pretraining, 1 = joint RL (default: 0).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_yaml_with_extends(args.config.resolve(), root=REPO_ROOT)

    seed = config.get("training", {}).get("seed", 42)
    set_seed(seed)

    run_id  = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    arm     = config.get("experiment", {}).get("name", "latent_grpo")
    run_dir = Path("artifacts") / arm / run_id

    logger.info("arm=%s  phase=%d  run_id=%s  seed=%d", arm, args.phase, run_id, seed)
    logger.info("run_dir=%s", run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "arm":         arm,
        "phase":       args.phase,
        "run_id":      run_id,
        "seed":        seed,
        "config_path": str(args.config),
    }, indent=2))

    if args.phase == 0:
        pretrain_vae_online(config, run_dir)
    else:
        train_latent(config, run_dir)


if __name__ == "__main__":
    main()
