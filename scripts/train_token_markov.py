"""Train the token-Markov GRPO arm (Delethink-style chunked rollouts).

Usage:
    # Full run (A100):
    python scripts/train_token_markov.py \
        --config configs/train_token_markov_grpo.yaml

    # Smoke run (local 4060 — uses smoke profile via the config):
    python scripts/train_token_markov.py \
        --config configs/train_token_markov_grpo.yaml \
        --smoke
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
from src.training.grpo_token_markov import train_token_markov

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train token-Markov GRPO arm.")
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_token_markov_grpo.yaml",
    )
    p.add_argument("--smoke", action="store_true",
                   help="Override profile to smoke (for local pipeline verification).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_yaml_with_extends(config_path.resolve(), root=REPO_ROOT)

    if args.smoke:
        config.setdefault("experiment", {})["profile"] = "smoke"

    seed = config["training"]["seed"]
    set_seed(seed)

    run_id = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    arm = config["experiment"]["name"]
    run_dir = Path("artifacts") / arm / run_id

    logger.info("arm=%s  run_id=%s  seed=%d", arm, run_id, seed)
    logger.info("run_dir=%s", run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "arm": arm,
        "run_id": run_id,
        "seed": seed,
        "config_path": str(args.config),
    }, indent=2))

    train_token_markov(config, run_dir)


if __name__ == "__main__":
    main()
