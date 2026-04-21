#!/usr/bin/env python3
"""Train the history-as-state GRPO baseline (`baseline_grpo`).

Wires config loading, run directory creation, and seeding. Core training logic lives in
``src/training/grpo_baseline.py``.

See ``PROJECT_CONTRACT.md`` Phase 1 and ``configs/train_baseline_grpo.yaml``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Repo root on sys.path for `src.*`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config_loader import load_yaml_with_extends
from src.utils.seeding import set_seed
from src.training.grpo_baseline import train_baseline

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "train_baseline_grpo.yaml",
        help="Training YAML (may extend configs/base_model.yaml).",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "baseline_grpo",
        help="Parent directory; each run creates a subdirectory run_id.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override auto-generated run id (default: UTC timestamp prefix).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    merged = load_yaml_with_extends(config_path.resolve(), root=REPO_ROOT)

    training = merged.get("training") or {}
    seed = int(training.get("seed", 0))
    set_seed(seed)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.artifacts_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "script": "train_baseline.py",
        "run_id": run_id,
        "profile": (merged.get("experiment") or {}).get("profile"),
        "config_path": str(config_path.resolve().relative_to(REPO_ROOT)),
        "seed": seed,
        "experiment": merged.get("experiment"),
        "primary_model_repo": (merged.get("primary") or {}).get("huggingface_repo_id"),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    print(f"Run directory: {run_dir}", file=sys.stderr)
    train_baseline(config=merged, run_dir=run_dir)


if __name__ == "__main__":
    main()
