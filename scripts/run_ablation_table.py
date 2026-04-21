#!/usr/bin/env python3
"""Generate the core ablation table from experiment artifacts.

Scans ``artifacts/<arm>/<run_id>/eval_metrics.json`` for all four arms,
picks the latest completed run per arm, and writes:

    reports/ablation_core.csv   — machine-readable (CI / downstream scripts)
    reports/ablation_core.md    — human-readable markdown table

Arms with no eval yet appear as ``—`` in the table.  Run this after each arm
completes eval; the table is always regenerated from artifacts, never hand-typed
(per PROJECT_CONTRACT.md Phase 4 gate).

Usage:
    python scripts/run_ablation_table.py [--artifacts-dir artifacts]
                                         [--output-dir reports]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ARMS: list[str] = [
    "baseline_grpo",
    "token_markov_grpo",
    "latent_grpo",
    "latent_grpo_uncertainty",
]

METRICS: list[str] = ["pass@1", "pass@16", "pass@1024"]


def _latest_eval(arm_dir: Path) -> dict | None:
    """Return the eval_metrics dict from the most recent run under arm_dir, or None."""
    candidates: list[tuple[str, Path]] = []
    for run_dir in arm_dir.iterdir():
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / "eval_metrics.json"
        if metrics_file.is_file():
            candidates.append((run_dir.name, metrics_file))
    if not candidates:
        return None
    # run_id is a timestamp string (e.g. 20260421T131720Z) — sort lexicographically
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, latest = candidates[0]
    with latest.open(encoding="utf-8") as f:
        data = json.load(f)
    data["_run_id"] = candidates[0][0]
    data["_metrics_path"] = str(latest.relative_to(REPO_ROOT))
    return data


def build_table(artifacts_dir: Path) -> list[dict]:
    rows = []
    for arm in ARMS:
        arm_dir = artifacts_dir / arm
        entry: dict = {"arm": arm}
        result = _latest_eval(arm_dir) if arm_dir.is_dir() else None
        if result:
            entry["run_id"] = result.get("_run_id", "—")
            entry["metrics_path"] = result.get("_metrics_path", "—")
            entry["n_problems"] = result.get("n_problems", "—")
            for m in METRICS:
                v = result.get("metrics", {}).get(m)
                entry[m] = f"{v:.4f}" if v is not None else "—"
        else:
            entry["run_id"] = "—"
            entry["metrics_path"] = "—"
            entry["n_problems"] = "—"
            for m in METRICS:
                entry[m] = "—"
        rows.append(entry)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "run_id", "n_problems"] + METRICS + ["metrics_path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Core ablation table",
        "",
        f"*Generated {generated_at} by `scripts/run_ablation_table.py` from `artifacts/`.*  ",
        "*Never hand-typed — re-run to refresh.*",
        "",
        "## MATH-B-I base pool",
        "",
        "| arm | pass@1 | pass@16 | pass@1024 | run |",
        "|-----|--------|---------|-----------|-----|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['arm']}` | {r['pass@1']} | {r['pass@16']} | {r['pass@1024']} | {r['run_id']} |"
        )
    lines += [
        "",
        "## Artifact paths",
        "",
        "| arm | eval_metrics.json |",
        "|-----|-------------------|",
    ]
    for r in rows:
        lines.append(f"| `{r['arm']}` | `{r['metrics_path']}` |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-dir", type=Path,
                        default=REPO_ROOT / "artifacts",
                        help="Root artifacts directory (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "reports",
                        help="Output directory for CSV and MD (default: %(default)s)")
    args = parser.parse_args()

    rows = build_table(args.artifacts_dir)

    csv_path = args.output_dir / "ablation_core.csv"
    md_path  = args.output_dir / "ablation_core.md"

    write_csv(rows, csv_path)
    write_md(rows, md_path)

    # Print the markdown table to stdout for quick inspection
    for row in rows:
        status = "✓" if row["pass@1024"] != "—" else "·"
        print(f"  {status}  {row['arm']:30s}  pass@1024={row['pass@1024']:>8}  run={row['run_id']}")

    print(f"\nWrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {md_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
