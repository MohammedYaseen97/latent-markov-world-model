#!/usr/bin/env python3
"""Compare per-step metrics and completion hashes from two runs to verify reproducibility.

Usage:
    python scripts/check_reproducibility.py \\
        artifacts/baseline_grpo/smoke_run_1/checkpoint-4/trainer_state.json \\
        artifacts/baseline_grpo/smoke_run_2/checkpoint-4/trainer_state.json

Two checks are performed:
  1. Metrics (entropy, loss, grad_norm, learning_rate, num_tokens) from trainer_state.json.
  2. SHA-256 hashes of the raw completions from completions_hashes.jsonl, auto-detected
     from the run root (two levels above the trainer_state.json checkpoint dir).
     If the hash file is absent for a run, that check is skipped with a warning.

Exits 0 if everything matches, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHECKED_KEYS = ["entropy", "loss", "grad_norm", "learning_rate", "num_tokens"]

col = "{:>5}  {:<26}  {:>16}  {:>16}  {}"


def load_log_history(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in data.get("log_history", []) if "step" in e]


def load_completion_hashes(trainer_state_path: Path) -> list[dict] | None:
    """Look for completions_hashes.jsonl two levels up (run_root/checkpoint-N/trainer_state.json)."""
    hash_path = trainer_state_path.parent.parent / "completions_hashes.jsonl"
    if not hash_path.is_file():
        return None
    entries = [json.loads(line) for line in hash_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return entries


def compare_section(label: str, rows_a: list[tuple], rows_b: list[tuple], name_a: str, name_b: str) -> int:
    """Print a comparison table and return the number of mismatches."""
    common = min(len(rows_a), len(rows_b))
    mismatches = 0
    print(f"\n── {label} ({'first ' + str(common) + ' steps' if len(rows_a) != len(rows_b) else str(common) + ' steps'}) ──")
    print(col.format("step", "metric", name_a, name_b, ""))
    print("-" * 74)
    for (step_a, key_a, va), (step_b, key_b, vb) in zip(rows_a[:common], rows_b[:common]):
        ok = va == vb
        if not ok:
            mismatches += 1
        tag = "OK" if ok else "!! MISMATCH"
        print(col.format(step_a, key_a, str(va)[:16], str(vb)[:16], tag))
    return mismatches


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_a", type=Path, help="Path to first trainer_state.json")
    p.add_argument("run_b", type=Path, help="Path to second trainer_state.json")
    p.add_argument("--keys", nargs="+", default=CHECKED_KEYS, help="Metrics to compare")
    args = p.parse_args()

    name_a = args.run_a.parent.parent.name
    name_b = args.run_b.parent.parent.name

    hist_a = load_log_history(args.run_a)
    hist_b = load_log_history(args.run_b)

    n_a, n_b = len(hist_a), len(hist_b)
    if n_a != n_b:
        print(f"Note: metric step counts differ ({n_a} vs {n_b}); comparing shared prefix.")

    # --- Section 1: metrics ---
    metric_rows_a = [(e["step"], k, e.get(k)) for e in hist_a for k in args.keys if not (e.get(k) is None)]
    metric_rows_b = [(e["step"], k, e.get(k)) for e in hist_b for k in args.keys if not (e.get(k) is None)]
    metric_mismatches = compare_section("Metrics (trainer_state.json)", metric_rows_a, metric_rows_b, name_a, name_b)

    # --- Section 2: completion hashes ---
    hashes_a = load_completion_hashes(args.run_a)
    hashes_b = load_completion_hashes(args.run_b)

    hash_mismatches = 0
    if hashes_a is None or hashes_b is None:
        missing = name_a if hashes_a is None else name_b
        print(f"\nNote: completions_hashes.jsonl not found for {missing} — skipping generation check.")
        print("      (Re-run training to generate hash logs.)")
    else:
        hn_a, hn_b = len(hashes_a), len(hashes_b)
        if hn_a != hn_b:
            print(f"\nNote: hash step counts differ ({hn_a} vs {hn_b}); comparing shared prefix.")
        hash_rows_a = [(e["step"], "completions_hash", e["completions_hash"]) for e in hashes_a]
        hash_rows_b = [(e["step"], "completions_hash", e["completions_hash"]) for e in hashes_b]
        hash_mismatches = compare_section("Completions (completions_hashes.jsonl)", hash_rows_a, hash_rows_b, name_a, name_b)

    # --- Summary ---
    total = metric_mismatches + hash_mismatches
    print()
    if total == 0:
        print("All checks passed — runs are reproducible.")
    else:
        print(f"{total} mismatch(es) found.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
