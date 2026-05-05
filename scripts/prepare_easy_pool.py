#!/usr/bin/env python3
"""Build the Phase 0 VAE pretraining pool from EleutherAI/hendrycks_math.

Loads all 7 subject configs, filters to the requested difficulty levels (default
Level 1–3), extracts the boxed answer from each solution, and writes a JSONL with
the same schema used by all other pool files in this repo.

Output: ``data/math_easy_pool.jsonl``

Schema (one JSON object per line):
  problem_id   : "math_easy_{i:04d}"
  source_index : i (sequential, 0-based)
  prompt       : problem text
  ground_truth : answer string extracted from the LaTeX \boxed{} in the solution
  data_source  : "hendrycks_math"
  topic        : subject category (Algebra, Geometry, etc.)
  difficulty   : level integer (1, 2, or 3)

Reproducibility: HuggingFace dataset revision is pinned via --hf-revision (defaults
to the value recorded in data/easy_pool_manifest.json on first run, or fetched live
with a warning). SHA-256 of the output file is written to data/easy_pool_manifest.json.

Usage:
    python scripts/prepare_easy_pool.py
    python scripts/prepare_easy_pool.py --levels 1 2         # only Level 1-2
    python scripts/prepare_easy_pool.py --max-per-level 300  # cap per level
    python scripts/prepare_easy_pool.py --output-dir data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.training.grpo_baseline import _extract_boxed as extract_boxed  # single source of truth

HF_DATASET = "EleutherAI/hendrycks_math"
CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

MANIFEST_PATH = REPO_ROOT / "data" / "easy_pool_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_library_versions() -> dict[str, str]:
    out = {}
    for pkg in ("datasets", "huggingface_hub"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2, 3],
        help="Difficulty levels to include (1–5). Default: 1 2 3",
    )
    p.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="HF splits to pull from. Default: train test (both)",
    )
    p.add_argument(
        "--max-per-level", type=int, default=1000,
        help="Cap on rows per difficulty level for equal level distribution (default 1000).",
    )
    p.add_argument(
        "--hf-revision", type=str, default="21a5633873b6a120296cce3e2df9d5550074f4a3",
        help="Pinned HF dataset git SHA (default: revision used to build the current pool).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Shuffle seed applied before --max-per-level cap (for reproducible subsampling).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    level_set = {f"Level {n}" for n in args.levels}
    out_path = args.output_dir / "math_easy_pool.jsonl"

    print(f"Loading {HF_DATASET} (levels {args.levels}, splits {args.splits})...", file=sys.stderr)

    all_records: list[dict] = []
    skipped_no_answer = 0
    level_counts: dict[str, int] = {}

    for config in CONFIGS:
        for split in args.splits:
            try:
                ds = load_dataset(HF_DATASET, config, split=split, revision=args.hf_revision)
            except Exception as e:
                print(f"  SKIP {config}/{split}: {e}", file=sys.stderr)
                continue

            for row in ds:
                lvl = row["level"]
                if lvl not in level_set:
                    continue
                answer = extract_boxed(row["solution"])
                if answer is None:
                    skipped_no_answer += 1
                    continue
                # Drop answers with \text{...} — catches MCQ letters (\text{(C)}),
                # word answers (\text{even}), and unit strings (\text{ ft}).
                # math_verify can't reliably grade these, so they'd give noisy
                # L_outcome labels. We have enough problems without them.
                if r"\text{" in answer:
                    skipped_no_answer += 1  # counts toward skipped total
                    continue
                level_counts[lvl] = level_counts.get(lvl, 0) + 1
                all_records.append({
                    "_level": lvl,
                    "_level_int": int(lvl.split()[-1]),
                    "prompt": row["problem"],
                    "ground_truth": answer,
                    "data_source": "hendrycks_math",
                    "topic": row["type"],
                    "difficulty": int(lvl.split()[-1]),
                    "_config": config,
                    "_split": split,
                })

    print(f"  Raw rows after level filter: {len(all_records)}", file=sys.stderr)
    print(f"  Skipped (no boxed answer):   {skipped_no_answer}", file=sys.stderr)
    for lvl in sorted(level_counts):
        print(f"    {lvl}: {level_counts[lvl]} rows", file=sys.stderr)

    # Shuffle for reproducible subsampling (so --max-per-level gives a stable subset)
    import random
    rng = random.Random(args.seed)
    rng.shuffle(all_records)

    # Optional per-level cap
    if args.max_per_level is not None:
        per_level: dict[str, list] = {}
        for r in all_records:
            per_level.setdefault(r["_level"], []).append(r)
        capped: list[dict] = []
        for lvl in sorted(per_level):
            subset = per_level[lvl][: args.max_per_level]
            capped.extend(subset)
            if len(per_level[lvl]) > args.max_per_level:
                print(
                    f"  Capped {lvl}: {len(per_level[lvl])} → {args.max_per_level}",
                    file=sys.stderr,
                )
        all_records = capped

    # Sort by level then problem for a deterministic stable order in the file
    all_records.sort(key=lambda r: (r["_level_int"], r["topic"], r["prompt"]))

    # Assign sequential IDs and strip internal bookkeeping fields
    final_records = []
    for i, r in enumerate(all_records):
        final_records.append({
            "problem_id": f"math_easy_{i:04d}",
            "source_index": i,
            "prompt": r["prompt"],
            "ground_truth": r["ground_truth"],
            "data_source": r["data_source"],
            "topic": r["topic"],
            "difficulty": r["difficulty"],
        })

    write_jsonl(out_path, final_records)
    sha = sha256_file(out_path)

    # Level breakdown in final file
    from collections import Counter
    level_final = Counter(r["difficulty"] for r in final_records)

    print(f"\nWrote {len(final_records)} rows → {out_path}", file=sys.stderr)
    for lvl in sorted(level_final):
        print(f"  Level {lvl}: {level_final[lvl]} rows", file=sys.stderr)
    print(f"SHA-256: {sha}", file=sys.stderr)

    manifest = {
        "hf_dataset": HF_DATASET,
        "hf_revision": args.hf_revision,
        "splits_used": args.splits,
        "levels_included": sorted(args.levels),
        "max_per_level": args.max_per_level,
        "shuffle_seed": args.seed,
        "row_count": len(final_records),
        "level_breakdown": {str(k): v for k, v in sorted(level_final.items())},
        "skipped_no_boxed_answer_or_text_answer": skipped_no_answer,
        "filter_text_answers": True,
        "output_path": str(out_path.relative_to(REPO_ROOT)),
        "sha256_math_easy_pool_jsonl": sha,
        "library_versions_at_build": pinned_library_versions(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest → {MANIFEST_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
