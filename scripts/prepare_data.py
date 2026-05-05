#!/usr/bin/env python3
"""Build MATH-Beyond JSONL artifacts from Hugging Face (reproducible, defensible).

Authoritative documentation for claims about these files: ``reports/DATA_PROTOCOL.md``.

**Primary evaluation pool (default output)**

``data/math_beyond_math_b_i_base.jsonl`` — **MATH-B-Intersection (MATH-B-I)** under Mayilvahanan et al.
(arXiv:2510.11653) **Section 3.2.3 Base Models**: rows where every listed base model has
``pass@1024 == 0`` on the Hub table. The paper reports **41** problems; at the pinned Hub revision
(``configs/math_beyond_hf_revision.txt``) this filter yields **40** rows — documented in
``data/benchmark_manifest.json``.

**Phase 0 VAE pretraining pool (always written)**

``data/math_beyond_complement_141.jsonl`` — complement of the MATH-B-I hard-40 within the full
181-row test split (181 - 40 = **141** rows at the pinned revision). Problems where at least one
base model achieved ``pass@1024 > 0`` — the easier stratum used to pretrain the VAE before RL
training on the hard pool. Always computed relative to the MATH-B-I base definition regardless
of ``--primary-mode``. See ``reports/latent_markov_design.md`` (Phase 0).

**Secondary pool (always written)**

``data/math_beyond_hf_strict_all_models.jsonl`` — **stricter** subset: logical AND of all **21**
``*_unsolved`` flags (equivalently all ``*_pass@1024 == 0`` on this Hub revision). Use for
appendix / robustness; **not** the paper’s MATH-B-I definition.

**Full pool**

``data/math_beyond_full_181.jsonl`` — entire ``test`` split (181 rows at pinned revision).

**Reproducibility**

- Pin ``hf_revision`` via ``configs/math_beyond_hf_revision.txt``, ``--hf-revision``, or env
  ``MATH_BEYOND_HF_REVISION``. Running without a pin prints a warning.
- ``source_index`` is the 0-based row index in ``load_dataset(..., split="test", revision=...)``.
  Row order is the Hub order; same revision + same ``datasets`` version → same JSONL bytes.
- ``benchmark_manifest.json`` records revision, column lists, row counts, and SHA-256 of outputs.

Dataset: https://huggingface.co/datasets/brendel-group/MATH-Beyond
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset

PROTOCOL_VERSION = "math_b_data_protocol_v1"

# Mayilvahanan et al., Sec. 3.2.3 "Base Models" → HF ``*_pass@1024`` column names.
MATH_B_I_BASE_PASS1024_COLUMNS: tuple[str, ...] = (
    "qwen2.5-1.5b_pass@1024",
    "qwen2.5-7b_pass@1024",
    "qwen2.5-math-1.5b_pass@1024",
    "qwen2.5-math-7b_pass@1024",
    "qwen3-4b-base_pass@1024",
    "qwen3_8b_base_pass@1024",
    "r1-1.5b_pass@1024",
    "r1-7b_pass@1024",
    "olmo_7b_pass@1024",
    "olmo_2_1124_7b_pass@1024",
    "llama3.1_8b_pass@1024",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVISION_FILE = REPO_ROOT / "configs" / "math_beyond_hf_revision.txt"


def resolve_hf_revision(cli_value: str | None) -> str | None:
    if cli_value is not None and cli_value.strip() != "":
        return cli_value.strip()
    env = os.environ.get("MATH_BEYOND_HF_REVISION", "").strip()
    if env:
        return env
    if DEFAULT_REVISION_FILE.is_file():
        line = DEFAULT_REVISION_FILE.read_text(encoding="utf-8").splitlines()[0].strip()
        return line or None
    return None


def list_unsolved_columns(feature_names: list[str]) -> list[str]:
    return sorted(c for c in feature_names if c.endswith("_unsolved"))


def _pass_at_1024_columns(feature_names: list[str]) -> list[str]:
    return sorted(c for c in feature_names if "_pass@1024" in c)


def row_to_record(idx: int, row: dict[str, Any]) -> dict[str, Any]:
    problem = row.get("problem")
    answer = row.get("answer")
    if problem is None or answer is None:
        raise ValueError(f"Row {idx} missing required fields 'problem' or 'answer'")
    return {
        "problem_id": f"math_b_{idx:04d}",
        "source_index": idx,
        "prompt": problem,
        "ground_truth": answer,
        "data_source": row.get("data_source"),
        "topic": row.get("topic"),
        "difficulty": row.get("difficulty"),
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pinned_library_versions() -> dict[str, str]:
    """Recorded in manifest for audit; pair with requirements.txt pins."""
    out: dict[str, str] = {}
    for pkg in ("datasets", "huggingface_hub"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not_installed"
    return out


def filter_indices_math_b_i_base(ds: Dataset) -> list[int]:
    missing = [c for c in MATH_B_I_BASE_PASS1024_COLUMNS if c not in ds.column_names]
    if missing:
        raise RuntimeError("Missing expected pass@1024 columns: " + ", ".join(missing))
    return [i for i in range(len(ds)) if all(ds[i][c] == 0 for c in MATH_B_I_BASE_PASS1024_COLUMNS)]


def filter_indices_strict_all_hf_models(ds: Dataset) -> list[int]:
    cols = list_unsolved_columns(ds.column_names)
    if not cols:
        raise RuntimeError("No *_unsolved columns found.")
    return [i for i in range(len(ds)) if all(bool(ds[i][c]) for c in cols)]


def filter_indices_unsolved_column(ds: Dataset, column: str) -> list[int]:
    if column not in ds.column_names:
        raise ValueError(
            f"Unknown column {column!r}. Available *_unsolved: {list_unsolved_columns(ds.column_names)}"
        )
    return [i for i in range(len(ds)) if bool(ds[i][column])]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument(
        "--primary-mode",
        choices=("paper_math_b_i_base", "intersection_all_unsolved", "unsolved_single"),
        default="paper_math_b_i_base",
        help="Primary JSONL subset. Default: paper MATH-B-I (base gauntlet).",
    )
    p.add_argument(
        "--unsolved-column",
        type=str,
        default="qwen2.5-1.5b-instruct_unsolved",
        help="For unsolved_single primary mode only.",
    )
    p.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Hub dataset git SHA. Default: configs/math_beyond_hf_revision.txt or MATH_BEYOND_HF_REVISION.",
    )
    p.add_argument(
        "--no-secondary-strict",
        action="store_true",
        help="Do not write math_beyond_hf_strict_all_models.jsonl (not recommended).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    hf_revision = resolve_hf_revision(args.hf_revision)

    if hf_revision is None:
        print(
            "WARNING: No hf_revision pin. Set configs/math_beyond_hf_revision.txt or --hf-revision for reproducibility.",
            file=sys.stderr,
        )

    ds: Dataset = load_dataset("brendel-group/MATH-Beyond", split="test", revision=hf_revision)

    full_path = out_dir / "math_beyond_full_181.jsonl"
    primary_path = out_dir / "math_beyond_math_b_i_base.jsonl"
    complement_path = out_dir / "math_beyond_complement_141.jsonl"
    strict_path = out_dir / "math_beyond_hf_strict_all_models.jsonl"
    smoke_path = out_dir / "math_beyond_smoke.jsonl"
    manifest_path = out_dir / "benchmark_manifest.json"

    all_records = [row_to_record(i, ds[i]) for i in range(len(ds))]
    write_jsonl(full_path, all_records)

    unsolved_cols = list_unsolved_columns(ds.column_names)
    p1024_cols = _pass_at_1024_columns(ds.column_names)

    if args.primary_mode == "paper_math_b_i_base":
        primary_indices = filter_indices_math_b_i_base(ds)
        primary_description = (
            "MATH-B-I (paper Sec. 3.2.3 Base Models): all listed base-model pass@1024 == 0."
        )
        primary_columns = list(MATH_B_I_BASE_PASS1024_COLUMNS)
    elif args.primary_mode == "intersection_all_unsolved":
        primary_indices = filter_indices_strict_all_hf_models(ds)
        primary_description = "Strict HF intersection: all 21 *_unsolved True."
        primary_columns = unsolved_cols
    else:
        primary_indices = filter_indices_unsolved_column(ds, args.unsolved_column)
        primary_description = f"Single *_unsolved column: {args.unsolved_column}"
        primary_columns = [args.unsolved_column]

    primary_records = [row_to_record(i, ds[i]) for i in primary_indices]
    write_jsonl(primary_path, primary_records)

    # Smoke pool: first 4 rows of the primary pool — deterministic, no extra filtering.
    write_jsonl(smoke_path, primary_records[:4])

    # Phase 0 VAE pretraining pool: complement of the MATH-B-I hard-40 in the full 181.
    # Always relative to the paper's MATH-B-I base definition, not --primary-mode, because
    # the Phase 0 pool is defined as "not the RL evaluation target" — which is always the
    # hard 40, regardless of what primary_mode the caller chose.
    math_b_i_indices_set = (
        set(primary_indices)
        if args.primary_mode == "paper_math_b_i_base"
        else set(filter_indices_math_b_i_base(ds))
    )
    complement_indices = [i for i in range(len(ds)) if i not in math_b_i_indices_set]
    write_jsonl(complement_path, [row_to_record(i, ds[i]) for i in complement_indices])

    strict_indices: list[int] | None = None
    if not args.no_secondary_strict:
        strict_indices = filter_indices_strict_all_hf_models(ds)
        write_jsonl(strict_path, [row_to_record(i, ds[i]) for i in strict_indices])

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p.resolve())

    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_doc": "reports/DATA_PROTOCOL.md",
        "hf_dataset": "brendel-group/MATH-Beyond",
        "hf_split": "test",
        "hf_revision": hf_revision,
        "hf_revision_source": (
            "cli"
            if args.hf_revision and args.hf_revision.strip()
            else (
                "env MATH_BEYOND_HF_REVISION"
                if os.environ.get("MATH_BEYOND_HF_REVISION")
                else "configs/math_beyond_hf_revision.txt"
            )
        ),
        "paper": {
            "title": "MATH-Beyond",
            "arxiv": "https://arxiv.org/abs/2510.11653",
            "math_b_i_reported_n": 41,
            "math_b_i_definition": "Section 3.2.3; intersection over Base Models at pass@1024",
        },
        "primary_pool": {
            "path": rel(primary_path),
            "mode": args.primary_mode,
            "description": primary_description,
            "row_count": len(primary_indices),
            "columns_used": primary_columns,
        },
        "phase0_pretrain_pool": {
            "path": rel(complement_path),
            "description": (
                "Phase 0 VAE pretraining pool: full-181 minus MATH-B-I hard-40. "
                "Problems where at least one base model has pass@1024 > 0. "
                "See reports/latent_markov_design.md."
            ),
            "row_count": len(complement_indices),
            "complement_of": "primary_pool (MATH-B-I base)",
        },
        "full_pool": {"path": rel(full_path), "row_count": len(ds)},
        "smoke_pool": {"path": rel(smoke_path), "row_count": min(4, len(primary_records))},
        "secondary_strict_pool": None,
        "hub_reference": "https://huggingface.co/datasets/brendel-group/MATH-Beyond",
        "reproducibility": {
            "source_index": "0-based index into HF test split at hf_revision",
            "row_order": "Hugging Face datasets default test split order",
            "library_versions_at_build": pinned_library_versions(),
            "requirements_pin_reference": "requirements.txt (datasets, huggingface_hub)",
        },
        "all_unsolved_columns": unsolved_cols,
        "all_pass_at_1024_columns_count": len(p1024_cols),
    }

    if strict_indices is not None:
        manifest["secondary_strict_pool"] = {
            "path": rel(strict_path),
            "description": "AND of all 21 *_unsolved (stricter than paper MATH-B-I).",
            "row_count": len(strict_indices),
        }

    if full_path.is_file():
        manifest["sha256_math_beyond_full_181_jsonl"] = sha256_file(full_path)
    if primary_path.is_file():
        manifest["sha256_math_beyond_math_b_i_base_jsonl"] = sha256_file(primary_path)
    if complement_path.is_file():
        manifest["sha256_math_beyond_complement_141_jsonl"] = sha256_file(complement_path)
    if smoke_path.is_file():
        manifest["sha256_math_beyond_smoke_jsonl"] = sha256_file(smoke_path)
    if strict_indices is not None and strict_path.is_file():
        manifest["sha256_math_beyond_hf_strict_all_models_jsonl"] = sha256_file(strict_path)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {len(ds)} rows -> {full_path}", file=sys.stderr)
    print(f"Wrote {len(primary_records)} rows -> {primary_path}  ({args.primary_mode})", file=sys.stderr)
    print(f"Wrote {len(complement_indices)} rows -> {complement_path}  (phase0 pretrain complement)", file=sys.stderr)
    print(f"Wrote {min(4, len(primary_records))} rows -> {smoke_path}  (smoke subset)", file=sys.stderr)
    if strict_indices is not None:
        print(f"Wrote {len(strict_indices)} rows -> {strict_path}", file=sys.stderr)
    print(f"Wrote {manifest_path}", file=sys.stderr)

    if args.primary_mode == "paper_math_b_i_base" and len(primary_records) != 41:
        print(
            f"NOTE: Paper reports N=41 MATH-B-I; at hf_revision={hf_revision!r} base gauntlet gives N={len(primary_records)}. "
            "See reports/DATA_PROTOCOL.md.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
