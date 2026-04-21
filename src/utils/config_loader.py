"""Load YAML configs that use ``extends:`` (shallow merge per key; nested dicts merged recursively)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_yaml_with_extends(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Load ``path``; if it contains ``extends``, load that file first (relative to ``path``'s parent), then merge."""
    path = path.resolve()
    base_dir = path.parent
    root = root.resolve() if root else base_dir
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root of {path}")
    extends = data.pop("extends", None)
    if extends is None:
        return data
    parent_path = (root / extends).resolve() if not Path(extends).is_absolute() else Path(extends)
    if not parent_path.is_file():
        parent_path = (base_dir / extends).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"extends target not found: {extends!r} (resolved from {path})")
    parent = load_yaml_with_extends(parent_path, root=root)
    return _deep_merge(parent, data)
