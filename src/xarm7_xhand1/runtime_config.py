"""Portable deployment configuration for the Quest capture supervisor."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PATH_FIELDS = {"root", "hardware_config", "vendor_dir", "vendor_config", "vendor_python", "lerobot_src"}
POSITIVE_FIELDS = {"rate_hz", "translation_scale", "max_translation_mm", "max_rotation_rad", "max_step_mm", "max_orientation_step_rad"}
ALLOWED_FIELDS = PATH_FIELDS | POSITIVE_FIELDS | {"task", "repo_id", "format", "no_cameras", "no_wear"}


def load_runtime_config(path: Path) -> dict[str, Any]:
    """Load defaults; resolve relative paths beside YAML, and reject unsafe keys."""
    path = path.expanduser().resolve()
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Runtime config must be a YAML mapping")
    unknown = set(values) - ALLOWED_FIELDS
    if unknown:
        raise ValueError("Unknown runtime fields: " + ", ".join(sorted(unknown)))
    result = dict(values)
    for name, value in result.items():
        if name in PATH_FIELDS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path")
            resolved = Path(value).expanduser()
            result[name] = (path.parent / resolved).resolve() if not resolved.is_absolute() else resolved
        elif name in POSITIVE_FIELDS:
            import math
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            if name == "rate_hz" and (not isinstance(value, int) or value < 1):
                raise ValueError("rate_hz must be a positive integer")
        elif name in {"no_cameras", "no_wear"}:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a YAML boolean")
        elif name == "format":
            if value not in {"lerobot-v2.1", "npz"}:
                raise ValueError("format must be lerobot-v2.1 or npz")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    return result
