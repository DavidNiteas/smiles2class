"""Model-directory helpers.

No download logic: the caller points ``from_pretrained`` at a local directory.
"""

from __future__ import annotations

from pathlib import Path


def resolve_model_dir(path: str | Path) -> Path:
    path = Path(path)
    required = [
        path / "label_map.json",
        path / "main_model" / "config.json",
        path / "main_model" / "model.safetensors",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "model directory is incomplete; missing: " + ", ".join(missing)
        )
    return path


def has_vice(path: str | Path) -> bool:
    return (Path(path) / "vice_model").is_dir()
