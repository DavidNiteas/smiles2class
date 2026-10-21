"""Repo-local data pipelines (not shipped in the wheel).

The package lives outside ``src/smiles2class`` on purpose: ``pip install
smiles2class`` does not pull it in, while the sdist still carries it.  It only
needs ``pandas`` (plus ``pyarrow`` for parquet and ``PyYAML`` for configs).

Helpers here are shared by the ``pipelines.hmdb.*`` entry points.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` (returns a new dict)."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(*paths: Path | str | None) -> dict[str, Any]:
    """Load and deep-merge YAML config files, left to right."""
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise ModuleNotFoundError(
            "PyYAML is required by the data pipelines: pip install pyyaml"
        ) from exc

    merged: dict[str, Any] = {}
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"config must be a mapping: {path}")
        merged = deep_merge(merged, data)
    return merged


def read_table(path: Path | str) -> pd.DataFrame:
    """Read a table by suffix (``.csv`` or ``.parquet``)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input format {suffix!r} for {path}; use .csv or .parquet")


def write_table(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write a table by suffix (``.csv`` or ``.parquet``), creating parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"unsupported output format {suffix!r} for {path}; use .csv or .parquet")
    return path


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], what: str
) -> None:
    """Raise a readable error when required columns are missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{what} table is missing required column(s): {missing}")


def as_python_scalar(value: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python values for JSON output."""
    if isinstance(value, (str, bytes)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, AttributeError):  # pragma: no cover - exotic dtypes
            return value
    return value


def python_ids(values: Sequence[Any]) -> list[Any]:
    """Normalise an id sequence for JSON serialisation and set operations."""
    return [as_python_scalar(value) for value in values]


__all__ = [
    "REPO_ROOT",
    "as_python_scalar",
    "deep_merge",
    "load_config",
    "python_ids",
    "read_table",
    "require_columns",
    "write_table",
]
