"""Build the 426-label multi-hot table from ``Alternative Parents``.

The label vocabulary and its order are authoritative in
``src/smiles2class/labels.json`` (same order as the model weights); this module
never re-derives or reorders them.

Input rows may encode ``Alternative Parents`` as a Python list literal
(``"['a', 'b']"``), a JSON array (``'["a", "b"]'``) or a delimited string
(``"a; b"`` / ``"a|b"``).  A bare ``,`` is accepted only when every piece is a
known label, because several ClassyFire terms contain commas themselves.

Outputs:

- ``label_map.json`` -- the authority list, item for item;
- ``labels.parquet`` (or ``.csv``) -- ``id``, ``smiles`` and ``label_0..label_N``.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import typer

from pipelines import REPO_ROOT, load_config, read_table, require_columns, write_table

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"
DEFAULT_LABEL_MAP = REPO_ROOT / "src" / "smiles2class" / "labels.json"

app = typer.Typer(
    add_completion=False,
    help="Turn Alternative Parents into a multi-hot table aligned with labels.json.",
)

_EMPTY_STRINGS = {"", "[]", "()", "null", "none", "nan"}


def load_authority_labels(path: Path | str | None = None) -> list[str]:
    """Load the label vocabulary; order is preserved exactly as written."""
    label_path = Path(path) if path else DEFAULT_LABEL_MAP
    labels = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ValueError(f"label map must be a JSON list of strings: {label_path}")
    return labels


def _clean_parts(values: Iterable[Any]) -> list[str]:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            parts.append(text)
    return parts


def _literal_parts(text: str) -> Optional[list[str]]:
    for loader in (ast.literal_eval, json.loads):
        try:
            parsed = loader(text)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(parsed, (list, tuple, set, frozenset)):
            return _clean_parts(parsed)
        if isinstance(parsed, str):
            return _clean_parts([parsed])
    return None


def parse_alternative_parents(
    value: Any, vocabulary: Optional[set[str]] = None
) -> list[str]:
    """Parse one ``Alternative Parents`` cell into a list of label names.

    ``vocabulary`` (the authority labels) enables comma disambiguation: a comma
    is treated as a separator only when every resulting piece is a known label.
    """
    if value is None:
        return []
    if not isinstance(value, (str, bytes)) and hasattr(value, "__iter__"):
        return _clean_parts(list(value))
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _EMPTY_STRINGS:
            return []
        parsed = _literal_parts(text)
        if parsed is not None:
            return parsed
        for separator in ("|", ";"):
            if separator in text:
                return _clean_parts(text.split(separator))
        if "," in text:
            pieces = _clean_parts(re.split(r",\s*", text))
            if vocabulary is None or all(piece in vocabulary for piece in pieces):
                return pieces
        return [text]

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):  # pragma: no cover - exotic cell types
        pass
    return _clean_parts([value])


def build_multi_hot(
    frame: pd.DataFrame,
    labels: Sequence[str],
    columns: Mapping[str, Any],
) -> pd.DataFrame:
    """Return ``id`` + ``smiles`` + one 0/1 column per label, in authority order."""
    config = columns["labels"]
    require_columns(frame, [config["smiles"], config["parent"]], "labels")

    vocabulary = set(labels)
    id_column = next(
        (candidate for candidate in config["id_candidates"] if candidate in frame.columns),
        None,
    )
    ids = (
        frame[id_column].tolist()
        if id_column is not None
        else list(range(len(frame)))
    )
    if id_column is None:
        warnings.warn(
            f"no id column found among {config['id_candidates']}; using positional ids",
            stacklevel=2,
        )

    matrix = np.zeros((len(frame), len(labels)), dtype="int8")
    index = {label: position for position, label in enumerate(labels)}
    for row, parents in enumerate(frame[config["parent"]]):
        for parent in set(parse_alternative_parents(parents, vocabulary)):
            position = index.get(parent)
            if position is not None:
                matrix[row, position] = 1

    prefix = config["prefix"]
    label_frame = pd.DataFrame(
        matrix, columns=[f"{prefix}{position}" for position in range(len(labels))]
    )
    head = pd.DataFrame(
        {
            config["output_id"]: ids,
            config["smiles"]: frame[config["smiles"]].to_numpy(),
        }
    )
    return pd.concat([head, label_frame], axis=1)


def _never_seen(table: pd.DataFrame, labels: Sequence[str], prefix: str) -> list[str]:
    label_columns = [f"{prefix}{position}" for position in range(len(labels))]
    totals = table[label_columns].sum(axis=0)
    return [name for name, total in zip(labels, totals) if total == 0]


def run(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    labels_output: Path | str | None = None,
    config: Path | str | None = None,
    label_map: Path | str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build and write ``label_map.json`` plus the multi-hot table."""
    settings = load_config(DEFAULT_CONFIG, config)
    section = settings.get("labels", {}) or {}
    columns = settings["columns"]

    input_path = input_path or section.get("input")
    output_path = output_path or section.get("output")
    labels_output = labels_output or section.get("labels_output")
    label_map = label_map or section.get("path")

    if input_path is None:
        raise ValueError("missing input: pass --input or set labels.input in the config")
    if output_path is None:
        raise ValueError("missing output: pass --output or set labels.output in the config")

    labels = load_authority_labels(label_map)
    expected = section.get("expected_count")
    if expected is not None and len(labels) != int(expected):
        warnings.warn(
            f"label map has {len(labels)} labels, expected {expected}: {label_map or DEFAULT_LABEL_MAP}",
            stacklevel=2,
        )

    table = build_multi_hot(read_table(input_path), labels, columns)

    map_path = Path(output_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if labels_output is None:
        labels_output = map_path.parent / "labels.parquet"
    write_table(table, labels_output)

    never_seen = _never_seen(table, labels, columns["labels"]["prefix"])
    if never_seen:
        preview = ", ".join(never_seen[:3])
        warnings.warn(
            f"{len(never_seen)} of {len(labels)} labels never appear in {input_path} "
            f"(e.g. {preview})",
            stacklevel=2,
        )
    return table, labels


@app.command()
def main(
    input: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Table with SMILES + Alternative Parents."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Where to write label_map.json."
    ),
    labels_output: Optional[Path] = typer.Option(
        None,
        "--labels-output",
        help="Multi-hot table (.csv/.parquet); default: sibling labels.parquet.",
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help=f"YAML config (defaults: {DEFAULT_CONFIG.name})."
    ),
    label_map: Optional[Path] = typer.Option(
        None,
        "--label-map",
        help="Label authority JSON; default: src/smiles2class/labels.json.",
    ),
) -> None:
    """Write the label map and the multi-hot label table."""
    table, labels = run(
        input_path=input,
        output_path=output,
        labels_output=labels_output,
        config=config,
        label_map=label_map,
    )
    positives = int(table.iloc[:, 2:].to_numpy().sum())
    typer.echo(
        f"wrote {len(labels)} labels and the multi-hot table "
        f"({len(table)} rows, {positives} positive cells)"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
