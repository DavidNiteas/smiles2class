"""Multi-label greedy train/vali/test split.

Walk the label columns in order; for each label assign its unassigned
molecules to test until that split covers ~20% of the label's molecules, then
to vali until ~20%, and put the rest in train.  A molecule is never assigned
twice.

Iteration order is sorted, so the result is deterministic; ``--seed`` is
recorded only.  Molecules that end up in no split (unlabelled, or labels
already saturated) are assigned to train, so train/vali/test is always a
complete partition.

Output ``splits.json`` contains exactly ``{"train": [...], "vali": [...],
"test": [...]}``; per-label coverage goes to the optional stats file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd
import typer

from pipelines import (
    as_python_scalar,
    load_config,
    python_ids,
    read_table,
)

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"

app = typer.Typer(
    add_completion=False,
    help="Greedy multi-label split into train/vali/test (20%/20%).",
)


def label_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return ``<prefix><n>`` columns ordered by their numeric suffix."""
    found = [
        column
        for column in frame.columns
        if column.startswith(prefix) and column[len(prefix) :].isdigit()
    ]
    return sorted(found, key=lambda column: int(column[len(prefix) :]))


def greedy_split(
    frame: pd.DataFrame,
    columns: Sequence[str],
    id_column: str = "id",
    test_ratio: float = 0.2,
    vali_ratio: float = 0.2,
) -> dict[str, list[Any]]:
    """Split ``frame`` by label coverage; returns ordered train/vali/test ids."""
    if id_column not in frame.columns:
        raise ValueError(f"split input is missing the id column {id_column!r}")
    if frame[id_column].isna().any():
        raise ValueError(f"split input contains missing ids in column {id_column!r}")
    if not columns:
        raise ValueError("no label columns found; check the label prefix")
    if not 0.0 <= test_ratio <= 1.0 or not 0.0 <= vali_ratio <= 1.0:
        raise ValueError("test_ratio and vali_ratio must be within [0, 1]")

    ids = python_ids(frame[id_column].tolist())
    if len(set(ids)) != len(ids):
        raise ValueError(f"split input contains duplicate ids in column {id_column!r}")

    train: set[Any] = set()
    vali: set[Any] = set()
    test: set[Any] = set()

    for column in columns:
        members = {
            as_python_scalar(value)
            for value in frame.loc[frame[column] == 1, id_column].tolist()
        }
        if not members:
            continue
        total = len(members)
        in_test = len(members & test)
        in_vali = len(members & vali)
        for item in sorted(members - (train | vali | test), key=str):
            if in_test / total < test_ratio:
                test.add(item)
                in_test += 1
            elif in_vali / total < vali_ratio:
                vali.add(item)
                in_vali += 1
            else:
                train.add(item)

    # Completeness: anything the label loop could not place goes to train.
    train |= set(ids) - (train | vali | test)

    return {
        split: [identifier for identifier in ids if identifier in members]
        for split, members in (("train", train), ("vali", vali), ("test", test))
    }


def compute_stats(
    frame: pd.DataFrame,
    splits: Mapping[str, Sequence[Any]],
    columns: Sequence[str],
    id_column: str = "id",
    test_ratio: float = 0.2,
    vali_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Build the statistics payload: split sizes and per-label coverage."""
    ids = python_ids(frame[id_column].tolist())
    total = len(ids)
    counts = {name: len(values) for name, values in splits.items()}

    split_sets = {name: set(values) for name, values in splits.items()}
    coverage: dict[str, dict[str, int]] = {}
    below_test: list[str] = []
    below_vali: list[str] = []
    without_samples: list[str] = []
    for column in columns:
        members = {
            as_python_scalar(value)
            for value in frame.loc[frame[column] == 1, id_column].tolist()
        }
        size = len(members)
        if size == 0:
            without_samples.append(column)
            continue
        entry = {
            "total": size,
            "train": len(members & split_sets["train"]),
            "vali": len(members & split_sets["vali"]),
            "test": len(members & split_sets["test"]),
        }
        coverage[column] = entry
        if entry["test"] / size < test_ratio:
            below_test.append(column)
        if entry["vali"] / size < vali_ratio:
            below_vali.append(column)

    label_sums = frame[list(columns)].sum(axis=1)
    return {
        "total": total,
        "counts": counts,
        "ratios": {
            name: round(value / total, 6) if total else 0.0
            for name, value in counts.items()
        },
        "params": {
            "seed": seed,
            "test_ratio": test_ratio,
            "vali_ratio": vali_ratio,
            "id_column": id_column,
        },
        "coverage": coverage,
        "labels_total": len(columns),
        "labels_without_samples": len(without_samples),
        "labels_without_samples_list": without_samples,
        "labels_below_target": {"test": below_test, "vali": below_vali},
        "rows_without_labels": int((label_sums == 0).sum()),
    }


def run(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    stats_output: Path | str | None = None,
    config: Path | str | None = None,
    seed: int | None = None,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    """Read the multi-hot table, write ``splits.json`` and optional stats."""
    settings = load_config(DEFAULT_CONFIG, config)
    section = settings.get("split", {}) or {}
    columns_config = settings["columns"]

    input_path = input_path or section.get("input")
    output_path = output_path or section.get("output")
    stats_output = stats_output or section.get("stats_output")
    if input_path is None:
        raise ValueError("missing input: pass --input or set split.input in the config")
    if output_path is None:
        raise ValueError("missing output: pass --output or set split.output in the config")

    seed = int(seed if seed is not None else section.get("seed", 42))
    test_ratio = float(section.get("test_ratio", 0.2))
    vali_ratio = float(section.get("vali_ratio", 0.2))
    id_column = columns_config["split"]["id"]
    prefix = columns_config["labels"]["prefix"]

    frame = read_table(input_path)
    columns = label_columns(frame, prefix)
    splits = greedy_split(frame, columns, id_column, test_ratio, vali_ratio)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    stats = compute_stats(
        frame, splits, columns, id_column, test_ratio, vali_ratio, seed
    )
    if stats_output:
        stats_path = Path(stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return splits, stats


@app.command()
def main(
    input: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Multi-hot table written by pipelines.hmdb.labels."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Where to write splits.json."
    ),
    stats_output: Optional[Path] = typer.Option(
        None, "--stats-output", help="Optional per-label coverage statistics JSON."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help=f"YAML config (defaults: {DEFAULT_CONFIG.name})."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="Recorded in stats; the greedy algorithm is deterministic."
    ),
) -> None:
    """Split the multi-hot table into train/vali/test."""
    _, stats = run(
        input_path=input,
        output_path=output,
        stats_output=stats_output,
        config=config,
        seed=seed,
    )
    typer.echo(
        "split: train={train} vali={vali} test={test} total={total}; "
        "labels with samples: {seen}/{labels}".format(
            total=stats["total"],
            seen=len(stats["coverage"]),
            labels=stats["labels_total"],
            **stats["counts"],
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
