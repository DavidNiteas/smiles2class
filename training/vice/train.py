"""Train the vice model: one AutoGluon predictor per label.

Consumes the table written by :mod:`training.vice.prepare` (``vec_*`` +
``label_*`` + ``cluster_id`` [+ ``balance_weight``]) and writes one predictor
per trained label::

    <output>/<i>/            # TabularPredictor.load(<output>/<i>)
    <output>/label_map.json  # labels of the trained subset, input order

``<output>`` is normally ``<model_dir>/vice_model``; the model root keeps its
own ``label_map.json`` (all labels) from main-model training.

    python -m training.vice.train --dataset vice.parquet \
        --label-map label_map.json --output <model_dir>/vice_model
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import typer
import yaml
from autogluon.tabular import TabularDataset, TabularPredictor

from training.vice.sampling import (
    WEIGHT_COLUMN,
    SingleClassifyDataset,
    label_columns,
)

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"

# AutoGluon 1.2 reserves the literal ``balance_weight`` for its built-in
# predefined weighting strategy and rejects a data column of that name, so the
# per-row weights are handed over under a neutral alias.
SAMPLE_WEIGHT_COLUMN = "vice_weight"

app = typer.Typer(add_completion=False, help="Train per-label AutoGluon vice predictors.")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read the vice YAML config (``sampling:`` / ``training:`` sections)."""
    data = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(data)


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    return dict(section) if isinstance(section, Mapping) else {}


def _read_labels(label_map: Path | str | list[str] | None, count: int) -> list[str]:
    """Return the label names for the first ``count`` label columns."""
    if label_map is None:
        return [f"label_{index}" for index in range(count)]
    if isinstance(label_map, (str, Path)):
        names = json.loads(Path(label_map).read_text(encoding="utf-8"))
    else:
        names = list(label_map)
    if len(names) < count:
        raise ValueError(f"label map has {len(names)} labels, need at least {count}")
    return list(names[:count])


def train_class(
    dataset: SingleClassifyDataset,
    index: int,
    output: Path | str,
    *,
    presets: Iterable[str] = ("medium_quality",),
    time_limit: Optional[float] = None,
    num_cpus: Optional[int] = None,
    fit_strategy: str = "parallel",
    test_size: float = 0.5,
    max_magnification: float = 9,
    seed: int = 42,
) -> TabularPredictor:
    """Fit one binary ``TabularPredictor`` for label column ``index``."""
    output = Path(output) / str(index)
    output.mkdir(parents=True, exist_ok=True)
    frame = dataset[index]
    train_set, _test_set = dataset.train_test_split(
        frame,
        test_size=test_size,
        max_magnification=max_magnification,
        random_state=seed,
    )

    options: dict[str, Any] = {
        "label": "label",
        "problem_type": "binary",
        "eval_metric": "f1",
        "path": str(output),
        "verbosity": 0,
    }
    if WEIGHT_COLUMN in train_set.columns:
        train_set = train_set.copy()
        train_set[SAMPLE_WEIGHT_COLUMN] = train_set[WEIGHT_COLUMN]
        train_set = train_set.drop(columns=[WEIGHT_COLUMN])
        options["sample_weight"] = SAMPLE_WEIGHT_COLUMN
    predictor = TabularPredictor(**options)

    fit_options: dict[str, Any] = {
        "presets": list(presets),
        "fit_strategy": fit_strategy,
        "num_cpus": num_cpus if num_cpus is not None else "auto",
    }
    if time_limit is not None:
        fit_options["time_limit"] = float(time_limit)
    predictor.fit(TabularDataset(train_set), **fit_options)
    return predictor


def train_vice(
    data: pd.DataFrame | SingleClassifyDataset,
    output: Path | str,
    *,
    label_map: Path | str | list[str] | None = None,
    max_classes: Optional[int] = None,
    presets: Iterable[str] = ("medium_quality",),
    time_limit: Optional[float] = None,
    num_cpus: Optional[int] = None,
    fit_strategy: str = "parallel",
    test_size: float = 0.5,
    max_magnification: float = 9,
    seed: int = 42,
) -> list[TabularPredictor]:
    """Train the first ``max_classes`` labels and write ``label_map.json``."""
    dataset = data if isinstance(data, SingleClassifyDataset) else SingleClassifyDataset(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    count = len(dataset)
    if max_classes is not None:
        count = min(count, int(max_classes))
    if count <= 0:
        raise ValueError("no label columns selected for training")
    labels = _read_labels(label_map, count)

    random.seed(seed)
    np.random.seed(seed)

    predictors: list[TabularPredictor] = []
    for index in range(count):
        predictor = train_class(
            dataset,
            index,
            output,
            presets=presets,
            time_limit=time_limit,
            num_cpus=num_cpus,
            fit_strategy=fit_strategy,
            test_size=test_size,
            max_magnification=max_magnification,
            seed=seed,
        )
        predictors.append(predictor)
        typer.echo(f"trained {index}/{count - 1}: {labels[index]} -> {output / str(index)}")

    (output / "label_map.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return predictors


@app.command()
def main(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Parquet from training.vice.prepare."),
    output: Path = typer.Option(
        ..., "--output", "-o", help="vice_model directory; predictors go to <output>/<i>."
    ),
    label_map: Optional[Path] = typer.Option(
        None, "--label-map", "-l", help="label_map.json for the label names."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML config; defaults to training/vice/configs/default.yaml."
    ),
    max_classes: Optional[int] = typer.Option(
        None, "--max-classes", min=1, help="Train only the first N labels."
    ),
    time_limit: Optional[float] = typer.Option(None, "--time-limit", help="Seconds per predictor."),
    presets: Optional[list[str]] = typer.Option(
        None, "--presets", help="AutoGluon presets, e.g. --presets medium_quality."
    ),
    num_cpus: Optional[int] = typer.Option(None, "--num-cpus", min=1),
    test_size: Optional[float] = typer.Option(None, "--test-size"),
    max_magnification: Optional[float] = typer.Option(None, "--max-magnification"),
    seed: Optional[int] = typer.Option(None, "--seed"),
):
    settings = load_config(config)
    sampling = _section(settings, "sampling")
    training = _section(settings, "training")

    frame = pd.read_parquet(dataset)
    if not label_columns(frame):
        raise ValueError(f"no label_* columns found in {dataset}")

    train_vice(
        frame,
        output,
        label_map=label_map,
        max_classes=max_classes if max_classes is not None else training.get("max_classes"),
        presets=presets if presets else training.get("presets", ["medium_quality"]),
        time_limit=time_limit if time_limit is not None else training.get("time_limit"),
        num_cpus=num_cpus if num_cpus is not None else training.get("num_cpus", os.cpu_count()),
        fit_strategy=training.get("fit_strategy", "parallel"),
        test_size=test_size if test_size is not None else sampling.get("test_size", 0.5),
        max_magnification=(
            max_magnification
            if max_magnification is not None
            else sampling.get("max_magnification", 9)
        ),
        seed=seed if seed is not None else sampling.get("seed", 42),
    )


if __name__ == "__main__":
    app()