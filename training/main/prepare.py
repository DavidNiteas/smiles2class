"""Build a pickled training datamodule from pipeline outputs.

Inputs are the ``labels.parquet`` (``id``, ``smiles``, ``label_0..label_N``)
and ``splits.json`` files produced by ``pipelines.hmdb.*``.  Every SMILES is
featurised with :func:`unimol_lite.data.smiles2unimolitem` (``coord_mode``
defaults to ``auto``) and stored as the four tensors the Uni-Mol encoder needs
plus the multi-hot ``label`` row.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch
import typer
from unimol_lite.data import smiles2unimolitem

from training.main.dataset import UniClassifyDataModule, UniClassifyTrainDataset

app = typer.Typer(add_completion=False, help="labels.parquet + splits.json -> training pickle")


def label_columns(frame: pd.DataFrame, prefix: str = "label_") -> list[str]:
    """Return ``<prefix><n>`` columns ordered by their numeric suffix."""
    found = [
        column
        for column in frame.columns
        if column.startswith(prefix) and column[len(prefix) :].isdigit()
    ]
    return sorted(found, key=lambda column: int(column[len(prefix) :]))


def _load_ids(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("train", "vali", "test")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing split keys: {missing}")
    return data


def _select(frame: pd.DataFrame, ids: Iterable[Any], id_column: str) -> pd.DataFrame:
    if id_column not in frame.columns:
        raise ValueError(f"labels table is missing the id column {id_column!r}")
    keyed = {str(value): position for position, value in enumerate(frame[id_column].tolist())}
    try:
        positions = [keyed[str(identifier)] for identifier in ids]
    except KeyError as exc:
        raise KeyError(f"split id {exc.args[0]!r} is not present in the labels table") from exc
    return frame.iloc[positions].reset_index(drop=True)


def _featurize_one(smiles: str, coord_mode: str, timeout: Optional[float]):
    return smiles2unimolitem(
        smiles,
        coord_mode=coord_mode,
        timeout=timeout,
        return_meta=True,
    )


def _split_dataset(
    frame: pd.DataFrame,
    ids: list[Any],
    *,
    id_column: str,
    smiles_column: str,
    columns: list[str],
    workers: int,
    coord_mode: str,
    timeout: Optional[float],
) -> UniClassifyTrainDataset:
    selected = _select(frame, ids, id_column)
    smiles = selected[smiles_column].tolist()

    if workers > 1 and len(smiles) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda item: _featurize_one(item, coord_mode, timeout), smiles))
    else:
        results = [_featurize_one(item, coord_mode, timeout) for item in smiles]

    items: list[Any] = []
    kept_ids: list[Any] = []
    labels: list[torch.Tensor] = []
    dropped = 0
    for identifier, (item, meta), row in zip(ids, results, selected[columns].itertuples(index=False)):
        if item is None:
            dropped += 1
            continue
        items.append(item)
        kept_ids.append(identifier)
        labels.append(torch.tensor([int(value) for value in row], dtype=torch.long))

    if dropped:
        typer.echo(f"warning: dropped {dropped} unfeaturisable SMILES", err=True)
    if not items:
        raise ValueError("no molecule could be featurised; check the input table")

    return UniClassifyTrainDataset(
        index=kept_ids,
        src_tokens=[item["src_tokens"] for item in items],
        src_distance=[item["src_distance"] for item in items],
        src_coord=[item["src_coord"] for item in items],
        src_edge_type=[item["src_edge_type"] for item in items],
        label=labels,
    )


def build_datamodule(
    labels_path: Path | str,
    splits_path: Path | str,
    *,
    label_map_path: Optional[Path | str] = None,
    id_column: str = "id",
    smiles_column: str = "smiles",
    workers: int = 8,
    coord_mode: str = "auto",
    timeout: Optional[float] = 30.0,
    batch_size: int = 32,
    num_workers: int = 0,
    expected_labels: Optional[int] = None,
) -> UniClassifyDataModule:
    """Featurise all splits and return the in-memory datamodule."""
    labels_path = Path(labels_path)
    splits_path = Path(splits_path)
    frame = pd.read_parquet(labels_path)
    columns = label_columns(frame)
    if not columns:
        raise ValueError(f"no label_* columns found in {labels_path}")
    if expected_labels is not None and len(columns) != int(expected_labels):
        raise ValueError(f"labels table has {len(columns)} labels, expected {expected_labels}")

    label_map_path = Path(label_map_path) if label_map_path else labels_path.parent / "label_map.json"
    if label_map_path.is_file():
        names = json.loads(label_map_path.read_text(encoding="utf-8"))
        if len(names) != len(columns):
            raise ValueError(
                f"{label_map_path} has {len(names)} labels but the table has {len(columns)}"
            )
    else:
        names = [f"{column}" for column in columns]
        typer.echo(f"warning: {label_map_path} not found; using column names as labels", err=True)

    splits = _load_ids(splits_path)
    options = dict(
        id_column=id_column,
        smiles_column=smiles_column,
        columns=columns,
        workers=workers,
        coord_mode=coord_mode,
        timeout=timeout,
    )
    trainset = _split_dataset(frame, splits["train"], **options)
    valiset = _split_dataset(frame, splits["vali"], **options)
    testset = _split_dataset(frame, splits["test"], **options)
    return UniClassifyDataModule(
        trainset,
        valiset,
        testset,
        batch_size=batch_size,
        num_workers=num_workers,
        labels=list(names),
    )


def prepare(
    labels_path: Path | str,
    splits_path: Path | str,
    output_path: Path | str,
    **kwargs,
) -> UniClassifyDataModule:
    """Build and pickle the datamodule."""
    datamodule = build_datamodule(labels_path, splits_path, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    datamodule.save_pickle(str(output_path))
    return datamodule


@app.command()
def main(
    labels: Path = typer.Option(..., "--labels", help="labels.parquet with id/smiles/label_*."),
    splits: Path = typer.Option(..., "--splits", help="splits.json from pipelines.hmdb.split."),
    output: Path = typer.Option(..., "--output", help="Where to write the pickled datamodule."),
    label_map: Optional[Path] = typer.Option(
        None, "--label-map", help="label_map.json; defaults to the labels file's directory."
    ),
    id_column: str = typer.Option("id", "--id-column"),
    smiles_column: str = typer.Option("smiles", "--smiles-column"),
    workers: int = typer.Option(8, "--workers", min=1, help="Featurisation worker threads."),
    coord_mode: str = typer.Option("auto", "--coord-mode", help="auto / 3d / 2d."),
    timeout: float = typer.Option(30.0, "--timeout", help="Per-molecule seconds; 0 disables."),
    batch_size: int = typer.Option(32, "--batch-size", help="Stored in the datamodule."),
    num_workers: int = typer.Option(0, "--num-workers", help="Dataloader workers."),
    expected_labels: Optional[int] = typer.Option(None, "--expected-labels"),
):
    datamodule = prepare(
        labels,
        splits,
        output,
        label_map_path=label_map,
        id_column=id_column,
        smiles_column=smiles_column,
        workers=workers,
        coord_mode=coord_mode,
        timeout=None if timeout <= 0 else timeout,
        batch_size=batch_size,
        num_workers=num_workers,
        expected_labels=expected_labels,
    )
    sizes = (len(datamodule.trainset), len(datamodule.valiset), len(datamodule.testset))
    typer.echo(f"wrote {output} (train/vali/test = {sizes[0]}/{sizes[1]}/{sizes[2]})")


if __name__ == "__main__":
    app()
