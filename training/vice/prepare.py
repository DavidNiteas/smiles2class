"""Build the vice training table from ``labels.parquet`` + the main model.

Every SMILES is run through the main model and stored as a 512-dimensional CLS
embedding (``vec_0..vec_511``), next to the ``label_*`` columns and a
``cluster_id`` column (zeros when the input has none).  A pre-existing
``balance_weight`` column is carried through untouched.

    python -m training.vice.prepare --input labels.parquet \
        --model-dir model_dir --output vice.parquet
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import typer
from unimol_lite.data import unimol_collate_fn

from smiles2class.featurize import featurize
from smiles2class.model import load_main_model
from training.vice.sampling import (
    CLUSTER_COLUMN,
    WEIGHT_COLUMN,
    label_columns,
    vector_columns,
)

app = typer.Typer(add_completion=False, help="labels.parquet + main model -> vice training table")

DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 30.0


def _chunks(seq: list, size: int) -> Iterator[list]:
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def _resolve_main_dir(model_dir: Path | str) -> Path:
    """Accept either a model root or a ``main_model/`` directory."""
    model_dir = Path(model_dir)
    candidate = model_dir / "main_model"
    for path in (candidate, model_dir):
        if (path / "config.json").is_file() and (path / "model.safetensors").is_file():
            return path
    raise FileNotFoundError(f"no main model (config.json + model.safetensors) under {model_dir}")


def embed_smiles(
    smiles: Iterable[str],
    model_dir: Path | str,
    *,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    coord_mode: str = "auto",
    device: str = "cpu",
) -> np.ndarray:
    """Return ``(n, encoder_embed_dim)`` embeddings, NaN for bad SMILES."""
    smiles = list(smiles)
    model = load_main_model(_resolve_main_dir(model_dir), device=device)
    dim = int(model.configs.encoder_embed_dim)
    embeds = np.full((len(smiles), dim), np.nan, dtype=np.float64)
    if not smiles:
        return embeds

    items, _metas = featurize(
        smiles, workers=workers, timeout=timeout, coord_mode=coord_mode
    )
    valid = [(position, item) for position, item in enumerate(items) if item is not None]
    for chunk in _chunks(valid, batch_size):
        batch = unimol_collate_fn([item for _, item in chunk])
        batch = {key: value.to(device) for key, value in batch.items()}
        _probs, batch_embeds = model.predict(**batch)
        batch_embeds = batch_embeds.detach().cpu().numpy()
        for row, (position, _item) in enumerate(chunk):
            embeds[position] = batch_embeds[row]
    dropped = int(np.isnan(embeds).any(axis=1).sum())
    if dropped:
        typer.echo(f"warning: dropped {dropped} unfeaturisable SMILES", err=True)
    return embeds


def build_table(
    input_path: Path | str,
    model_dir: Path | str,
    *,
    smiles_column: str = "smiles",
    label_map_path: Optional[Path | str] = None,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    coord_mode: str = "auto",
    device: str = "cpu",
    expected_labels: Optional[int] = None,
    embeds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Embed every row of ``input_path`` and return the vice training table."""
    input_path = Path(input_path)
    frame = pd.read_parquet(input_path)
    if smiles_column not in frame.columns:
        raise ValueError(f"{input_path} is missing the SMILES column {smiles_column!r}")
    columns = label_columns(frame)
    if not columns:
        raise ValueError(f"no label_* columns found in {input_path}")
    if expected_labels is not None and len(columns) != int(expected_labels):
        raise ValueError(f"{input_path} has {len(columns)} labels, expected {expected_labels}")

    label_map_path = (
        Path(label_map_path) if label_map_path else Path(model_dir) / "label_map.json"
    )
    if label_map_path.is_file():
        names = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
        if len(names) != len(columns):
            typer.echo(
                f"warning: {label_map_path} has {len(names)} labels but the table has "
                f"{len(columns)}",
                err=True,
            )

    if embeds is None:
        embeds = embed_smiles(
            frame[smiles_column].tolist(),
            model_dir,
            workers=workers,
            batch_size=batch_size,
            timeout=timeout,
            coord_mode=coord_mode,
            device=device,
        )
    embeds = np.asarray(embeds)
    keep = ~np.isnan(embeds).any(axis=1)
    if not keep.all():
        typer.echo(f"warning: keeping {int(keep.sum())}/{len(frame)} featurised rows", err=True)
    selected = frame.loc[keep].reset_index(drop=True)

    table = pd.DataFrame(
        embeds[keep].astype(np.float32),
        columns=[f"vec_{index}" for index in range(embeds.shape[1])],
    )
    for column in columns:
        table[column] = selected[column].astype(int).to_numpy()
    if CLUSTER_COLUMN in selected.columns:
        table[CLUSTER_COLUMN] = selected[CLUSTER_COLUMN].to_numpy()
    else:
        table[CLUSTER_COLUMN] = 0
    if WEIGHT_COLUMN in selected.columns:
        table[WEIGHT_COLUMN] = selected[WEIGHT_COLUMN].to_numpy()
    return table


def prepare(
    input_path: Path | str,
    model_dir: Path | str,
    output_path: Path | str,
    **kwargs,
) -> pd.DataFrame:
    """Build the table and write it as parquet; returns the frame."""
    table = build_table(input_path, model_dir, **kwargs)
    vectors = vector_columns(table)
    if not vectors:
        raise ValueError("embedding computation produced no vec_* columns")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    return table


@app.command()
def main(
    input: Path = typer.Option(..., "--input", help="labels.parquet with smiles + label_*."),
    model_dir: Path = typer.Option(..., "--model-dir", help="Model root or main_model/ directory."),
    output: Path = typer.Option(..., "--output", help="Where to write the vice training parquet."),
    smiles_column: str = typer.Option("smiles", "--smiles-column"),
    label_map: Optional[Path] = typer.Option(
        None, "--label-map", help="label_map.json; defaults to <model-dir>/label_map.json."
    ),
    workers: int = typer.Option(DEFAULT_WORKERS, "--workers", min=1, help="Featurisation threads."),
    batch_size: int = typer.Option(DEFAULT_BATCH_SIZE, "--batch-size", min=1, help="Model batch size."),
    timeout: float = typer.Option(DEFAULT_TIMEOUT, "--timeout", help="Per-molecule seconds; 0 disables."),
    coord_mode: str = typer.Option("auto", "--coord-mode", help="auto / 3d / 2d."),
    device: str = typer.Option("cpu", "--device", help="Torch device for embedding."),
    expected_labels: Optional[int] = typer.Option(None, "--expected-labels"),
):
    table = prepare(
        input,
        model_dir,
        output,
        smiles_column=smiles_column,
        label_map_path=label_map,
        workers=workers,
        batch_size=batch_size,
        timeout=None if timeout <= 0 else timeout,
        coord_mode=coord_mode,
        device=device,
        expected_labels=expected_labels,
    )
    typer.echo(
        f"wrote {output} ({len(table)} rows, {len(vector_columns(table))} embedding dims, "
        f"{len(label_columns(table))} labels)"
    )


if __name__ == "__main__":
    app()