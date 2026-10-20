"""Command line entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .classifier import MainClassifier, ViceClassifier

app = typer.Typer(help="SMILES → molecular class probabilities")


@app.command()
def predict(
    model_path: Path = typer.Option(..., "--model-path", "-m", help="Model directory"),
    input_csv: Path = typer.Option(..., "--input-csv", "-i", help="Input CSV"),
    output: Path = typer.Option(..., "--output", "-o", help="Output CSV"),
    model: str = typer.Option("main", "--model", help="main | vice"),
    smiles_column: str = typer.Option("smiles", "--smiles-column", "-s"),
    workers: int = typer.Option(8, "--workers", help="Featurization workers"),
    timeout: float = typer.Option(30.0, "--timeout", help="Per-conformer timeout (s)"),
    device: str = typer.Option("cpu", "--device", help="cpu | cuda"),
    batch_size: int = typer.Option(64, "--batch-size"),
    vice_workers: int = typer.Option(1, "--vice-workers"),
    vice_load_mode: str = typer.Option(
        "lazy", "--vice-load-mode", help="lazy | eager (eager preloads 426 predictors)"
    ),
):
    """Predict per-class probabilities for every row of a CSV."""
    if model not in ("main", "vice"):
        raise typer.BadParameter("--model must be 'main' or 'vice'")
    frame = pd.read_csv(input_csv)
    if smiles_column not in frame.columns:
        raise typer.BadParameter(f"column {smiles_column!r} not found")
    smiles = frame[smiles_column].astype(object).tolist()

    if model == "main":
        clf = MainClassifier.from_pretrained(model_path, device=device)
        probs, metas = clf.predict(
            smiles,
            workers=workers,
            timeout=timeout,
            batch_size=batch_size,
            return_meta=True,
        )
    else:
        clf = ViceClassifier.from_pretrained(
            model_path, device=device, load_mode=vice_load_mode
        )
        probs, metas = clf.predict(
            smiles,
            workers=workers,
            timeout=timeout,
            batch_size=batch_size,
            vice_workers=vice_workers,
            return_meta=True,
        )

    out = probs.copy()
    out.insert(0, "smiles", smiles)
    out.insert(1, "status", [meta["status"] for meta in metas])
    out.to_csv(output, index=False)
    typer.echo(f"wrote {output} ({len(out)} rows)")


if __name__ == "__main__":
    app()
