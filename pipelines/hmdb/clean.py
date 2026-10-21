"""Wash HMDB, convert LIPID MAPS and merge both sources.

Behaviour matches the original compound-processing flow, but every column name
and path comes from the CLI or ``--config``.

- ``wash_hmdb_data``: add ``CANON_SMILES`` (copy of ``SMILES``), optionally drop
  ``Alternative Parents``, sort by the HMDB id;
- ``transform_lipidmaps_data``: drop rows without SMILES and map columns onto
  the unified schema;
- ``merge_hmdb_lipidmaps``: concatenate HMDB first, LIPID MAPS after.

Usage::

    python -m pipelines.hmdb.clean --input raw/hmdb.csv --output clean.parquet
    python -m pipelines.hmdb.clean --input raw/hmdb.csv --lipidmaps-input raw/lm.csv \
        --output compounds.parquet
    python -m pipelines.hmdb.clean --input raw/hmdb.csv \
        --output train.parquet --keep-alternative-parents
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import typer

from pipelines import load_config, read_table, require_columns, write_table

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"

app = typer.Typer(
    add_completion=False,
    help="Wash HMDB, convert LIPID MAPS and merge them into one table.",
)


def wash_hmdb_data(
    hmdb_df: pd.DataFrame,
    columns: Mapping[str, Any],
    keep_alternative_parents: bool = False,
) -> pd.DataFrame:
    """Return the washed HMDB table (all rows kept, including empty SMILES)."""
    config = columns["hmdb"]
    require_columns(hmdb_df, [config["id"], config["smiles"]], "HMDB")

    washed = hmdb_df.copy()
    washed[config["canon_smiles"]] = washed[config["smiles"]]

    parent = config["parent"]
    if not keep_alternative_parents and parent in washed.columns:
        washed = washed.drop(columns=[parent])

    washed = washed.sort_values(config["id"]).reset_index(drop=True)
    return washed


def transform_lipidmaps_data(
    lipidmaps_df: pd.DataFrame,
    columns: Mapping[str, Any],
) -> pd.DataFrame:
    """Map LIPID MAPS columns onto the unified schema, dropping empty SMILES."""
    source = columns["lipidmaps"]
    unified = columns["unified"]
    require_columns(
        lipidmaps_df,
        [source[key] for key in ("id", "name", "smiles", "kingdom", "super_class", "class", "sub_class")],
        "LIPID MAPS",
    )

    kept = lipidmaps_df[
        lipidmaps_df[source["smiles"]].notna()
        & lipidmaps_df[source["smiles"]].astype(str).str.strip().ne("")
    ].copy()
    rename = {source[key]: unified[key] for key in unified}
    return kept[list(rename)].rename(columns=rename).reset_index(drop=True)


def merge_hmdb_lipidmaps(
    hmdb_df: pd.DataFrame,
    lipidmaps_df: pd.DataFrame,
    columns: Mapping[str, Any],
    keep_alternative_parents: bool = False,
) -> pd.DataFrame:
    """Concatenate HMDB (first) and transformed LIPID MAPS (after)."""
    source = columns["hmdb"]
    unified = columns["unified"]
    required = [source[key] for key in ("id", "name", "smiles", "kingdom", "super_class", "class", "sub_class")]
    require_columns(hmdb_df, required, "HMDB")

    rename = {source[key]: unified[key] for key in unified}
    hmdb_unified = hmdb_df[list(rename)].rename(columns=rename).copy()

    parent = source["parent"]
    if keep_alternative_parents:
        require_columns(hmdb_df, [parent], "HMDB")
        hmdb_unified[parent] = hmdb_df[parent]

    merged = pd.concat([hmdb_unified, lipidmaps_df], ignore_index=True)
    return merged.reset_index(drop=True)


def run(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    lipidmaps_input: Path | str | None = None,
    config: Path | str | None = None,
    keep_alternative_parents: bool = False,
    hmdb_washed_output: Path | str | None = None,
) -> pd.DataFrame:
    """Execute the cleaning pipeline and return the written table."""
    settings = load_config(DEFAULT_CONFIG, config)
    clean = settings.get("clean", {}) or {}
    columns = settings["columns"]

    input_path = input_path or clean.get("input")
    output_path = output_path or clean.get("output")
    lipidmaps_input = lipidmaps_input or clean.get("lipidmaps_input")
    hmdb_washed_output = hmdb_washed_output or clean.get("hmdb_washed_output")

    if input_path is None:
        raise ValueError("missing HMDB input: pass --input or set clean.input in the config")
    if output_path is None:
        raise ValueError("missing output: pass --output or set clean.output in the config")

    washed = wash_hmdb_data(read_table(input_path), columns, keep_alternative_parents)
    if hmdb_washed_output:
        write_table(washed, hmdb_washed_output)

    if lipidmaps_input:
        lipidmaps = transform_lipidmaps_data(read_table(lipidmaps_input), columns)
        result = merge_hmdb_lipidmaps(washed, lipidmaps, columns, keep_alternative_parents)
    else:
        result = washed

    write_table(result, output_path)
    return result


@app.command()
def main(
    input: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Raw HMDB table (.csv/.parquet)."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output table (.csv/.parquet)."
    ),
    lipidmaps_input: Optional[Path] = typer.Option(
        None, "--lipidmaps-input", "-l", help="Raw LIPID MAPS table; omit to only wash HMDB."
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help=f"YAML config (defaults: {DEFAULT_CONFIG.name})."
    ),
    keep_alternative_parents: bool = typer.Option(
        False,
        "--keep-alternative-parents",
        help="Keep the Alternative Parents column (training tables need it).",
    ),
    hmdb_washed_output: Optional[Path] = typer.Option(
        None, "--hmdb-washed-output", "-w", help="Optional intermediate washed HMDB table."
    ),
) -> None:
    """Wash HMDB, optionally merge LIPID MAPS, and write the result."""
    result = run(
        input_path=input,
        output_path=output,
        lipidmaps_input=lipidmaps_input,
        config=config,
        keep_alternative_parents=keep_alternative_parents,
        hmdb_washed_output=hmdb_washed_output,
    )
    typer.echo(f"wrote {len(result)} rows, {len(result.columns)} columns")


if __name__ == "__main__":  # pragma: no cover
    app()
