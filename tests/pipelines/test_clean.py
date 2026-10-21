"""Batch 1: HMDB wash, LIPID MAPS conversion and merge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipelines import read_table, write_table
from pipelines.hmdb.clean import (
    merge_hmdb_lipidmaps,
    run,
    transform_lipidmaps_data,
    wash_hmdb_data,
)


def test_wash_drops_parents_and_sorts(hmdb_frame, columns):
    washed = wash_hmdb_data(hmdb_frame, columns)
    assert len(washed) == 12
    assert "CANON_SMILES" in washed.columns
    assert "Alternative Parents" not in washed.columns
    assert washed["CANON_SMILES"].tolist() == washed["SMILES"].tolist()
    assert washed["HMDB ID"].tolist() == sorted(washed["HMDB ID"].tolist())


def test_wash_keeps_parents_on_request(hmdb_frame, columns):
    washed = wash_hmdb_data(hmdb_frame, columns, keep_alternative_parents=True)
    assert "Alternative Parents" in washed.columns
    assert len(washed) == 12


def test_transform_lipidmaps_drops_empty_smiles(lipidmaps_frame, columns):
    transformed = transform_lipidmaps_data(lipidmaps_frame, columns)
    assert list(transformed.columns) == list(columns["unified"].values())
    assert len(transformed) == 2
    assert transformed["ID"].tolist() == ["LM0001", "LM0002"]
    assert transformed.loc[0, "Kingdom"] == "Lipids"
    assert transformed.loc[0, "Super Class"] == "Fatty Acyls"
    assert transformed.loc[1, "Sub Class"] == "Triacylglycerols"


def test_merge_hmdb_first_and_unified_schema(hmdb_frame, lipidmaps_frame, columns):
    washed = wash_hmdb_data(hmdb_frame, columns)
    lipidmaps = transform_lipidmaps_data(lipidmaps_frame, columns)
    merged = merge_hmdb_lipidmaps(washed, lipidmaps, columns)

    assert list(merged.columns) == list(columns["unified"].values())
    assert len(merged) == 14
    assert merged["ID"].iloc[0] == "HMDB00001"
    assert merged["ID"].iloc[-1] == "LM0002"
    assert merged["ID"].iloc[:12].tolist() == sorted(merged["ID"].iloc[:12].tolist())


def test_merge_can_keep_parents(hmdb_frame, lipidmaps_frame, columns):
    washed = wash_hmdb_data(hmdb_frame, columns, keep_alternative_parents=True)
    lipidmaps = transform_lipidmaps_data(lipidmaps_frame, columns)
    merged = merge_hmdb_lipidmaps(
        washed, lipidmaps, columns, keep_alternative_parents=True
    )

    assert list(merged.columns) == list(columns["unified"].values()) + [
        "Alternative Parents"
    ]
    assert merged["Alternative Parents"].notna().sum() == 12
    # LIPID MAPS rows have no Alternative Parents.
    assert pd.isna(merged["Alternative Parents"].iloc[-1])


def test_run_writes_both_suffixes(tmp_path, hmdb_raw, lipidmaps_raw):
    for suffix in (".csv", ".parquet"):
        output = tmp_path / f"compounds{suffix}"
        result = run(
            input_path=hmdb_raw,
            lipidmaps_input=lipidmaps_raw,
            output_path=output,
        )
        assert output.exists()
        assert len(read_table(output)) == len(result) == 14
        assert "Alternative Parents" not in result.columns


def test_run_keep_parents_roundtrip(tmp_path, hmdb_raw):
    output = tmp_path / "train.parquet"
    run(
        input_path=hmdb_raw,
        output_path=output,
        keep_alternative_parents=True,
    )
    table = read_table(output)
    assert len(table) == 12
    assert "Alternative Parents" in table.columns
    assert "CANON_SMILES" in table.columns


def test_run_uses_config_column_mapping(tmp_path, hmdb_frame):
    raw = tmp_path / "renamed.csv"
    hmdb_frame.rename(
        columns={
            "HMDB ID": "hmdb_id",
            "SMILES": "smiles",
            "IUPAC Name": "name",
            "Kingdom": "kingdom",
            "Super Class": "super_class",
            "Class": "klass",
            "Sub Class": "sub_class",
            "Alternative Parents": "parents",
        }
    ).to_csv(raw, index=False)

    config = {
        "columns": {
            "hmdb": {
                "id": "hmdb_id",
                "smiles": "smiles",
                "name": "name",
                "parent": "parents",
                "kingdom": "kingdom",
                "super_class": "super_class",
                "class": "klass",
                "sub_class": "sub_class",
            }
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output = tmp_path / "out.csv"
    run(
        input_path=raw,
        output_path=output,
        config=str(config_path),
        keep_alternative_parents=True,
    )
    table = pd.read_csv(output)
    assert len(table) == 12
    assert "parents" in table.columns


def test_write_table_rejects_unknown_suffix(tmp_path, hmdb_frame):
    with pytest.raises(ValueError, match="unsupported output format"):
        write_table(hmdb_frame, Path(tmp_path) / "table.xlsx")
