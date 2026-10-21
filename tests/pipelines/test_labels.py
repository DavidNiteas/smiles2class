"""Batch 2: Alternative Parents parsing and the 426-label multi-hot table."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pipelines.hmdb.labels import (
    build_multi_hot,
    load_authority_labels,
    parse_alternative_parents,
    run,
)


@pytest.mark.parametrize("form", ["python", "json", "semicolon", "pipe"])
def test_parse_supported_forms(form, label_a, label_b):
    value = {
        "python": repr([label_a, label_b]),
        "json": json.dumps([label_a, label_b]),
        "semicolon": f"{label_a}; {label_b}",
        "pipe": f"{label_a}|{label_b}",
    }[form]
    assert parse_alternative_parents(value) == [label_a, label_b]


def test_parse_single_label_and_empty_values(label_a):
    assert parse_alternative_parents(label_a) == [label_a]
    assert parse_alternative_parents("") == []
    assert parse_alternative_parents("[]") == []
    assert parse_alternative_parents(None) == []
    assert parse_alternative_parents(float("nan")) == []


def test_parse_accepts_sequences(label_a, label_b):
    assert parse_alternative_parents([label_a, label_b]) == [label_a, label_b]
    assert parse_alternative_parents((label_a, label_b)) == [label_a, label_b]


def test_parse_comma_delimiter_uses_vocabulary():
    comma_label = "1,2-diols"
    assert parse_alternative_parents(
        "1,2-diols; Organic oxides", {"1,2-diols", "Organic oxides"}
    ) == ["1,2-diols", "Organic oxides"]
    # A single ClassyFire term containing a comma stays intact.
    assert parse_alternative_parents(comma_label, {comma_label}) == [comma_label]
    # Without a vocabulary the comma is treated as a separator.
    assert parse_alternative_parents("a, b") == ["a", "b"]


def test_multi_hot_matches_expected_matrix(multi_hot_frame, columns, label_a):
    expected_columns = ["id", "SMILES", "label_0", "label_1", "label_2"]
    assert list(multi_hot_frame.columns) == expected_columns
    expected = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ]
    assert multi_hot_frame[["label_0", "label_1", "label_2"]].to_numpy().tolist() == expected
    assert multi_hot_frame["SMILES"].iloc[0] == "CCO"
    assert multi_hot_frame["id"].iloc[0] == "HMDB00001"


def test_authority_labels_are_used_in_order(authority):
    assert load_authority_labels() == authority
    assert len(authority) == 426


def test_multi_hot_uses_authority_order(authority, label_a, label_b):
    frame = pd.DataFrame(
        {
            "id": ["m1", "m2"],
            "SMILES": ["CCO", "CCN"],
            "Alternative Parents": [[label_a], [label_b]],
        }
    )
    columns = {
        "labels": {
            "id_candidates": ["id"],
            "output_id": "id",
            "smiles": "SMILES",
            "parent": "Alternative Parents",
            "prefix": "label_",
        }
    }
    table = build_multi_hot(frame, authority, columns)

    assert len(table.columns) == 2 + 426
    assert list(table.columns[:2]) == ["id", "SMILES"]
    assert list(table.columns[2:]) == [f"label_{i}" for i in range(426)]

    values = table.drop(columns=["id", "SMILES"]).to_numpy()
    assert np.flatnonzero(values[0]).tolist() == [0]
    assert np.flatnonzero(values[1]).tolist() == [1]
    assert values.sum() == 2


def test_run_writes_label_map_and_multi_hot(tmp_path, hmdb_raw, authority):
    label_map = tmp_path / "label_map.json"
    labels_output = tmp_path / "labels.parquet"
    with pytest.warns(UserWarning, match="never appear"):
        table, labels = run(
            input_path=hmdb_raw,
            output_path=label_map,
            labels_output=labels_output,
        )

    assert json.loads(label_map.read_text(encoding="utf-8")) == authority
    assert labels == authority
    written = pd.read_parquet(labels_output)
    assert list(written.columns) == ["id", "SMILES"] + [
        f"label_{i}" for i in range(426)
    ]
    assert len(written) == 12
    assert int(written["label_0"].sum()) == 6
    assert int(written["label_1"].sum()) == 4
    assert int(written["label_2"].sum()) == 4


def test_run_defaults_labels_output_beside_label_map(tmp_path, hmdb_raw):
    label_map = tmp_path / "nested" / "label_map.json"
    with pytest.warns(UserWarning):
        run(input_path=hmdb_raw, output_path=label_map)
    assert (tmp_path / "nested" / "labels.parquet").exists()


def test_run_honours_custom_label_map(tmp_path, hmdb_raw, label_a, label_b, label_c):
    custom = tmp_path / "three.json"
    custom.write_text(
        json.dumps([label_a, label_b, label_c]), encoding="utf-8"
    )
    label_map = tmp_path / "label_map.json"
    with pytest.warns(UserWarning, match="expected 426"):
        table, labels = run(
            input_path=hmdb_raw,
            output_path=label_map,
            label_map=custom,
        )
    assert labels == [label_a, label_b, label_c]
    assert json.loads(label_map.read_text(encoding="utf-8")) == labels
    assert list(table.columns) == ["id", "SMILES", "label_0", "label_1", "label_2"]


def test_run_rejects_missing_parent_column(tmp_path):
    csv = tmp_path / "no_parents.csv"
    pd.DataFrame({"id": ["a"], "SMILES": ["CCO"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Alternative Parents"):
        run(input_path=csv, output_path=tmp_path / "label_map.json")
