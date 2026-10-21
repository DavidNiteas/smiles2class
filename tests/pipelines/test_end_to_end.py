"""End-to-end: fixture -> clean -> labels -> split, plus CLI entry points."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from pipelines.hmdb import clean, labels, split


def test_end_to_end(tmp_path, hmdb_raw, lipidmaps_raw, authority):
    clean_output = tmp_path / "compounds.parquet"
    clean.run(
        input_path=hmdb_raw,
        lipidmaps_input=lipidmaps_raw,
        output_path=clean_output,
        keep_alternative_parents=True,
    )
    cleaned = pd.read_parquet(clean_output)
    assert len(cleaned) == 14
    assert "Alternative Parents" in cleaned.columns

    label_map = tmp_path / "label_map.json"
    labels_output = tmp_path / "labels.parquet"
    with pytest.warns(UserWarning, match="never appear"):
        label_table, label_names = labels.run(
            input_path=clean_output,
            output_path=label_map,
            labels_output=labels_output,
        )
    assert label_names == authority
    assert json.loads(label_map.read_text(encoding="utf-8")) == authority
    assert len(label_table) == 14
    assert len(pd.read_parquet(labels_output)) == 14

    splits_path = tmp_path / "splits.json"
    stats_path = tmp_path / "split_stats.json"
    splits, stats = split.run(
        input_path=labels_output,
        output_path=splits_path,
        stats_output=stats_path,
    )

    ids = set(pd.read_parquet(labels_output)["id"])
    groups = [set(values) for values in splits.values()]
    assert set().union(*groups) == ids
    assert sum(len(group) for group in groups) == len(ids)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    assert json.loads(splits_path.read_text(encoding="utf-8")) == splits
    assert stats["total"] == len(ids) == 14
    assert stats["rows_without_labels"] == 2  # the two LIPID MAPS rows
    assert stats["counts"] == {name: len(values) for name, values in splits.items()}


@pytest.mark.parametrize(
    "module", ["pipelines.hmdb.clean", "pipelines.hmdb.labels", "pipelines.hmdb.split"]
)
def test_module_help(module, repo_root):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
