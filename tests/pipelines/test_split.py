"""Batch 2: the multi-label greedy split."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipelines import write_table
from pipelines.hmdb.split import (
    compute_stats,
    greedy_split,
    label_columns,
    run,
)

LABELS = ["label_0", "label_1", "label_2"]


def test_greedy_split_exact_fixture(multi_hot_frame):
    splits = greedy_split(multi_hot_frame, LABELS)

    assert splits == {
        "train": ["HMDB00005", "HMDB00006", "HMDB00008", "HMDB00011", "HMDB00012"],
        "vali": ["HMDB00003", "HMDB00004", "HMDB00010"],
        "test": ["HMDB00001", "HMDB00002", "HMDB00007", "HMDB00009"],
    }


def test_split_is_a_partition(multi_hot_frame):
    splits = greedy_split(multi_hot_frame, LABELS)
    groups = [set(values) for values in splits.values()]
    all_ids = set(multi_hot_frame["id"])

    assert sum(len(group) for group in groups) == len(all_ids)
    assert set().union(*groups) == all_ids
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])


def test_each_label_reaches_about_twenty_percent(multi_hot_frame):
    splits = greedy_split(multi_hot_frame, LABELS)
    for column in LABELS:
        members = set(multi_hot_frame.loc[multi_hot_frame[column] == 1, "id"])
        for name in ("test", "vali"):
            covered = len(members & set(splits[name]))
            assert covered / len(members) >= 0.2 - 1e-9


def test_split_is_deterministic(multi_hot_frame):
    first = greedy_split(multi_hot_frame, LABELS)
    second = greedy_split(multi_hot_frame, LABELS)
    assert first == second


def test_unlabelled_rows_fall_into_train(multi_hot_frame):
    extra = multi_hot_frame.iloc[[0]].copy()
    extra["id"] = "UNLABELLED"
    extra[LABELS] = 0
    frame = pd.concat([multi_hot_frame, extra], ignore_index=True)

    splits = greedy_split(frame, LABELS)
    assert "UNLABELLED" in splits["train"]
    assert len(set().union(*(set(v) for v in splits.values()))) == len(frame)


def test_duplicate_ids_are_rejected(multi_hot_frame):
    frame = multi_hot_frame.copy()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="duplicate ids"):
        greedy_split(frame, LABELS)


def test_label_columns_are_ordered_by_suffix():
    frame = pd.DataFrame(columns=["id", "label_10", "label_2", "label_1", "other"])
    assert label_columns(frame, "label_") == ["label_1", "label_2", "label_10"]


def test_stats_report_sizes_and_coverage(multi_hot_frame):
    splits = greedy_split(multi_hot_frame, LABELS)
    stats = compute_stats(multi_hot_frame, splits, LABELS)

    assert stats["total"] == 12
    assert sum(stats["counts"].values()) == 12
    assert stats["labels_total"] == 3
    assert stats["labels_without_samples"] == 0
    assert stats["rows_without_labels"] == 0
    assert stats["coverage"]["label_0"] == {
        "total": 6,
        "train": 2,
        "vali": 2,
        "test": 2,
    }
    assert stats["coverage"]["label_1"]["train"] == 2


def test_run_writes_splits_and_stats(tmp_path, multi_hot_frame):
    input_path = tmp_path / "labels.parquet"
    write_table(multi_hot_frame, input_path)
    output = tmp_path / "splits.json"
    stats_output = tmp_path / "stats.json"

    splits, stats = run(
        input_path=input_path,
        output_path=output,
        stats_output=stats_output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == splits
    assert set(splits) == {"train", "vali", "test"}
    on_disk = json.loads(stats_output.read_text(encoding="utf-8"))
    assert on_disk["counts"] == stats["counts"]
    assert on_disk["params"]["seed"] == 42
