"""Smoke tests for the migrated vice-model training code.

Two synthetic labels over 64 rows of 8-dimensional "embeddings"; the vice
sampler splits them, AutoGluon fits one predictor per label, and the result is
loaded back with ``ViceModel``.  No full 426-class run here.

Measured on the dev host: ~13 s for the 2-class smoke (well under the 60 s
budget), so the suite is not marked ``slow``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from smiles2class.vice import ViceModel
from training.vice.sampling import WEIGHT_COLUMN, SingleClassifyDataset
from training.vice.train import train_vice

N_ROWS = 64
N_DIMS = 8
LABELS = ["class_a", "class_b"]
PRESETS = ["medium_quality"]
TIME_LIMIT = 5
NUM_CPUS = 2


def synthetic_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(size=(N_ROWS, N_DIMS)).astype(np.float32),
        columns=[f"vec_{index}" for index in range(N_DIMS)],
    )
    # Balanced positives/negatives so both halves of every split see both classes.
    frame["label_0"] = [index % 2 for index in range(N_ROWS)]
    frame["label_1"] = [1 if index < N_ROWS // 2 else 0 for index in range(N_ROWS)]
    frame["cluster_id"] = 0
    return frame


def test_split_keeps_both_classes_and_adds_weights():
    dataset = SingleClassifyDataset(synthetic_table())
    assert len(dataset) == len(LABELS)
    train_set, test_set = dataset.train_test_split(dataset[0])

    for split in (train_set, test_set):
        assert set(split["label"].unique()) == {0, 1}
        assert WEIGHT_COLUMN in split.columns
        # Positives and negatives each carry half of the total weight.
        totals = split.groupby("label")[WEIGHT_COLUMN].sum()
        assert totals[0] == pytest.approx(0.5)
        assert totals[1] == pytest.approx(0.5)


def test_split_is_deterministic():
    dataset = SingleClassifyDataset(synthetic_table())
    first = dataset.train_test_split(dataset[1])
    second = dataset.train_test_split(dataset[1])
    for left, right in zip(first, second):
        assert left.index.tolist() == right.index.tolist()


# One AutoGluon fit per label; shared by the layout/reload assertions below.
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    output = tmp_path_factory.mktemp("vice_model")
    frame = synthetic_table()
    train_vice(
        frame,
        output,
        label_map=LABELS,
        max_classes=len(LABELS),
        presets=PRESETS,
        time_limit=TIME_LIMIT,
        num_cpus=NUM_CPUS,
        seed=42,
    )
    return output, frame


def test_trained_layout_and_label_map(trained):
    output, _frame = trained
    assert json.loads((output / "label_map.json").read_text(encoding="utf-8")) == LABELS
    for index in range(len(LABELS)):
        assert (output / str(index) / "predictor.pkl").is_file()


def test_vice_model_reloads_and_predicts(trained):
    output, frame = trained
    vice = ViceModel.from_pretrained(output, output / "label_map.json")
    embeds = frame[[f"vec_{index}" for index in range(N_DIMS)]].to_numpy()
    probs = vice.predict(embeds)

    assert probs.shape == (len(frame), len(LABELS))
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()