"""Per-class sampling for the vice model.

Ported from ``dev/v1/training_single_models.py``.  A vice training table holds
one row per molecule with ``vec_0..vec_N`` (main-model embeddings),
``label_0..label_M`` (binary multi-hot) and ``cluster_id``.

The split keeps the original behaviour: split the positive and negative rows
separately (so both sides are represented in train/test) and, when the class
is heavily imbalanced, downsample the majority side cluster-by-cluster down to
``max_magnification`` times the minority side.

When a table arrives without a ``balance_weight`` column, every split gets
per-row weights so that positives and negatives each carry half of the total
weight (inverse-frequency weighting, reused as AutoGluon ``sample_weight``).
"""

from __future__ import annotations

import random
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_PREFIX = "label_"
VECTOR_PREFIX = "vec_"
CLUSTER_COLUMN = "cluster_id"
WEIGHT_COLUMN = "balance_weight"

__all__ = [
    "CLUSTER_COLUMN",
    "LABEL_PREFIX",
    "VECTOR_PREFIX",
    "WEIGHT_COLUMN",
    "SingleClassifyDataset",
    "label_columns",
    "vector_columns",
]


def label_columns(frame: pd.DataFrame, prefix: str = LABEL_PREFIX) -> list[str]:
    """Return ``<prefix><n>`` columns ordered by their numeric suffix."""
    found = [
        column
        for column in frame.columns
        if column.startswith(prefix) and column[len(prefix) :].isdigit()
    ]
    return sorted(found, key=lambda column: int(column[len(prefix) :]))


def vector_columns(frame: pd.DataFrame, prefix: str = VECTOR_PREFIX) -> list[str]:
    """Return ``<prefix><n>`` embedding columns ordered by their numeric suffix."""
    found = [
        column
        for column in frame.columns
        if column.startswith(prefix) and column[len(prefix) :].isdigit()
    ]
    return sorted(found, key=lambda column: int(column[len(prefix) :]))


def add_balance_weight(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with ``balance_weight`` giving each class half the weight.

    Rows already carrying weights are returned unchanged.  The function never
    mutates its input.
    """
    if WEIGHT_COLUMN in frame.columns or frame.empty:
        return frame
    if "label" not in frame.columns:
        raise ValueError("frame must have a 'label' column to compute balance weights")
    counts = frame["label"].value_counts()
    weights = frame["label"].map(lambda value: 0.5 / counts[value])
    weighted = frame.copy()
    weighted[WEIGHT_COLUMN] = weights.astype("float64")
    return weighted


class SingleClassifyDataset:
    """A multi-label table viewed as one binary dataset per label column."""

    def __init__(self, datas: pd.DataFrame):
        self.datas = datas
        self.label_cols = label_columns(datas)
        self.vec_cols = vector_columns(datas)
        if not self.label_cols:
            raise ValueError("no label_* columns found")
        if not self.vec_cols:
            raise ValueError("no vec_* columns found")

    def __len__(self) -> int:
        return len(self.label_cols)

    def __getitem__(self, idx: int) -> pd.DataFrame:
        """Return ``vec_*`` + ``label`` (+ ``balance_weight`` when available)."""
        label_col = self.label_cols[idx]
        columns = self.vec_cols + [label_col]
        weighted = WEIGHT_COLUMN in self.datas.columns
        if weighted:
            columns.append(WEIGHT_COLUMN)
        df = self.datas[columns].copy()
        df.columns = self.vec_cols + ["label"] + ([WEIGHT_COLUMN] if weighted else [])
        return df

    @property
    def vectors(self) -> pd.DataFrame:
        return self.datas[self.vec_cols]

    @property
    def cluster_ids(self) -> pd.Series:
        if CLUSTER_COLUMN in self.datas.columns:
            return self.datas[CLUSTER_COLUMN]
        return pd.Series(0, index=self.datas.index, name=CLUSTER_COLUMN)

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.5,
        max_magnification: float = 9,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split one per-class frame into ``(train, test)``.

        Positives and negatives are split independently.  If the imbalance
        ratio exceeds ``max_magnification``, the majority side is downsampled
        cluster-by-cluster first.  Missing ``balance_weight`` columns are
        filled in on both returned splits.
        """
        if "label" not in df.columns:
            raise ValueError("frame must have a 'label' column")
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        if df_0.empty or df_1.empty:
            raise ValueError("frame must contain both label 0 and label 1 rows")

        magnification = len(df_0) / len(df_1)
        if magnification < 1:
            magnification = 1 / magnification
            down_set = df_0
            up_set = df_1
        else:
            down_set = df_1
            up_set = df_0

        if magnification > max_magnification:
            rng = random.Random(random_state)
            selected_idx_list: list = []
            cluster_ids = self.cluster_ids.loc[up_set.index]
            unique_cluster_ids = cluster_ids.unique()
            while len(selected_idx_list) / len(down_set) < max_magnification:
                if len(cluster_ids) == 0:
                    break
                for cluster_id in unique_cluster_ids:
                    tag_idx = cluster_ids[cluster_ids == cluster_id]
                    if len(tag_idx) > 0:
                        selected_idx = rng.choice(list(tag_idx.index))
                        selected_idx_list.append(selected_idx)
                        cluster_ids = cluster_ids.drop(selected_idx)
            up_set = up_set.loc[selected_idx_list]

        down_set_train, down_set_test = train_test_split(
            down_set, test_size=test_size, random_state=random_state
        )
        up_set_train, up_set_test = train_test_split(
            up_set, test_size=test_size, random_state=random_state
        )
        train_set = pd.concat([down_set_train, up_set_train], axis=0)
        test_set = pd.concat([down_set_test, up_set_test], axis=0)
        if WEIGHT_COLUMN not in df.columns:
            train_set = add_balance_weight(train_set)
            test_set = add_balance_weight(test_set)
        return train_set, test_set