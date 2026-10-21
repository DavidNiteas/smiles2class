"""Training dataset and datamodule for the main model.

Ported from ``uni_classify.data_tools.dataset`` (the HDF5 helpers are dropped;
``prepare.py`` writes the pickled datamodule this code reads back).
"""

from __future__ import annotations

import pickle
from typing import Dict, Hashable, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from unimol_lite.data.collate_fn import unimol_collate_fn
from unimol_lite.model import UniMolTokenizer


class UniClassifyTrainDataset(Dataset):
    """Per-molecule Uni-Mol item tensors plus a multi-hot ``label`` row."""

    def __init__(
        self,
        dataframe: Optional[pd.DataFrame] = None,
        index: Optional[List[Hashable]] = None,
        src_tokens: Optional[List[torch.Tensor]] = None,
        src_distance: Optional[List[torch.Tensor]] = None,
        src_coord: Optional[List[torch.Tensor]] = None,
        src_edge_type: Optional[List[torch.Tensor]] = None,
        label: Optional[List[torch.Tensor]] = None,
        tokenizer: UniMolTokenizer = UniMolTokenizer(),
    ):
        if isinstance(dataframe, pd.DataFrame):
            self.datas = dataframe
        else:
            self.datas = pd.DataFrame(
                {
                    "src_tokens": src_tokens if src_tokens is not None else [],
                    "src_distance": src_distance if src_distance is not None else [],
                    "src_coord": src_coord if src_coord is not None else [],
                    "src_edge_type": src_edge_type if src_edge_type is not None else [],
                    "label": label if label is not None else [],
                },
                index=index,
            )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.Datas)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        data = self.Datas.iloc[index]
        data = data.to_dict()
        for key, value in data.items():
            if not isinstance(value, torch.Tensor):
                data[key] = torch.tensor(value)
        return data

    @property
    def Datas(self) -> pd.DataFrame:
        return self.datas

    @Datas.setter
    def Datas(self, dataframe: pd.DataFrame):
        self.datas = dataframe

    @property
    def DataCols(self) -> pd.Index:
        return self.Datas.columns

    @property
    def LabelTensor(self) -> torch.Tensor:
        label_list = self.Datas["label"].tolist()
        tensor_list = []
        for label in label_list:
            if not isinstance(label, torch.Tensor):
                tensor_list.append(torch.tensor(label))
            else:
                tensor_list.append(label)
        return torch.stack(tensor_list, dim=0)

    @classmethod
    def from_pickle(cls, data_path: str) -> "UniClassifyTrainDataset":
        with open(data_path, "rb") as handle:
            return pickle.load(handle)

    def save_pickle(self, data_path: str):
        with open(data_path, "wb") as handle:
            pickle.dump(self, handle)

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        new_batch = unimol_collate_fn(batch, self.tokenizer)
        new_batch["label"] = torch.stack([data["label"] for data in batch], dim=0)
        return new_batch


class UniClassifyDataModule(pl.LightningDataModule):
    """Pickle-friendly container for the train/vali/test datasets."""

    default_configs = {
        "batch_size": 32,
        "num_workers": 0,
    }

    def __init__(
        self,
        trainset: UniClassifyTrainDataset,
        valiset: UniClassifyTrainDataset,
        testset: UniClassifyTrainDataset,
        configs: Optional[pd.Series] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()
        self.trainset = trainset
        self.valiset = valiset
        self.testset = testset
        if isinstance(configs, pd.Series):
            self.configs = configs
        else:
            self.configs = pd.Series(self.default_configs)
        if isinstance(batch_size, int):
            self.BatchSize = batch_size
        if isinstance(num_workers, int):
            self.NumWorkers = num_workers
        self.configs.update(kwargs)
        # Ordered label names (``label_map.json`` content); written out on export.
        self.labels = labels

    @property
    def BatchSize(self) -> int:
        return int(self.configs["batch_size"])

    @BatchSize.setter
    def BatchSize(self, batch_size: int):
        self.configs["batch_size"] = batch_size

    @property
    def NumWorkers(self) -> int:
        return int(self.configs["num_workers"])

    @NumWorkers.setter
    def NumWorkers(self, num_workers: int):
        self.configs["num_workers"] = num_workers

    @classmethod
    def from_pickle(cls, data_path: str) -> "UniClassifyDataModule":
        with open(data_path, "rb") as handle:
            return pickle.load(handle)

    def save_pickle(self, data_path: str):
        with open(data_path, "wb") as handle:
            pickle.dump(self, handle)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset,
            batch_size=self.BatchSize,
            num_workers=self.NumWorkers,
            collate_fn=self.trainset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valiset,
            batch_size=self.BatchSize,
            num_workers=self.NumWorkers,
            collate_fn=self.valiset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset,
            batch_size=self.BatchSize,
            num_workers=self.NumWorkers,
            collate_fn=self.testset.collate_fn,
        )
